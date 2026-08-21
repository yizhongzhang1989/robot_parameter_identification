#!/usr/bin/env python3
"""Sweep joint one at a series of gravity loads, and record what it takes.

Joint one's gravity torque goes as sin(j1) x sin(j2), which has two
consequences that shape this whole experiment.

The first is that joint two sets the level: at j1 = 0 the torque is 0.17 Nm
whatever joint two does, so the load cannot be dialled in from downstream
alone. The second is more awkward. Sweeping joint one moves joint one, so it
moves its own load: a seventy degree pass centred on zero takes the torque from
2.9 Nm down to 0.2 and back up to 2.7, which is the joint's entire range inside
a single pass. "One sweep at one load" is not something that exists there.

It does exist at ninety degrees, where the sine is stationary. Centred there a
ten degree pass holds the load to 0.8 per cent, twenty degrees to 2.2, forty to
7.5. That is the trade this experiment cannot escape: a pass needs room to
accelerate, cruise and stop, so holding the load still costs top speed. Rather
than pick one compromise, the arc is sized per speed as the campaign sizes it,
the resulting drift is computed for every pass, and passes whose drift exceeds
``--drift-limit`` are dropped and named. The friction minimum this is meant to
resolve sits between 3 and 13 deg/s, well inside what a steady load allows.

Every pose and every arc is screened against the collision scene before the arm
moves, and the screen is proved able to reject before it is believed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from robot_parameter_identification import autoprofile, campaign
from robot_parameter_identification import identification as ident
from robot_parameter_identification.interfaces import MotionFailed
from robot_parameter_identification.obstacles import ObstacleScene
from robot_parameter_identification.plants.ros_control import (
    HardwareConfig, HardwarePlant)

JOINT = 0
PARTNER = 1
# Where joint one's load is stationary in its own angle, so a pass about this
# point does not move the thing it is trying to hold still.
STATIONARY_DEG = 90.0


def read_urdf(path):
    """From a file if given, else from /robot_description."""
    if path:
        return Path(path).read_text(encoding="utf-8")
    import rclpy  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rclpy.qos import (  # noqa: PLC0415
        DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
    from std_msgs.msg import String  # noqa: PLC0415

    rclpy.init()
    try:
        node = Node("j1_load_sweep_description")
        seen: list[str] = []
        # The description is published once and latched, so a plain
        # subscription joins too late and waits for a message that never comes.
        node.create_subscription(
            String, "/robot_description", lambda m: seen.append(m.data),
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        deadline = time.monotonic() + 10.0
        while not seen and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    if not seen:
        raise RuntimeError("/robot_description did not arrive in 10 s")
    return seen[0]


def build_scene(arm, urdf, obstacles):
    scene = ObstacleScene(arm.model, urdf_text=urdf)
    if obstacles and Path(obstacles).is_file():
        payload = json.loads(Path(obstacles).read_text(encoding="utf-8"))
        scene.replace_all(payload if isinstance(payload, list)
                          else payload.get("obstacles", []))
    report = scene.geometry_report()
    if not report.get("self_collision_checked"):
        raise RuntimeError(
            "the collision scene cannot see the arm "
            f"({report.get('robot_geometry_error') or 'no link shapes'}); "
            "source the workspace so package:// mesh paths resolve")
    # A screen that has never said no is indistinguishable from one that
    # cannot, and this experiment drives the arm far from home.
    folded = np.full(arm.joint_count, 150.0)
    if scene.collision_free(folded):
        raise RuntimeError("the collision screen passed a folded pose; "
                           "it is not actually checking anything")
    return scene


def load_of(arm: ident.ArmModel, one: float, two: float) -> float:
    pose = np.zeros(arm.joint_count)
    pose[JOINT], pose[PARTNER] = one, two
    return abs(float(arm.joint_loads(pose)[JOINT][JOINT]))


def design_levels(arm: ident.ArmModel, count: int, centre: float) -> list[dict]:
    """Loads evenly spaced from the lightest pose to the heaviest, set by j2."""
    lightest = load_of(arm, 0.0, 0.0)
    heaviest = load_of(arm, centre, 90.0)
    grid = np.linspace(0.0, 90.0, 1801)
    reachable = np.array([load_of(arm, centre, two) for two in grid])
    levels = []
    for wanted in np.linspace(lightest, heaviest, count):
        two = float(grid[int(np.argmin(np.abs(reachable - wanted)))])
        levels.append({"wanted_nm": round(float(wanted), 4),
                       "partner_deg": round(two, 3),
                       "load_nm": round(load_of(arm, centre, two), 4)})
    return levels


def arc_drift(arm: ident.ArmModel, centre: float, two: float,
              arc: float) -> tuple[float, float]:
    """Mean load over a pass and how far it moves across it."""
    values = [load_of(arm, centre + offset, two)
              for offset in np.linspace(-arc / 2.0, arc / 2.0, 21)]
    return float(np.mean(values)), float(max(values) - min(values))


def pose_at(arm: ident.ArmModel, one: float, two: float) -> np.ndarray:
    pose = np.zeros(arm.joint_count)
    pose[JOINT], pose[PARTNER] = one, two
    return pose


def screen(arm, scene, centre: float, two: float, arc: float,
           steps: int = 61) -> bool:
    """The whole pass, not just its ends."""
    return all(
        scene.collision_free(pose_at(arm, centre + offset, two))
        for offset in np.linspace(-arc / 2.0, arc / 2.0, steps))


def plan_passes(arm, scene, levels, speeds, centre, drift_share, ceiling):
    """Every pass to drive, with its arc and the load it will actually hold.

    One arc cap for all ten levels, not one per level. The experiment exists to
    compare curves across loads, and a level that reached a higher speed than
    its neighbour would differ from it for that reason as much as for any
    physical one.

    Drift is judged against the gap between levels rather than against each
    level's own size. The absolute swing is nearly the same at every level --
    0.06 Nm across a twenty degree arc, whether the joint carries 0.17 Nm or
    4.89 -- so a percentage test is vacuous at the top and impossible at the
    bottom, where it left the lightest level with five passes and a top speed
    of 1.4 deg/s.
    """
    loads = [level["load_nm"] for level in levels]
    spacing = ((max(loads) - min(loads)) / (len(loads) - 1)
               if len(loads) > 1 else max(loads))
    allowed = drift_share * spacing

    def worst(arc: float) -> float:
        return max(arc_drift(arm, centre, level["partner_deg"], arc)[1]
                   for level in levels)

    low, high = 1.0, ceiling
    if worst(high) > allowed:
        for _ in range(40):
            middle = 0.5 * (low + high)
            if worst(middle) > allowed:
                high = middle
            else:
                low = middle
        cap = low
    else:
        cap = ceiling

    passes, refused = [], []
    kept_speeds = []
    for speed in speeds:
        arc = campaign.pass_amplitude_deg(speed, ceiling)
        if arc > cap:
            refused.append({"speed_deg_s": speed, "arc_deg": round(arc, 2),
                            "why": "needs a longer arc than the load allows"})
            continue
        kept_speeds.append((speed, arc))
    for index, level in enumerate(levels):
        two = level["partner_deg"]
        for speed, arc in kept_speeds:
            mean, drift = arc_drift(arm, centre, two, arc)
            if not screen(arm, scene, centre, two, arc):
                refused.append({"level": index + 1, "speed_deg_s": speed,
                                "arc_deg": round(arc, 2), "why": "collision"})
                continue
            passes.append({"level": index + 1, "partner_deg": two,
                           "speed_deg_s": speed, "arc_deg": round(arc, 3),
                           "load_nm": round(mean, 4),
                           "load_drift_nm": round(drift, 4),
                           "load_drift": round(drift / max(mean, 1e-9), 4)})
    return passes, refused, cap, allowed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="j1_load_sweep")
    parser.add_argument("--urdf", default=None,
                        help="URDF file; read from /robot_description if absent")
    parser.add_argument("--obstacles", default="/tmp/right_arm_obstacles.json")
    parser.add_argument("--action", default="/right_arm_joint_trajectory_"
                                            "controller/follow_joint_trajectory")
    parser.add_argument("--prefix", default="right_")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--centre", type=float, default=STATIONARY_DEG)
    parser.add_argument("--speeds", type=int, default=18)
    parser.add_argument("--slowest", type=float, default=0.5)
    parser.add_argument("--fastest", type=float, default=40.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drift-limit", type=float, default=0.35,
                        help="largest load swing across a pass, as a fraction "
                             "of the gap between load levels")
    parser.add_argument("--arc-ceiling", type=float, default=70.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="design and screen everything, move nothing")
    args = parser.parse_args(argv)

    urdf = read_urdf(args.urdf)
    arm = ident.ArmModel.from_urdf_text(urdf, args.prefix)
    scene = build_scene(arm, urdf, args.obstacles)
    names = list(arm.joint_names)

    levels = design_levels(arm, args.levels, args.centre)
    speeds = [round(float(s), 3) for s in np.geomspace(
        args.slowest, args.fastest, args.speeds)]
    passes, refused, cap, allowed = plan_passes(
        arm, scene, levels, speeds, args.centre, args.drift_limit,
        args.arc_ceiling)

    print(f"joint {names[JOINT]}, load set by {names[PARTNER]}")
    print(f"sweep centred at {args.centre:g} deg, where the load is stationary")
    print(f"arc capped at {cap:.1f} deg so no pass moves its load by more "
          f"than {allowed:.3f} Nm")
    print(f"\n{'level':>6}{'j2 deg':>9}{'load Nm':>10}{'drift Nm':>10}"
          f"{'passes':>8}{'fastest':>9}")
    for index, level in enumerate(levels):
        mine = [p for p in passes if p["level"] == index + 1]
        fastest = max((p["speed_deg_s"] for p in mine), default=0.0)
        drift = max((p["load_drift_nm"] for p in mine), default=0.0)
        print(f"{index + 1:>6}{level['partner_deg']:>9.1f}"
              f"{level['load_nm']:>10.3f}{drift:>10.3f}"
              f"{len(mine):>8}{fastest:>9.1f}")
    if refused:
        print(f"\n{len(refused)} passes refused; fastest kept is "
              f"{max(p['speed_deg_s'] for p in passes):g} deg/s")
        reasons = {}
        for entry in refused:
            reasons[entry["why"]] = reasons.get(entry["why"], 0) + 1
        for why, count in reasons.items():
            print(f"  {count:>4}  {why}")

    total = len(passes) * max(1, args.repeats) * 2
    print(f"\n{total} passes to drive, about {total * 6.0 / 3600:.1f} hours")
    if args.dry_run:
        print("dry run: nothing was driven")
        return 0

    # The workspace cap has to let joint one reach the far end of its arc while
    # keeping every other joint where this experiment expects it.
    widest = max(p["arc_deg"] for p in passes)
    caps = [90.0] * len(names)
    caps[JOINT] = min(abs(args.centre) + widest / 2.0 + 5.0,
                      float(np.min(np.abs(arm.limits_deg()[1]))))
    profile = autoprofile.derive_profile(
        urdf, names, workspace_limit_deg=caps,
        speed_limit_deg_s=max(args.fastest, 10.0))
    plant = HardwarePlant(
        profile,
        config=HardwareConfig(action=args.action,
                              maximum_speed_deg_s=max(args.fastest, 10.0),
                              require_neutral_start=True),
        collision_model=scene)
    plant.open()

    folder = Path(args.output) / time.strftime("j1-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "design.json").write_text(json.dumps(
        {"joint": names[JOINT], "partner": names[PARTNER],
         "centre_deg": args.centre, "levels": levels, "speeds_deg_s": speeds,
         "repeats": args.repeats, "drift_limit": args.drift_limit,
         "passes": passes, "refused": refused}, indent=1), encoding="utf-8")

    columns = ["level", "load_nm", "load_drift", "partner_deg", "speed_deg_s",
               "direction", "repeat", "arc_deg", "window",
               "position_deg", "velocity_deg_s", "acceleration_deg_s2",
               "effort", "temperature_c", "load_here_nm",
               "window_frames", "window_fit_rms_deg"]
    handle = (folder / "samples.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(columns)

    print(f"\nwriting to {folder}")
    plant.park()
    driven = skipped = 0
    started = time.monotonic()
    try:
        for entry in passes:
            centre_pose = pose_at(arm, args.centre, entry["partner_deg"])
            for repeat in range(max(1, args.repeats)):
                for distance in (entry["arc_deg"], -entry["arc_deg"]):
                    origin = centre_pose.copy()
                    origin[JOINT] = args.centre - distance / 2.0
                    try:
                        frames = list(plant.traverse(
                            JOINT, origin, distance, entry["speed_deg_s"]))
                    except MotionFailed as failure:
                        skipped += 1
                        print(f"  skipped level {entry['level']} at "
                              f"{entry['speed_deg_s']:g} deg/s: {failure}")
                        continue
                    driven += 1
                    for index, frame in enumerate(frames):
                        here = pose_at(
                            arm, float(frame["position_deg"][JOINT]),
                            entry["partner_deg"])
                        writer.writerow([
                            entry["level"], entry["load_nm"],
                            entry["load_drift"], entry["partner_deg"],
                            entry["speed_deg_s"],
                            "+" if distance > 0 else "-", repeat + 1,
                            entry["arc_deg"], index,
                            round(float(frame["position_deg"][JOINT]), 4),
                            round(float(frame["speed_deg_s"][JOINT]), 4),
                            round(float((frame.get("acceleration_deg_s2")
                                         or [0.0] * len(names))[JOINT]), 4),
                            round(float(frame["current_a"][JOINT]), 5),
                            round(float(frame["temperature_c"][JOINT]), 2),
                            round(abs(float(
                                arm.joint_loads(here)[JOINT][JOINT])), 4),
                            frame.get("window_frames", 0),
                            round(float(frame.get("window_fit_rms_deg") or 0.0),
                                  5)])
                    handle.flush()
            done = passes.index(entry) + 1
            print(f"  level {entry['level']} at {entry['speed_deg_s']:>5g} "
                  f"deg/s done  ({done}/{len(passes)}, "
                  f"{(time.monotonic() - started) / 60:.0f} min)")
    except KeyboardInterrupt:
        print("\nstopped by the operator; what was measured is kept")
    finally:
        handle.close()
        try:
            plant.park()
        finally:
            plant.close()
    print(f"\n{driven} passes driven, {skipped} refused by the arm")
    print(f"wrote {folder}/samples.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
