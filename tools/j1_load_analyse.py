#!/usr/bin/env python3
"""Read one joint's load sweep and say what its friction actually does.

Works only from the recorded file. The design is read back for the load each
level was meant to hold, but every number reported here comes from what the arm
did, not from what it was asked to do.

Friction is separated from gravity by direction rather than by a model. Each
pass is driven both ways at the same speed and pose, so the half-difference
between them is the part that reverses -- friction -- and the half-sum is the
part that does not, which is gravity plus whatever the gravity estimate got
wrong. Reporting both is the point: a load effect that shows up in the half-sum
is not friction, and this is what tells the two apart.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(folder):
    rows = list(csv.DictReader((folder / "samples.csv").open(encoding="utf-8")))
    design = json.loads((folder / "design.json").read_text(encoding="utf-8"))
    for row in rows:
        for key in ("load_nm", "speed_deg_s", "effort", "velocity_deg_s",
                    "position_deg", "temperature_c", "load_here_nm",
                    "load_drift", "arc_deg"):
            row[key] = float(row[key])
        row["level"] = int(row["level"])
    return rows, design


def split(rows):
    """Half-difference and half-sum per level and speed rung.

    Anything that reverses with direction is friction; anything that does not
    is gravity and the error in the estimate of it.
    """
    grouped = defaultdict(lambda: {"+": [], "-": []})
    for row in rows:
        grouped[(row["level"], row["speed_deg_s"])][row["direction"]].append(row)
    table = []
    for (level, speed), sides in sorted(grouped.items()):
        if not sides["+"] or not sides["-"]:
            continue
        forward = float(np.mean([r["effort"] for r in sides["+"]]))
        back = float(np.mean([r["effort"] for r in sides["-"]]))
        every = sides["+"] + sides["-"]
        table.append({
            "level": level, "speed": speed,
            "friction": (forward - back) / 2.0,
            "gravity": (forward + back) / 2.0,
            "load": float(np.mean([r["load_here_nm"] for r in every])),
            "measured_speed": float(np.mean(
                [abs(r["velocity_deg_s"]) for r in every])),
            "temperature": float(np.mean([r["temperature_c"] for r in every])),
            "passes": len(every),
        })
    return table


def curve_of(table, level):
    rows = sorted((r for r in table if r["level"] == level),
                  key=lambda r: r["speed"])
    return (np.array([r["speed"] for r in rows]),
            np.array([r["friction"] for r in rows]),
            np.array([r["gravity"] for r in rows]))


def fit_curve(speed, friction):
    """Fit the friction law to one level, rather than describing its shape.

    friction(v) = coulomb x tanh(v / width)
                  + dip x exp(-v / dip_speed)
                  + viscous x v

    Descriptive summaries will not do here, and the reason is worth stating.
    Read the trough position off the curve and it moves with load even when the
    planted dip speed is a constant, because a deeper dip pushes the minimum
    later against the viscous rise; read the slope off the fast end and it falls
    with load for the same reason. On synthetic data with a fixed dip speed
    those two summaries reported correlations of +0.89 and -1.00 with load, both
    of them artefacts. Fitting the parameters separates amplitude from position,
    which is the whole question.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    speed = np.asarray(speed, dtype=float)
    friction = np.asarray(friction, dtype=float)

    def residual(theta):
        coulomb, width, dip, dip_speed, viscous = theta
        model = (coulomb * np.tanh(speed / max(width, 1e-3))
                 + dip * np.exp(-speed / max(dip_speed, 1e-3))
                 + viscous * speed)
        return model - friction

    guess = [float(np.median(friction)), 1.0,
             float(max(friction.max() - friction.min(), 1e-3)), 6.0, 1e-3]
    found = least_squares(
        residual, guess,
        bounds=([0.0, 0.05, 0.0, 0.2, 0.0], [10.0, 20.0, 10.0, 200.0, 1.0]))
    coulomb, width, dip, dip_speed, viscous = found.x
    return {"coulomb": float(coulomb), "width": float(width),
            "dip": float(dip), "dip_speed": float(dip_speed),
            "viscous": float(viscous),
            "rms": float(np.sqrt(np.mean(found.fun ** 2)))}


def fit_global(speeds, stack, loads):
    """One law for the whole joint, with load as a parameter.

    The per-level fits are the diagnostic; this is the deliverable. Width and
    viscous belong to the joint and are shared, because they are properties of
    it rather than of what it happens to be carrying. Letting the width fit per
    level instead lets it collapse from 1.06 to 0.05 across the loads and hand
    the low-speed shape to the dip term, which then reports a load dependence
    that is an artefact of the trade.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    def model(theta):
        width, viscous, c0, ck, d0, dk, vs0, vsk = theta
        return np.array([
            (c0 + ck * load) * np.tanh(speeds / max(width, 1e-3))
            + (d0 + dk * load) * np.exp(
                -speeds / max(vs0 + vsk * load, 1e-3))
            + viscous * speeds
            for load in loads])

    found = least_squares(
        lambda t: (model(t) - stack).ravel(),
        [0.8, 0.004, 0.24, 0.12, 0.04, 0.19, 0.5, 0.25],
        bounds=([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
                [20.0, 1.0, 5.0, 5.0, 5.0, 5.0, 200.0, 50.0]))
    width, viscous, c0, ck, d0, dk, vs0, vsk = found.x
    return {"width": width, "viscous": viscous, "c0": c0, "ck": ck,
            "d0": d0, "dk": dk, "vs0": vs0, "vsk": vsk,
            "rms": float(np.sqrt(np.mean(found.fun ** 2)))}


def global_curve(model, speed, load):
    """The law over signed speed, which is where the measurement lives."""
    speed = np.asarray(speed, dtype=float)
    return ((model["c0"] + model["ck"] * load)
            * np.tanh(speed / model["width"])
            + (model["d0"] + model["dk"] * load)
            * np.exp(-np.abs(speed) / (model["vs0"] + model["vsk"] * load))
            * np.sign(speed)
            + model["viscous"] * speed)


def report_against_load(summary, field, label):
    load = np.array([s["load"] for s in summary])
    values = np.array([s[field] for s in summary])
    if values.std() < 1e-12 or load.std() < 1e-12:
        print(f"  {label:<22} flat")
        return
    slope, intercept = np.polyfit(load, values, 1)
    print(f"  {label:<22} {intercept:9.4f} + {slope:8.4f} x load"
          f"   r = {np.corrcoef(load, values)[0, 1]:+.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--plot", default=None,
                        help="write a PNG of the curves here")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    rows, design = read(folder)
    table = split(rows)
    levels = sorted({r["level"] for r in table})
    print(f"{len(rows)} samples, {len(table)} level-speed pairs, "
          f"{len(levels)} levels")

    print("\nWhat each level actually held, against what it was designed to")
    print(f"{'level':>6}{'designed':>10}{'measured':>10}{'drift':>8}"
          f"{'temp C':>8}{'rungs':>7}")
    for level in levels:
        mine = [r for r in table if r["level"] == level]
        wanted = design["levels"][level - 1]["load_nm"]
        got = float(np.mean([r["load"] for r in mine]))
        drift = max(r["load"] for r in mine) - min(r["load"] for r in mine)
        print(f"{level:>6}{wanted:>10.3f}{got:>10.3f}{drift:>8.3f}"
              f"{np.mean([r['temperature'] for r in mine]):>8.1f}"
              f"{len(mine):>7}")

    print("\nFriction against speed, one row per rung, one column per level")
    speeds = sorted({r["speed"] for r in table})
    print(f"{'speed':>7}" + "".join(f"{level:>8}" for level in levels))
    for speed in speeds:
        cells = []
        for level in levels:
            hit = [r["friction"] for r in table
                   if r["level"] == level and r["speed"] == speed]
            cells.append(f"{hit[0]:8.3f}" if hit else f"{'':>8}")
        print(f"{speed:>7.2f}" + "".join(cells))

    print("\nThe friction law fitted to each load, one fit per level")
    print(f"{'level':>6}{'load Nm':>9}{'coulomb':>9}{'width':>8}{'dip':>8}"
          f"{'dip v':>8}{'viscous':>10}{'fit rms':>9}")
    summary = []
    for level in levels:
        speed, friction, _gravity = curve_of(table, level)
        if speed.size < 6:
            continue
        found = fit_curve(speed, friction)
        found["level"] = level
        found["load"] = float(
            np.mean([r["load"] for r in table if r["level"] == level]))
        summary.append(found)
        print(f"{level:>6}{found['load']:>9.3f}{found['coulomb']:>9.3f}"
              f"{found['width']:>8.2f}{found['dip']:>8.3f}"
              f"{found['dip_speed']:>8.2f}{found['viscous']:>10.5f}"
              f"{found['rms']:>9.4f}")

    if len(summary) >= 3:
        print("\nEach fitted term against the load, which is the question")
        for field, label in (("coulomb", "Coulomb"),
                             ("dip", "dip amplitude"),
                             ("dip_speed", "dip speed"),
                             ("width", "reversal width"),
                             ("viscous", "viscous")):
            report_against_load(summary, field, label)

    whole = None
    if len(summary) >= 3:
        rungs = curve_of(table, levels[0])[0]
        stack = np.array([curve_of(table, level)[1] for level in levels])
        carried = np.array([s["load"] for s in summary])
        whole = fit_global(rungs, stack, carried)
        print("\nOne law for the joint, load as a parameter")
        print(f"  friction(v, L) = ({whole['c0']:.4f} + {whole['ck']:.4f} L)"
              f" tanh(v / {whole['width']:.3f})")
        print(f"                 + ({whole['d0']:.4f} + {whole['dk']:.4f} L)"
              f" exp(-|v| / ({whole['vs0']:.3f} + {whole['vsk']:.3f} L))"
              " sign(v)")
        print(f"                 + {whole['viscous']:.5f} v"
              "          [A, v in deg/s, L in Nm]")
        print(f"  rms {whole['rms']:.4f} A over {stack.size} points, "
              f"{100 * whole['rms'] / float(np.mean(np.abs(stack))):.1f}% of "
              "the mean friction")

        # The gravity-removed panel shows dots either side of that line, and
        # the gap is two different things. Only one of them is fit error.
        print("\nWhy the dots sit off the line, split into what the law can\n"
              "fit and what it structurally cannot: no function that reverses\n"
              "with direction can produce a term that does not.")
        print(f"{'load Nm':>9}{'odd (fit error)':>17}{'even (asymmetry)':>18}"
              f"{'even share':>12}")
        for index, level in enumerate(levels):
            odd = stack[index] - global_curve(whole, rungs, carried[index])
            even = [r["gravity"] for r in table if r["level"] == level]
            even = np.array(even) - float(np.mean(even))
            o = float(np.sqrt(np.mean(odd ** 2)))
            e = float(np.sqrt(np.mean(even ** 2)))
            print(f"{carried[index]:>9.2f}{o:>17.4f}{e:>18.4f}"
                  f"{100 * e ** 2 / (o ** 2 + e ** 2):>11.0f}%")

    print("\nThe half-sum, which is gravity and not friction. It should not "
          "move with speed;\nif it does, the gravity estimate is speed "
          "dependent and the split above is contaminated.")
    print(f"{'level':>6}{'mean A':>9}{'swing over the ladder':>24}")
    for level in levels:
        _speed, _friction, gravity = curve_of(table, level)
        print(f"{level:>6}{float(np.mean(gravity)):>9.3f}"
              f"{float(gravity.max() - gravity.min()):>24.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, grid = plt.subplots(2, 2, figsize=(14.0, 10.4))
        axes = grid.ravel()
        colours = plt.cm.viridis(np.linspace(0, 0.92, len(levels)))

        def branches(level):
            """Each direction's rungs, repeats averaged, sorted by speed.

            Drawing the repeats as separate points and joining them turns three
            readings of one rung into a zigzag that looks like structure.
            """
            mine = [r for r in rows if r["level"] == level]
            for direction in ("+", "-"):
                rungs = {}
                for row in mine:
                    if row["direction"] == direction:
                        rungs.setdefault(row["speed_deg_s"], []).append(row)
                yield direction, sorted(
                    (float(np.mean([r["velocity_deg_s"] for r in group])),
                     float(np.mean([r["effort"] for r in group])))
                    for group in rungs.values())

        # Gravity is one number per posture: it does not know how fast the
        # joint is going. Estimating it per speed instead -- the half-sum at
        # each rung -- would force the two branches to mirror each other
        # exactly, and the question of whether they do is the reason to draw
        # them.
        gravity_of = {level: float(np.mean([r["effort"] for r in rows
                                            if r["level"] == level]))
                      for level in levels}

        for index, level in enumerate(levels):
            load = float(np.mean([r["load_here_nm"] for r in rows
                                  if r["level"] == level]))
            for direction, points in branches(level):
                mark = f"{load:.2f} N\u00b7m" if direction == "+" else None
                axes[0].plot([p[0] for p in points], [p[1] for p in points],
                             "-o", color=colours[index], linewidth=1.2,
                             markersize=2.5, alpha=0.9, label=mark)
                axes[1].plot([p[0] for p in points],
                             [p[1] - gravity_of[level] for p in points],
                             "o", color=colours[index], markersize=3.0,
                             alpha=0.95, label=mark)
            if whole is not None:
                # The law drawn through the measurement it was fitted to. It is
                # odd in speed by construction, so where the two branches are
                # not mirror images it can only split the difference, and that
                # gap is the part of the reading friction does not explain.
                reach = max(abs(p[0]) for _d, pts in branches(level)
                            for p in pts)
                fine = np.linspace(-reach, reach, 401)
                axes[1].plot(fine, global_curve(whole, fine, load), "-",
                             color=colours[index], linewidth=1.1, alpha=0.75)
        for axis in (axes[0], axes[1]):
            axis.axvline(0.0, color="#999", linewidth=0.8, zorder=0)
            axis.set_xlabel("joint speed (deg/s), both directions")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        axes[1].axhline(0.0, color="#999", linewidth=0.8, zorder=0)
        axes[0].set_ylabel("measured current (A)")
        axes[0].set_title("As measured, gravity still in it\n"
                          "the gap between branches is twice the friction")
        axes[1].set_ylabel("current less gravity (A)")
        axes[1].set_title("Gravity removed, one constant per posture\n"
                          "dots measured, lines the one fitted law")

        for index, level in enumerate(levels):
            speed, friction, _g = curve_of(table, level)
            load = float(np.mean([r["load"] for r in table
                                  if r["level"] == level]))
            axes[2].plot(speed, friction, "-o", color=colours[index],
                         markersize=3.5, label=f"{load:.2f} N\u00b7m")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("speed (deg/s, log)")
        axes[2].set_ylabel("friction (A)")
        axes[2].set_title("Half the difference between the branches\n"
                          "which is the friction, gravity cancelled")
        axes[2].grid(alpha=0.25, which="both")
        axes[2].legend(fontsize=7, ncol=2)
        if summary:
            load = np.array([s["load"] for s in summary])
            coulomb = np.array([s["coulomb"] for s in summary])
            axes[3].plot(load, coulomb, "o", color="#4da3ff",
                         label="fitted Coulomb")
            slope, intercept = np.polyfit(load, coulomb, 1)
            axes[3].plot(load, intercept + slope * load, "-", color="#e0b341",
                         label=f"{intercept:.3f} + {slope:.3f} x load")
            axes[3].set_xlabel("load (N\u00b7m)")
            axes[3].set_ylabel("friction (A)")
            axes[3].set_title("Coulomb against load")
            axes[3].grid(alpha=0.25)
            axes[2].legend(fontsize=9)
        figure.tight_layout()
        figure.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
