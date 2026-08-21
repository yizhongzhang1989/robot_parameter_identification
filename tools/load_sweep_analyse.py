#!/usr/bin/env python3
"""Read a load sweep and say what each joint's friction does with load.

Works on a run that is still going. The sweep writes one self-describing record
per line as it measures, so an analysis half way through is an analysis of half
the data rather than an error, and the same code answers at the end.

The measurement is a pair of passes in opposite directions at one posture and
one speed. Anything that reverses with direction is friction; anything that does
not is gravity, plus whatever error there is in holding it still. Splitting them
that way is what makes the gravity term cancel without ever having to be
modelled, which matters here because the postures are chosen for their load and
not for their conditioning.

What comes out is one law per joint,

    friction(v, L) = (c0 + ck L) tanh(v / w)
                   + (d0 + dk L) exp(-|v| / (vs0 + vsk L)) sign(v)
                   + b v

with the width and the viscous slope shared across loads, because they are
properties of the joint rather than of what it happens to be carrying. Letting
the width fit per level instead lets it collapse and hand the low-speed shape to
the Stribeck term, which then reports a load dependence that is an artefact of
that trade. A joint whose load could not be varied gets the same law with the
load terms held at zero, and is said to have no load axis rather than being
given one fitted to rounding error.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Measured on this arm from the gravity branch of the joint one sweep: the
# current needed per newton metre held, r = -0.99996.
AMPS_PER_NM = 0.4186


def read(folder):
    """Records and manifest. Records may be mid-flight; a torn last line is
    the run still writing, not a corrupt file."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    rows = []
    with (folder / "records.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows, manifest


def split(rows):
    """Half-difference and half-sum, per joint, level and speed rung."""
    grouped = defaultdict(lambda: {"+": [], "-": []})
    for row in rows:
        key = (row["joint"], row["level"], row["speed_deg_s"])
        grouped[key][row["direction"]].append(row)
    table = []
    for (joint, level, speed), sides in sorted(grouped.items()):
        if not sides["+"] or not sides["-"]:
            continue
        forward = float(np.mean([r["current_a"] for r in sides["+"]]))
        back = float(np.mean([r["current_a"] for r in sides["-"]]))
        every = sides["+"] + sides["-"]
        table.append({
            "joint": joint, "level": level, "speed": speed,
            "friction": (forward - back) / 2.0,
            "gravity": (forward + back) / 2.0,
            "load": float(np.mean([r["load"]["axial_nm"] for r in every])),
            "radial": float(np.mean([r["load"]["radial_n"] for r in every])),
            "measured_speed": float(np.mean(
                [abs(r["velocity_deg_s"]) for r in every])),
            "temperature": float(np.mean([r["temperature_c"] for r in every])),
            "passes": len(every),
        })
    return table


def grid(table, joint):
    """Speeds, one friction curve per level, and the load each level held."""
    mine = [r for r in table if r["joint"] == joint]
    if not mine:
        return None
    levels = sorted({r["level"] for r in mine})
    speeds = sorted({r["speed"] for r in mine})
    stack, loads, kept = [], [], []
    for level in levels:
        row = {r["speed"]: r for r in mine if r["level"] == level}
        if len(row) < len(speeds):
            continue           # a level still being measured
        stack.append([row[s]["friction"] for s in speeds])
        loads.append(float(np.mean([row[s]["load"] for s in speeds])))
        kept.append(level)
    if not stack:
        return None
    # Eight parameters against three points is not a fit, it is an
    # interpolation that reports zero residual and means nothing.
    if len(speeds) < 6:
        return None
    return {"speeds": np.array(speeds, dtype=float),
            "stack": np.array(stack, dtype=float),
            "loads": np.array(loads, dtype=float), "levels": kept}


def fit(speeds, stack, loads, with_load=True):
    """One law for the joint, load as a parameter when there is a load axis."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    def model(theta):
        width, viscous, c0, ck, d0, dk, vs0, vsk = theta
        return np.array([
            (c0 + ck * load) * np.tanh(speeds / max(width, 1e-3))
            + (d0 + dk * load) * np.exp(
                -speeds / max(vs0 + vsk * load, 1e-3))
            + viscous * speeds
            for load in loads])

    high = [20.0, 1.0, 5.0, 5.0, 5.0, 5.0, 200.0, 50.0]
    if not with_load:
        # No load axis: a slope fitted across levels that differ by less than
        # the noise is not a measurement of anything.
        high[3] = high[5] = high[7] = 1e-9
    found = least_squares(
        lambda t: (model(t) - stack).ravel(),
        [0.8, 0.004, 0.24, 0.12 if with_load else 0.0, 0.04,
         0.19 if with_load else 0.0, 0.5, 0.25 if with_load else 0.0],
        bounds=([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0], high))
    width, viscous, c0, ck, d0, dk, vs0, vsk = found.x
    return {"width": width, "viscous": viscous, "c0": c0, "ck": ck,
            "d0": d0, "dk": dk, "vs0": vs0, "vsk": vsk,
            "rms": float(np.sqrt(np.mean(found.fun ** 2)))}


def curve(model, speed, load):
    speed = np.asarray(speed, dtype=float)
    return ((model["c0"] + model["ck"] * load)
            * np.tanh(speed / model["width"])
            + (model["d0"] + model["dk"] * load)
            * np.exp(-np.abs(speed) / (model["vs0"] + model["vsk"] * load))
            * np.sign(speed)
            + model["viscous"] * speed)


def progress(rows, manifest):
    """How much of the designed sweep is actually on disk."""
    want = {}
    for joint in manifest["joints"]:
        want[joint["joint"]] = len(joint["passes"])
    repeats = int(manifest["plan"].get("repeats", 1))
    done = defaultdict(set)
    for row in rows:
        done[row["joint"]].add(row["key"])
    lines = []
    for index, joint in enumerate(manifest["joints"]):
        number = joint["joint"]
        total = want[number] * repeats * 2
        have = len(done.get(number, ()))
        lines.append({"joint": number, "name": joint["name"],
                      "have": have, "total": total,
                      "levels": len(joint["levels"]),
                      "span": joint["span_nm"],
                      "loadable": joint["loadable"],
                      "note": joint.get("note", "")})
        del index
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args(argv)

    rows, manifest = read(args.folder)
    table = split(rows)
    print(f"{len(rows)} records, {len(table)} level-speed pairs measured\n")

    print("How much of the sweep is on disk")
    print(f"{'joint':>18}{'measured':>12}{'levels':>8}{'span Nm':>10}  note")
    for line in progress(rows, manifest):
        print(f"{line['name']:>18}{line['have']:>6}/{line['total']:<5}"
              f"{line['levels']:>8}{line['span']:>10.3f}  "
              f"{'' if line['loadable'] else 'no load axis'}")

    fits = {}
    print("\nThe law fitted to each joint, load as a parameter\n")
    header = (f"{'joint':>18}{'levels':>7}{'loads Nm':>16}{'coulomb':>9}"
              f"{'per Nm':>9}{'dip':>8}{'per Nm':>9}{'width':>8}"
              f"{'viscous':>9}{'rms A':>8}{'of mean':>9}")
    print(header)
    for entry in manifest["joints"]:
        joint = entry["joint"]
        data = grid(table, joint)
        if data is None:
            print(f"{entry['name']:>18}      -  not enough measured yet")
            continue
        loadable = entry["loadable"] and len(data["levels"]) >= 3
        model = fit(data["speeds"], data["stack"], data["loads"], loadable)
        fits[joint] = (model, data, loadable)
        mean = float(np.mean(np.abs(data["stack"])))
        print(f"{entry['name']:>18}{len(data['levels']):>7}"
              f"{data['loads'].min():>7.2f}-{data['loads'].max():<8.2f}"
              f"{model['c0']:>9.4f}{model['ck']:>9.4f}"
              f"{model['d0']:>8.4f}{model['dk']:>9.4f}"
              f"{model['width']:>8.3f}{model['viscous']:>9.5f}"
              f"{model['rms']:>8.4f}{100 * model['rms'] / mean:>8.1f}%")

    print("\nWhat compensation would be worth, in joint torque")
    print(f"{'joint':>18}{'friction Nm':>13}{'residual Nm':>13}{'left':>7}"
          f"{'gain':>7}   at the lightest and heaviest load measured")
    for joint, (model, data, loadable) in sorted(fits.items()):
        name = manifest["joints"][joint]["name"]
        for which, index in (("light", 0), ("heavy", len(data["levels"]) - 1)):
            measured = data["stack"][index]
            residual = measured - curve(model, data["speeds"],
                                        data["loads"][index])
            friction = float(np.mean(np.abs(measured))) / AMPS_PER_NM
            error = float(np.sqrt(np.mean(residual ** 2))) / AMPS_PER_NM
            share = f"{100 * error / friction:>6.1f}%" if friction > 1e-9 else "     -"
            gain = f"{friction / error:>7.1f}x" if error > 1e-6 else "      -"
            print(f"{name if which == 'light' else '':>18}"
                  f"{friction:>13.3f}{error:>13.4f}{share}{gain}"
                  f"   {which}, {data['loads'][index]:.2f} Nm")

    print("\nDoes friction rise with load, and by how much")
    print(f"{'joint':>18}{'A per Nm':>10}{'as a share of the load':>24}"
          f"{'over span':>11}")
    for joint, (model, data, loadable) in sorted(fits.items()):
        name = manifest["joints"][joint]["name"]
        if not loadable:
            print(f"{name:>18}         -  load could not be varied")
            continue
        span = data["loads"].max() - data["loads"].min()
        print(f"{name:>18}{model['ck']:>10.4f}"
              f"{model['ck'] / AMPS_PER_NM:>23.1%}"
              f"{model['ck'] * span / AMPS_PER_NM:>10.3f} Nm")

    if args.plot:
        draw(args.plot, manifest, fits)
        print(f"\nwrote {args.plot}")
    return 0


def draw(path, manifest, fits):
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    count = len(fits)
    if not count:
        return
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns,
                                                        3.8 * rows),
                                squeeze=False)
    for slot, (joint, (model, data, loadable)) in enumerate(sorted(fits.items())):
        axis = axes[slot // columns][slot % columns]
        colours = plt.cm.viridis(np.linspace(0, 0.92, len(data["levels"])))
        fine = np.geomspace(max(data["speeds"].min(), 1e-3),
                            data["speeds"].max(), 200)
        for index, load in enumerate(data["loads"]):
            axis.plot(data["speeds"], data["stack"][index], "o", ms=3.4,
                      color=colours[index], alpha=0.85,
                      label=f"{load:.2f} Nm")
            # Black casing so the fitted line reads as a line over a
            # measurement rather than as more measurement.
            axis.plot(fine, curve(model, fine, load), "-", lw=2.6,
                      color="black", alpha=0.45)
            axis.plot(fine, curve(model, fine, load), "-", lw=1.4,
                      color=colours[index])
        axis.set_xscale("log")
        axis.set_xlabel("speed deg/s")
        axis.set_ylabel("friction A")
        axis.set_title(f"{manifest['joints'][joint]['name']}"
                       f"{'' if loadable else '  (no load axis)'}", fontsize=10)
        axis.grid(alpha=0.25, which="both")
        if len(data["levels"]) <= 6:
            axis.legend(fontsize=6, ncol=2)
    for spare in range(count, rows * columns):
        axes[spare // columns][spare % columns].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
