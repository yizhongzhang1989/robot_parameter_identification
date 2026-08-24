#!/usr/bin/env python3
"""Read a load sweep, print what it found, and write the page.

Works on a run that is still going: the sweep writes one self-describing record
per line as it measures, so an analysis half way through is an analysis of half
the data rather than an error.

The computation lives in the package, beside the sweep that produces it, so the
report a run writes for itself and the numbers printed here cannot drift apart.
This file is only the terminal view of it.
"""

import argparse
import sys

from robot_parameter_identification.loadsweep_report import (summarise,
                                                             write_report)


def _maybe(value, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--no-report", action="store_true",
                        help="print only; do not rewrite report.html")
    args = parser.parse_args(argv)

    story = summarise(args.folder)
    print(f"{story['records']} records, {story['pairs']} level-speed pairs, "
          f"{story['measured']}/{story['designed']} passes measured")
    heat = story["temperature"]
    print(f"joints ran {heat['low']}-{heat['high']} C\n")

    print("What was measured, and the torque constant it gave for free")
    print(f"{'joint':>18}{'levels':>8}{'span Nm':>10}{'measured':>13}"
          f"{'A per Nm':>10}{'r':>10}")
    for joint in story["joints"]:
        print(f"{joint['name']:>18}{len(joint['levels']):>8}"
              f"{joint['span_nm']:>10.3f}"
              f"{joint['measured']:>7}/{joint['designed']:<5}"
              f"{_maybe(joint['amps_per_nm'], 4):>10}"
              f"{_maybe(joint['amps_per_nm_fit'], 5):>10}")
        if joint["note"]:
            print(f"{'':>18}  {joint['note']}")

    print("\nThe law fitted to each joint")
    print(f"{'joint':>18}{'coulomb':>9}{'per Nm':>9}{'stribeck':>10}"
          f"{'per Nm':>9}{'width':>8}{'viscous':>9}{'rms A':>8}{'of mean':>9}"
          f"  load terms")
    for joint in story["joints"]:
        fit = joint.get("fit")
        if not fit:
            print(f"{joint['name']:>18}   not enough measured yet")
            continue
        print(f"{joint['name']:>18}{fit['c0']:>9.4f}{fit['ck']:>9.4f}"
              f"{fit['d0']:>10.4f}{fit['dk']:>9.4f}{fit['width']:>8.3f}"
              f"{fit['viscous']:>9.5f}{fit['rms']:>8.4f}"
              f"{_maybe(fit['share_of_mean'], 1):>7}%  {fit['load_terms']}")

    print("\nWhat compensation would be worth, in joint torque")
    print(f"{'joint':>18}{'friction Nm':>13}{'left Nm':>10}{'left':>7}"
          f"{'gain':>8}   at")
    for joint in story["joints"]:
        for index, worth in enumerate(joint.get("worth") or []):
            print(f"{joint['name'] if not index else '':>18}"
                  f"{worth['friction_nm']:>13.3f}{worth['residual_nm']:>10.4f}"
                  f"{_maybe(worth['left'], 1):>6}%{_maybe(worth['gain'], 1):>7}x"
                  f"   {worth['which']}, {worth['load_nm']:.2f} Nm")

    print("\nDoes friction rise with load")
    print(f"{'joint':>18}{'A per Nm':>10}{'share of load':>15}   pinned down by")
    for joint in story["joints"]:
        if joint.get("per_nm") is None:
            print(f"{joint['name']:>18}{'-':>10}   load could not be varied")
            continue
        terms = joint["fit"]["load_terms"]
        how = ("two levels: an offset, not a shape" if terms == "coulomb"
               else f"{len(joint['levels'])} levels")
        print(f"{joint['name']:>18}{joint['per_nm']:>10.4f}"
              f"{joint['share_of_load']:>14.1f}%   {how}")

    print("\nWhether the load terms can be told apart at all")
    print(f"{'joint':>18}{'condition':>11}   radial / thrust / tilt against axial")
    for joint in story["joints"]:
        regression = joint.get("regression")
        if not regression:
            continue
        seen = regression["correlation"]
        text = " / ".join(
            "constant" if seen[term] is None else f"{seen[term]:+.3f}"
            for term in ("radial_n", "thrust_n", "tilt_nm"))
        verdict = "" if regression["trustworthy"] else "   <- confounded"
        print(f"{joint['name']:>18}{regression['condition']:>11.1f}"
              f"   {text}{verdict}")
    print("\nA condition number over about ten means the coefficients trade")
    print("against each other, so the split between the load terms is")
    print("arithmetic rather than measurement. The report says why.")

    if not args.no_report:
        print(f"\nwrote {write_report(args.folder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
