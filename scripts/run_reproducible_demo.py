#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _bootstrap import ROOT
from assemble_results import assemble
from generate_method_partial import generate_method
from safe_ac_repro.simulation import METHOD_ORDER, package_zip


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete V17.1 severe-sensor benchmark")
    ap.add_argument("--out", default="outputs/reproduction")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2, help="secondary moderate perception multiplier")
    ap.add_argument("--stress-test", type=float, default=6.0, help="additional exploratory extreme perception multiplier")
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument("--display-seed", type=int, default=4)
    ap.add_argument(
        "--display-seconds", type=float, default=10.0,
        help="plot/animation window; does not shorten the 15.4 s evaluation horizon",
    )
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--keep-partials", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    partial = ROOT / "_partials"
    if args.clean:
        if out.exists():
            shutil.rmtree(out)
        if partial.exists():
            shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)

    for method in METHOD_ORDER:
        generate_method(
            method, partial, args.epochs, args.seeds, args.moderate_test,
            args.stress_test, args.disturbance_stress, clean=True,
        )
    assemble(
        ROOT, partial, out, args.epochs, args.seeds, args.moderate_test,
        args.stress_test, args.disturbance_stress, args.display_seed,
        args.animate, clean=True, display_time_s=args.display_seconds,
    )
    if not args.keep_partials:
        shutil.rmtree(partial, ignore_errors=True)
    if args.zip:
        print(package_zip(ROOT))
    print(f"[done] results in {out}")


if __name__ == "__main__":
    main()
