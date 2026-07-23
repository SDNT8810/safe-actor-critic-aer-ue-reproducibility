#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import ROOT
from safe_ac_repro.simulation import make_animation


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the informative active-control interval")
    ap.add_argument("--out", default="outputs/reproduction")
    ap.add_argument("--display-seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=9)
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    make_animation(
        out,
        "stress",
        frames=max(2, round(args.display_seconds * args.fps)),
        fps=args.fps,
        display_time_s=args.display_seconds,
    )


if __name__ == "__main__":
    main()
