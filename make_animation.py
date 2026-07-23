#!/usr/bin/env python3
import argparse
from pathlib import Path
from safe_ac_static_obstacle_uncertainty_v17 import make_animation


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the informative active-control interval")
    ap.add_argument("--display-seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=9)
    args = ap.parse_args()
    make_animation(
        Path(__file__).resolve().parent / "demo_results",
        "stress",
        frames=max(2, round(args.display_seconds * args.fps)),
        fps=args.fps,
        display_time_s=args.display_seconds,
    )


if __name__ == "__main__":
    main()
