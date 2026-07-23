#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from safe_ac_repro.simulation import (
    make_animation, make_plots, make_tables, write_acceptance_audit,
    write_oracle_invariance_audit, write_protocol_and_readme,
    write_sensor_fairness_audit,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate V17.1 tables, plots, audits, and animation without retraining")
    ap.add_argument("--out", default="outputs/reproduction")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2)
    ap.add_argument("--stress-test", type=float, default=6.0)
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument(
        "--display-seconds", type=float, default=10.0,
        help="plot/animation window; does not shorten the evaluation horizon",
    )
    ap.add_argument("--animate", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    write_sensor_fairness_audit(out)
    write_oracle_invariance_audit(out)
    make_tables(out)
    make_plots(out, display_time_s=a.display_seconds)
    write_protocol_and_readme(ROOT, out, a.epochs, a.seeds, a.moderate_test, a.stress_test, a.disturbance_stress)
    if a.animate:
        make_animation(
            out,
            "stress",
            frames=max(2, round(a.display_seconds * 9)),
            fps=9,
            display_time_s=a.display_seconds,
        )
    write_acceptance_audit(out)
    print(f"[done] postprocessed {out}")


if __name__ == "__main__":
    main()
