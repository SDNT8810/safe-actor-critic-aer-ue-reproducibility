#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from safe_ac_static_obstacle_uncertainty_v17 import (
    MethodState, World, clone_for_eval, method_specs, simulate_episode,
)


def generate_display_trajectories(
    out_dir: Path,
    epochs: int,
    seeds: int,
    moderate_test: float,
    stress_test: float,
    disturbance_stress: float,
    display_seed: int | None = None,
) -> Path:
    if display_seed is None:
        display_seed = min(4, seeds - 1)
    if not 0 <= display_seed < seeds:
        raise ValueError(f"display_seed={display_seed} must be in [0,{seeds-1}]")
    world = World()
    frames: List[pd.DataFrame] = []
    for spec in method_specs():
        state = MethodState(spec, world, display_seed)
        for ep in range(epochs):
            simulate_episode(state, ep, 1.0 + 0.06 * ep, disturbance_stress, True, False)
        trained = clone_for_eval(state)
        _, nominal = simulate_episode(clone_for_eval(trained), 1000, 1.0, disturbance_stress, False, True)
        _, moderate = simulate_episode(clone_for_eval(trained), 1001, moderate_test, disturbance_stress, False, True)
        _, stress = simulate_episode(clone_for_eval(trained), 1001, stress_test, disturbance_stress, False, True)
        assert nominal is not None and moderate is not None and stress is not None
        nominal["eval_type"] = "nominal"
        moderate["eval_type"] = "moderate"
        stress["eval_type"] = "stress"
        frames += [nominal, moderate, stress]
    out = out_dir / f"evaluation_trajectories_display_seed{display_seed}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8")
    (out_dir / "display_seed.txt").write_text(str(display_seed), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2)
    ap.add_argument("--stress-test", type=float, default=6.0)
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument("--display-seed", type=int, default=4)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(a.out)
    out = out if out.is_absolute() else root / out
    print(generate_display_trajectories(
        out, a.epochs, a.seeds, a.moderate_test, a.stress_test,
        a.disturbance_stress, a.display_seed,
    ))
