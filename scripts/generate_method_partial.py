#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from _bootstrap import ROOT
from safe_ac_repro.simulation import (
    METHOD_ORDER, MethodState, World, clone_for_eval, method_specs, simulate_episode,
)

# One fixed episode seed is used at every sweep level so severity is not
# confounded with a different random realization.
SWEEP_LEVELS = [0.8, 1.0, 1.3, 1.6, 2.0, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5]


def safe_name(method: str) -> str:
    return method.replace("+", "_").replace(" ", "_")


def generate_method(
    method: str,
    partial_root: Path,
    epochs: int,
    seeds: int,
    moderate_test: float,
    stress_test: float,
    disturbance_stress: float,
    clean: bool = True,
) -> Path:
    world = World()
    specs = {s.key: s for s in method_specs()}
    if method not in specs:
        raise ValueError(f"unknown method {method!r}; choose from {METHOD_ORDER}")
    spec = specs[method]
    out = partial_root / safe_name(method)
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    training_rows: List[dict] = []
    nominal_rows: List[dict] = []
    moderate_rows: List[dict] = []
    stress_rows: List[dict] = []
    replay_rows: List[dict] = []
    waypoint_rows: List[dict] = []
    trajectory_frames: List[pd.DataFrame] = []
    sweep_rows: List[dict] = []

    print(f"[method] {method}", flush=True)
    for seed in range(seeds):
        print(f"  [seed] {seed}", flush=True)
        state = MethodState(spec, world, seed)
        for ep in range(epochs):
            row, _ = simulate_episode(
                state,
                episode_id=ep,
                stress=1.0 + 0.06 * ep,
                disturbance_stress=disturbance_stress,
                train=True,
                record=False,
            )
            row["epoch"] = ep
            training_rows.append(row)

        trained = clone_for_eval(state)
        nrow, ntraj = simulate_episode(
            clone_for_eval(trained), episode_id=1000, stress=1.0,
            disturbance_stress=disturbance_stress, train=False, record=(seed == 0),
        )
        # Moderate and severe use the same episode id. Their random seed is the
        # same, while severity legitimately changes amplitudes and hold probability.
        mrow, mtraj = simulate_episode(
            clone_for_eval(trained), episode_id=1001, stress=moderate_test,
            disturbance_stress=disturbance_stress, train=False, record=(seed == 0),
        )
        srow, straj = simulate_episode(
            clone_for_eval(trained), episode_id=1001, stress=stress_test,
            disturbance_stress=disturbance_stress, train=False, record=(seed == 0),
        )
        nrow["eval_type"] = "nominal"
        mrow["eval_type"] = "moderate"
        srow["eval_type"] = "stress"
        nominal_rows.append(nrow)
        moderate_rows.append(mrow)
        stress_rows.append(srow)
        replay_rows.append(state.replay.audit(seed))

        for j, point in enumerate(trained.actor.waypoints):
            waypoint_rows.append({
                "method": method, "short": spec.short, "seed": seed,
                "waypoint": j, "x": point[0], "y": point[1],
            })

        if seed == 0:
            assert ntraj is not None and mtraj is not None and straj is not None
            ntraj["eval_type"] = "nominal"
            mtraj["eval_type"] = "moderate"
            straj["eval_type"] = "stress"
            trajectory_frames += [ntraj, mtraj, straj]

        for multiplier in SWEEP_LEVELS:
            row, _ = simulate_episode(
                clone_for_eval(trained), episode_id=2000,
                stress=multiplier,
                disturbance_stress=disturbance_stress,
                train=False,
                record=False,
            )
            sweep_rows.append({
                "stress_multiplier": multiplier,
                "method": method,
                "short": spec.short,
                "seed": seed,
                "sweep_episode_id": 2000,
                "violation_rate": row["violation_rate"],
                "violation_count": row["violation_count"],
                "known_static_violation_rate": row["known_static_violation_rate"],
                "known_static_violation_count": row["known_static_violation_count"],
                "uncertain_true_violation_rate": row["uncertain_true_violation_rate"],
                "uncertain_true_violation_count": row["uncertain_true_violation_count"],
                "min_clearance_cm": row["min_clearance_cm"],
                "min_known_static_clearance_cm": row["min_known_static_clearance_cm"],
                "min_uncertain_true_clearance_cm": row["min_uncertain_true_clearance_cm"],
                "cost": row["cost"],
                "belief_center_rmse_cm": row["belief_center_rmse_cm"],
                "goal_error_cm": row["goal_error_cm"],
                "reached": row["reached"],
            })

    outputs = {
        "training_episode_metrics.csv": pd.DataFrame(training_rows),
        "nominal_eval_metrics.csv": pd.DataFrame(nominal_rows),
        "moderate_eval_metrics.csv": pd.DataFrame(moderate_rows),
        "stress_eval_metrics.csv": pd.DataFrame(stress_rows),
        "replay_finite_buffer_audit.csv": pd.DataFrame(replay_rows),
        "learned_waypoints.csv": pd.DataFrame(waypoint_rows),
        "evaluation_trajectories_seed0.csv": pd.concat(trajectory_frames, ignore_index=True),
        "robustness_sweep_seed_metrics.csv": pd.DataFrame(sweep_rows),
    }
    for name, df in outputs.items():
        df.to_csv(out / name, index=False, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate one V17.1 method partial")
    ap.add_argument("--method", required=True, choices=METHOD_ORDER)
    ap.add_argument("--partial-root", default="_partials")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2)
    ap.add_argument("--stress-test", type=float, default=6.0)
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument("--no-clean", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    root = Path(a.partial_root)
    if not root.is_absolute():
        root = ROOT / root
    print(generate_method(
        a.method, root, a.epochs, a.seeds, a.moderate_test,
        a.stress_test, a.disturbance_stress, clean=not a.no_clean,
    ))
