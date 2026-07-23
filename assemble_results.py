#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from generate_method_partial import safe_name
from generate_display_trajectories import generate_display_trajectories
from safe_ac_static_obstacle_uncertainty_v17 import (
    METHOD_ORDER,
    make_animation,
    make_plots,
    make_tables,
    package_zip,
    write_acceptance_audit,
    write_oracle_invariance_audit,
    write_protocol_and_readme,
    write_sensor_fairness_audit,
)

FILES = [
    "training_episode_metrics.csv",
    "nominal_eval_metrics.csv",
    "moderate_eval_metrics.csv",
    "stress_eval_metrics.csv",
    "replay_finite_buffer_audit.csv",
    "learned_waypoints.csv",
    "evaluation_trajectories_seed0.csv",
    "robustness_sweep_seed_metrics.csv",
]


def assemble(
    root: Path,
    partial_root: Path,
    out: Path,
    epochs: int,
    seeds: int,
    moderate_test: float,
    stress_test: float,
    disturbance_stress: float,
    display_seed: int,
    animate: bool,
    clean: bool,
    display_time_s: float = 10.0,
) -> Path:
    if clean and out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "animations").mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}

    for filename in FILES:
        frames: List[pd.DataFrame] = []
        for method in METHOD_ORDER:
            p = partial_root / safe_name(method) / filename
            if not p.exists():
                raise FileNotFoundError(f"missing partial: {p}")
            frames.append(pd.read_csv(p))
        df = pd.concat(frames, ignore_index=True)
        if "method" in df.columns:
            df["_order"] = df.method.map(order)
            sort_cols = [
                c for c in [
                    "_order", "seed", "epoch", "eval_type", "step",
                    "stress_multiplier", "waypoint",
                ] if c in df.columns
            ]
            df = df.sort_values(sort_cols).drop(columns="_order")
        df.to_csv(out / filename, index=False, encoding="utf-8")

    sweep_seed = pd.read_csv(out / "robustness_sweep_seed_metrics.csv")
    sweep = sweep_seed.groupby(["stress_multiplier", "method", "short"], as_index=False).agg(
        violation_rate_mean=("violation_rate", "mean"),
        violation_rate_std=("violation_rate", "std"),
        violation_count_mean=("violation_count", "mean"),
        known_static_violation_rate_mean=("known_static_violation_rate", "mean"),
        known_static_violation_rate_std=("known_static_violation_rate", "std"),
        known_static_violation_count_mean=("known_static_violation_count", "mean"),
        uncertain_true_violation_rate_mean=("uncertain_true_violation_rate", "mean"),
        uncertain_true_violation_rate_std=("uncertain_true_violation_rate", "std"),
        uncertain_true_violation_count_mean=("uncertain_true_violation_count", "mean"),
        min_clearance_cm_mean=("min_clearance_cm", "mean"),
        min_clearance_cm_std=("min_clearance_cm", "std"),
        min_known_static_clearance_cm_mean=("min_known_static_clearance_cm", "mean"),
        min_known_static_clearance_cm_std=("min_known_static_clearance_cm", "std"),
        min_uncertain_true_clearance_cm_mean=("min_uncertain_true_clearance_cm", "mean"),
        min_uncertain_true_clearance_cm_std=("min_uncertain_true_clearance_cm", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        belief_center_rmse_cm_mean=("belief_center_rmse_cm", "mean"),
        belief_center_rmse_cm_std=("belief_center_rmse_cm", "std"),
        goal_error_cm_mean=("goal_error_cm", "mean"),
        reached_rate=("reached", "mean"),
    )
    sweep["_order"] = sweep.method.map(order)
    sweep = sweep.sort_values(["stress_multiplier", "_order"]).drop(columns="_order")
    sweep.to_csv(out / "robustness_sweep.csv", index=False, encoding="utf-8")

    # Standalone severe common sensor stream for direct inspection.
    traj = pd.read_csv(out / "evaluation_trajectories_seed0.csv")
    ref = traj[(traj.eval_type == "stress") & (traj.method == "AC")].copy()
    sensor_cols = [
        "step", "time", "stress", "sensor_stream_hash", "sensor_valid_fraction",
        "sensor_observable_jump_score", "sensor_hidden_event",
    ]
    for i in (1, 2):
        sensor_cols += [
            f"obs{i}_true_x", f"obs{i}_true_y", f"obs{i}_raw_x", f"obs{i}_raw_y",
            f"obs{i}_valid", f"obs{i}_jump_score", f"obs{i}_hidden_event",
            f"obs{i}_reported_sigma", f"obs{i}_raw_error",
        ]
    ref[sensor_cols].to_csv(
        out / "shared_sensor_stream_severe_seed0.csv", index=False, encoding="utf-8"
    )

    write_sensor_fairness_audit(out)
    write_oracle_invariance_audit(out)
    make_tables(out)
    generate_display_trajectories(
        out, epochs, seeds, moderate_test, stress_test, disturbance_stress,
        display_seed=display_seed,
    )
    make_plots(out, display_time_s=display_time_s)
    write_protocol_and_readme(
        root, out, epochs, seeds, moderate_test, stress_test, disturbance_stress,
    )

    # Known-map collision audit for the additional extreme evaluation.
    stress = pd.read_csv(out / "stress_eval_metrics.csv")
    known = stress.groupby("method", as_index=False).agg(
        known_static_violation_count_mean=("known_static_violation_count", "mean"),
        known_static_violation_count_max=("known_static_violation_count", "max"),
        min_known_static_clearance_cm_min=("min_known_static_clearance_cm", "min"),
    )
    known.to_csv(out / "known_static_collision_audit.csv", index=False, encoding="utf-8")
    (out / "known_static_collision_audit.md").write_text(
        "# Known static-map collision audit\n\n" + known.to_markdown(index=False),
        encoding="utf-8",
    )

    sweep_known = sweep_seed.groupby(["stress_multiplier", "method"], as_index=False).agg(
        known_static_violation_count_mean=("known_static_violation_count", "mean"),
        known_static_violation_count_max=("known_static_violation_count", "max"),
        uncertain_true_violation_count_mean=("uncertain_true_violation_count", "mean"),
        uncertain_true_violation_count_max=("uncertain_true_violation_count", "max"),
        min_known_static_clearance_cm_min=("min_known_static_clearance_cm", "min"),
        min_uncertain_true_clearance_cm_min=("min_uncertain_true_clearance_cm", "min"),
    )
    sweep_known.to_csv(
        out / "robustness_sweep_collision_source_audit.csv", index=False, encoding="utf-8"
    )

    belief = stress.groupby("method", as_index=False).agg(
        raw_sensor_rmse_cm=("raw_sensor_rmse_cm", "mean"),
        belief_center_rmse_cm=("belief_center_rmse_cm", "mean"),
        min_true_clearance_cm=("min_clearance_cm", "mean"),
        violation_count=("violation_count", "mean"),
        goal_error_cm=("goal_error_cm", "mean"),
        reached_rate=("reached", "mean"),
    )
    belief["estimation_improvement_vs_raw_pct"] = 100 * (
        1 - belief.belief_center_rmse_cm / belief.raw_sensor_rmse_cm
    )
    belief.to_csv(out / "belief_estimation_audit.csv", index=False, encoding="utf-8")

    contract = {
        "controller_visible": [
            "robot state", "goal", "known rectangle map", "uncertain obstacle radii",
            "shared raw/held obstacle-center measurement", "validity bit",
            "observable frame-to-frame jump score", "method belief and sigma",
            "observed contact after collision", "nominal and executed actions",
        ],
        "evaluator_only": [
            "true uncertain-obstacle centers", "true signed clearance",
            "hidden sensor fault-event label", "raw/belief center error",
        ],
        "replay_serialized_fields": {
            "executed_transition": [
                "x", "y", "x_next", "y_next", "u_nom_x", "u_nom_y",
                "u_exec_x", "u_exec_y", "instant_cost", "td_error", "td_score",
            ],
            "required_by_replayed_actor_update": [
                "exploration_x", "exploration_y", "actor_waypoint_index",
                "belief_grad_x", "belief_grad_y",
            ],
            "safety_and_replay": [
                "collision_observed", "cbf_active", "cbf_estimated_h",
                "estimated_active_constraint", "correction_norm", "safety_score",
                "uncertainty_score", "novelty_score", "is_boundary", "is_uncertainty",
            ],
            "per_obstacle_controller_visible": [
                "raw center", "belief center", "posterior sigma", "reported sigma",
                "innovation", "outlier flag", "validity bit", "observable jump score", "radius",
            ],
            "metadata": [
                "method", "seed", "episode_id", "step", "time", "train",
                "stress", "disturbance_stress", "sensor_stream_hash",
            ],
        },
        "behavior_policy_likelihood_stored": False,
        "evaluator_truth_stored_in_replay": False,
    }
    (out / "control_information_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    if animate:
        make_animation(
            out,
            eval_type="stress",
            frames=max(2, round(display_time_s * 9)),
            fps=9,
            display_time_s=display_time_s,
        )
    write_acceptance_audit(out)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assemble V17.1 severe-sensor method partials")
    ap.add_argument("--partial-root", default="_partials")
    ap.add_argument("--out", default="demo_results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2)
    ap.add_argument("--stress-test", type=float, default=6.0)
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument("--display-seed", type=int, default=4)
    ap.add_argument(
        "--display-seconds", type=float, default=10.0,
        help="plot/animation window; does not shorten the evaluation horizon",
    )
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--zip", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    root = Path(__file__).resolve().parent
    partial = Path(a.partial_root)
    out = Path(a.out)
    if not partial.is_absolute():
        partial = root / partial
    if not out.is_absolute():
        out = root / out
    assemble(
        root, partial, out, a.epochs, a.seeds, a.moderate_test,
        a.stress_test, a.disturbance_stress, a.display_seed,
        a.animate, a.clean, a.display_seconds,
    )
    if a.zip:
        print(package_zip(root))
    print(f"[done] {out}")
