#!/usr/bin/env python3
"""Descriptive post-hoc sensitivity check for the Full method's AER weights.

This diagnostic is deliberately separate from the manuscript experiment runner.
It is not a tuning, model-selection, or confirmatory procedure, and it does not
overwrite any headline result.  Starting from the fixed AER vector, it changes
one component by -20% or +20% and renormalizes all four weights to sum to one.
Each configuration is retrained from scratch with the same five seeds, ten
training episode IDs, curriculum, and post-training evaluation episode ID used
by ``generate_method_partial.py``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from safe_ac_repro.simulation import (
    MethodState,
    ReplayBuffer,
    World,
    clone_for_eval,
    method_specs,
    simulate_episode,
)


EPOCHS = 10
SEEDS = tuple(range(5))
MODERATE_MULTIPLIER = 2.2
EXTREME_MULTIPLIER = 6.0
DISTURBANCE_STRESS = 1.0
TRAIN_EPISODE_IDS = tuple(range(EPOCHS))
EVALUATION_EPISODE_ID = 1001
COMPONENTS = ("td", "safety", "uncertainty", "novelty")


def sensitivity_configurations(
    nominal: Sequence[float],
) -> List[Dict[str, object]]:
    """Return nominal plus one-at-a-time relative +/-20% perturbations."""
    base = np.asarray(nominal, dtype=float)
    if base.shape != (4,) or np.any(base <= 0.0):
        raise ValueError(f"expected four positive AER weights, got {nominal!r}")
    if not np.isclose(base.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"nominal AER weights must sum to one, got {base.sum():.16g}")

    configs: List[Dict[str, object]] = [{
        "configuration": "nominal",
        "perturbed_component": "none",
        "perturbation_percent": 0,
        "weights": tuple(float(x) for x in base),
    }]
    for index, component in enumerate(COMPONENTS):
        for percent in (-20, 20):
            perturbed = base.copy()
            perturbed[index] *= 1.0 + percent / 100.0
            perturbed /= perturbed.sum()
            configs.append({
                "configuration": f"{component}_{'minus' if percent < 0 else 'plus'}20",
                "perturbed_component": component,
                "perturbation_percent": percent,
                "weights": tuple(float(x) for x in perturbed),
            })
    return configs


def _weight_columns(weights: Sequence[float]) -> Dict[str, float]:
    return {f"weight_{name}": float(value) for name, value in zip(COMPONENTS, weights)}


def run_diagnostic() -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the bounded 9-configuration x 5-seed diagnostic."""
    original_weights = tuple(float(x) for x in ReplayBuffer.AER_WEIGHTS)
    configs = sensitivity_configurations(original_weights)
    full_spec = next(spec for spec in method_specs() if spec.key == "Full")
    world = World()
    seed_rows: List[Dict[str, object]] = []
    configuration_runtime: Dict[str, float] = {}
    diagnostic_start = time.perf_counter()

    try:
        for config_order, config in enumerate(configs):
            name = str(config["configuration"])
            weights = tuple(float(x) for x in config["weights"])
            ReplayBuffer.AER_WEIGHTS = weights
            config_start = time.perf_counter()
            print(
                f"[configuration {config_order + 1}/{len(configs)}] "
                f"{name}: {weights}",
                flush=True,
            )

            for seed in SEEDS:
                print(f"  [seed {seed}]", flush=True)
                state = MethodState(full_spec, world, seed)
                for episode_id in TRAIN_EPISODE_IDS:
                    simulate_episode(
                        state,
                        episode_id=episode_id,
                        stress=1.0 + 0.06 * episode_id,
                        disturbance_stress=DISTURBANCE_STRESS,
                        train=True,
                        record=False,
                    )

                # Match the main runner: freeze the trained state, then use an
                # independent evaluation clone for each severity.  Moderate and
                # extreme deliberately share episode ID 1001.
                trained = clone_for_eval(state)
                for evaluation, multiplier in (
                    ("moderate", MODERATE_MULTIPLIER),
                    ("exploratory_extreme", EXTREME_MULTIPLIER),
                ):
                    result, _ = simulate_episode(
                        clone_for_eval(trained),
                        episode_id=EVALUATION_EPISODE_ID,
                        stress=multiplier,
                        disturbance_stress=DISTURBANCE_STRESS,
                        train=False,
                        record=False,
                    )
                    result.update({
                        "configuration_order": config_order,
                        "configuration": name,
                        "perturbed_component": config["perturbed_component"],
                        "perturbation_percent": config["perturbation_percent"],
                        **_weight_columns(weights),
                        "evaluation": evaluation,
                        "posthoc_descriptive_only": True,
                        "training_epochs": EPOCHS,
                        "training_episode_ids": "0-9",
                        "evaluation_episode_id": EVALUATION_EPISODE_ID,
                    })
                    result["violation_rate_pct"] = 100.0 * float(result["violation_rate"])
                    result["cbf_intervention_pct"] = 100.0 * float(result["cbf_intervention_rate"])
                    seed_rows.append(result)

            configuration_runtime[name] = time.perf_counter() - config_start
            print(
                f"  completed {name} in {configuration_runtime[name]:.3f} s",
                flush=True,
            )
    finally:
        # The simulator stores these as a class-level setting.  Always restore
        # the original vector, including after interruption or an exception.
        ReplayBuffer.AER_WEIGHTS = original_weights

    total_runtime = time.perf_counter() - diagnostic_start
    if tuple(float(x) for x in ReplayBuffer.AER_WEIGHTS) != original_weights:
        raise RuntimeError("ReplayBuffer.AER_WEIGHTS was not restored")

    seed_df = pd.DataFrame(seed_rows)
    leading = [
        "configuration_order", "configuration", "perturbed_component",
        "perturbation_percent", "weight_td", "weight_safety",
        "weight_uncertainty", "weight_novelty", "evaluation", "method",
        "seed", "training_epochs", "training_episode_ids",
        "evaluation_episode_id", "stress", "disturbance_stress",
        "posthoc_descriptive_only",
    ]
    seed_df = seed_df[leading + [c for c in seed_df.columns if c not in leading]]
    seed_df = seed_df.sort_values(
        ["configuration_order", "evaluation", "seed"], kind="stable",
    ).reset_index(drop=True)

    aggregate_rows: List[Dict[str, object]] = []
    for (config_order, configuration, evaluation), group in seed_df.groupby(
        ["configuration_order", "configuration", "evaluation"], sort=True,
    ):
        first = group.iloc[0]
        row: Dict[str, object] = {
            "configuration_order": int(config_order),
            "configuration": configuration,
            "perturbed_component": first["perturbed_component"],
            "perturbation_percent": int(first["perturbation_percent"]),
            **{f"weight_{name}": float(first[f"weight_{name}"]) for name in COMPONENTS},
            "evaluation": evaluation,
            "stress_multiplier": float(first["stress"]),
            "n_seeds": int(len(group)),
            "contact_samples_total": int(group["violation_count"].sum()),
            "colliding_seeds": int((group["violation_count"] > 0).sum()),
            "known_static_contact_samples_total": int(group["known_static_violation_count"].sum()),
            "successful_seeds": int(group["reached"].sum()),
            "success_rate_pct": 100.0 * float(group["reached"].mean()),
            "configuration_runtime_s": configuration_runtime[configuration],
            "posthoc_descriptive_only": True,
        }
        for metric in (
            "cost", "violation_rate_pct", "min_clearance_cm",
            "min_uncertain_true_clearance_cm", "belief_center_rmse_cm",
            "goal_error_cm", "cbf_intervention_pct", "runtime_ms_per_episode",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["evaluation", "configuration_order"], kind="stable",
    ).reset_index(drop=True)

    # Paired configurations use identical seeds.  Deltas are descriptive shifts
    # in aggregate means relative to the fixed nominal-weight rerun.
    delta_metrics = (
        "cost_mean", "violation_rate_pct_mean", "min_clearance_cm_mean",
        "min_uncertain_true_clearance_cm_mean", "belief_center_rmse_cm_mean",
        "goal_error_cm_mean", "success_rate_pct", "cbf_intervention_pct_mean",
    )
    for evaluation in aggregate_df["evaluation"].unique():
        mask = aggregate_df["evaluation"].eq(evaluation)
        nominal_row = aggregate_df[mask & aggregate_df["configuration"].eq("nominal")].iloc[0]
        for metric in delta_metrics:
            aggregate_df.loc[mask, f"delta_vs_nominal_{metric}"] = (
                aggregate_df.loc[mask, metric] - float(nominal_row[metric])
            )

    return seed_df, aggregate_df, total_runtime


def _mean_sd(row: pd.Series, stem: str, digits: int = 2) -> str:
    return f"{row[f'{stem}_mean']:.{digits}f} +/- {row[f'{stem}_std']:.{digits}f}"


def markdown_report(aggregate_df: pd.DataFrame, total_runtime: float) -> str:
    sections: List[str] = [
        "# Descriptive post-hoc AER replay-weight sensitivity diagnostic",
        "",
        "This is a bounded, descriptive post-hoc diagnostic. It was not used for "
        "tuning, model selection, method definition, or selection of reported "
        "headline results. The nominal replay weights remain fixed at "
        "(TD, safety, uncertainty, novelty) = (0.18, 0.39, 0.30, 0.13).",
        "",
        "Each alternative changes exactly one nominal component by -20% or +20% "
        "and then renormalizes all four components to sum to one. Every row uses "
        "five paired seeds (0-4), ten training episodes (IDs 0-9), the unchanged "
        "training curriculum `stress = 1.0 + 0.06 * episode_id`, and the same "
        "post-training evaluation episode ID 1001 for moderate multiplier 2.2 "
        "and exploratory extreme multiplier 6.0. Disturbance stress is 1.0.",
        "",
        "Values are mean +/- sample standard deviation across the five seeds. "
        "Contact samples and successful seeds are exact totals. Runtime is "
        "machine-dependent and is included only as an execution record.",
    ]

    for evaluation, title in (
        ("moderate", "Moderate evaluation (multiplier 2.2)"),
        ("exploratory_extreme", "Exploratory extreme evaluation (multiplier 6.0)"),
    ):
        group = aggregate_df[aggregate_df["evaluation"].eq(evaluation)].copy()
        table = pd.DataFrame({
            "Configuration": group["configuration"],
            "Weights (TD/S/U/N)": group.apply(
                lambda r: "/".join(f"{r[f'weight_{name}']:.6f}" for name in COMPONENTS), axis=1,
            ),
            "Cost": group.apply(lambda r: _mean_sd(r, "cost"), axis=1),
            "Contacts (samples/colliding seeds)": group.apply(
                lambda r: f"{int(r.contact_samples_total)}/{int(r.colliding_seeds)}", axis=1,
            ),
            "Violation rate (%)": group.apply(lambda r: _mean_sd(r, "violation_rate_pct"), axis=1),
            "Global clearance (cm)": group.apply(lambda r: _mean_sd(r, "min_clearance_cm"), axis=1),
            "Uncertain clearance (cm)": group.apply(
                lambda r: _mean_sd(r, "min_uncertain_true_clearance_cm"), axis=1,
            ),
            "Belief RMSE (cm)": group.apply(lambda r: _mean_sd(r, "belief_center_rmse_cm"), axis=1),
            "Goal error (cm)": group.apply(lambda r: _mean_sd(r, "goal_error_cm"), axis=1),
            "Successes": group.apply(lambda r: f"{int(r.successful_seeds)}/5", axis=1),
            "CBF intervention (%)": group.apply(lambda r: _mean_sd(r, "cbf_intervention_pct"), axis=1),
        })
        sections.extend(["", f"## {title}", "", table.to_markdown(index=False)])

    runtime_table = aggregate_df[
        aggregate_df["evaluation"].eq("moderate")
    ][["configuration", "configuration_runtime_s"]].copy()
    runtime_table.columns = ["Configuration", "Training + two evaluations for five seeds (s)"]
    runtime_column = "Training + two evaluations for five seeds (s)"
    runtime_table[runtime_column] = runtime_table[runtime_column].map(lambda x: f"{x:.3f}")
    sections.extend([
        "",
        "## Execution record",
        "",
        runtime_table.to_markdown(index=False),
        "",
        f"Total diagnostic wall-clock runtime: {total_runtime:.3f} s.",
        "",
        "The seed-level CSV contains the unrounded simulator outputs. The "
        "aggregate CSV contains unrounded means, sample standard deviations, and "
        "descriptive deltas from the nominal rerun. `ReplayBuffer.AER_WEIGHTS` was "
        "restored to its original tuple after execution.",
        "",
    ])
    return "\n".join(sections)


def seed_markdown_report(seed_df: pd.DataFrame) -> str:
    """Render the key seed-level outcomes; the companion CSV is unrounded."""
    sections = [
        "# Seed-level descriptive post-hoc AER weight sensitivity results",
        "",
        "These are the paired seed-level outcomes behind the aggregate report. "
        "The diagnostic is descriptive and post hoc, not a tuning or selection "
        "exercise. Values in this Markdown view are printed to six decimals; "
        "the companion CSV retains the unrounded simulator outputs.",
    ]
    for evaluation, title in (
        ("moderate", "Moderate evaluation (multiplier 2.2)"),
        ("exploratory_extreme", "Exploratory extreme evaluation (multiplier 6.0)"),
    ):
        group = seed_df[seed_df["evaluation"].eq(evaluation)].copy()
        table = pd.DataFrame({
            "Configuration": group["configuration"],
            "Seed": group["seed"].astype(int),
            "Cost": group["cost"].map(lambda x: f"{x:.6f}"),
            "Contact samples": group["violation_count"].astype(int),
            "Violation rate (%)": group["violation_rate_pct"].map(lambda x: f"{x:.6f}"),
            "Global clearance (cm)": group["min_clearance_cm"].map(lambda x: f"{x:.6f}"),
            "Uncertain clearance (cm)": group["min_uncertain_true_clearance_cm"].map(
                lambda x: f"{x:.6f}",
            ),
            "Belief RMSE (cm)": group["belief_center_rmse_cm"].map(lambda x: f"{x:.6f}"),
            "Goal error (cm)": group["goal_error_cm"].map(lambda x: f"{x:.6f}"),
            "Reached": group["reached"].astype(int),
            "CBF intervention (%)": group["cbf_intervention_pct"].map(lambda x: f"{x:.6f}"),
        })
        sections.extend(["", f"## {title}", "", table.to_markdown(index=False)])
    sections.append("")
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed descriptive post-hoc Full/AER weight diagnostic",
    )
    parser.add_argument(
        "--out",
        default="outputs/sensitivity",
        help="output directory (default: outputs/sensitivity)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    seed_df, aggregate_df, total_runtime = run_diagnostic()
    seed_path = out / "posthoc_aer_weight_sensitivity_seed_metrics.csv"
    aggregate_path = out / "posthoc_aer_weight_sensitivity_aggregate.csv"
    seed_report_path = out / "posthoc_aer_weight_sensitivity_seed_metrics.md"
    report_path = out / "posthoc_aer_weight_sensitivity_aggregate.md"
    seed_df.to_csv(seed_path, index=False, encoding="utf-8", float_format="%.17g")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8", float_format="%.17g")
    seed_report_path.write_text(seed_markdown_report(seed_df), encoding="utf-8")
    report_path.write_text(markdown_report(aggregate_df, total_runtime), encoding="utf-8")
    print(f"wrote {seed_path}", flush=True)
    print(f"wrote {aggregate_path}", flush=True)
    print(f"wrote {seed_report_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    print(f"total runtime: {total_runtime:.3f} s", flush=True)


if __name__ == "__main__":
    main()
