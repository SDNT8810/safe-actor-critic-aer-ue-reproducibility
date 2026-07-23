# Safe actor-critic AER/UE reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21515579.svg)](https://doi.org/10.5281/zenodo.21515579)

Public code, deterministic simulation data, figures, animations, and audit
artifacts for the manuscript:

> *Toward Integrating Adaptive Experience Replay and Online Uncertainty
> Estimation in Safe Actor-Critic Optimal Control*

Repository:
<https://github.com/SDNT8810/safe-actor-critic-aer-ue-reproducibility>

This repository contains the paper-consistent V17.1 benchmark release. The
bundled results use the registered 280-transition evaluation horizon and match
the numerical tables reported in the manuscript.

## Scientific purpose

The two circular obstacles are **physically static**, while their true centers are hidden from every controller. Every method receives exactly the same raw detector stream. V17.1 strengthens only the perception corruption so that the benchmark exposes the difference between raw-center CBF protection, online filtering, adaptive replay, and their integration.

This package does **not** weaken a baseline, move the true obstacles, give Full extra labels, or select a different sensor stream per method. The moderate post-training evaluation at multiplier `2.2` is retained alongside the additional exploratory extreme rerun at `6`, and the robustness sweep is also saved. The software history records that `6` was added after an earlier 4.0 run; it is not presented as a preregistered confirmatory test. A baseline is allowed to remain safe; failures are not an acceptance requirement.

## What is shown

- red filled circle: fixed true physical obstacle, evaluator-only;
- dark-red outer circle: true robot-collision boundary;
- yellow `×`: common noisy/held/jumping detector center;
- green marker: UE-only filtered belief;
- blue marker and dashed envelope: Full replay-calibrated belief and robust safety envelope;
- path `×`: a physical collision evaluated against hidden truth.

The figures and animation use the same trajectory CSV, so their geometry and safety curves cannot silently disagree.

## Evaluation horizon versus display window

The evaluation protocol always runs for 280 transitions:

```text
280 × 0.055 s = 15.4 s
```

All visualization-seed methods enter the goal region during the first 10 s.
Figure 3 and the animation therefore default to a 10 s display window, omitting
the uninformative post-goal dwell tail. This is a visualization crop only; it
does not shorten the evaluation or change any reported metric. Use
`--display-seconds` to select a different visualization window.

## Severe perception profile

At `--stress-test 6`, the common detector has:

- episode registration-bias amplitude: 6.0–11.4 cm;
- smooth drift amplitudes: up to 5.4 cm in x and 4.8 cm in y;
- jump-outlier initial amplitude: 30–42 cm with a nominal 0.88–1.21 s decay per injection; scheduled injections can overlap;
- intermittent 2–4-frame holds, whose onset probability while inactive is 5.5% per step;
- detector-reported standard deviation fixed at 1.0 cm, deliberately representing an overconfident detector under distribution shift.

The true obstacle centers never move. `--disturbance-stress` independently scales the small common Gaussian AR(1)-plus-sinusoid plant disturbance. The CBF uses 0.012 m/s as a design allowance at disturbance multiplier 1.0; the stochastic generator is not hard bounded, so this allowance is not a safety certificate.

## Information available to each method

- `AC`, `AC+CBF`, `AC+CBF+PER`, and `AC+CBF+AER` use the common raw/held center;
- `AC+CBF+UE` uses a robust online constant-position filter;
- `Full` uses the same filter plus a static-center anchor learned self-supervised from adaptively replayed raw measurements.

No true center or true clearance enters the actor, CBF, critic, replay priority, estimator, or replay buffer. See `control_information_contract.json`, `sensor_fairness_audit.csv`, and `oracle_invariance_audit.csv`.

## Actor--critic update

The linear state-value critic is fitted from the realized cost and next state produced by the executed post-filter action. The waypoint actor uses a replay-weighted cost-advantage score-function update: the critic's clipped TD residual multiplies the normalized common Gaussian exploration draw, with the sign chosen to reduce cost. A controller-visible estimated-barrier gradient supplies an auxiliary safety-shaping term. Thus the critic directly changes the policy; it is not used only for replay priority. However, replayed transitions do not carry behavior-policy likelihoods and no importance ratios are applied, so this clipped, safety-shaped update is a biased off-policy surrogate rather than an unbiased on-policy policy-gradient estimator. The benchmark does not isolate a performance gain due to the value baseline alone. The common nominal policy also contains disclosed known-map and perceived-obstacle potential-field terms.

## Replay scores and priorities

Every stored transition uses controller-visible quantities only. The normalized scores are:

```text
TD score          = min(1, |δ_t| / 2.5)
safety score      = min(1, exp(-max(h_hat_t,0)/0.075)
                           + 0.34 I_CBF + 0.30 min(1, ||u-u_a||/0.11)
                           + 0.75 I_observed-contact)
uncertainty score = min(1, 0.58 mean(innovation_i / max(2.5 sigma_i,0.030))
                           + 0.30 mean(observable-jump_i)
                           + 0.28 (1-mean(valid_i)))
novelty score     = clipped state-cell novelty in [0,1]
```

Uniform replay (including `AC+CBF+UE`) samples uniformly. `AC+CBF+PER` uses TD priority only. `AC+CBF+AER` and `Full` use

```text
p_t = 0.18 TD + 0.39 safety + 0.30 uncertainty + 0.13 novelty + 1e-5.
```

Prioritized mini-batches are sampled **without replacement**. At each sequential draw, `AC+CBF+PER` uses probability proportional to `p_t**0.68`, while `AC+CBF+AER` and `Full` use probability proportional to `p_t**0.78`; the denominator is recomputed over entries not yet selected for that batch. Uniform rows sample without replacement at equal probability. Uniform insertion uses reservoir replacement. Prioritized replacement chooses a random candidate with probability `0.16` and otherwise the current minimum-priority entry; the new entry is accepted when its priority is no smaller, or with fallback probability `0.035`. Only sampled entries have their TD-dependent priority refreshed after each critic update.

`demo_results/experimental_protocol.json` records the complete plant, cost, actor, critic, waypoint/corridor, potential-field, estimator, CBF, sensor, disturbance, replay, seed, and selection-provenance constants. No formal validation-set or method-specific hyperparameter search is claimed.

The finite-buffer retention and sampling fractions are saved in `replay_finite_buffer_audit.csv` and `paper_ready_replay_audit_table.md`.

The separate `run_posthoc_aer_weight_sensitivity.py` diagnostic perturbs each nominal AER weight by -20% and +20% one at a time, renormalizes the four weights, and reruns Full at the moderate and exploratory extreme tiers. It is descriptive and was not used for tuning or headline selection. Its aggregate and seed-level CSV/Markdown outputs are stored under `demo_results/posthoc_aer_weight_sensitivity_*`.

## Reproduce the included run

```bash
python run_reproducible_demo.py --out demo_results --epochs 10 --seeds 5 --moderate-test 2.2 --stress-test 6.0 --disturbance-stress 1.0 --display-seed 4 --display-seconds 10 --animate --clean
```

Windows Command Prompt:

```bat
run_demo.bat
```

## Included moderate result

| Method | Mean violations | Seeds with violation | Mean minimum clearance (cm) | Belief RMSE (cm) | Goal error (cm) | Goal success | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| AC | 1.00 | 2/5 | 0.03 | 6.01 | 4.67 | 100% | 11.62 |
| AC+CBF | 0.00 | 0/5 | 1.00 | 6.01 | 4.44 | 100% | 6.89 |
| AC+CBF+PER | 0.00 | 0/5 | 1.41 | 6.01 | 4.43 | 100% | 6.68 |
| AC+CBF+UE | 0.00 | 0/5 | 3.23 | 5.09 | 4.52 | 100% | 7.36 |
| AC+CBF+AER | 0.00 | 0/5 | 1.97 | 6.01 | 4.39 | 100% | 6.72 |
| Full | 0.00 | 0/5 | 3.73 | 3.44 | 4.40 | 100% | 6.94 |

## Included severe result

| Method | Mean violations | Seeds with violation | Mean minimum clearance (cm) | Belief RMSE (cm) | Goal error (cm) | Goal success | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| AC | 7.80 | 5/5 | -1.76 | 16.71 | 4.00 | 100% | 46.99 |
| AC+CBF | 4.60 | 5/5 | -1.26 | 16.71 | 4.62 | 100% | 31.00 |
| AC+CBF+PER | 6.20 | 4/5 | -1.04 | 16.71 | 4.47 | 100% | 39.08 |
| AC+CBF+UE | 0.00 | 0/5 | 3.22 | 11.08 | 17.72 | 80% | 8.96 |
| AC+CBF+AER | 3.00 | 4/5 | -0.35 | 16.71 | 4.37 | 100% | 22.62 |
| Full | 0.00 | 0/5 | 2.71 | 3.52 | 4.70 | 100% | 7.63 |

Interpret the severe result jointly. The intended evidence is not “every ablation must collide.” It is that Full preserves safety, reaches the target, maintains positive true clearance, reduces hidden-center estimation error using only shared measurements, and remains competitive in cost. UE-only may remain collision-free but can become conservative or fail to finish; that is a legitimate ablation outcome rather than something to hide.

## Main outputs

- `demo_results/moderate_eval_metrics.csv`
- `demo_results/stress_eval_metrics.csv`
- `demo_results/evaluation_trajectories_seed0.csv`
- `demo_results/evaluation_trajectories_display_seed4.csv`
- `demo_results/shared_sensor_stream_severe_seed0.csv`
- `demo_results/sensor_severity_manifest.json`
- `demo_results/sensor_fairness_audit.csv`
- `demo_results/oracle_invariance_audit.csv`
- `demo_results/replay_finite_buffer_audit.csv`
- `demo_results/robustness_sweep.csv`
- `demo_results/robustness_sweep_collision_source_audit.csv`
- `demo_results/paper_ready_evaluation_table.md/.tex`
- `demo_results/paper_ready_moderate_evaluation_table.md/.tex`
- `demo_results/figures/fig01...fig13`
- `demo_results/animations/v17_1_shared_sensor_uncertainty_stress_seed4.mp4`
- `demo_results/animations/v17_1_shared_sensor_uncertainty_stress_seed4.gif`
- `demo_results/simulation_acceptance_audit_v17_1.md`
- `demo_results/RUN_README.md`
- `demo_results/DESIGN_CORRECTION.md`

Aggregate tables use all seeds. Display seed 4 is visualization-only and is explicitly recorded in the protocol.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

## Postprocess an existing completed run

```bash
python postprocess_results.py --out demo_results --epochs 10 --seeds 5 --moderate-test 2.2 --stress-test 6.0 --disturbance-stress 1.0 --animate
```

## License and citation

The software and bundled data are released under the MIT License. Citation
metadata are provided in `CITATION.cff`. The immutable `v1.0.0` release is
archived by Zenodo at <https://doi.org/10.5281/zenodo.21515579>.

See `DATA_AVAILABILITY.md` for the public-release scope and
`REPRODUCIBILITY.md` for the verification procedure.
