# Safe actor-critic AER/UE reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21515579.svg)](https://doi.org/10.5281/zenodo.21515579)

Code and results for:

> *Toward Integrating Adaptive Experience Replay and Online Uncertainty
> Estimation in Safe Actor-Critic Optimal Control*

## Start here

There is one public entry point:

```bash
python run.py verify
```

That command verifies every tracked file and runs the automated scientific
contract tests. It does not retrain the agents.

To run the complete benchmark:

```bash
python run.py reproduce --animate
```

The new run is written to `outputs/reproduction/`; the published results are
never overwritten.

## What should I look at?

| If you want... | Open... |
|---|---|
| the corrected manuscript safety-margin figure | [`results/figures/fig03_safety_margin_over_time.png`](results/figures/fig03_safety_margin_over_time.png) |
| the main numerical results | [`results/tables/paper_ready_evaluation_table.md`](results/tables/paper_ready_evaluation_table.md) |
| the animation | [`results/animations/v17_1_shared_sensor_uncertainty_stress_seed4.mp4`](results/animations/v17_1_shared_sensor_uncertainty_stress_seed4.mp4) |
| all published outputs | [`results/README.md`](results/README.md) |
| exact reproduction instructions | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| the implementation | [`src/safe_ac_repro/simulation.py`](src/safe_ac_repro/simulation.py) |
| the experimental protocol | [`results/protocol/experimental_protocol.json`](results/protocol/experimental_protocol.json) |

## Repository map

```text
.
|-- run.py                  # the only command users need
|-- src/safe_ac_repro/      # simulation and learning implementation
|-- scripts/                # workflow internals called by run.py
|-- results/                # published paper-consistent outputs
|   |-- figures/
|   |-- animations/
|   |-- tables/
|   |-- data/
|   |-- audits/
|   `-- protocol/
|-- docs/                   # reproducibility and scientific notes
`-- tests/                  # fairness, no-oracle, and release-contract tests
```

## Evaluation horizon versus display window

The registered evaluation always runs for 280 transitions:

```text
280 x 0.055 s = 15.4 s
```

The corrected Figure 4 safety-margin panel and animation show the informative
first 10 seconds. The remaining 5.4 seconds are post-goal dwell and remain in
the CSV data and all numerical metrics. Display cropping never changes the
evaluation.

## Reproduction commands

Install Python 3.11 or newer and the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Then choose one command:

```bash
python run.py verify                         # seconds; no retraining
python run.py reproduce --animate            # complete independent rerun
python run.py postprocess --animate           # rebuild an existing run
python run.py sensitivity                     # separate post-hoc diagnostic
python run.py help                            # command summary
```

The complete rerun uses the paper settings by default: 10 training episodes,
five seeds, moderate multiplier 2.2, exploratory extreme multiplier 6.0,
display seed 4, and a 10-second visualization window.

## Scientific scope

The uncertain physical obstacles are static, but their centers are hidden from
the controllers. Every method receives the same raw detector stream. True
centers and true clearance are evaluator-only and are excluded from the actor,
CBF, critic, estimator, replay priority, and replay buffer.

This is a two-dimensional simulation proof of concept. It does not claim
hardware validation, an unbiased off-policy policy gradient, or an
unconditional stochastic safety certificate. Detailed design and severity
notes are indexed in [`docs/README.md`](docs/README.md).

## Citation and license

The immutable paper-consistent release is
[`v1.0.0`](https://github.com/SDNT8810/safe-actor-critic-aer-ue-reproducibility/releases/tag/v1.0.0),
archived at [DOI 10.5281/zenodo.21515579](https://doi.org/10.5281/zenodo.21515579).
Machine-readable citation metadata are in [`CITATION.cff`](CITATION.cff).

Code and bundled data are released under the [MIT License](LICENSE).
