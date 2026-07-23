# Published results

This directory contains the paper-consistent outputs archived in release
`v1.0.0`. Readers do not need to inspect every file.

## Recommended entry points

1. [Main evaluation table](tables/paper_ready_evaluation_table.md)
2. [Corrected safety-margin figure](figures/fig03_safety_margin_over_time.png)
3. [Trajectory comparison](figures/fig01_full_trajectory_comparison.png)
4. [Animation](animations/v17_1_shared_sensor_uncertainty_stress_seed4.mp4)
5. [Acceptance audit](audits/simulation_acceptance_audit_v17_1.md)
6. [Complete experimental protocol](protocol/experimental_protocol.json)

## Directory guide

| Directory | Contents |
|---|---|
| [`figures/`](figures/) | The 13 paper and diagnostic figures |
| [`animations/`](animations/) | MP4, GIF, and preview image |
| [`tables/`](tables/) | Reader-friendly Markdown and manuscript-ready LaTeX tables |
| [`data/`](data/) | Raw trajectories, seed-level metrics, aggregated metrics, and CSV audits |
| [`audits/`](audits/) | Human-readable fairness, no-oracle, collision, and acceptance reports |
| [`protocol/`](protocol/) | Experimental constants, runtime environment, severity manifest, and information contract |

## Time convention

The evaluation CSVs retain all 280 transitions (15.4 seconds). Figure 3 and the
animation display only the first 10 seconds because the remaining 5.4 seconds
are post-goal dwell. No metric is computed from a shortened run.

## Recreate these results

Do not write into this directory. From the repository root, run:

```bash
python run.py reproduce --animate
```

Fresh outputs are created under `outputs/reproduction/`.
