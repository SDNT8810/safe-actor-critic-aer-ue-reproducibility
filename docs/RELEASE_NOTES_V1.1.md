# Version 1.1.0 release notes

Version 1.1.0 is the paper-consistent, reorganized reproducibility release.
It preserves the registered evaluation and numerical results from version
1.0.0 while making the public package substantially easier to navigate.

## Repository redesign

- Added `run.py` as the single public command-line entry point.
- Moved the implementation into `src/safe_ac_repro/`.
- Grouped workflow internals under `scripts/`.
- Organized paper outputs under `results/figures`, `results/animations`,
  `results/tables`, `results/data`, `results/audits`, and `results/protocol`.
- Consolidated reproducibility and scientific notes under `docs/`.
- Updated automated tests and the SHA-256 manifest for the new layout.

## Visualization clarification

The safety-margin figure and animation use a configurable display interval.
The registered evaluation horizon and all numerical metrics remain unchanged.
Visualization settings affect only plots and animations.

## Reproduction

Run:

```bash
python run.py verify
python run.py reproduce --animate
```

The first command verifies the release manifest and runs ten scientific
contract tests. The second performs the complete paper-consistent benchmark
and writes new output without overwriting the published results.
