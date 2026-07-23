# Reproducibility and verification

Run every command below from the repository root.

## 1. Install

Python 3.11 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## 2. Fast verification

```bash
python run.py verify
```

This verifies `MANIFEST.sha256` and runs all automated tests. It checks the
registered 280-transition horizon, paper headline values, shared-sensor
fairness, hidden-truth isolation, replay sanitization, and known-map guard.

## 3. Complete independent rerun

```bash
python run.py reproduce --animate
```

Equivalent explicit form:

```bash
python run.py reproduce \
  --out outputs/reproduction \
  --epochs 10 \
  --seeds 5 \
  --moderate-test 2.2 \
  --stress-test 6.0 \
  --disturbance-stress 1.0 \
  --display-seed 4 \
  --display-seconds 10 \
  --animate \
  --clean
```

Evaluation metrics use every configured transition. `--display-seconds`
changes only plots and animations.

## 4. Rebuild an existing run

```bash
python run.py postprocess --out outputs/reproduction --animate
```

To rebuild only the animation:

```bash
python run.py animate --out outputs/reproduction
```

## 5. Compare with the published results

Fresh results are written under `outputs/reproduction/`. Published reference
files are organized under `results/`:

- aggregate tables: `results/tables/`;
- numerical CSVs and trajectories: `results/data/`;
- figures and animations: `results/figures/` and `results/animations/`;
- protocol and runtime metadata: `results/protocol/`; and
- human-readable audits: `results/audits/`.

Machine-dependent runtime fields can differ. Deterministic numerical and
categorical outputs should otherwise match.

## Maintainer-only manifest update

After an intentional tracked-file change:

```bash
python scripts/verify_manifest.py --write
python run.py verify
```
