# Reproducibility and verification

## Environment

Python 3.11 or newer is recommended. Install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

## Fast verification

Compile all Python sources and run the information-flow tests:

```bash
python -m compileall -q .
python -m unittest -v tests/test_shared_sensor_and_no_oracle.py
```

## Regenerate tables, audits, and figures from bundled results

```bash
python postprocess_results.py --out demo_results --epochs 10 --seeds 5 \
  --moderate-test 2.2 --stress-test 6.0 --disturbance-stress 1.0 \
  --display-seconds 10 --animate
```

## Full independent rerun

```bash
python run_reproducible_demo.py --out reproduction_check --epochs 10 \
  --seeds 5 --moderate-test 2.2 --stress-test 6.0 \
  --disturbance-stress 1.0 --display-seed 4 \
  --display-seconds 10 --animate --clean
```

The full evaluation horizon is 280 transitions (15.4 s). The
`--display-seconds 10` option crops only Figure 3 and the animation; it does not
change simulation length or numerical metrics.

Machine-dependent runtime columns can differ. Deterministic numerical and
categorical outputs should otherwise match the bundled CSV results.

## Integrity manifest

`MANIFEST.sha256` records SHA-256 hashes for every tracked release file except
the manifest itself. Verify it with:

```bash
python verify_manifest.py
```

Release maintainers can regenerate it after intentional changes:

```bash
python verify_manifest.py --write
```
