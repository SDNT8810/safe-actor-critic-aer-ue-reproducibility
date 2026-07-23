# V17.1 exploratory extreme-noise rerun (sensor multiplier 6.0)

Only the perception-stress parameter is increased. Method definitions, training budget, map, fixed true obstacles, disturbance scale, seeds, and fairness rules are unchanged. Multiplier 6.0 was added after an earlier multiplier-4.0 run and is therefore reported descriptively, not as a preregistered confirmatory test.

## Reproduction

```bash
python run.py reproduce --out outputs/reproduction --epochs 10 --seeds 5 --moderate-test 2.2 --stress-test 6.0 --disturbance-stress 1.0 --display-seed 4 --animate --clean
```

## Extreme detector profile

- persistent registration-bias amplitude: 6.0--11.4 cm;
- sinusoidal-drift amplitude: 5.4 cm in x and 4.8 cm in y;
- jump-outlier initial amplitude: 30--42 cm;
- nominal jump decay: 16--22 samples (0.88--1.21 s), with scheduled injections allowed to overlap;
- colored-noise innovation standard deviation: 0.784 cm;
- white-noise standard deviation: 0.637 cm;
- hold-onset probability while inactive: 5.5% per step, followed by 2--4 repeated frames;
- detector-reported standard deviation: 1.0 cm, representing an overconfident detector under distribution shift.

The hidden physical obstacles remain static. Every method receives exactly the same raw center, validity, observable-jump, reported-scale, exploration, and disturbance arrays for a fixed seed and episode. The common Gaussian plant disturbance is not hard bounded; 0.012 m/s is a CBF design allowance at disturbance multiplier 1.0.

## Five-seed results

| Method | Seeds with contact | Mean contact samples | Global min. clearance (cm) | Uncertain-obstacle clearance (cm) | Belief RMSE (cm) | Goal success | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| AC | 5/5 | 7.8 | -1.76 +/- 0.80 | -1.76 +/- 0.80 | 16.71 +/- 1.55 | 5/5 | 46.99 +/- 19.28 |
| AC+CBF | 5/5 | 4.6 | -1.26 +/- 0.40 | -1.26 +/- 0.40 | 16.71 +/- 1.55 | 5/5 | 31.00 +/- 12.43 |
| AC+CBF+PER | 4/5 | 6.2 | -1.04 +/- 1.91 | -1.04 +/- 1.91 | 16.71 +/- 1.55 | 5/5 | 39.08 +/- 24.08 |
| AC+CBF+UE | 0/5 | 0.0 | 3.22 +/- 2.81 | 3.94 +/- 3.37 | 11.08 +/- 1.23 | 4/5 | 8.96 +/- 2.08 |
| AC+CBF+AER | 4/5 | 3.0 | -0.35 +/- 1.55 | -0.35 +/- 1.55 | 16.71 +/- 1.55 | 5/5 | 22.62 +/- 12.82 |
| Full | 0/5 | 0.0 | 2.71 +/- 0.23 | 6.95 +/- 0.26 | 3.52 +/- 0.55 | 5/5 | 7.63 +/- 0.44 |

No method contacts a known rectangular obstacle or workspace wall. Full and UE both have zero contact samples; UE has the slightly larger mean global minimum clearance but misses the goal in one seed, whereas Full reaches the goal in all five seeds and has lower mean cost and belief error. With five paired seeds, these outcomes are an exploratory joint trend rather than evidence of universal or statistical superiority.

The actor update is documented in `README.md` as a biased replay-weighted score-function surrogate: replayed samples have no behavior-policy importance correction. The archive does not claim an isolated performance gain from the learned value baseline.
