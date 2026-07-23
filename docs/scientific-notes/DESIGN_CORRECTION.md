# V17.1 design and severity correction

V17.1 preserves the information-flow correction: hidden static truth is separated from one common noisy detector stream and method-specific post-processing. The software history records an earlier severe multiplier of 4.0 and the later addition of the reported 6 stress case while retaining the moderate 2.2 case. The 6 case was not independently preregistered; it is reported as a designed stress benchmark, not as an untouched confirmatory test.

## Fair comparison

All methods share the map, hidden truth, start and goal, actor initialization, exploration, raw sensor arrays, validity bits, detector jumps, disturbance, cost, horizon, and evaluation seeds. No method-specific sensor corruption is permitted.

## Why stronger uncertainty is legitimate

The severe setting models a detector under registration shift, correlated error, temporary holds, and large but visible outliers. It is an out-of-distribution perception test, not an attempt to sabotage baselines. The same episode seed is evaluated at moderate and severe multipliers, and the full robustness sweep is reported.

## Why Full may estimate better

Full receives no ground-truth center labels. Adaptive replay retains boundary and innovation events, and the estimator forms a robust geometric-median anchor from replayed raw measurements. The claim must be supported jointly by common-stream fairness, lower belief RMSE, safety, target completion, clearance, cost, and replay audits.
