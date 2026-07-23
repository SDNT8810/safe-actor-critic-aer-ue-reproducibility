# V17.1 release regression audit

**Regression status: PASS**

This outcome-dependent audit checks reproduction of the released run; it is not independent validation or a confirmatory acceptance criterion.

- Shared raw sensor stream verified across methods and evaluation tiers: PASS
- Hidden true-center oracle invariance: PASS
- Full zero collisions, positive clearance, and target completion in every severe seed: PASS
- Zero known-static-wall collisions across all methods: PASS
- Belief RMSE ordering Full < UE-only < raw/no-UE: PASS
- Required figures and MP4/GIF exist: PASS

## Benchmark informativeness diagnostic

- Non-Full methods with at least one severe-test collision: 4/5.
- At least three non-Full methods exhibit a collision: YES.
- This diagnostic is deliberately not an acceptance criterion; a valid ablation is allowed to remain safe.

## Severe-test means

| method     |   violation_count |   violating_seeds |   min_clearance_cm |   worst_seed_clearance_cm |   min_uncertain_true_clearance_cm |     cost |   belief_center_rmse_cm |   goal_error_cm |   reached_rate |   known_static_violation_count |
|:-----------|------------------:|------------------:|-------------------:|--------------------------:|----------------------------------:|---------:|------------------------:|----------------:|---------------:|-------------------------------:|
| AC         |               7.8 |                 5 |           -1.75682 |                 -2.64881  |                          -1.75682 | 46.9924  |                16.708   |         4.00001 |            1   |                              0 |
| AC+CBF     |               4.6 |                 5 |           -1.25964 |                 -1.80773  |                          -1.25964 | 31.0005  |                16.708   |         4.62495 |            1   |                              0 |
| AC+CBF+AER |               3   |                 4 |           -0.34709 |                 -1.65169  |                          -0.34709 | 22.6182  |                16.708   |         4.36879 |            1   |                              0 |
| AC+CBF+PER |               6.2 |                 4 |           -1.04239 |                 -2.80506  |                          -1.04239 | 39.0838  |                16.708   |         4.46622 |            1   |                              0 |
| AC+CBF+UE  |               0   |                 0 |            3.21564 |                  0.658334 |                           3.94422 |  8.96101 |                11.0791  |        17.7187  |            0.8 |                              0 |
| Full       |               0   |                 0 |            2.70715 |                  2.37186  |                           6.952   |  7.63205 |                 3.51668 |         4.69578 |            1   |                              0 |

## Moderate-test means

| method     |   violation_count |   min_clearance_cm |   belief_center_rmse_cm |   goal_error_cm |   reached_rate |     cost |
|:-----------|------------------:|-------------------:|------------------------:|----------------:|---------------:|---------:|
| AC         |                 1 |          0.0268019 |                 6.00879 |         4.67359 |              1 | 11.6165  |
| AC+CBF     |                 0 |          1.00068   |                 6.00879 |         4.44212 |              1 |  6.8948  |
| AC+CBF+AER |                 0 |          1.96608   |                 6.00879 |         4.38659 |              1 |  6.71923 |
| AC+CBF+PER |                 0 |          1.41264   |                 6.00879 |         4.42924 |              1 |  6.68005 |
| AC+CBF+UE  |                 0 |          3.22941   |                 5.09433 |         4.51933 |              1 |  7.35668 |
| Full       |                 0 |          3.72743   |                 3.44227 |         4.40013 |              1 |  6.9388  |

## Information-flow audit

- True uncertain-obstacle centers are used only for sensor generation, physical evaluation, and figures.
- The replay buffer stores a sanitized controller-visible transition without true centers, true clearances, hidden fault labels, or evaluator errors.
- Actor and CBF receive method beliefs, not evaluator truth.
- Replay safety priority uses estimated barrier slack, correction, CBF activation, and observed contact.
- Replay uncertainty priority uses innovation, visible frame-to-frame measurement change, and validity bits.
- The state-value critic uses realized cost and next state produced by the executed safe action; its TD residual directly drives a biased replay-weighted score-function actor surrogate without behavior-policy importance correction.
- Full replay calibration uses only raw measurement samples and a geometric median.
- Moderate and additional extreme tests use the same episode seed; severity scales error amplitudes and detector hold-onset probability.
- Robustness-sweep levels use one fixed episode seed, reducing the severity/random-realization confound.
