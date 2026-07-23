# Hidden-truth oracle invariance audit

The evaluator-only true centers are translated while all controller-visible beliefs, uncertainty radii, state, and known-map geometry are held fixed.
Nominal and executed actions must remain unchanged.

| method     |   hidden_truth_translation_x_m |   hidden_truth_translation_y_m |   max_nominal_action_difference |   max_executed_action_difference |   pass_no_oracle_dependence |
|:-----------|-------------------------------:|-------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|
| AC         |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |
| AC+CBF     |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |
| AC+CBF+PER |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |
| AC+CBF+UE  |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |
| AC+CBF+AER |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |
| Full       |                           0.43 |                          -0.31 |                               0 |                                0 |                           1 |

**Status: PASS**