# Shared-sensor fairness audit

**Summary status: PASS**

For each seed/evaluation, all six methods must have one common SHA-256 sensor-stream hash, identical raw-sensor RMSE, identical hold fraction, and identical observable jump statistics.
The seed-0 pointwise audit additionally requires exactly zero x/y measurement spread across methods at every step.

| eval_type   |   seed |   methods_present |   unique_sensor_stream_hashes |   raw_sensor_rmse_range_cm |   sensor_hold_fraction_range |   sensor_jump_mean_range |   pass_identical_common_sensor |
|:------------|-------:|------------------:|------------------------------:|---------------------------:|-----------------------------:|-------------------------:|-------------------------------:|
| nominal     |      0 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| nominal     |      1 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| nominal     |      2 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| nominal     |      3 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| nominal     |      4 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| moderate    |      0 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| moderate    |      1 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| moderate    |      2 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| moderate    |      3 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| moderate    |      4 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| stress      |      0 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| stress      |      1 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| stress      |      2 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| stress      |      3 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |
| stress      |      4 |                 6 |                             1 |                          0 |                            0 |                        0 |                              1 |

Pointwise seed-0 equality: PASS.