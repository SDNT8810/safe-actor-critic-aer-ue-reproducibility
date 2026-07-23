# Descriptive post-hoc AER replay-weight sensitivity diagnostic

This is a bounded, descriptive post-hoc diagnostic. It was not used for tuning, model selection, method definition, or selection of reported headline results. The nominal replay weights remain fixed at (TD, safety, uncertainty, novelty) = (0.18, 0.39, 0.30, 0.13).

Each alternative changes exactly one nominal component by -20% or +20% and then renormalizes all four components to sum to one. Every row uses five paired seeds (0-4), ten training episodes (IDs 0-9), the unchanged training curriculum `stress = 1.0 + 0.06 * episode_id`, and the same post-training evaluation episode ID 1001 for moderate multiplier 2.2 and exploratory extreme multiplier 6.0. Disturbance stress is 1.0.

Values are mean +/- sample standard deviation across the five seeds. Contact samples and successful seeds are exact totals. Runtime is machine-dependent and is included only as an execution record.

## Moderate evaluation (multiplier 2.2)

| Configuration       | Weights (TD/S/U/N)                  | Cost          | Contacts (samples/colliding seeds)   | Violation rate (%)   | Global clearance (cm)   | Uncertain clearance (cm)   | Belief RMSE (cm)   | Goal error (cm)   | Successes   | CBF intervention (%)   |
|:--------------------|:------------------------------------|:--------------|:-------------------------------------|:---------------------|:------------------------|:---------------------------|:-------------------|:------------------|:------------|:-----------------------|
| nominal             | 0.180000/0.390000/0.300000/0.130000 | 6.94 +/- 0.30 | 0/0                                  | 0.00 +/- 0.00        | 3.73 +/- 0.36           | 4.55 +/- 1.26              | 3.44 +/- 0.22      | 4.40 +/- 0.75     | 5/5         | 21.50 +/- 3.27         |
| td_minus20          | 0.149378/0.404564/0.311203/0.134855 | 6.97 +/- 0.29 | 0/0                                  | 0.00 +/- 0.00        | 3.76 +/- 0.37           | 4.58 +/- 1.25              | 3.44 +/- 0.21      | 4.46 +/- 0.65     | 5/5         | 21.79 +/- 3.66         |
| td_plus20           | 0.208494/0.376448/0.289575/0.125483 | 6.99 +/- 0.27 | 0/0                                  | 0.00 +/- 0.00        | 3.77 +/- 0.34           | 4.59 +/- 1.24              | 3.44 +/- 0.20      | 4.47 +/- 0.65     | 5/5         | 21.64 +/- 3.57         |
| safety_minus20      | 0.195228/0.338395/0.325380/0.140998 | 6.96 +/- 0.24 | 0/0                                  | 0.00 +/- 0.00        | 3.77 +/- 0.37           | 4.53 +/- 1.25              | 3.46 +/- 0.20      | 4.45 +/- 0.64     | 5/5         | 21.64 +/- 3.41         |
| safety_plus20       | 0.166976/0.434137/0.278293/0.120594 | 6.98 +/- 0.29 | 0/0                                  | 0.00 +/- 0.00        | 3.74 +/- 0.40           | 4.54 +/- 1.28              | 3.43 +/- 0.22      | 4.46 +/- 0.63     | 5/5         | 21.57 +/- 3.17         |
| uncertainty_minus20 | 0.191489/0.414894/0.255319/0.138298 | 6.99 +/- 0.28 | 0/0                                  | 0.00 +/- 0.00        | 3.78 +/- 0.36           | 4.60 +/- 1.23              | 3.45 +/- 0.21      | 4.46 +/- 0.65     | 5/5         | 21.93 +/- 3.44         |
| uncertainty_plus20  | 0.169811/0.367925/0.339623/0.122642 | 6.96 +/- 0.29 | 0/0                                  | 0.00 +/- 0.00        | 3.78 +/- 0.35           | 4.59 +/- 1.22              | 3.46 +/- 0.19      | 4.43 +/- 0.64     | 5/5         | 21.71 +/- 3.07         |
| novelty_minus20     | 0.184805/0.400411/0.308008/0.106776 | 6.98 +/- 0.22 | 0/0                                  | 0.00 +/- 0.00        | 3.76 +/- 0.39           | 4.56 +/- 1.26              | 3.43 +/- 0.22      | 4.46 +/- 0.65     | 5/5         | 22.21 +/- 3.42         |
| novelty_plus20      | 0.175439/0.380117/0.292398/0.152047 | 6.96 +/- 0.25 | 0/0                                  | 0.00 +/- 0.00        | 3.76 +/- 0.33           | 4.57 +/- 1.24              | 3.42 +/- 0.21      | 4.46 +/- 0.64     | 5/5         | 21.86 +/- 4.08         |

## Exploratory extreme evaluation (multiplier 6.0)

| Configuration       | Weights (TD/S/U/N)                  | Cost          | Contacts (samples/colliding seeds)   | Violation rate (%)   | Global clearance (cm)   | Uncertain clearance (cm)   | Belief RMSE (cm)   | Goal error (cm)   | Successes   | CBF intervention (%)   |
|:--------------------|:------------------------------------|:--------------|:-------------------------------------|:---------------------|:------------------------|:---------------------------|:-------------------|:------------------|:------------|:-----------------------|
| nominal             | 0.180000/0.390000/0.300000/0.130000 | 7.63 +/- 0.44 | 0/0                                  | 0.00 +/- 0.00        | 2.71 +/- 0.23           | 6.95 +/- 0.26              | 3.52 +/- 0.55      | 4.70 +/- 0.51     | 5/5         | 29.14 +/- 4.69         |
| td_minus20          | 0.149378/0.404564/0.311203/0.134855 | 7.65 +/- 0.44 | 0/0                                  | 0.00 +/- 0.00        | 2.70 +/- 0.21           | 6.95 +/- 0.23              | 3.54 +/- 0.70      | 4.61 +/- 0.67     | 5/5         | 28.79 +/- 5.22         |
| td_plus20           | 0.208494/0.376448/0.289575/0.125483 | 7.66 +/- 0.44 | 0/0                                  | 0.00 +/- 0.00        | 2.68 +/- 0.19           | 6.99 +/- 0.17              | 3.50 +/- 0.62      | 4.67 +/- 0.49     | 5/5         | 28.14 +/- 5.22         |
| safety_minus20      | 0.195228/0.338395/0.325380/0.140998 | 7.65 +/- 0.35 | 0/0                                  | 0.00 +/- 0.00        | 2.68 +/- 0.21           | 6.95 +/- 0.33              | 3.58 +/- 0.67      | 4.60 +/- 0.65     | 5/5         | 28.43 +/- 5.28         |
| safety_plus20       | 0.166976/0.434137/0.278293/0.120594 | 7.64 +/- 0.42 | 0/0                                  | 0.00 +/- 0.00        | 2.70 +/- 0.22           | 6.96 +/- 0.25              | 3.49 +/- 0.65      | 4.62 +/- 0.65     | 5/5         | 28.14 +/- 4.94         |
| uncertainty_minus20 | 0.191489/0.414894/0.255319/0.138298 | 7.66 +/- 0.43 | 0/0                                  | 0.00 +/- 0.00        | 2.71 +/- 0.22           | 6.96 +/- 0.25              | 3.55 +/- 0.69      | 4.53 +/- 0.80     | 5/5         | 28.79 +/- 4.81         |
| uncertainty_plus20  | 0.169811/0.367925/0.339623/0.122642 | 7.64 +/- 0.43 | 0/0                                  | 0.00 +/- 0.00        | 2.72 +/- 0.22           | 6.97 +/- 0.27              | 3.58 +/- 0.70      | 4.52 +/- 0.80     | 5/5         | 28.86 +/- 4.99         |
| novelty_minus20     | 0.184805/0.400411/0.308008/0.106776 | 7.67 +/- 0.39 | 0/0                                  | 0.00 +/- 0.00        | 2.69 +/- 0.22           | 6.97 +/- 0.25              | 3.53 +/- 0.68      | 4.59 +/- 0.66     | 5/5         | 28.57 +/- 5.10         |
| novelty_plus20      | 0.175439/0.380117/0.292398/0.152047 | 7.64 +/- 0.39 | 0/0                                  | 0.00 +/- 0.00        | 2.71 +/- 0.22           | 6.98 +/- 0.24              | 3.47 +/- 0.57      | 4.68 +/- 0.49     | 5/5         | 28.64 +/- 4.91         |

## Execution record

| Configuration       |   Training + two evaluations for five seeds (s) |
|:--------------------|------------------------------------------------:|
| nominal             |                                           7.85  |
| td_minus20          |                                           8.136 |
| td_plus20           |                                           8.097 |
| safety_minus20      |                                           8.128 |
| safety_plus20       |                                           8.197 |
| uncertainty_minus20 |                                           8.325 |
| uncertainty_plus20  |                                           8.342 |
| novelty_minus20     |                                           8.302 |
| novelty_plus20      |                                           8.377 |

Total diagnostic wall-clock runtime: 73.753 s.

The seed-level CSV contains the unrounded simulator outputs. The aggregate CSV contains unrounded means, sample standard deviations, and descriptive deltas from the nominal rerun. `ReplayBuffer.AER_WEIGHTS` was restored to its original tuple after execution.
