# Severe perception-stress results (mean ± standard deviation across seeds)

| Method | Cost ↓ | Violation rate (%) ↓ | Minimum true clearance (cm) ↑ | Minimum true uncertain-obstacle clearance (cm) ↑ | Obstacle-belief RMSE (cm) ↓ | Goal error (cm) ↓ | Goal success (%) ↑ | CBF intervention (%) ↓ | Runtime/episode (ms) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AC | 46.99 ± 19.28 | 2.79 ± 1.32 | -1.76 ± 0.80 | -1.76 ± 0.80 | 16.71 ± 1.55 | **4.00 ± 0.69** | **100.0 ± 0.0** | N/A | **142.03 ± 13.87** |
| AC+CBF | 31.00 ± 12.43 | 1.64 ± 0.86 | -1.26 ± 0.40 | -1.26 ± 0.40 | 16.71 ± 1.55 | 4.62 ± 0.69 | **100.0 ± 0.0** | 19.79 ± 4.38 | 223.44 ± 10.32 |
| AC+CBF+PER | 39.08 ± 24.08 | 2.21 ± 1.64 | -1.04 ± 1.91 | -1.04 ± 1.91 | 16.71 ± 1.55 | 4.47 ± 0.76 | **100.0 ± 0.0** | **18.36 ± 2.38** | 220.59 ± 15.82 |
| AC+CBF+UE | 8.96 ± 2.08 | **0.00 ± 0.00** | **3.22 ± 2.81** | 3.94 ± 3.37 | 11.08 ± 1.23 | 17.72 ± 29.14 | 80.0 ± 44.7 | 47.64 ± 28.19 | 236.34 ± 17.71 |
| AC+CBF+AER | 22.62 ± 12.82 | 1.07 ± 0.87 | -0.35 ± 1.55 | -0.35 ± 1.55 | 16.71 ± 1.55 | 4.37 ± 0.67 | **100.0 ± 0.0** | 21.64 ± 3.66 | 217.80 ± 12.77 |
| Full | **7.63 ± 0.44** | **0.00 ± 0.00** | 2.71 ± 0.23 | **6.95 ± 0.26** | **3.52 ± 0.55** | 4.70 ± 0.51 | **100.0 ± 0.0** | 29.14 ± 4.69 | 228.76 ± 15.32 |

Raw sensor-center RMSE is omitted because the raw stream is exactly identical across methods; see `sensor_fairness_audit.csv`.
`AC` has no CBF, so its intervention rate is not applicable. All tied mean optima are bolded.
Runtime is platform-specific and is not an algorithm-independent complexity bound.