# V17.1 release notes -- shared sensor-position uncertainty

## Scope and chronology

V17.1 evaluates two fixed hidden circular obstacles through one common noisy detector stream. For a fixed seed, episode, and severity level, all six methods receive the same raw centers, validity bits, observable jump scores, reported detector scales, exploration draws, and plant disturbance. The package retains the moderate test at multiplier 2.2 and the eleven-level robustness sweep through 4.5. Multiplier 6.0 was added after an earlier multiplier-4.0 run; the multiplier-6 results are an additional exploratory stress rerun rather than a preregistered confirmatory test.

## Corrections represented by this release

- The linear state-value critic is trained from realized cost and next state produced by the executed post-filter action.
- Its TD residual directly enters the actor update. The actor step is explicitly described as a biased replay-weighted clipped score-function surrogate because replay samples carry no behavior-policy likelihoods or importance ratios.
- Goal success includes entry at the terminal post-transition state.
- Raw centers, validity bits, observable jump scores, and reported scales are covered by the within-run composite fairness hash.
- Uniform, PER, and adaptive replay use a common per-seed base replay RNG; method-specific contents and probabilities legitimately produce different later samples.
- TD-dependent replay priorities are refreshed after critic fitting.
- The two-sided paired Wilcoxon and Friedman outputs, SciPy version, complete sensor protocol, and finite-buffer audits are exported.
- The release-regression audit is outcome-dependent reproduction checking, not independent validation or a result-gated acceptance criterion.

The replay-weighted actor surrogate is not an unbiased policy-gradient estimator, and the benchmark does not isolate a material performance contribution from the learned value baseline alone.

## Current exploratory multiplier-6 result

Full and AC+CBF+UE record zero contact samples across all five seeds. Full reaches the goal in five of five seeds, compared with four of five for UE, and reports mean cost 7.63 +/- 0.44, global minimum clearance 2.71 +/- 0.23 cm, uncertain-obstacle clearance 6.95 +/- 0.26 cm, and belief RMSE 3.52 +/- 0.55 cm. UE has the slightly larger mean global minimum clearance, 3.22 +/- 2.81 cm. The remaining four configurations have contact in at least four seeds. No method contacts the exact known rectangular map or workspace wall.

These results are interpreted jointly and descriptively. The release does not claim that Full is best in every column, statistically superior with five pairs, universally convergent, or unconditionally safe.

## Scientific safeguards

- Hidden true centers and evaluator clearances never enter the actor, critic, estimator, CBF, replay priority, or replay buffer.
- The actor and CBF receive each method's causal belief, not evaluator truth.
- The oracle-invariance test changes hidden truth while holding controller-visible inputs fixed and verifies unchanged actions.
- Known-map contacts are audited separately from contacts with uncertain obstacles.
- The CBF disturbance term is a design allowance, not a hard bound on the Gaussian disturbance.
- The shared-stream hash certifies equality within a run and is not claimed to be portable across arbitrary numerical-library platforms.

## Visualization and reproducibility

The figures and MP4/GIF animation are generated from the same display-seed trajectory file. Aggregate tables use all five seeds. From the current organized `main` branch, run `python run.py reproduce --animate` to reproduce the included experiment. `MANIFEST.sha256` authenticates every tracked file other than the manifest itself.
