# Dispersion branches

Call `ModePoint::continue_with` with the source backend, a sequence of real targets in cm⁻¹, and explicit `BranchSettings`. Start the schedule after the accepted seed and advance in one direction. Propagation varies real k₀ and solves complex β. Both spectral strategies vary real β and solve complex k₀. The predictor uses the physical implicit derivative. The corrector accepts a fresh root inside a local chart. Choose the chart half-width in native units (dimensionless conformal coordinate or direct k₀ in cm⁻¹).

Before correcting, the boundary winding of the chart counts its roots. The step is accepted only if the count is one. With none, the predictor has lost the branch; with several, neighbouring roots lie within the half-width, and the corrector could converge onto any of them. Either way the step is refused with `BranchFailure::RootCount` and subdivided like any other failure, so a smaller step brings the prediction closer to the branch. If the half-width itself exceeds the spacing of a crowded ladder, every step is refused and the trace stops on `RootCount`. `BranchTrace::near_degenerate` lists the refused steps and stop whose chart held several roots, with the count and the chart region: the near-degenerate neighbours of the branch along the path. A root too close to the chart boundary to count is `BranchFailure::Uncounted`, also subdivided.

The caller's local and domain declarations must cover the swept neighbourhood, including a regular exterior sheet along the short segment to each predictor. The algorithm transports the two stored exterior roots in small increments and checks the relative change. Use small control intervals near a light line, and a half-width below the spacing to the nearest other root. Continuation gives local numerical evidence for the path reached from the seed. Near an avoided crossing inspect both traces and their fields before assigning a physical identity.

The trace owns its seed and accepted points. Each subsequent point has one local chart and one correction candidate, indexed zero. Its provenance identifies the new station. Compare the point's control coordinate to its entry in `destinations` to identify inserted steps. Inspect `retries` for failed trials and `stop` for the first target that exhausted the policy. The prefix can be sampled for fields or used to resume.

## Through folds

`continue_with` holds k₀ fixed at its targets, so it stops at a fold, where dβ/dk₀ diverges and the branch turns back in k₀ (for a lossless mode, a zero-group-velocity point). `ModePoint::continue_through_folds_with` follows the branch as a curve F(k₀, β) = 0 instead. Its tangent (dk₀, dβ) ∝ (∂F/∂β, −∂F/∂k₀) is finite through a fold. Each step holds fixed the coordinate along which the tangent moves most: k₀ (β solved on the in-plane chart) away from a fold, β (k₀ solved directly, or on the spectral chart for a point found there) near and through it. Steps are arclength in the real (k₀, Re β) plane in cm⁻¹, halved on failure and doubled back up to `ArclengthSettings::step` on success. The tangent keeps its orientation from point to point, so past a fold k₀ decreases. Every station counts the roots in its chart and accepts only one, as in `continue_with`.

A sign change of dk₀/dβ between two accepted points is a fold. It is located by regula falsi on Re dk₀/dβ over stations at fixed β, to `fold_tolerance` of |β|, and reported as a `Fold`: its k₀ and β, the station there and its slope, and the accepted points on both sides with their slopes. A trace ends when k₀ or Re β leaves its bounds, after `max_points`, or when a step fails at `minimum_step` (`ArclengthEnd`). Stations solving k₀ directly need frequency-domain evidence; a point found at fixed k₀ has none, and takes `ArclengthSettings::frequency`. Their k₀ boxes have half-width `frequency_radius` and must keep clear of the singularities of the materials (`FrequencyProblem::known_singularities`). For a lossless 100 nm hBN film in air, the fundamental TM mode's type I fold lies at ν̃ = 829.0767638 cm⁻¹ (exact 829.076764); the isotropic films with hBN's lower and upper phonon axes fold at 829.1366259 and 1605.267328 cm⁻¹.

## Physical report

Run these from lamina-mode, supplying a new output directory. The plotter requires Python 3 and gnuplot 6 on PATH.

```sh
cargo test --locked --test branch_continuation
cargo run --release --locked --example branch_report -- /tmp/modal-branches
python3 scripts/plot_branch_report.py /tmp/modal-branches
```

Open `report.html`. Each CSV row records the accepted complex coordinates in cm⁻¹, exterior wavevectors, residual and correction limits, backend, and physical slope. The interface-plasmon runs also record independent reference eigenvalues, unsquared TM mismatch and backend disagreement, each checked against a relative or scaled tolerance of 10⁻⁸. The two dielectric guides have decaying air exteriors. Their isolated TE₀ modes coincide at β = 7 µm⁻¹. The independent matching equation and widths are in `examples/support/coupled_slabs.py`.

The passive stopped-light benchmark uses the 110 nm semiconductor core and 50 nm ITO cap from Page's thesis. The comparison uses sign changes of d Re(ω)/dβ and linear interpolation between neighbouring stations. The imaginary slope is recorded separately. The benchmark tolerances reflect the precision of the reference table and the sampled grid (0.05 µm⁻¹ in β, 0.5 and 0.01 rad/ps in real frequency and decay rate). See `examples/support/page_stopped_light.json` and [Page, chapter 3, tables 3.1–3.2](https://doi.org/10.25560/30760).

Each field CSV starts with its normalization status and physical root signature. The figures use the magnitudes of the three complex electric or magnetic components at the stored depths. Duplicate interface depths are the two one-sided field limits.
