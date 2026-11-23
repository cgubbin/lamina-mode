# Dispersion branches

Call `ModePoint::continue_with` with the source backend, a sequence of real targets in cm⁻¹, and explicit `BranchSettings`. Start the schedule after the accepted seed and advance in one direction. Propagation varies real k₀ and solves complex β. Both spectral strategies vary real β and solve complex k₀. The predictor uses the physical implicit derivative. The corrector accepts a fresh root inside a local chart. Choose the chart half-width in native units (dimensionless conformal coordinate or direct k₀ in cm⁻¹).

The caller's local and domain declarations must cover the swept neighbourhood, including a regular exterior sheet along the short segment to each predictor. The algorithm transports the two stored exterior roots in small increments and checks the relative change. Use small control intervals near a light line or another root. Continuation gives local numerical evidence for the path reached from the seed. Near an avoided crossing inspect both traces and their fields before assigning a physical identity.

The trace owns its seed and accepted points. Each subsequent point has one local chart and one correction candidate, indexed zero. Its provenance identifies the new station. Compare the point's control coordinate to its entry in `destinations` to identify inserted steps. Inspect `retries` for failed trials and `stop` for the first target that exhausted the policy. The prefix can be sampled for fields or used to resume.

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
