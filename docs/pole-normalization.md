# Modal pole normalization conventions

We define the scalar Green function by `L G = delta(z-z')`, with a positive unit
flux jump. For primary field psi=Ey (TE) or Hy (TM),

```
L psi = (p psi')' + q psi
p=1/f, q=k0² g-beta²/f
TE: f=mu, g=epsilon; TM: f=epsilon, g=mu.
```

The pole variable is explicitly k0 at fixed beta, or beta at fixed k0, in cm⁻¹.
This is one planar Fourier component, with no extra area or Fourier prefactor.
It does not define a finite 3D mode volume. The complete scalar factor satisfies
`Res G = psi(z) psi(z') / N_lambda`. Angular-frequency residues multiply the
k0-plane residue by c, since omega=c k0; derivatives transform oppositely.

With outgoing boundary fluxes B_left=-i kappa_left/f_left and
B_right=+i kappa_right/f_right,

```
N_lambda = integral[-p_lambda (psi')² + q_lambda psi²] dz
           + psi(right)² B_right,lambda - psi(left)² B_left,lambda.
```

There is no complex conjugation and no numerical integration over infinite
exteriors. Scalar boundary terms encode their outgoing dependence. Constitutive
and selected-branch derivatives are analytic. This expression is independent of
where finite truncation planes are placed within homogeneous exteriors.

## The 1D cancellation matters

For a normal-incidence cavity with nondispersive homogeneous exteriors, the
outgoing Maxwell bilinear exterior density E·epsilon E-H·mu H vanishes. The
finite-layer Maxwell integral M already contains the scalar boundary contribution
through the derivative-field identity. In core's primary-field conventions,
`N_k0=k0 M` for TE and `N_k0=-k0 M` for TM in this case. Adding scalar boundary
terms to M again would double-count. The analytic n=2, d=.3 cm cavity with unit
right-outgoing primary amplitude has M=.9 cm and N_k0=±.9 k0.

This equivalence needs its assumptions. Dispersive exteriors, nonzero beta and
beta-pole normalization require the declared operator/reciprocal convention;
vanishing exterior terms must be established, not assumed for every planar mode.
For the reciprocal vector partner at reversed beta, the longitudinal Hz (TE) or
Ez (TM) changes sign while the scalar primary and transverse partner do not.
The legacy same-beta E/H integral is retained numerically and tested in the case
where it is equivalent. The new scalar pole normalization is labelled explicitly;
its normalized fields need not have the legacy M=1 scaling.

A common complex seed multiplier a changes N by a². E and H receive the same
1/sqrt(N) scale; the principal square root leaves a possible common sign change
under phase variation. Residue products are invariant. Near-zero, singular,
nonfinite or ill-conditioned factors fail rather than implying an exceptional
point. Original accepted points and arbitrary-amplitude field recovery remain
available, and no new radiation qualification is inferred.

## Public API

```rust,ignore
let normalized = point.normalize_with(
    lamina_core::Transfer2::new(),
    lamina_mode::PoleVariable::K0AtFixedBeta, // or BetaAtFixedK0
    lamina_mode::AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    lamina_mode::PoleNormalizationSettings::default(),
    lamina_mode::FieldRecoverySettings::default(),
)?;
let fields = normalized.fields(&sampling)?;
```

The pole variable must match the source solve's held-fixed coordinate. The
explicit new declaration concerns the local constitutive/branch variation used
for normalization, even if a search declaration already exists. It is a caller
assumption, not proof of causality, a zero-free contour or a simple pole.
`PoleNormalizationSettings::amplitude` sets the raw seed multiplier; the recovery
settings supply residual limits, with their amplitude replaced by the computed
normalized amplitude. The default inverse-square-root scale normalizes the
**scalar pole factor** to one. It does not silently impose the legacy Maxwell
integral M=1 convention; the analytic conversion above explains the difference.

The owned `NormalizedMode` retains the accepted point, original and normalization
backends, local declaration and complete factor evidence. `normalization.layers`,
`exterior_segments`, `outgoing_boundaries` and `total` expose the calculation;
`legacy_finite_maxwell` is comparison evidence, never another term added to it.
Normalization does not modify original acceptance, radiation provenance or fields.
The analytic exterior segment entries are zero at the default physical boundaries;
finite outward partition shifts are available in centimetres for auditing.

Factor thresholds use `factor_atol * factor_reference_scale + factor_rtol *
contribution_scale`, in squared primary-field units for the declared cm⁻¹ pole
variable. `condition` is the sum of contribution magnitudes divided by the total
magnitude. Settings, cutoff bands and condition limits are finite and explicit.
A small or nonfinite factor returns a source-preserving failure with available
contributions and thresholds; no legacy fallback or exceptional-point conclusion
is made. Sampling rechecks normalization for detectable material changes. Stable
analytical materials remain a caller obligation.

## Reproduce validation

```sh
cargo test --locked --test pole_normalization
cargo run --release --locked --example pole_normalization -- --suite quick --output /tmp/norm-quick
cargo run --release --locked --example pole_normalization -- --suite full --output /tmp/norm-full
python scripts/plot_pole_normalization.py /tmp/norm-full
```

Both modes use twelve fixed accepted-point fixtures and seven decreasing offsets
in two complex directions. Full mode uses three source/observation pairs and
256/512-point independent contour estimates; quick mode uses two pairs and
128/256 points. Numerical checks need no Python. Plotting uses the optional
Matplotlib environment documented for the plasmon validation examples.
Directories must be new. Failed cases remain identified in `cases.csv` and
`failures.txt`; the command returns failure and plotting rejects failed or
incomplete runs. Settings, models, physical coordinates, contributions and source
provenance are recorded in JSON/CSV and per-case evidence files.

Reference Green functions use value-only sine/cosine fundamental solutions and
weighted Wronskians, with an independently tested positive source-flux jump.
They do not use production normalization, its derivative or normalized fields as
input. Contour residues must agree with normalized-mode predictions to 1e-7;
contour refinement, gauge and partition discrepancies must be below 1e-8.
Offset sequences must show an improvement window before root/roundoff errors
rise; no claim of monotonic improvement at arbitrarily tiny offsets is made.
Negative controls remove a scalar boundary term or material dispersion,
conjugate one required bilinear residue factor, or use the wrong pole variable.
`unavailable` means the deliberately incorrect factor could not be normalized;
an empty dispersion-control cell means that control was inapplicable to the case.

This increment proceeds before the general local-derivative API at the user's
request, using existing analytic core constitutive derivatives. General gradients,
branch continuation, mode volumes and excitation observables remain separate work.
