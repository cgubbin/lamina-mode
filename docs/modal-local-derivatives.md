# Physical modal derivatives

`ModePoint::local_derivatives_with` evaluates the checked characteristic at an accepted physical point and returns separately fallible first partials. It does not rediscover, refine, or replace the accepted root. The report owns the source point, source and evaluation backend labels, explicit local analyticity declaration, characteristic value, requested parameters, units, and per-parameter outcomes.

## Parameters and units

- `PhysicalParameter::K0` differentiates vacuum angular wavenumber `k0` in `cm^-1`, holding `beta`, all thicknesses, and accepted exterior branch labels fixed.
- `PhysicalParameter::Beta` differentiates in-plane angular wavenumber `beta` in `cm^-1`, holding `k0`, all thicknesses, and accepted branch labels fixed.
- `PhysicalParameter::LayerThickness(i)` differentiates finite layer `i` in the unit stored on the source stack. Core seeds that variable before converting layer thickness to canonical centimetres.

The coordinate partials are `F_k0` and `F_beta`, not the native chart derivative `dF/dz`. Actual spectral dispersion of both permittivity and permeability is included in `F_k0`. Exterior wavevector derivatives use the accepted roots through `d(kappa)/dp = d(q)/dp / (2 kappa)`, retaining the selected sign. A configurable cutoff rejects roots too close to zero, and a relation check rejects stored roots inconsistent with source materials.

Equivalent stacks represented in centimetres and nanometres have the same characteristic. Their thickness derivatives convert by the unit scale; for example, `F_d[nm] = 10^-7 F_d[cm]`.

## Local assumptions

Every call requires `LocalAnalyticityAssumption`. It is a caller declaration that the characteristic is holomorphic and pole-free in the local `(k0, beta)` neighborhood and regular under requested real thickness changes. A source found at fixed real frequency does not by itself establish a complex-frequency derivative. The declaration does not expand the original search domain's coverage or prove material analyticity.

## Implicit slopes

`LocalDerivativeReport::implicit_slope` forms conditional simple-root slopes from the returned partials:

```text
d beta / d k0 = -F_k0 / F_beta
d k0 / d beta = -F_beta / F_k0
d p / d d_j = -F_dj / F_p, where p is k0 or beta
```

Each call supplies a finite denominator threshold in the denominator partial's own units. The result retains its numerator, denominator, and applied threshold. A missing partial, non-finite slope, or denominator inside the exclusion band is returned as unavailable; valid gradients remain available.

These complex local slopes are not energy velocity or a certified real group velocity. This API does not identify zero-group-velocity points, exceptional points, branch identity, or optimization sensitivities. It also does not calculate field derivatives or second derivatives.

## Runnable example

Run `cargo run --example local_derivatives`. It evaluates a maintained finite-stack fixture and prints the accepted point, backend provenance, characteristic, each successful or unavailable partial, and a guarded slope.

The essential call pattern is:

```rust,no_run
use lamina_core::Scatter2;
use lamina_mode::{
    LocalAnalyticityAssumption, LocalDerivativeSettings, PhysicalParameter,
};

fn inspect(point: &lamina_mode::ModePoint) -> Result<(), Box<dyn std::error::Error>> {
    let local = point.local_derivatives_with(
        Scatter2::new(),
        &[PhysicalParameter::K0, PhysicalParameter::Beta],
        LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
        LocalDerivativeSettings::default(),
    )?;
    for outcome in &local.outcomes {
        match &outcome.result {
            Ok(partial) => println!("{:?}: {} {:?}", partial.parameter, partial.value, partial.units),
            Err(error) => println!("{:?}: unavailable ({error})", outcome.parameter),
        }
    }
    Ok(())
}
```
