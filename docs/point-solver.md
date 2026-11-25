# Public point solvers

Lamina-mode searches planar stacks for propagation modes at fixed real frequency,
or complex-frequency modes at fixed real in-plane wavenumber. It uses core device
primitives and returns numerical evidence with explicit branch and coverage limits.

| Public problem | Input | Native coordinate |
| --- | --- | --- |
| `PropagationProblem` | Core analytical stack and fixed real k0 | Dimensionless conformal z |
| `ConformalSpectralProblem` | Constant exterior evidence, core analytical layers and fixed real beta | Dimensionless conformal z |
| `FrequencyProblem` | Core analytical stack, including dispersive exteriors, and fixed real beta | Physical k0 in cm^-1 |

A core stack may close either side with a perfect electric or perfect magnetic
wall. Only open sides carry an exterior wavevector, and `PropagationProblem`
charts them: with both sides open, the two-exterior conformal map; with one,
the same map built from the open side's material alone; with none,
`direct_domain` over a β rectangle in cm^-1 (`CoordinateKind::Wavenumber`),
since the characteristic of a stack walled on both sides is entire and even in
β and needs no sheet or radiation reference. Conformal domains of such a stack,
and `FrequencyProblem` on any walled stack, fail with typed errors. A walled
side is classified `Closed` with no flux; with one open side the mode type
follows from that side alone, and with none it is `ModeType::Closed`.

`new` uses core Scatter2. `with_backend` accepts an explicit core backend, including
supported Transfer2 evaluation; there is no backend or strategy fallback. Problems
use complex f64 scalars. Physical k0, beta and exterior kappa use vacuum angular
wavenumbers in cm^-1; thickness uses core's length units. A direct characteristic
derivative is dF/dk0; a conformal derivative is dF/dz. Public evidence and settings
call the varying chart coordinate `native_coordinate`/`native`, with units in
provenance. Relative tolerances are dimensionless. The characteristic residual
uses core's documented gauge and must be interpreted with its configured scale.

## Quickstart

This complete workflow is compiled and run as a doctest. It finds an independently
known outgoing-basis cavity root on an explicit complex branch. An explicit
complex reference alone leaves physical radiation unspecified; use a real
`radiation_domain` anchor when supported outgoing qualification is needed.

```rust
use lamina_core::{AnalyticalMaterialHandle, Constant, Layer, Polarisation};
use lamina_mode::{
    AnalyticAssumption, Complex, ConformalChartConfig, ConformalRegion,
    ConformalSpectralProblem, ConstantExterior, DiscoveryConfig, MatchConfig,
    ReferenceSelection, RefinementConfig,
};
use lamina_units::Length;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let exterior = ConstantExterior::new(Complex::new(1.0, 0.0), Complex::new(1.0, 0.0))?;
let problem = ConformalSpectralProblem::new(
    exterior, exterior,
    [Layer::new(AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                Length::centimetres(0.3))],
    0.0, Polarisation::TransverseElectric,
)?;
let expected = Complex::new(std::f64::consts::PI, -3.0_f64.ln()) / 0.6;
let reference = (-Complex::i() * expected).ln();
let domain = problem.domain(
    ConformalRegion::new(reference.re - 0.1, reference.re + 0.1,
                         reference.im - 0.1, reference.im + 0.1)?,
    ReferenceSelection::Coordinate { reference, coordinate: expected },
    ConformalChartConfig::default(),
    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
)?;
let report = problem.plan(vec![domain], vec![])?.solve(
    DiscoveryConfig::default(), RefinementConfig::default(), MatchConfig::default(),
)?;
assert_eq!(report.observations().len(), 1);
assert!((report.observations()[0].physical_coordinate() - expected).norm() < 1e-8);
assert_eq!(report.discovery().entries().len(), 1);
# Ok(())
# }
```

## Domains, references and settings

Conformal inputs use `ConformalRegion` and `ReferenceSelection` or
`RadiationReference` with dimensionless `native_coordinate`. Direct-frequency
inputs use `FrequencyDomainInput` and `FrequencyRegion`. The latter requires
`ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero` in addition to the
explicit characteristic/material assumption passed to `domain`. Neither has a
convenience default that declares a domain analytic for the caller.

`FrequencyReference::Explicit` supplies finite nonzero ordered exterior wavevectors
at a complex anchor, checked against the actual supplied stack. Its radiation is
unspecified. `FrequencyReference::RealRadiation` instead verifies outgoing/decaying
or incoming/growing choices at the actual positive real dielectric anchor. Lossless
exterior dispersion is allowed; lossy or magnetic unsupported anchors retain typed
failures and can use valid explicit complex branch data without inventing outgoing
qualification. Conformal `radiation_domain` performs its existing branch checks.
Coordinate sign and radiation sign are distinct choices.

Known frequency exclusions have typed material-pole, side-specific threshold or
caller reasons. They must miss the closed requested region and its evaluation
halo, including boundary contact. Empty exclusions and successful contour samples
do not certify the interior. Caller assumptions cover a neighbourhood of the whole
evaluation region and require stable material behavior throughout a problem's
lifetime. The halo only permits boundary roundoff and cannot enlarge searched
coverage, root enclosures or acceptance.

`ContinuationConfig` exposes direct-frequency quadrature/consistency tolerances,
physical cutoff, depth and per-side evaluation budgets. `DiscoveryConfig` exposes
validated discovery settings. `RefinementConfig.native_atol` uses the domain's
native units; residual criteria use a fixed explicit scale. `MatchConfig.native`
also uses native units; physical and exterior absolute matching tolerances use
cm^-1. Numerical defaults never establish physical assumptions or global accuracy.

## Execution, ownership and evidence

Use `problem.plan(domains, exclusions)` for a finite ordered plan. Supported
canonical conformal builders are available through `canonical_plan`; symmetric or
normal-incidence degeneracy returns its diagnostic and requires explicit domains.
Frequency plans use explicit domains. Foreign problem/domain/plan associations and
repeated exact domains are rejected before evaluation. Cloning a domain preserves
its identity, reference and halo.

```text
plan.discover(settings)                       -> owned Discovery
plan.refine(&discovery, refinement, matching) -> owned SolveReport
plan.solve(discovery, refinement, matching)   -> owned SolveReport
```

Reports and standalone points retain the selected backend type name for
diagnostics (not as a serialization identifier), and provenance identifies the
explicit strategy.

Reports can be returned from helpers after local problem, plan and discovery
objects are dropped. They share the original discovery once and own candidate
outcomes and accepted observations, without self-references or repeated numerical
execution. `report.discovery().entries()` retains one successful chart discovery or
typed error per chart. Successful discoveries expose estimates, multiplicities,
leaves, summary and termination. `report.charts()` retains exact provenance and all
candidate outcomes; a failed discovery has no fabricated refinement result.

`report.observations()` contains numerically accepted points only. A `ModePoint`
can be cloned and retained independently, with its original enclosure, configuration,
work, evidence and source domain. Its construction is controlled by numerical
acceptance. It is neither a globally unique mode identity nor a continued branch.
`groups()` references all contributing observations; no multiplicity is summed or
erased. Opposite or ambiguous branches remain separate. `exclusions()`, `geometry()`
and reconciliation `diagnostics()` retain local scope limits. Success in an overlap
does not erase a failed chart. `unresolved()` lists, with its chart, native region
and cause, every failed discovery, discovery stopped before convergence, leaf above
the score tolerance and estimate refinement did not accept; `is_complete()` holds
only when that list and the reconciliation diagnostics are empty. It covers the
charts' requested regions, never the exclusions.

## Diagnostics and interpretation

`Diagnostic.is_foreign_association()` identifies ownership failures.
`Diagnostic.stage()` distinguishes construction, domain, discovery, refinement,
evaluation and interpretation failures. `frequency_error()` exposes typed
continuation side/kind/work and domain causes; `optical_failure()` distinguishes
checked input/backend/output failures. `find_source::<T>()` and `Error::source`
retain original causes, with selected public cause types available from the API.
Failure never becomes a successful empty result, fabricated root or silent backend
switch. Valid finite plans retain independent successful and failed charts.

`point.describe(config)` and `report.describe_points(config)` are optional fallible
physical descriptions over retained branch signatures and the source materials.
They preserve point acceptance and report evidence if data are unavailable. Supported
fixed-frequency dielectric points can receive guided, leaky/source, absorptive or
gain labels with applied thresholds and component evidence. Spectral points report
temporal/exterior behavior and radiation provenance while named fixed-frequency
classification and instantaneous complex-frequency flux remain unsupported.
Temporal decay alone does not prove an outgoing QNM. No point has computed fields,
normalization or mode volume merely because it was accepted or described.

## Runnable examples

Run these from the lamina-mode development environment:

```sh
cargo run --example guided_mode --locked --offline
cargo run --example conformal_cavity --locked --offline
cargo run --example dispersive_substrate --locked --offline
```

The [guided example](../examples/guided_mode.rs) uses a core analytical stack and
independent TE phase matching; the [constant cavity](../examples/conformal_cavity.rs)
checks an analytic pole with a verified outgoing real reference. Their shared
[example helpers](../examples/support/mod.rs) use public imports only. The
[dispersive substrate](../examples/dispersive_substrate.rs) uses lamina-materials'
analytical Lorentz model with epsilon=2.25+100/(400-k0²), explicit assumptions and
pole guards. Its accepted pole agrees with an independent scalar characteristic
and differs from freezing the exterior at the anchor. Each example checks its
result and prints acceptance, radiation/classification and finite-coverage evidence.

The [mathematical guide](modal-transform.md) documents the transform correspondence,
sheets and physical support matrix.

See [modal field recovery](modal-fields.md) for separately fallible postprocessing of owned accepted points.
