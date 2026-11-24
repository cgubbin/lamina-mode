# lamina-mode

Public physical-input point solvers using `lamina-core` and `argloc`, supporting
fixed-frequency propagation and both constant-exterior conformal and dispersive
direct-frequency spectral strategies. Start with the [public API quickstart](docs/point-solver.md)
and [runnable examples](docs/point-solver.md#runnable-examples).

The public facade reuses core device primitives and returns owned reports with
numerical evidence, radiation provenance and finite coverage. Existing internals
are described below for mathematical and implementation context.

## Scope

lamina-mode finds the modes of planar multilayers (guided, leaky, lossy, gain,
plasmonic and quasi-normal) and reports each answer with its numerical evidence:
accepted roots, residuals, coverage and the completeness of a stated window. It
provides their fields, normalisation, regional power and overlaps between
stacks, derivatives of modal quantities with respect to stack, wavelength and
material parameters, a certified group index, and continuation of modes and
modal sets along a parameter. It builds on `lamina-units`, `lamina-core` and
`argloc`.

Read the [modal transform guide](docs/modal-transform.md) for the
correspondence to Bakhtazad et al. (1997), field conventions and physical scope.
The squared coordinate repeats after `i*pi`; the exterior sheets repeat after
`2*i*pi`. Canonical width-pi charts select only part of that surface, with the
omitted sheet sectors reported explicitly.

Run from this repository's Rust/Nix development environment:

```sh
cargo test --all-targets --locked
cargo test --doc --locked
cargo clippy --all-targets --locked
cargo doc --no-deps --locked
```

A single-chart search keeps the requested conformal rectangle unchanged and
validates a small surrounding evaluation margin for floating-point boundary
roundoff. The margin preserves the reference point and coordinate sheet; it
must not cross a coordinate branch point. It does not enlarge the searched
region or certify a pole-free material domain.

The baseline had 196 passing tests and two failures: exact floating-point
coordinate equality and an adaptive evaluation 1.1e-16 beyond a rectangle edge.
Coordinate comparisons now use explicit tolerances, every derivative check
asserts its Result, and the independently validated margin handles boundary
roundoff without clamping evaluation coordinates.

Search results remain estimates. Multiplicity can describe an unresolved cluster;
termination, leaves, and run diagnostics must be inspected before interpreting
coverage. Optional candidate refinement and multi-chart orchestration preserve
that evidence, as described below. Optional internal physical classification is
available; fields and continuation remain later work.

Focused, runnable physical and error-chain regressions:

```sh
cargo test --locked checked_tests::physical_in_plane_te_and_tm_searches
cargo test --locked checked_tests::physical_spectral_te_and_tm_searches
cargo test --locked checked_tests::checked_failures_survive_dispersion_and_adaptive_search
cargo test --locked checked_tests::budget_limited_search_preserves_cluster_and_unresolved_diagnostics
```

The in-plane fixture engineers an asymmetric slab thickness from the independent
TE/TM phase-matching equation. The spectral fixture checks the outgoing vacuum /
index-two slab / vacuum cavity pole `k0 = (pi - i ln(3)) / (2 d)` on an explicit
chart. This explicit normal-incidence fixture does not implement automatic
symmetric-exterior chart construction. Both fixtures check conformal derivatives
and the physical coordinate on the selected branch.

The compiled optical adapter calls `checked_determinant` once per seeded point,
passing exterior jets through unchanged. Input, backend, and output failures
propagate through `DispersionError` into `ModalSearchError`; the complete engine
error is retained alongside the typed evaluation cause because the locked
upstream oracle error omits that cause from `Error::source`.

Coordinate validation establishes a single analytic coordinate branch. The caller
must additionally provide a pole-free material domain and its neighborhood;
finite point samples cannot establish that contract. Checked failures stop the
search, and errors do not imply a successful partial result. Retained ordinary
scattering amplitudes at repaired exceptional points can remain unavailable even
when the characteristic and its derivatives continue finitely. Exact directional
cutoffs and finite homogeneous dynamic range remain limitations.

## Bound problem and domain construction

The internal `problem` module underlying the public facade provides three coherent construction flows:

- `InPlaneProblem::in_plane(stack, fixed_k0, polarisation, backend)` evaluates the
  actual analytical stack's exterior epsilon and mu at fixed k0 and constructs
  its own map and checked solver.
- `SpectralProblem::spectral(left, right, layers, fixed_k_parallel,
  polarisation, backend)` takes validated `ConstantExterior` descriptors and
  ordered analytical finite layers. It builds both stack and map from those same
  exteriors. Complex lossy/magnetic constants are supported; finite layers retain
  their dispersion and derivatives. Arbitrary erased exterior handles cannot be
  supplied as evidence of constancy, and sampling does not prove it.

- `FrequencyProblem::frequency(stack, fixed_beta, polarisation, backend)` reuses
  the supplied analytical stack, including dispersive exteriors, in direct k0
  charts; see the dispersive spectral section below.

Fixed coordinates are real and expressed in cm^-1, matching the existing
maps. This first internal binding uses complex f64 scalar jets. A problem's
physical context is immutable; custom materials must keep their behavior stable
throughout its lifetime, since cloned handles cannot freeze interior mutable
state in user models.

Call `problem.domain(rectangle, reference_selection, chart_config,
AnalyticAssumption::CallerAssumedHolomorphicPoleFree)` to construct one executable
domain. Reference selection supports the existing sheet selection or an explicit
reference point and physical coordinate. The explicit assumption has no Default
and is a required argument. It covers a neighborhood of the entire closed
searched domain containing the validated evaluation halo, including material
validity and the characteristic representation. For in-plane search this is an
obligation at fixed k0; for spectral search it covers the mapped k0 domain.
Coordinate validation and finite checked samples establish no such certificate.

Then use `problem.evaluate(&domain, z)` or `problem.search(&domain, config)`.
Domains belong to their creating problem; clones share ownership, while a domain
from a separately constructed problem is rejected before optical evaluation.
Reusing a domain does not add another halo. Construction errors retain their
causes and are distinct from later candidate acceptance.

A bound result wraps the complete existing estimates and diagnostics together
with polarization, direction, fixed coordinate, domain/reference information, and
the caller assumption. Cluster centroids remain unresolved estimates and budget
termination remains visible. These modules are private.

Run the bound regressions from this repository:

```sh
cargo test --locked problem::tests
cargo test --locked problem::tests::bound_physical_in_plane_te_and_tm
cargo test --locked problem::tests::bound_physical_spectral_te_and_tm
cargo test --locked problem::tests::foreign_domains_fail_before_evaluation_and_optical_sources_survive
cargo test --locked problem::tests::incomplete_results_preserve_provenance_clusters_and_domain_reuse
```

Next comes candidate refinement and validation, then multi-chart coverage and
overlap handling, the public physical-input API, modal fields, continuation, and
profiling-led performance work. Automatic symmetric/normal-incidence chart
construction and dispersive spectral exteriors remain later work; explicit charts
already supported by the kernel remain usable.

## Numerical refinement and acceptance

Discovery still returns estimates. The optional next operation is
`problem.refine(&domain, &discovery, RefinementConfig::default())` after constructing
the problem/domain and calling `problem.search(&domain, search_config)`. The report
borrows the complete original discovery and contains one outcome per estimate;
accepted roots are a separate type with private construction and fresh checked
numerical evidence. Refinement requires the exact originating domain identity.
Domain clones share identity; a separately constructed equal rectangle does not.

Only zero targets marked as single points with multiplicity one are eligible.
Clusters/repeated multiplicities remain unresolved; contradictory metadata and
invalid enclosures are ineligible. Damped Newton steps are evaluated strictly
inside the original enclosure and searched rectangle. The evaluation halo never
expands acceptance geometry, and points are not clamped or moved to another chart.

Acceptance requires a fresh checked bundle satisfying both:

```text
|F| <= residual_atol + residual_rtol * residual_scale
|F/F_prime| <= z_atol + z_rtol * max(1, |z|)
```

The derivative must be finite and nonzero; the correction, thresholds, and physical
projection must be finite. Defaults are residual_atol=1e-10, residual_rtol=0,
residual_scale=1, z_atol=z_rtol=1e-10. Residual units follow the core characteristic
gauge; configure a positive fixed residual scale when needed. It is never inferred
from a nearly zero initial residual. Small residuals or small damped moves alone
do not establish convergence. Numerical acceptance remains conditional on the
caller's domain assumption, with no mathematical isolation/error-bound certificate.

Default per-candidate limits are 32 successful updates, 12 step halvings after the
full-step proposal, and 256 checked bundles. Counters include all trial proposals
(even those rejected before evaluation), while bundle evaluations include initial,
evaluated trial, and fresh final checks. Any zero work limit disables candidate
work. Ineligible metadata and clusters are classified first, then geometry and
budgets, then evaluation/derivative/correction checks. At an admissible point,
convergence is tested before spending another update. Exhausted update/evaluation
limits yield BudgetExceeded; unsuccessful bounded backtracking yields BoundaryLimited
if all proposals were outside, NoProgress if evaluated trials could not improve,
or Stagnated when floating-point arithmetic cannot move the point.

Checked failures at initial, trial, or fresh final evaluations and physical
projection errors are retained as typed per-candidate failures with last-valid
evidence and counters. Optical errors stop that candidate immediately rather than
being silently hidden by backtracking. Independent candidates still receive their
own outcomes. The report preserves estimates, multiplicities, leaves, elapsed time,
summary, termination, and provenance. An accepted root from budget-limited discovery
does not imply complete coverage, and refinement cannot create a successful partial
result from an unsuccessful discovery call.

```sh
cargo test --locked refinement
cargo test --locked problem::tests::bound_physical_in_plane_te_and_tm
cargo test --locked problem::tests::bound_physical_spectral_te_and_tm
```

Next address chart orchestration and overlap/coverage handling. Cluster splitting,
multiple-root treatment, automatic retries, and cross-chart deduplication remain
separate work before broader coverage claims.

## Finite chart orchestration

The private `problem::orchestration` layer coordinates ordered domains for one
physical problem. Build a `ChartPlan::explicit(problem.clone(), domains,
exclusions)` from already validated domains, or call the problem-owned
`canonical_in_plane_plan` / `canonical_spectral_plan` geometry helpers. Canonical
construction requires explicit branch/reference selections and analytic caller
assumptions, returning one in-plane domain or lower/upper spectral domains in
stable order. Plan clones preserve plan and domain identity; empty, foreign-domain,
and repeated exact-domain plans are rejected. Distinct overlapping domains are
allowed.

Canonical metadata retains physical truncation inputs, finite real-z bounds,
imaginary margins/excluded strips, and omitted exterior-sheet sectors.
`OmittedExteriorSheetSectors` records the nominal selected imaginary interval
`[-pi/2, pi/2]`, the exterior period `2*pi`, and omitted intervals
`[-pi, -pi/2]` and `[pi/2, pi]` in that representative. Endpoints do not establish
seam coverage, and contour margins are separate exclusions. Those inputs
choose finite conformal geometry, not an exhaustive rectangular complex physical
window. Canonical geometry that is unsupported or degenerate retains its original
typed cause and requires explicit finite domains. This supports symmetric exteriors
and normal incidence without inventing bounds for a preimage at negative infinity.
Explicit exclusion records are caller declarations, not a computed complement of
all physical space.

The staged flow is:

```text
problem -> owned domains -> chart plan
plan.discover(DiscoveryConfig) -> owned discovery batch
plan.refine(&batch, RefinementConfig, MatchConfig) -> borrowed batch report
```

Discovery runs sequentially, retaining one complete result or typed error per
chart, and continues after an independent chart fails. Settings are validated
before each stage runs. `DiscoveryConfig` exposes finite numeric tolerances and
budgets while retaining the upstream default quadrature rule and undeformed
contour; an opaque arbitrary integrator configuration cannot be validated through
its current interface. Failed discoveries have no fabricated refinement result.
Batch refinement uses each successful result's exact originating domain.

The report is also its coverage ledger: `charts()` exposes each domain's provenance,
original discovery result/error, and optional complete refinement report.
`geometry()` and `exclusions()` preserve scope limitations. Leaves, multiplicities,
elapsed time, summaries, termination, and non-accepted outcomes remain available.
A successful overlapping chart does not erase a failed chart or recover excluded
seams. There is no unconditional completeness flag.

Accepted `observations()` retain chart/candidate indices, local evidence, original
enclosures, and branch signatures. `groups()` references those observations;
representatives are deterministic and every contributor remains available. Matching
requires different overlapping charts, mutual enclosure containment, close conformal
and physical coordinates, and matching ordered left/right exterior branches.
Physical/exterior absolute tolerances use cm^-1. Native tolerances are dimensionless
for conformal charts and cm^-1 for direct-frequency charts, as recorded in provenance.
Defaults are absolute=relative=1e-8 for each comparison. They are configurable
numerical conventions, not root-uncertainty certificates. Near-zero sign ambiguity,
incompatible enclosure evidence, and signature errors remain visible through
`diagnostics()` and observation signatures. Groups use complete pairwise compatibility,
so proximity chains do not collapse. Same-chart candidates and disjoint periodic
representations are not merged, and multiplicities are never summed or discarded.
These are duplicate observation groups, not a certified globally unique mode count.

Canonical spectral lower/upper chart selections need not be the all-outgoing
exterior sheet. Physical tests independently check the selected exterior signs;
the outgoing normal-incidence pole is tested separately using explicit charts.

```sh
cargo test --locked orchestration
cargo test --locked problem::orchestration::tests::overlapping_physical_te_tm_guided_and_normal_spectral_roots
cargo test --locked problem::orchestration::tests::canonical_spectral_lower_upper_keep_physical_te_tm_evidence
cargo test --locked problem::orchestration::tests::independent_chart_failures_stay_in_coverage_ledger
```

Physical point descriptions, supported dielectric classification and verified
radiation references are implemented internally. Direct frequency charts with actual dispersive exteriors are also implemented.
The public point API and executable examples expose these capabilities.

## Physical point interpretation (internal)

`problem.domain_with_radiation_reference(...)` verifies explicit per-exterior
outgoing/decaying or incoming/growing choices at a real physical anchor strictly
inside a supplied chart. This is distinct from the linear-coordinate sign.
Unsupported material, cutoff and branch mismatch errors precede discovery;
existing `domain(...)` calls keep unspecified radiation provenance.

After refinement, `problem.describe_root(&domain, root, config)` or
`report.describe_points(config)` provides physical descriptions without changing
numerical acceptance or coverage. Each description retains its accepted root even
when physical data are unavailable. Supported fixed-frequency dielectric types
include guided, left/right leaky/source, two-sided leaky/source, absorptive and
gain modes. Boundary and unsupported cases are explicit. Spectral points report
temporal behaviour and radiation provenance, not the fixed-frequency paper label.
See the [guide](docs/modal-transform.md#radiation-references-and-physical-point-interpretation)
for the support matrix, tolerances and private usage example.

```sh
cargo test --locked radiation
cargo test --locked physical::tests
cargo test --locked physical_interpretation_keeps_original_signature_errors_and_batch_evidence
```

## Dispersive spectral charts (internal)

There are two explicit spectral strategies. The conformal constructor requires
constant exteriors by construction. `FrequencyProblem::frequency(stack, beta,
polarisation, backend)` accepts an existing core `AnalyticalMaterialStack` and
uses its actual complex exterior values and derivatives throughout. Core owns
stack, layer, material, polarization and unit primitives; no device conversion or
replacement exterior descriptors are required.

`FrequencyRegion` bounds and direct-chart native corrections/enclosures are in
vacuum angular wavenumber `k0` (cm^-1). Each exterior is continued from its own
fixed reference using bounded integration of `q'/q`. Explicit complex branch data
carry unspecified radiation; a real positive dielectric anchor can verify
outgoing/decaying or incoming/growing choices using the actual anchor material.

Construction requires explicit characteristic/material analyticity and separate
nonzero-holomorphic `q` assumptions over the whole evaluation neighbourhood.
Known pole/threshold guard rectangles are retained and must miss both the closed
requested region and its roundoff halo. Empty exclusions and successful samples
do not certify an analytic domain. Per-exterior tolerances, cutoff and recursion/
evaluation budgets are validated; typed failures survive discovery, refinement
and descriptions. The halo preserves the fixed reference and never extends
root acceptance or searched coverage. Only listed frequency regions/branches
are searched; there is no inferred periodic or global sheet coverage.

See the [direct-frequency guide](docs/modal-transform.md#direct-frequency-charts-with-dispersive-exteriors)
for equations, private usage and runnable TE/TM regressions.

Physical regression and plots: [plasmon validation](docs/plasmon-validation.md).

Accepted-point [modal fields](docs/modal-fields.md) use core sampling and explicit arbitrary-amplitude gauges.

[Physical local modal derivatives](docs/modal-local-derivatives.md) expose `F_k0`, `F_beta`, stored-unit thickness partials, and guarded conditional simple-root slopes. Run `cargo run --example local_derivatives` for a report.

See [pole normalization](docs/pole-normalization.md) for complete scalar-pole factors, normalized fields and the 1D exterior-cancellation checks.

[Dispersion branches and physical report](docs/modal-branches.md)

[Enumerating a guided basis](docs/guided-basis.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
