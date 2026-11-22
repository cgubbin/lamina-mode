# Modal transforms and exterior sheets

The exponential map formulas agree with the fixed-frequency transform of
Bakhtazad et al. after translating notation. Earlier Lamina documentation
incorrectly identified a width-π strip as the full exterior-sheet surface.
That strip remains a useful finite search selection, but excludes other sheets.
This distinction does not invalidate roots found on the selected sheets.

The solver modules are currently private. Point discovery, refinement and finite
chart orchestration, explicit radiation references and supported physical point
classification are implemented. A public point-solver API, field recovery, branch
continuation and QNM normalisation remain future work.

## Reference and notation

A. Bakhtazad, H. Abiri and R. Ghayour, **“A General Transform for Regularizing
Planar Open Waveguide Dispersion Relation”**, *Journal of Lightwave Technology*
15(2), 383–390 (1997), [DOI: 10.1109/50.554392](https://doi.org/10.1109/50.554392).
Equations 4–9 define the transform; Figure 2 and section III discuss mode types.

Use `z` for Lamina's dimensionless conformal coordinate. This is not the physical
propagation axis `z` in the paper. Lamina's physical stack-normal coordinate
increases from left to right; the in-plane propagation wavenumber is
`k_parallel` (called `kx` in map source). Angular wavenumbers are in cm^-1 and
layer thicknesses in cm. Frequency is represented by vacuum angular wavenumber
`k0 = omega/c`.

| Paper | Lamina, fixed real k0 |
| --- | --- |
| cladding, substrate | ordered left, right exterior |
| beta, beta-bar | k_parallel, k_parallel/k0 |
| n_cl², n_sub² | chi_L, chi_R, with chi = epsilon*mu; n² identification assumes nonmagnetic dielectrics |
| eta*Z | z; eta is the paper's arbitrary positive coordinate scale |
| R = (n_sub²-n_cl²)/4 | -r, with r = (chi_L-chi_R)/4 |
| S | s = (chi_L+chi_R)/2 |
| w(Z) | k_parallel²(z)/k0² |
| g(Z), h(Z) | alpha_L/k0 = -i*kappa_L/k0, alpha_R/k0 = -i*kappa_R/k0 |

Thus equation 7 becomes

```text
k_parallel² = k0² [s + exp(2z) + r² exp(-2z)]
kappa_L = i k0 [exp(z) - r exp(-z)]
kappa_R = i k0 [exp(z) + r exp(-z)].
```

Substituting `R=-r` into the paper's equations 6a and 6b gives these two
normalised decay constants directly. Each obeys
`kappa_j² = chi_j*k0² - k_parallel²`. Swapping left and right requires also
translating the sign and physical labels; do not copy the Figure 2 regions after
silently reversing the exterior order.

## Field and radiation conventions

Both use the time factor `exp(-i omega t)` and in-plane factor
`exp(+i k_parallel u)` for a physical propagation coordinate `u`. Core's
`waves/propagation.rs` defines forward/backward stack-normal factors
`exp(+i kappa x)` and `exp(-i kappa x)` respectively. In the homogeneous
outgoing-basis characteristic, the surviving exterior factors are

```text
left:  exp[-i kappa_L (x-x_L)] = exp[+alpha_L (x-x_L)]
right: exp[+i kappa_R (x-x_R)] = exp[-alpha_R (x-x_R)]
alpha_j = -i kappa_j.
```

These are the paper's equation 1 exterior factors. A positive real alpha gives
decay away from the stack. Calling the characteristic outgoing refers to the
selected directional basis: arbitrary analytically selected kappa signs do not
guarantee the physically outgoing sheet. For propagating positive-index real
exteriors the usual outgoing choice has positive real kappa on both sides.
For complex frequency its choice is an analytic radiation continuation, not a
universal pointwise sign rule.

The independent asymmetric TE/TM slab phase condition uses interior transverse
`q`, decay constants `alpha_L, alpha_R` and thickness `d`:

```text
q*d = atan(alpha_L*f_core/(q*f_L))
    + atan(alpha_R*f_core/(q*f_R))       (fundamental guided branch)
f_j = mu_j (TE), epsilon_j (TM).
```

The physical regressions construct a slab from this condition and recover its
guided root. Separately, the normal-incidence vacuum / index-two slab / vacuum
pole is `k0 = (pi - i ln(3))/(2d)` for both TE and TM. These tests establish the
field/sign correspondence without using the mapped determinant as its own
reference.

## Coordinate periods and four exterior sheets

Under `z -> z+i*pi`, the squared physical coordinate repeats, but both exterior
wavevectors reverse sign. The pair of exterior wavevectors repeats only after
`z -> z+2*i*pi`. This is the paper's `2*pi/eta` period in Z (equations 8–9).
The boundaries at Im(z)=±pi/2 are therefore **not identified as exterior sheets**.
One full representative can instead use `-pi <= Im(z) < pi`.

For a generic asymmetric exterior and a noncritical squared coordinate, the
inverse quadratic in `t=exp(2z)` has two roots. Each nonzero root has two
representatives modulo `2*i*pi`, separated by `i*pi`; together these give four
exterior sign pairs. The helper `physical_preimages` returns just two selected
representatives in the closed width-π strip, not all four sheet preimages.
At a light line some sheets meet; coincident quadratic roots and signed-zero
boundary choices must not be counted as four distinct preimages. Equal exteriors
give `r=0`: one t-root escapes to Re(z)=-infinity and the map describes the
reduced two-sheet continuation with linked exterior signs. It does not
independently encode every boundary-sign choice when the two branch points
coincide. The remaining finite signs still reverse after `i*pi`.

The core consumes a signed linear physical coordinate, not its square. Zeros of
the squared coordinate are additional branch points for that square root.
Their locations do repeat after `i*pi`; validators must continue to check these
copies. A `2*i*pi` exterior identity alone does not prove that analytic
continuation of the signed coordinate around a loop returns to the same sign.
Each domain retains its coordinate branch reference separately.

## Spectral extension and material restrictions

The fixed-real-k_parallel map searches complex k0 using

```text
r = k_parallel² (chi_L-chi_R)/(4 chi_L chi_R)
s = k_parallel² (chi_L+chi_R)/(2 chi_L chi_R)
k0² = s - exp(2z) - r² exp(-2z)
kappa_L = i sqrt(chi_L) [exp(z) - r exp(-z)]
kappa_R = i sqrt(chi_R) [exp(z) + r exp(-z)].
```

This is a Lamina extension, not the paper's fixed-frequency chart. The square
roots of the exterior chi values are fixed at construction and form branch
data. For this conformal strategy, the exterior constitutive products must remain constant
throughout the search. Sampling dispersive exterior models and freezing their
values is not an equivalent problem. Interiors may be dispersive provided the
caller supplies an appropriate analytic domain. The same squared-coordinate and
exterior-sheet period distinction applies.

The algebra supports more general complex/magnetic exterior constants than the
paper's Figure 2 assumptions. This does not extend its mode labels automatically.
Nor does the paper's regular dielectric characteristic certify pole-free
dispersive materials or prevent overflow, basis singularities or backend
representation failures. Checked evaluation is a pointwise numerical check;
holomorphy and pole-freedom on a neighbourhood of the search remain caller
assumptions.

## What canonical searches cover

Current canonical geometry requires positive real exterior chi and a finite
physical extent. It selects `-pi/2 < Im(z) < pi/2` with configurable margins.
The in-plane plan uses one rectangle; the spectral plan splits it into lower
and upper rectangles and also excludes a margin around Im(z)=0. Real bounds
come from selected physical preimages of the requested maximum coordinate.
These are finite conformal truncations, not exhaustive physical windows.

The coverage ledger records finite real bounds, excluded boundary/seam strips,
and an `OmittedExteriorSheetSectors` entry. That entry gives the nominal selected
imaginary interval, a full exterior period of `2*pi`, and the omitted intervals
within `[-pi,pi]`. Canonical margins remain separate entries. Reports preserve
this ledger even when discovery succeeds or an overlapping chart finds a root.
Success does not recover omitted sectors, failed charts or unresolved candidates.

The canonical spectral sheets need not both be physically outgoing: the existing
asymmetric TE/TM fixtures find opposite exterior real signs on their lower and
upper charts. The all-outgoing normal-incidence cavity fixture uses an explicit
domain outside the canonical strip. Valid explicit domains are allowed there;
their own coordinate branch reference and exterior signs are retained.

Equal-exterior canonical extents cannot infer finite real bounds from a maximum
alone, and spectral normal incidence is also degenerate. Supply explicit bounds.
Complex exterior chi similarly requires explicit geometry. Automatic full-surface
chart generation and seam recovery are not implemented. Duplicate grouping only
compares compatible observations from overlapping charts; it neither merges
opposite sheets nor identifies disjoint periodic copies.

## Physical interpretation

Figure 2 assumes real exterior dielectric n², with substrate n² greater than
cladding n², at fixed real frequency. The core may be active or absorptive.
Its regions depend on real/imaginary parts of the exterior decay constants and
on propagation convention. The paper includes cladding/substrate leaky modes,
cladding/substrate source modes, two-sided source and leaky modes, absorptive
modes and gain modes. These labels describe exterior flux and spatial behaviour;
they are not merely the sign of the core material's absorption.

For `exp(+i beta u)`, positive Im(beta) gives spatial attenuation. For
`exp(-i omega t)`, negative Im(omega) gives temporal decay. These are different
eigenproblems. Exterior field growth/decay, radiation direction and temporal or
spatial attenuation should be reported separately. Boundary cases and unsupported
material assumptions receive explicit boundary or unsupported outcomes in the
classifier below.

An accepted numerical root has local residual/correction and isolation evidence
on its selected chart. Optional classification is separate from acceptance.
Acceptance alone establishes neither a complete mode count nor an outgoing
frequency quasinormal mode. In particular, the fixed-frequency
diagram cannot simply be applied to the spectral extension.

## Radiation references and physical point interpretation

Radiation verification and classification are available through the public
point-solver facade. Existing non-radiation `domain(...)` calls retain
`RadiationProvenance::Unspecified`; selecting `CoordinateSheet` only chooses the
signed physical coordinate and does not establish outgoing exterior branches.

`domain_with_radiation_reference` is an opt-in path. Supply a conformal anchor
strictly inside the requested rectangle, its real varying physical coordinate,
per-side `OutgoingOrDecaying` or `IncomingOrGrowing` choices, explicit comparison
and cutoff tolerances, chart settings and the existing analytic assumption.
It requires positive real k0 and exactly positive real exterior epsilon with
mu=1. At the anchor, q=epsilon*k0²-beta² gives the outward/decaying choice
`+sqrt(q)` for an open channel and `+i*sqrt(-q)` for a closed channel; the opposite
choice negates it. The actual map is checked against those values.

Invalid settings, non-finite data, an exterior cutoff, an unsupported material,
an inconsistent coordinate or mismatched exterior signs fail before discovery.
The constructor never translates the rectangle or flips a wavevector to make a
request pass. A symmetric map cannot represent arbitrary mixed sign requests.
Clones, evaluation halos, refinement and chart plans retain the verified anchor
on the original domain identity. This establishes the selected analytic radiation
continuation within that chart, not material analyticity, a universal causal-sheet
proof or global coverage. Transporting radiation evidence to a disjoint chart is
future continuation work.

The outgoing normal-incidence cavity example uses the independent pole
`k0=(pi-i*ln(3))/(2*d)`. Its conformal point is `z=ln(-i*k0)` and lies below the
canonical strip. A real outgoing anchor is
`z_anchor=ln(abs(k0))-i*pi/2`, with real k0 at the anchor equal to `abs(k0)`.
Choose a finite explicit rectangle containing both points. The TE/TM regression
`anchored_cavity_preserves_reference_through_search_refinement_and_batch` checks
this continuation and retains the exact domain through discovery and refinement.
The real anchor shifted by i*pi has incoming exterior signs and fails the
outgoing request.

Internal use, after constructing the physical problem and rectangle:

```text
reference = RadiationReference {
    z: z_anchor,
    coordinate: abs(k0_pole),
    choices: [OutgoingOrDecaying, OutgoingOrDecaying],
    tolerance: RadiationTolerance::default(),
}
domain = problem.domain_with_radiation_reference(
    rectangle, reference, chart_config, analytic_assumption)?
// Search and refine on this same domain; root must be numerically accepted.
description = problem.describe_root(&domain, root, InterpretationConfig::default())
// For a batch: report.describe_points(InterpretationConfig::default()).
```

Each `PointDescription` borrows its accepted root, keeping polarization, numerical
evidence, domain and radiation provenance available even if its `result` is an
error. Batch descriptions additionally retain chart/candidate indices and borrow
original signature errors; they do not silently re-evaluate a failed signature.
Foreign problem/domain associations are rejected before physical evaluation.
Successful details contain raw k0, beta, ordered kappa and alpha, exterior factors,
independent physical components, thresholds and classification. They are snapshots
under the existing material-immutability obligation, not globally unique mode IDs.

| Component | Supported interpretation |
| --- | --- |
| Exterior growth/decay | Sign of Im(kappa)=Re(alpha), on either solve direction |
| Spatial behaviour | At fixed real frequency, Im(beta)>0 attenuates along positive propagation; Im(beta)<0 amplifies |
| Temporal behaviour | At fixed real beta, Im(k0)<0 decays; Im(k0)>0 grows |
| Outward flux coefficient | Fixed positive real frequency and positive real nonmagnetic exterior dielectrics only; Re[kappa/(k0*f)], with f=mu for TE and epsilon for TM |
| Named dielectric type | The preceding fixed-frequency assumptions and positive Re(beta), away from boundary bands |
| Frequency radiation qualification | Verified outgoing continuation, verified other choices, or unspecified; separate from temporal behaviour |

The dimensionless flux coefficient suppresses the positive unit-system
prefactor and is per squared primary-field amplitude. It is not integrated mode
power, normalised leakage or a pointwise flux rule for complex frequency. Left
and right outward coefficients use the same sign convention because the left
travelling-wave sign and outward normal both reverse.

`InterpretationConfig` records finite nonnegative absolute/relative wavenumber
bands and separate dimensionless flux bands, plus a positive cutoff band.
The wavenumber band is `absolute + relative*max(abs(k0),abs(beta),abs(kappa_L),
abs(kappa_R))`, in cm^-1, with no dimensionless floor of one. Defaults are 1e-9
for these five settings. Flux bands scale with the largest absolute coefficient.
Anchor comparison defaults are absolute=relative=cutoff=1e-10, with absolute and
cutoff in cm^-1. These are numerical conventions, not certified root uncertainties.
Non-finite data or invalid/unrepresentable thresholds are diagnostic errors.

The following strict-sector labels translate the paper's sign evidence into the
actual left/right exterior order. O/I denote outward/inward flux:

| Field behaviour (left, right) | Flux (left, right) | Spatial behaviour | Type |
| --- | --- | --- | --- |
| growth, decay | O, I | attenuation | left leaky |
| decay, growth | O, I | amplification | right source |
| decay, growth | I, O | attenuation | right leaky |
| growth, decay | I, O | amplification | left source |
| growth, growth | I, I | amplification | two-sided source |
| decay, decay | I, I | attenuation | absorptive |
| decay, decay | O, O | amplification | gain |
| growth, growth | O, O | attenuation | two-sided leaky |

For a real beta above both light lines, fields decaying on both sides and zero
flux within the bands give `Guided`. Cutoffs, improper real-axis solutions,
near-sector boundaries and inconsistent physical evidence receive explicit
boundary reasons. Reversed exterior index ordering is supported through these
physical signs; equal exteriors have only their represented reduced sectors.
Complex or magnetic exteriors, nonpositive propagation direction and spectral
searches withhold the paper's named labels with explicit unsupported reasons.
Available raw values and growth/attenuation components remain accessible.

For a decaying frequency root, all-outgoing verified provenance supplies an
outgoing-continuation frequency-mode candidate under the chart and material
assumptions. A negative imaginary frequency or apparently outgoing endpoint signs
alone do not establish that qualification. No result implies a normalised QNM,
mode volume or exhaustive search. Interpretation never revokes numerical
acceptance, promotes clusters, filters groups or erases coverage. Batch
interpretation proceeds independently for each accepted observation.

## Regression evidence

Run `cargo test --locked` with these test-name filters:

- `radiation`: real-anchor sign checks in both solve directions, outgoing TE/TM
  and incoming cavity continuation, domain identity and retained provenance.
- `physical::tests`: the eight strict sectors, component tolerances, supported
  limits, and post-acceptance failure with ownership checks before evaluation.
- `physical_interpretation_keeps_original_signature_errors_and_batch_evidence`:
  original error sources, independent outcomes and preserved batch evidence.

- `sheet_period_and_inverse_multiplicity`: full versus half periods, generic
  four-sheet inverse, light-line and symmetric limits in both maps.
- `overlapping_physical_te_tm_guided_and_normal_spectral_roots`: independent
  guided phase condition and outgoing cavity pole with retained provenance.
- `canonical_plans_match_existing_geometry_and_record_exclusions`: finite bounds
  and structured omitted-sheet ledger for both solve directions.
- `canonical_spectral_lower_upper_keep_physical_te_tm_evidence`: independent
  selected-sheet slab characteristic and unchanged coverage in reports.
- `matching_preserves_radiation_branches_order_and_signature_failures`:
  incompatible exterior signs remain distinct, including simultaneous reversal.
- `validator_detects_periodic`: squared-coordinate branch-point copies remain
  excluded even though the exterior period is twice as large.

## Direct frequency charts with dispersive exteriors

The constant-exterior restriction above belongs to the conformal spectral
strategy. A second, local strategy accepts an unchanged core
`AnalyticalMaterialStack<C>`, fixed real beta, polarization and backend. The same
stack is compiled for checked optics and supplies the exterior analytical handles.
Finite layers and exteriors retain their actual constitutive models. A material
must provide complex continuation and derivatives through core's analytical
material contract; real-axis samples alone do not qualify and are never silently
fitted, extrapolated or frozen.

The native variable is `k=k0`, vacuum angular wavenumber in cm^-1 (the core unit
convention), with identity derivative and fixed beta. `FrequencyRegion` is a
separate input wrapper over the existing rectangle machinery. Domain/result
provenance records `CoordinateKind::Frequency` and its units. Requested and
numerical evaluation bounds, discovery enclosures, the legacy `z` evidence field,
`RefinementConfig.z_atol` and `MatchConfig.conformal.absolute` all use that native
coordinate. Those names are private compatibility names; they do not turn k into
dimensionless z. Existing conformal domains retain dimensionless semantics.
Relative tolerances remain dimensionless; physical/exterior comparison tolerances
remain in cm^-1 for either strategy.

For each ordered exterior, using its actual complex epsilon and mu,

```text
q_j(k)  = epsilon_j(k) mu_j(k) k² - beta²
q'_j(k) = (epsilon'_j mu_j + epsilon_j mu'_j) k²
          + 2 epsilon_j mu_j k
kappa_j(k) = kappa_j_ref exp(1/2 integral[k_ref -> k] q'_j/q_j ds)
kappa'_j(k) = q'_j(k) / (2 kappa_j(k)).
```

The reference lies strictly inside the requested rectangle. Its finite, nonzero
ordered wavevectors must satisfy kappa_ref²=q(k_ref). Every sample integrates
along the straight segment from that same anchor, independently for each side.
Shuffled adaptive evaluations therefore cannot change sheet selection. Negating
one reference reverses only that exterior branch. No endpoint principal-root or
previous-sample fallback is used. A local algebraic square-root jet is signed to
agree with the continued scalar branch and checked against the analytic derivative.
Constitutive lifting includes both epsilon and mu dispersion. The checked optical
jet returns **dF/dk0**, without differentiating the numerical quadrature. For a
conformal comparison, use `(dF/dz)/(dk0/dz)` on matching physical branches. The
identity coordinate adds no artificial branch point at k0=0; real material or
backend limitations there still apply.

The private fallible `AnalyticChart` boundary shares the existing characteristic,
argument-principle discovery, bounded refinement, physical description and chart
plan pipeline. It supplies region, reference, native-coordinate projection,
seeded map and validated halo. Integration failure cannot be represented by the
old infallible raw-map interface without losing the cause; it remains a typed
chart error inside the shared optical/search error chain.

### Domains and continuation controls

Two explicit declarations are required: the existing characteristic/material
holomorphic pole-free assumption, and the non-default
`ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero` for both q_j on a
neighbourhood of the entire closed evaluation region. All q zeros are excluded
conservatively, including even-order zeros whose square root might be removable.
The characteristic assumption also covers material/backend singularities.
Successful samples, consistency checks and empty known-exclusion lists are not
domain certification.

Known exclusions are typed finite guard rectangles: material singularity,
exterior threshold with side, or caller-declared reason. Boundary contact with the
requested region or its roundoff halo rejects construction before discovery and
retains the exclusion. The caller supplies separate valid rectangles around
excluded scope. The solver neither subtracts holes nor automatically locates poles,
thresholds or a global dispersive Riemann surface. The halo retains the exact
anchor and branch data; it does not extend the search or refinement enclosure.
Genuine out-of-region evaluations remain errors.

`ContinuationConfig` defaults are a 1e-10 absolute tolerance on the dimensionless
logarithmic integral, 1e-8 relative squared-value/jet consistency tolerance,
1e-10 cm^-1 cutoff, maximum recursion depth 20 and 4096 material-jet evaluations
per exterior per point (including the local jet). Settings must be finite and
positive, depth is capped at 32 and evaluation budget must be nonzero. Sampled
cutoffs, non-finite values/derivatives, inconsistent branches, exhausted sample
budgets or recursion depth fail with exterior identity and available work count.
There is no automatic strategy, principal-root, constant-material or backend
substitution. These are numerical controls, not error or domain certificates.

### Reference modes and private core-stack flow

An explicit complex anchor supplies `ExteriorReferenceValues`; squared-relation
consistency establishes branch provenance, with radiation `Unspecified`. This
supports lossy and magnetic models without inventing an outgoing interpretation.
A `RealRadiation` anchor instead verifies per-side outgoing/decaying or incoming/
growing choices at positive real k0 using the supplied stack's actual positive
real epsilon and mu=1. Lossless exterior dispersion is allowed away from that
anchor. Lossy/magnetic anchors, cutoffs and invalid settings remain typed failures.
A matching real reference is evidence at that anchor, conditional on the declared
analytic continuation, rather than global causal-sheet certification.

The following historical illustration uses private implementation names.
Use the public point-solver quickstart and examples for executable public calls:

```rust,ignore
let stack = Stack::from_analytical_materials(left_model, right_model)
    .analytical_layer(core_model, Length::centimetres(thickness_cm))
    .finalise();
let problem = FrequencyProblem::frequency(stack, beta_cm_inverse, pol, Scatter2::new())?;
let domain = problem.frequency_domain(
    FrequencyDomainInput {
        bounds: FrequencyRegion::new(k_re_min, k_re_max, k_im_min, k_im_max)?,
        reference: FrequencyReference::RealRadiation {
            frequency: real_anchor,
            choices: [RadiationChoice::OutgoingOrDecaying; 2],
            tolerance: RadiationTolerance::default(),
        },
        continuation: ContinuationConfig::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: known_guard_rectangles,
    },
    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
)?;
let discovery = problem.search(&domain, ArgumentConfig::new(1e-6))?;
let refined = problem.refine(&domain, &discovery, RefinementConfig::default())?;
// Accepted roots carry this domain's identity, assumptions, units and evidence.
// problem.describe_root(&domain, root, InterpretationConfig::default())
```

Explicit frequency plans use `ChartPlan::explicit` over domains of one problem.
They retain finite requested rectangles and known exclusions rather than inventing
periodic conformal sectors. Each outside-region entry records an individual finite
chart scope, not a claim that its exterior is absent from every other chart.
Successful overlaps do not erase failed charts, terminations, multiplicities,
unresolved clusters, assumptions or missing coverage. Refinement cannot accept
outside its original enclosure or use the halo to expand it. Accepted observations
keep exact chart identity and ordered branch signatures; overlap groups retain
all contributors and never merge opposite exterior sheets. Mixed strategy plans
and cross-strategy reconciliation are not implemented.

Discovery errors preserve the original chart cause. Candidate evaluation errors
remain non-accepted refinement outcomes; failed physical-data evaluations keep
the accepted root and the typed error. Batch descriptions reuse retained signature
evidence. Spectral descriptions report temporal growth/decay, exterior behaviour
and radiation provenance; instantaneous complex-frequency flux and the paper's
fixed-frequency named classification remain unsupported. This supplies neither
fields nor QNM normalization yet.

### Independent dispersive regressions

Run `cargo test --locked problem::frequency`. The tests include:

- `dispersive_branches_cross_principal_cut_and_keep_both_material_derivatives`:
  independent q=exp(k) continuation, both material derivatives, opposite signs,
  shuffled points and finite differences across the principal cut.
- `dispersive_te_tm_slab_roots_and_derivatives_are_not_frozen`: vacuum / index-two
  slab of thickness 0.3 cm / Lorentz substrate with
  epsilon_R=2.25+100/(400-k²), mu=1, beta=0. The chart stays away from the known
  poles at ±20 and q zeros. Independent TE/TM Fabry-Perot roots and full derivatives
  agree; freezing at the real anchor shifts the root by more than 1e-3 cm^-1.
- `constant_te_tm_limit_matches_conformal_chain_rule_and_roots`: zero oscillator
  strength agrees with the conformal strategy after the coordinate chain rule.
- `frequency_overlap_keeps_native_evidence_and_finite_coverage` and
  `frequency_plans_keep_opposite_sheets_failures_and_clusters`: local evidence,
  overlap contributors, branch separation, failed charts and budget terminations.
- `frequency_refinement_and_description_retain_original_material_failures`:
  deliberately injected material faults remain typed while discovery/root evidence
  survives. Mutable models here are fault-injection fixtures, not a supported way
  to change an immutable problem between stages.

The independent characteristic uses TE admittance y=kappa/mu or TM y=kappa/epsilon,
core phase delta=kappa_core*d and

```text
D = 1 - [(y_c-y_L)/(y_c+y_L)] [(y_c-y_R)/(y_c+y_R)] exp(2i delta)
F = -i (y_L+y_c)(y_c+y_R)/(2 y_c) exp(-i delta) D.
```

The frequency-dependent prefactor is core's characteristic gauge (−2i y_L/t21).
It matters when comparing derivatives; equal root locations alone would not
validate dF/dk0. Tests require root agreement within 1e-8 cm^-1 and independent
finite-difference derivatives within 1e-7 in that gauge. Numerical continuation
adds work per optical evaluation; budgets bound each continuation and discovery
retains its own work/termination settings. Profiling and caching are future
performance choices, with fixed immutable references retained.
