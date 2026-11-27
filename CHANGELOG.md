# Changelog

The lamina crates (lamina-units, lamina-core, lamina-materials, lamina-mode) are
released in step until 1.0. Each entry names any shared convention it changes.

## Unreleased

Fixed:

- Guided enumeration refuses an open exterior that couples TE and TM (an ε or
  μ entry `xy`, `yx`, `yz` or `zy`) before any search, on every family and
  backend, with `EnumerationError::Constitutive(ConstitutiveError::Unsupported
  { region, class })`. In 0.2.0 such an exterior was accepted, and the 4×4
  reference basis of its incoming waves could lose rank at a real β above the
  light line: a zero of the characteristic for every stack, reported as a
  guided mode in a complete basis. For a uniaxial crystal (ε_o = 2.25,
  ε_e = 2.56) with its optic axis in the interface plane against ε = 2.40, a
  spurious mode was reported at β/k0 = 1.5732133 (60°), beside the true one at
  45.9271° and 45.85°. Isotropic, diagonal and separable full-class exteriors
  (an optic axis in the xz plane) are unaffected: their reference loses rank
  only on their light lines, outside the guided interval. Their results are
  unchanged bit for bit. Anisotropic open exteriors will be supported by
  a later change.
- lamina-py: such a stack raises `InvalidInput` ("anisotropic open exteriors are
  not supported yet").

## 0.2.0 (2026-11-27)

Added:

- `ModePoint::hybrid_power_field_tangent_with` gives the rates of unit-power
  fields and regional power fractions of guided modes of the coupled channel
  along a `HybridPerturbation` (thickness, k0, full tensor rates, rotations),
  material-attached or fixed in the lab frame, as `HybridPowerSensitivity`
  (β rate, phase reference, rate of the unit-power multiplier, samples and
  regional powers). The rates come from the closed-form field and form jets,
  with the moving ends of layers whose thickness changes; they match central
  differences to 1e-8 on a general-tensor film and 4e-7 at an avoided
  crossing, and on a separable stack equal the scalar path's to 2e-10.
  `HybridPhaseReference` selection is shared with `hybrid_power_profile_with`.
  `HybridPerturbation` gains `origin_velocity` (the first interface's lab-frame
  velocity, moving fixed-lab samples; a walled stack refuses it with
  `HybridFieldError::MovesAgainstWalls`), a breaking change for struct
  literals.
  A TE or TM mode solved through a 4×4 backend refuses a direction that
  couples TE and TM (`HybridFieldError::CouplesPolarisations { region }`),
  for its β rate as for its fields: count it on the coupled channel.
- lamina-py: sensitivities of modes counted on the coupled channel return the
  unit-power fields at `z_m`, their derivatives in either frame and the
  regional power fractions with their rates, along every parameter, instead
  of dβ alone; `z_m` on a TE or TM mode along a rate that couples TE and TM
  still raises `InvalidInput`, now naming the coupled channel.
- `ModePoint::continue_through_folds_with` follows a branch by arclength in
  the (k0, Re β) plane, holding fixed k0 or β, whichever the tangent moves
  along most, so it passes folds (zero-group-velocity points of lossless
  modes) that stop `continue_with`. Folds are located on fixed-β stations and
  reported as `Fold` (location, station, slope, and the points and slopes on
  both sides); `ArclengthSettings`, `ArclengthTrace`, `ArclengthEnd`,
  `Control` and `FrequencyStation` configure and report it.

- Continuation counts the roots in each corrector chart by its boundary
  winding before correcting, and refuses a step whose chart does not hold
  exactly one: `BranchFailure::RootCount { zeros, bounds }` (none: the branch
  was lost; several: neighbours within the chart radius) and
  `BranchFailure::Uncounted(RootCountFailure)`, subdivided like other
  failures. A step in a crowded hyperbolic ladder no longer converges silently
  onto a neighbouring root. `BranchTrace::near_degenerate` reports the steps
  whose chart held several roots, with the count and region.

- `GuidedBetaError::IneligibleEntry` and `GroupIndexError::IneligibleEntry`
  name the region, the entry (`ε_xx`) and its value when a sensitivity or group
  index is refused for an entry that is not lossless (complex, not finite or
  zero); a lossy entry is named before the mode's complex β is refused.
- Exterior sheets on the in-plane chart: `ExteriorSector` (proper, outgoing or
  incoming improper, real) and `BranchSignature::sectors`, so every point names
  the sheet of each exterior; `PropagationProblem::full_period_plan` and
  `full_period_domains` add the iπ-shifted strip, which holds the improper
  sheets the canonical strip omits. A lossy mode that crosses the light line
  and leaves the proper sheet is found there.
- `DiscoveryConfig::max_subdivisions` (default 1000): plan discovery splits
  each converged leaf that holds several roots into quadrants searched again,
  as modal bases do, and splits a chart whose search fails numerically. One
  chart over a crowded hyperbolic ladder (10 roots) is now complete; quadrant
  counts that do not add up to their cell's are unresolved with
  `WindingViolation::Inconsistent`. `ChartDiscovery::subdivisions` reports the
  splits.
- `UnresolvedCause::AssumptionViolated`: a discovery leaf with a negative
  winding is unresolved, so a box holding a pole is never complete.
- `FrequencyProblem::known_singularities`: the declared poles and zeros of the
  stack's regions at which the channel's characteristic is singular (poles of
  every entry; zeros of ε_xx, ε_zz for TM, μ_xx, μ_zz for TE, every zero for
  the coupled channel), and the regions that declare none.
- `FrequencyProblem::singularity_exclusions` and `FrequencyProblem::domains`:
  exclusions around the known singularities, and a box less the exclusions
  inside it as rectangular domains, each with its exterior reference continued
  from the box's through adjacent tiles. The holes are recorded in
  `FrequencyEvidence::holes` and in the plan's coverage.
- Guided modes of stacks with full-tensor regions (docs/hybrid-modes.md):
  `ModeFamily::All` solves the coupled channel of Transfer4 or Scatter4 and
  labels each mode TE, TM (separable stacks) or `Hybrid`; TE and TM of a
  separable stack with full-class regions use the 4×4 backends' scalar
  channels. Counts by the argument principle on graded β rectangles above the
  light line, with exterior roots continued over each rectangle (closed form
  for separable exteriors, the quartic otherwise), sampled region evidence
  (`GuidedRegionEvidence`), Transfer4 for rectangles Scatter4 cannot continue
  over, `NonRealRoot` and `NearDegenerate` gaps, and a separable cross-check
  against the TE and TM counts (`HybridCountEvidence`). Eligibility: Hermitian,
  positive-definite ε and μ; upper bound k0 max √(λmax(ε) λmax(μ)).
- Unit-power six-component fields (`hybrid_power_profile_with`), with the
  phase rule `E_y + Z₀H_y` real and positive at the left interface (fallback
  `E_x + Z₀H_x`, recorded); the unconjugated reciprocity pairing and power
  overlaps; β rates along thickness, wavenumber, tensor-rate and rotation
  directions (`HybridPerturbation`); the group index with its Hermitian
  dispersive energy certificate; tracking of bases with near-degenerate pairs
  followed as subspaces.
- Leaky and quasi-normal problems take a channel: the coupled channel with
  full-tensor layers and isotropic exteriors.

- Guided modes of stacks whose regions are isotropic or carry ε and μ
  diagonal in the stack frame (lamina-core `Region`; principal axes along
  x, y and z). A polarisation sees three entries per region: μ_xx, μ_zz and
  ε_yy for TE, ε_xx, ε_zz and μ_yy for TM. The oscillation count uses
  q² = s (line² − β²) with line = k0 √(ε_yy μ_zz) and s = μ_xx/μ_zz (TE), or
  line = k0 √(ε_zz μ_yy) and s = ε_xx/ε_zz (TM); light lines and upper bounds
  are per polarisation. The in-plane and spectral charts keep their branch
  points at the per-polarisation lines and scale each exterior wavevector by
  √s. Fields, unit power and its phase rule, local derivatives, branch
  continuation, sensitivities, overlaps, reciprocity and local coupling carry
  over; the group-index energy uses per-component dispersive weights
  ε_ii + k0 ∂ε_ii/∂k0 and μ_ii + k0 ∂μ_ii/∂k0, and local coupling weighs each
  field component by its own permittivity rate, with ε_xx and ε_yy in the
  tangential and ε_zz in the normal interface terms.
- Modal bases in a β window and their pairings accept diagonal regions,
  through walls and PMLs: the PML normalisation scale, its near-zero and
  residual limits and the PML quadrature split use each exterior's channel,
  κ² = s (χ k0² − β²).
- Eligibility of guided enumeration: the entries a polarisation sees must be
  real and positive, so a hyperbolic or lossy entry is refused with
  `EnumerationError::Material { region, entry }` (`TensorEntry`); entries it
  does not see are not inspected.
- `enumerate_guided_modes_with(backend, stack, window)` selects the 2×2
  backend; `enumerate_guided_modes` keeps Scatter2.
- `ConstantExterior::diagonal` for spectral problems with diagonal constant
  exteriors.
- `ConstitutiveRate::diagonal` and the fields `permittivity_entries` and
  `permeability_entries`: the rate of ε_ii is `relative_permittivity +
  permittivity_entries[i]`, and likewise for μ; `ConstitutiveRate::scaled`.
  Wavelength directions add each tensor entry's dispersion to its own rate.
- Regions with off-diagonal entries are refused with
  `ConstructionError::Constitutive`, `EnumerationError::Constitutive` or
  `DirectChartError::Constitutive` (lamina-core `ConstitutiveError`).

- Complete modal bases in a window for stacks closed by walls on both sides:
  `modal_basis_in_window_with` with `ModalWindow::Wavenumber` (and `forward`)
  finds every mode in a β rectangle on the direct chart, guided,
  radiation-like, evanescent and complex. The expected count comes from the
  argument principle over the window boundary; each region left unresolved is
  a `BasisGap` with its `BasisLimit` (`BoundaryRoot`, `NonIntegerCount`,
  `Budget`, `Refinement`, `Multiplicity`, `Cutoff`); `BasisStatus` and
  `BasisScope::Discrete` state the claim. `BasisStatus::Complete` also needs
  the counted and found modes to agree and every mode to be normalised, so it
  claims more than `SolveReport::is_complete`. Modes have unit unconjugated self-pairing from a canonical seed,
  a sign rule at the first interface (with a wall reference when a wall lies
  on it), the order decreasing Re β², Im β², Re β, and unconjugated Löwdin
  bases of near-degenerate groups (1e-6 k0); ill-conditioned self-pairings
  (below 1e-8 of the field scale) are reported per mode. Results are
  reproducible bit for bit (docs/decisions/0011).
- `modal_pairing_with` and `basis_pairing_with`: the unconjugated pairing
  ¼∫(E_a×H_b + E_b×H_a)·x̂ dz and the conjugated product between modes of two
  stacks whose walls coincide in a registered frame, with regional
  contributions, singular values, leading condition numbers and self-pairing
  defects from one shared grid. The guided pairings share their integrator.
- Modal bases and pairings of stacks closed by wall-backed PMLs, on either or
  both sides. Inside a PML the unconjugated pairing is the contour integral
  along the stretched coordinate, ∫ (…) s dζ = ∫ (…) dζ̃, which depends on the
  PML only through its complex thickness τ; the conjugated product integrates
  the equivalent-medium fields in real coordinates (normal components s E_z,
  s H_z). PML intervals are split by their transverse phase |κ τ|. A basis
  counts every mode in the window: those converging to the open stack's
  guided and exposed leaky modes as the PML thickens and those near the
  rotated continuum. The near-zero threshold of the search and the residual
  limit of root correction follow the PML scale of the characteristic, and a
  window on which that scale leaves the representable range is refused with
  `ModalBasisError::Scale`; a cell that does so during subdivision is
  unresolved with `BasisLimit::Scale`. Pairings accept PMLs equal in start,
  thickness, profile and backing in the registered frame; their materials may
  differ. Accepted roots of a basis are stepped once more by their own Newton
  correction, self-pairings converge relative to themselves, and a modal
  pairing's regional contributions converge with its totals.
- Modal bases of stacks with one or two open sides, on the conformal chart of
  the open exteriors: `ModalWindow::Chart`, a rectangle in z less than one
  period tall, and `ModalWindow::covering`, the chart window covering a β
  rectangle on stated sheets with its extra area (`WindowError`, including
  `CrossesCut` for a rectangle that meets a stated sheet's cut). Guided and
  leaky modes carry their `ExteriorSheet` per open side; scope
  `BasisScope::DiscreteWithoutContinuum` states that the radiation continuum
  is not included; a root within 1e-4 k0 of a branch point is unresolved with
  `BasisLimit::BranchPoint`. Open tails: unconjugated i v₀/(κ_a + κ_b) by
  analytic continuation, conjugated i v₀/(κ_a − κ_b*) only where
  Im κ_a + Im κ_b > 0, otherwise absent (`Absent::Unbounded`), and
  `TailSingular` near κ_a + κ_b = 0. Pairings accept sides open in both
  stacks. Open exteriors must be isotropic; an anisotropic one is refused
  with `WindowError::AnisotropicExterior`, naming the region.
- `basis_projection_with`: a supplied field F expanded in the normalised
  modes of a `ModalBasis` by solving G a = b, G_mn = ⟨f_m, f_n⟩ and
  b_m = ⟨f_m, F⟩ unconjugated, on the basis's partition with dz̃ in a PML and
  continued tails on open sides, all at one refinement level
  (`BasisProjectionInput`, `BasisFieldProjection`). Inside a PML F is the
  continued field, as modal profiles give it; on an open side the field
  beyond the support may be stated as `ExteriorWave`s, whose pairings are
  closed-form tails. The result has G's singular values, condition number
  and defect, the equation residual, and the field scale of
  F − Σ a_m f_m over the bounded partition, absolute and relative to F's;
  `residual_includes_continuum` states that with an open side it holds F's
  radiation content, which no discrete mode represents.
- `basis_local_coupling_with`: the complex-symmetric perturbation matrix
  K_mn = (ω ε0/4) [∫ Ẽ_m · ∂ε/∂p · E_n dz̃ + interface terms] of a modal
  basis, Ẽ = (−E_x, E_y, E_z) the mirrored field, with one-sided fields at
  moving interfaces, the stretch in a PML and continued tails on open sides;
  its diagonal is each mode's dβ/dp, and C_mn = K_mn/(β_n − β_m) is the rate
  of the pairing of mode m with the moving mode n. Modes within the gap
  (default the basis's degeneracy gap) form groups with absent mutual
  coefficients. Wavelength and permeability directions, moving jumps in μ and
  interfaces moving against walls or PMLs are refused (`BasisCouplingError`).
- lamina-py: `ModalBasis.project_field` and `ModalBasis.local_coupling`, with
  the record kinds `lamina.basis_projection` and
  `lamina.basis_local_coupling` at schema version 1.
- Breaking: `ModalPairing::conjugated_w_per_m` is a
  `Result<Complex, Absent>` and `BasisPairing::conjugated` a
  `DMatrix<Result<Complex, Absent>>`.
- lamina-py: `lamina.ChartWindow`, `lamina.chart_window_covering`, open stacks
  in `lamina.modal_basis` (charts `"one_exterior"` and `"two_exterior"`,
  sheets, `"guided"` and `"leaky"` mode types, scope
  `"discrete_without_continuum"`, cause `"branch_point"`), and `null`
  conjugated entries with cause `"unbounded"` in `lamina.basis_pairing`. A
  `BetaWindow` on an open stack raises `InvalidInput` naming `ChartWindow`.
- lamina-py: `lamina.modal_basis` and `lamina.basis_pairing` accept PML-closed
  stacks; profiles name `"pml"` samples and carry their stretched coordinate
  and stretch (`stretched_z_m`, `stretch`, `pml_fields`; omitted for stacks
  without a PML); unresolved regions may have cause `"scale"`; an
  out-of-range window raises `InvalidInput`.
- lamina-py: `lamina.BetaWindow`, `lamina.modal_basis`, `ModalBasis.profile`
  and `lamina.basis_pairing`, with the record kinds `lamina.modal_basis_input`,
  `lamina.modal_basis`, `lamina.modal_profile` and `lamina.basis_pairing` at
  schema version 1, each with fixed `definitions`.
- Stacks closed by perfect electric or perfect magnetic walls (lamina-core
  `Termination`). The search chart follows the open sides: the two-exterior
  conformal map when both are open, the same map with both constants set to
  the open side's material (r = 0) when one is, and a direct β chart in cm⁻¹
  (`PropagationProblem::direct_domain`, `CoordinateKind::Wavenumber`) when
  none is: there the characteristic is entire and even in β, with no sheet or
  radiation reference.
- `ExteriorBehaviour::Closed`, `Flux::Closed` and `ModeType::Closed`. With one
  open side the mode type follows from that side alone.
- `enumerate_guided_modes` over walls: the phase count starts and ends with
  the wall condition, the gaps count as layers of the exterior material, and
  the light line comes from open sides only. Between two walls the basis holds
  every real-β mode in the window, radiation-like ones included.
- Unit-power profiles, regional power, local derivatives, sensitivities and
  the group index integrate each wall gap, with the walls fixed in the stack
  frame. `DirectionError::MovesAgainstWalls` refuses a direction that moves the
  first interface of a walled stack.
- lamina-py: `lamina.Wall` and `Stack(left_termination=, right_termination=)`;
  optional `left_termination`/`right_termination` in the input record,
  internally tagged by `"kind"` and omitted when open, so every open-stack
  record keeps its hash (docs/decisions/0009).
- Stacks closed by wall-backed PMLs (lamina-core `Pml`). A PML side counts as
  closed: it takes no exterior wavevector, classifies as `Closed`, and a stack
  closed by walls and PMLs is searched on the direct β chart. `direct_domain`
  refuses a rectangle on which the PML normalisation scale leaves e^±300
  (`DirectChartError::PmlScale`, naming the side). Fields are sampled through a
  PML on its stretched coordinate. The convergence of guided and leaky modes to
  the open stack's is in docs/pml-convergence-report.md.
- A mode of a stack with a PML side infers no gain or loss from Im β:
  `Evolution::PmlTruncation { truncation }` carries the largest
  |exp(2iκ̃(g + τ))| over the PML sides instead. Beside an open side the type
  follows from that side's exterior and flux alone (guided when decaying,
  leaky when growing outward). `Evolution` is no longer `Eq`.
- lamina-py: `lamina.Pml` with every field required, and the `"pml"` record
  variant `{"kind": "pml", "start_z_m", "thickness_m", "profile": {"kind":
  "polynomial", "order", "strength"}, "backing"}`; walled and open records keep
  their hashes. Enumerating a PML-terminated stack raises
  `UnsupportedTermination`, an `InvalidInput`.
- Modal bases and coupled guided bases localise by the boundary winding they
  already count: a cell holding one zero is corrected from the winding's first
  moment, a cell holding more is split into counted quadrants, and argloc runs
  only where that fails. Counts, roots and completeness are unchanged; a
  coupled basis of a one-layer slab takes about 1,200 characteristic
  evaluations instead of 19,000.
  
Changed:

- Single-stack mode integrals are closed forms from lamina-core's
  `directional_forms` instead of refined composite Simpson sums of sampled
  fields: the hybrid unit-power profile, reciprocity pairing, power overlap
  (and tracking) and group-index energy; the scalar unit-power profile and
  group-index stored energy; regional powers and their rates
  (`power_field_tangent_with`). Values agree with the previous quadrature to
  its tolerance and are exact to round-off (the power of a unit profile is
  1 ± 1e-13 against tight Gauss–Legendre, from 1 ± 3e-11). A hybrid profile
  takes about 7 ms instead of 0.75 s, a pairing or overlap about 4 ms instead
  of 2 s; the hybrid_fields test target runs in 32 s instead of 231 s. A side
  closed by a PML is still refused. Two-stack registered pairings, modal
  bases, projection and local coupling keep their quadrature.
- `guided_power_profile_with`, `guided_power_overlap_with`,
  `guided_reciprocity_with` and `guided_local_coupling_with` also require
  `ModalDerivativeBackend` of their backend (every built-in backend has it).
- Error variants: `GuidedPowerError::Quadrature` becomes `Integrals`;
  `GroupIndexError::{Quadrature, Recovery}` become `Integrals` and
  `IllConditioned`; `GuidedPowerSensitivityError::Quadrature` becomes
  `Integrals`; `HybridFieldError::Quadrature` is gone and `Mismatched` reports
  two modes whose stacks differ in layer thicknesses. Of the
  `guided_group_index_with` field-recovery settings only the amplitude enters.
- `guided_beta_sensitivity_with`, `guided_group_index_with` and
  `guided_power_profile_with` (and the field derivatives and power fractions
  built on them) accept lossless entries of either sign: real, finite and
  non-zero instead of real and positive. Modes of hyperbolic and
  negative-permittivity layers (a lossless hBN film in either Reststrahlen
  band, a surface phonon polariton) get a certified group index and unit-power
  profiles; the energy certificate uses the Brillouin weight ∂(k0 ε)/∂k0,
  positive in a lossless Lorentz medium also where ε < 0. Backward modes have
  a negative group index and power. Requests that failed with
  `IneligibleEntry` for a negative entry now succeed.
- With cluster splitting enabled (`DiscoveryConfig::max_subdivisions > 0`,
  the default), argloc's refinement stops at the first step at which every
  leaf is resolved instead of after two more confirming steps, in plans,
  modal bases and guided enumeration. On a tiled chart search of a 100 nm hBN
  film the same roots are found with 13–44 % fewer characteristic evaluations
  (1501 cm⁻¹: 88 639 → 58 525; 1375 cm⁻¹: 228 325 → 182 883); root values agree
  to refinement round-off. With `max_subdivisions = 0` argloc's default window
  is kept, since an early-converged leaf holding a cluster would otherwise be
  reported rather than split.


Fixed:

- The rates of the coupled channel's exterior wavevectors along a direction
  are Cauchy integrals over a radius scaled to the direction (k0, β and the
  exterior tensors change by 1e-5 of themselves), not a fixed 1e-4 in the
  parameter: at optical k0 in cm⁻¹, or along a fast direction such as a
  thickness at its β rate, the fixed radius lost digits or overshot.
- Refinement accepts a root whose residual is at the characteristic's
  round-off floor: once the Newton correction is within 64 units of round-off
  of the coordinate and a central difference of values confirms the
  derivative, the residual is judged against `64 ε |F′| max(|z|, 1)` where
  that exceeds the configured limit (`NumericalEvidence::residual_floor`; the
  applied limit is `residual_limit`). Deep in the type II ladder of a 100 nm
  hBN film near ν̃_TO⊥ the round-off of the characteristic exceeds the default
  limit 1e-10 k0/2π, and refinement stopped with `NoProgress` (lossless) or
  `Stagnated` (lossy) at roots located to round-off: continuing every root
  with 1370.5 ≤ ν̃ < 1380 cm⁻¹ up by 2 cm⁻¹ in 0.01 cm⁻¹ steps, 38 of 93
  lossless and 31 of 83 lossy branches stopped, and searches about 71 and 57
  further roots of the same window found none. Now all 164 and 140 roots are
  found and continued, with no stop and no jump between branches.
  An explicit absolute residual limit too small for the scale (the old
  1e-10 at 1500 cm⁻¹) likewise no longer stagnates at a root located to
  round-off.
- A solve whose discovery failed on a chart (for example a direct β rectangle
  with an edge on the core light line, where lamina-core returns
  `NonFiniteOutput`) returned a report with no sign that it was incomplete.
  `SolveReport::unresolved` now lists each failed chart, stopped discovery,
  unresolved leaf and unaccepted estimate as an `UnresolvedRegion` with its
  `UnresolvedCause`, and `SolveReport::is_complete` holds only when there are
  none and reconciliation raised no diagnostic.

Changed (breaking):

- `DiscoveryConfig::zero_tolerance` and `RefinementConfig::residual_atol` are
  `Option<f64>`. The default, `None`, is `DEFAULT_ZERO_TOLERANCE` (1e-12) or
  `DEFAULT_RESIDUAL_ATOL` (1e-10) at a 1 cm wavelength scaled by |k|/2π, with k
  the plan's wavenumber (fixed k0, or the largest |k0| or |β| of the domains),
  as `ModalBasisRequest::new` already did; a given value is used as given. At
  k0 ≈ 10⁴ cm⁻¹ the old absolute defaults left roots `Stagnated`. Modal bases
  and hybrid counts keep their thresholds.
- `GuidedWindow::polarisation` becomes `family: ModeFamily`
  (`From<Polarisation>`); `ModeFamily` gains `All`. `GuidedBasis` gains
  `labels` and `hybrid`; `GuidedLimit` gains the argument-principle causes.
- Problems carry a `Channel`: `Provenance::polarisation` returns
  `Option<Polarisation>` (`None` for the coupled channel) and
  `Provenance::channel` is added; scalar-only capabilities refuse the coupled
  channel (`CoupledChannel` causes). `BranchSignature` gains `pairs`.
  `enumerate_guided_modes_with` requires a `GuidedBackend` (the four core
  backends). A region with off-diagonal entries is no longer refused by guided
  enumeration.

- Stacks are `ModeStack` (lamina-core `AnalyticalTensorStack` of `Region`s).
  Functions that take a stack accept `impl Into<ModeStack>`, so scalar stacks
  convert; `ProblemContext::stack`, `ModalProfile::stack`, `ModalBasis::stack`
  and `GuidedTrackingStep::stack` hold a `ModeStack`, and field backends are
  bound as `ModalFieldBackend<Region>`. Spectral problems take layers of
  `impl Into<Region>`.
- `EnumerationError::Material(region)` becomes `Material { region, entry }`.
- `ConstantExterior::epsilon` and `mu` return the stack-frame diagonal, and
  `ConstantExterior` no longer implements the scalar material traits.
- An exterior that is not isotropic is refused by `FrequencyProblem`
  (`ConstructionError::Constitutive`; `ExteriorFailure::Anisotropic` on its
  domain) and by QNM scattering windows, and a radiation reference refuses an
  exterior whose channel is scaled (`RadiationReferenceError::UnsupportedMaterial`).
  Interpretation reports such an open exterior as
  `Unsupported(ExteriorMaterial)`.

- `BranchSignature::left`/`right`, `PhysicalData::kappa`,
  `RadiationReference::choices` and `VerifiedRadiationReference::observed`
  hold an `Option` per side, `None` on a walled side; `BranchSignature` gains
  `exteriors` and `open_exteriors`.
- A side closed by a termination other than a wall or a PML is refused when
  the problem is built, with `ConstructionError::UnsupportedTermination`
  naming the side.
- Guided enumeration refuses PML sides with `EnumerationError::Termination`,
  and unit-power profiles, the group index and power tangents refuse them with
  `GuidedPowerError::Termination`, `GroupIndexError::Termination` and
  `GuidedPowerSensitivityError::Termination` (`PmlIntegral`, naming the side).
- `FrequencyProblem` and guided-mode pairings refuse PML-closed stacks as they
  refuse walled ones.
- `FrequencyProblem` refuses walled stacks (`ConstructionError::ClosedSide`);
  conformal domains of a stack walled on both sides fail with
  `ConformalChartError::ClosedStack`; pairings refuse walled stacks
  (`GuidedOverlapError::ClosedStack`).
- Built against lamina-core's per-side `ExteriorWavevectors`.

Isotropic stacks keep their roots, fields, sensitivities, group indices,
overlaps and couplings bit for bit. Open stacks keep their roots, evidence and enumerated records bit for bit. These breaking
changes, with lamina-core's, make the next release 0.2.0; lamina-core's version
is still 0.1.0, so the package versions and the lamina-core requirement are
unchanged here.

## 0.1.0

First release on crates.io.

Conventions fixed by this release (see `docs/modal-fields.md`):

- the canonical spectral coordinate is vacuum angular wavenumber `k₀ = 2π/λ₀`
  in `cm⁻¹`; in-plane wavenumber β uses the same unit internally;
- the stack normal is z and the in-plane wavevector points along x; TE has
  Ey, Hx, Hz and TM has Hy, Ex, Ez;
- phasors use `exp(+iβx − iωt)`;
- guided modes are normalised to unit signed longitudinal power per metre of
  invariant width, with the left-interface primary field real and positive.
