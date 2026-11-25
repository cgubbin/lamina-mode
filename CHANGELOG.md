# Changelog

The lamina crates (lamina-units, lamina-core, lamina-materials, lamina-mode) are
released in step until 1.0. Each entry names any shared convention it changes.

## Unreleased

Added:

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

Fixed:

- A solve whose discovery failed on a chart (for example a direct β rectangle
  with an edge on the core light line, where lamina-core returns
  `NonFiniteOutput`) returned a report with no sign that it was incomplete.
  `SolveReport::unresolved` now lists each failed chart, stopped discovery,
  unresolved leaf and unaccepted estimate as an `UnresolvedRegion` with its
  `UnresolvedCause`, and `SolveReport::is_complete` holds only when there are
  none and reconciliation raised no diagnostic.

Changed (breaking):

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

## 0.1.0 (unreleased)

First release on crates.io.

Conventions fixed by this release (see `docs/modal-fields.md`):

- the canonical spectral coordinate is vacuum angular wavenumber `k₀ = 2π/λ₀`
  in `cm⁻¹`; in-plane wavenumber β uses the same unit internally;
- the stack normal is z and the in-plane wavevector points along x; TE has
  Ey, Hx, Hz and TM has Hy, Ex, Ez;
- phasors use `exp(+iβx − iωt)`;
- guided modes are normalised to unit signed longitudinal power per metre of
  invariant width, with the left-interface primary field real and positive.
