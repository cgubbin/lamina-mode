# Changelog

The lamina crates (lamina-units, lamina-core, lamina-materials, lamina-mode) are
released in step until 1.0. Each entry names any shared convention it changes.

## Unreleased

Added:

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

Fixed:

- A solve whose discovery failed on a chart (for example a direct β rectangle
  with an edge on the core light line, where lamina-core returns
  `NonFiniteOutput`) returned a report with no sign that it was incomplete.
  `SolveReport::unresolved` now lists each failed chart, stopped discovery,
  unresolved leaf and unaccepted estimate as an `UnresolvedRegion` with its
  `UnresolvedCause`, and `SolveReport::is_complete` holds only when there are
  none and reconciliation raised no diagnostic.

Changed (breaking):

- `BranchSignature::left`/`right`, `PhysicalData::kappa`,
  `RadiationReference::choices` and `VerifiedRadiationReference::observed`
  hold an `Option` per side, `None` on a walled side; `BranchSignature` gains
  `exteriors` and `open_exteriors`.
- A side closed by anything other than a wall is refused when the problem is
  built, with `ConstructionError::UnsupportedTermination` naming the side.
- `FrequencyProblem` refuses walled stacks (`ConstructionError::ClosedSide`);
  conformal domains of a stack walled on both sides fail with
  `ConformalChartError::ClosedStack`; pairings refuse walled stacks
  (`GuidedOverlapError::ClosedStack`).
- Built against lamina-core's per-side `ExteriorWavevectors`.

Open stacks keep their roots, evidence and enumerated records bit for bit. These breaking
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
