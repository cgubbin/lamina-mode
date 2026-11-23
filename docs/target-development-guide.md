# Target development guide

This guide sets the targets for lamina-mode as a component of larger photonic design tools. It states what the
crate owns and what it never owns, the contract its results keep, and an ordered set of milestones with acceptance
tests. It complements the [integrated photonics roadmap](integrated-photonics-roadmap.md), which covers the modal
internals; this guide covers what consumers need from them.

## The boundary

lamina-mode answers one question: **given a planar multilayer, what are its modes, how certain is that answer, and
how does it change?** Its vocabulary is layers, thicknesses, materials, frequency and in-plane wavenumber. Anything
expressed in polygons, devices, ports or fabrication processes belongs to a consumer.

| lamina-mode owns | lamina-mode never owns |
| --- | --- |
| Modes of a 1D stack: guided, leaky, lossy, gain, plasmonic, quasi-normal | Polygons, layouts, GDS, cross-section geometry |
| Evidence: accepted roots, residuals, coverage, completeness of a stated window | Fabrication processes, foundry data, material databases |
| Fields, normalisation, regional power, overlaps between stacks | Ports, S-matrices, device-level assembly |
| Derivatives of modal quantities with respect to stack parameters | FDTD, full-vector 2D or 3D solvers |
| Continuation of modes and modal sets along a parameter | Objectives, optimisation drivers, device synthesis rules |

Two further families of models are wanted (milestones 4 and 5): approximate 2D confinement from 1D solves, and
local-mode propagation for tapers, MMIs and couplers. Both are built only from lamina-mode's public results, so
they belong one level up:

```text
lamina-units, lamina-materials, lamina-core   primitives: units, materials, stacks and transfer/scattering matrices
argloc                                        complex root localisation (published)
lamina-mode                                   1D modal engine (this crate)
lamina-reduce (proposed)                      separable 2D approximations; local-mode propagation
lamina-design                                 device physics: taper, MMI and coupler models; objectives
lamina-py (proposed)                          Python bindings over the public APIs above
downstream simulation stacks                  geometry to stacks, processes, verification against full solvers
```

Rules that keep the boundary:

1. **Stacks in, modal results out.** A consumer converts its geometry into ordered layers and thicknesses; lamina
   never sees the geometry.
2. **Public APIs only, one direction.** lamina-reduce and lamina-design use lamina-mode's public API and nothing
   private. lamina-mode imports none of them.
3. **Physics, not policy.** lamina-mode returns quantities and their evidence. Choosing a mode, a taper profile or
   an MMI length is a decision for a consumer.
4. **No silent approximation.** Every approximate result (milestone 4) states the approximation and an error
   indicator, so a consumer can decide whether to trust it or verify it with a full solver.

## The result contract

Consumers cache results, compare them against other solvers and differentiate through them, so these conventions
are part of the API. They are the most expensive to change later, so fix them before milestone 1 ships.

- **Units at the boundary.** Bindings take and return SI with the unit named in every field (lengths in metres or
  micrometres, chosen once; β and k0 in rad/m). Internal cm⁻¹ stays internal.
- **Frame and polarisation.** The stack normal and propagation axis are named in every result. TE and TM are
  relative to the stack normal. A consumer composing two stages (milestone 4) maps them to its own frame; the
  reduction states which stage polarisation produced which result.
- **Phasor sign, phase and normalisation.** One documented time convention, one phase-fixing rule and unit
  longitudinal power per unit invariant width for guided modes, as the roadmap specifies. Quasi-normal modes carry
  their own normalisation, named in the result.
- **Evidence travels with the answer.** Every mode set carries its window, expected and found counts, unresolved
  intervals and the completeness status. Every root carries its residual and acceptance thresholds. An answer
  without its evidence is not a result.
- **Typed failure, never a partial success.** As now: work limits and failed corrections are recorded with their
  cause.
- **Determinism and identity.** The same inputs give the same outputs, bit for bit, on one platform. Inputs have a
  canonical serialisation, so consumers can key caches on a hash of them.
- **Versioned results.** Serialised results carry a schema version; a change to any convention above bumps it.

## Milestones

Each milestone is usable on its own and builds only on earlier ones.

### 0. Publish the family

A consumer can't depend on path dependencies, so the crates are published to crates.io in dependency order:

| Crate | State | Work |
| --- | --- | --- |
| argloc | published (0.3.0) | none |
| lamina-units | no public repository | package and publish |
| lamina-materials | no public repository | package and publish |
| lamina-core | public repository, unpublished | update, then publish |
| lamina-mode | public repository, unpublished | replace path dependencies with versions; publish |

- **One licence** across the family. lamina-core is MIT OR Apache-2.0; lamina-mode is MIT. Choose one before the
  first release, as relicensing later needs every contributor's agreement.
- **Version together until 1.0.** The family shares result conventions (the contract below), so release the
  crates in step while those change, and say in each changelog which convention changed.
- **Each crate states its own boundary** in its README, as this guide does for lamina-mode, so a change that
  crosses one is visible in review.

*Accept when:* `cargo publish --dry-run` passes for each crate in order; a fresh project depending on lamina-mode
by version builds and runs the guided-basis example; `cargo deny` passes across the family.

### 1. Python bindings over the guided basis

A `lamina-py` crate (PyO3 and maturin) exposing what already exists: guided-mode enumeration, field sampling,
unit-power normalisation and the local derivatives, with numpy arrays and typed results that keep their evidence.

*Accept when:* analytic symmetric and asymmetric slab ladders match to 1e-10 in n_eff through the bindings;
results round-trip through their serialised form; equal inputs give equal hashes; wheels build for Linux on
x86-64 and are installable without a Rust toolchain.

### 2. Sensitivities

Total derivatives, not only characteristic partials:

- **eigenvalues:** dβ/dp for layer thickness, the motion of one interface with its neighbours fixed, rigid
  translation, wavelength, and material-model parameters (including dispersive models);
- **normalised fields:** dE/dp and dH/dp, including the motion of sample coordinates, the unit-power scale and the
  phase rule;
- **group index** for lossless guided modes, as a certified real quantity, distinct from the complex local slopes
  available now;
- **regional power fractions** (confinement factors) per layer and exterior, and their derivatives.

*Accept when:* every derivative converges against finite differences at the expected order; regional fractions
sum to one and their derivatives to zero; the group index matches a finite-difference dβ/dk0 on dispersive slabs.

### 3. Overlaps, projections and coupling coefficients

Roadmap steps 5 and 6, plus the quantity local-mode theory needs:

- the conjugated power product and the unconjugated reciprocity pairing, as separate named APIs, across two
  different stacks at one frequency;
- ordered bases with a conditioned Gram matrix, and the projection of a supplied field onto a basis;
- **local coupling coefficients per unit parameter change**, C_mn(p) = ⟨m|∂ε/∂p|n⟩/(β_m − β_n), with moving
  interfaces handled through one-sided fields (continuous tangential E and normal D), and the near-degenerate case
  returned as a coupled subspace instead of a large number.

*Accept when:* self-power and orthogonality hold to tolerance; coupling coefficients agree with finite
differences of overlaps between neighbouring stacks; near-degenerate pairs are reported as subspaces.

### 4. Approximate 2D confinement (lamina-reduce)

Separable reduction of a cross-section given as **lateral regions, each a vertical stack with a width**:

1. solve each region's vertical stack (milestones 1 and 2), giving complex effective indices and vertical profiles;
2. form the lateral stack of those indices and solve it, giving 2D-like modes, their effective indices and
   separable field approximations;
3. return an error indicator for each mode. Start with the first-order correction from the coupling the
   separation neglects, computed from the vertical profiles. Return both the uncorrected and corrected indices.

Derivatives with respect to widths and thicknesses come from milestone 2 through both stages. Leaky vertical
stages (substrate leakage) carry their complex indices into the lateral stage.

*Accept when:* on reference strip, rib and slot cross-sections, the reduced n_eff and confinement factors fall
within stated error bands of a full-vector 2D solver; the error indicator ranks the cases correctly (larger
indicated error where the true error is larger); completeness from both stages is carried into the result.

### 5. Local-mode propagation (lamina-reduce)

Propagation through structures that vary slowly or step along the propagation axis, from milestones 3 and 4:

- **multimode sections** (generalising the existing slab propagator): modal amplitudes advanced by their β along a
  section, with excitation from a supplied input field by projection;
- **slowly varying sections** (tapers): coupled local-mode equations with the coefficients of milestone 3, and the
  adiabaticity parameter of each mode pair along the profile;
- **abrupt junctions:** mode matching between neighbouring bases, with the power not captured by the basis
  reported.

*Accept when:* MMI self-imaging lengths match the beat-length estimate L_π = π/(β₀ − β₁) on reduced MMIs; a linear
taper's conversion efficiency converges in retained modes and in section count; power balance holds across
junctions to the reported residual.

### 6. Device physics models (lamina-design)

Models that turn milestones 2–5 into design quantities, each with its sensitivity:

- **tapers:** minimum adiabatic length for a target loss, and profiles that equalise the adiabaticity parameter;
- **MMIs:** imaging lengths and their tolerance to width, imaging quality from the deviation of the β ladder from
  the ideal quadratic spacing, and the input coupling into the multimode basis;
- **directional couplers:** coupling length from the supermode split, and its sensitivity to gap and thickness.

These are fast first estimates and design rules. A consumer verifies a final design with a full solver.

*Accept when:* each model's predictions agree with milestone 5 propagation on its validity range, and its stated
range is checked by a case just outside it.

## What a consumer gets

| Need | From |
| --- | --- |
| Exact slab effective indices, with completeness, for effective-index rules in 2.5D simulation | 1 |
| Substrate leakage, absorption, metal layers and dispersion, which plane-wave mode solvers can't model | 1, existing QNM work |
| Process sensitivity: dn_eff and dΓ per nanometre of layer thickness; analytic derivatives for tolerance analysis | 2 |
| Fast approximate 2D confinement with an error indicator, as a first stage before a full solver | 4 |
| Taper, MMI and coupler first designs with sensitivities: physics-informed starting points for optimisation | 5, 6 |

## Out of scope, permanently

- Full-vector 2D or 3D mode solving. Milestone 4 approximates it and says so; it doesn't replace it.
- Geometry, layout and process descriptions, in any form.
- Optimisation drivers and objectives beyond the device models of milestone 6.
- Material property data. lamina-materials owns material models; their parameters come from the consumer.
