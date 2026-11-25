# 0011: Modal bases in a window and modal pairing

Status: accepted

## Context

A stack closed by walls on both sides (0009) is a regular problem on a bounded interval: its spectrum is discrete and
its modes, guided, radiation-like, evanescent and complex, span its fields. A modal basis states which of them it
holds: every mode in a window of the complex β plane, with evidence that none is missing. Evanescent and complex
modes carry no signed power, so the guided normalisation (unit signed power, 0008) does not apply; a basis needs a
normalisation, sign, order and treatment of near-degenerate modes that hold for every mode, and a pairing under which
its modes are bi-orthogonal. This note fixes those choices for lamina-mode and the lamina-py records.

## Decision

**Definitions.** Every `lamina.modal_basis`, `lamina.modal_profile` and `lamina.basis_pairing` record carries one
fixed `definitions` object, and parsing rejects any other text:

| key | text |
| --- | --- |
| `unconjugated_pairing` | (1/4) ∫ (E1 × H2 + E2 × H1) · x dz̃; dz̃ = s dz in a PML; open tails by analytic continuation |
| `conjugated_pairing` | (1/4) ∫ (E1 × H2* + E2* × H1) · x dz, real coordinate, PML equivalent-medium fields |
| `normalisation` | unit unconjugated self-pairing, 1 W/m |
| `sign_rule` | Re(Ey + Z0 Hy) > 0 at the first interface; at a wall on it Re(Ey + Z0 Hy + i(Z0 Hx − Ex)) > 0; ties by Im |
| `ordering` | decreasing Re β², then decreasing Im β², then decreasing Re β |
| `degenerate_basis` | unconjugated Löwdin within groups, principal G^(-1/2), then the sign rule |

The clauses on PMLs and open tails state how the same definitions extend to those terminations; between walls
dz̃ = dz and there are no tails.

**Pairings.** ⟨a, b⟩ is symmetric and in W/m per metre of invariant width. With M the mirror x → −x,
(Ex, Ey, Ez, Hx, Hy, Hz) → (−Ex, Ey, Ez, Hx, −Hy, −Hz) at −β, the antisymmetric flux of 0008 is
R(a, b) = −4 ⟨a, M b⟩ and ⟨Ma, Mb⟩ = −⟨a, b⟩. For a lossless guided mode with real fields, ⟨f, f⟩ is its signed
power, and its backward partner σ M f (σ = +1 in TE, −1 in TM) gives R = −4 W/m in TE and +4 W/m in TM. Lorentz
reciprocity between two modes of one stack leaves boundary terms (E_a × H_b − E_b × H_a) · z at the walls, which
vanish at PEC and PMC walls alike, so ⟨a, b⟩ = 0 for β_a ≠ β_b with loss, gain or dispersion at the fixed k0. The
conjugated product is orthogonal only for a self-adjoint problem; it is power bookkeeping and never normalises or
orders. Both pairings integrate over the union of the two stacks' interfaces and wall planes in the registered frame,
up to the walls. Two stacks pair only when each side's walls coincide in kind and registered position; materials,
gaps included, may differ.

**Window.** A wavenumber window is a β rectangle (rad/m in records, cm⁻¹ in lamina-mode) on the direct chart:
`{"kind": "wavenumber", "re_rad_per_m": [a, b], "im_rad_per_m": [c, d]}`. `BetaWindow.forward(β_max, decay_max, m)`
spans Re β ∈ [−m, β_max] and Im β ∈ [−m, decay_max], so real and imaginary roots of a lossless stack lie strictly
inside and the window meets its mirror image only in the square |Re β|, |Im β| ≤ m.

**Counts and evidence.** `expected` is the number of zeros inside the window from the change of arg D around its
boundary, in steps over which the phase turns by less than π/4 and agrees with the integrated logarithmic
derivative; it is formed before the search. `found` is the number of distinct accepted modes. Every region that does
not end as accepted roots matching its count is unresolved, with bounds, β image, count and one cause:
`boundary_root` (the boundary passes within 1e-6 of the window size of a zero), `non_integer_count`, `budget`,
`refinement`, `multiplicity` (roots together at 1e-9 k0, or a near-degenerate group with a singular Gram) and
`cutoff` (a root in the mirror square, where a mode and its mirror cannot be told apart). The status is `complete`
when expected equals found, nothing is unresolved and every mode is normalised; `count_failed` when the boundary
count fails; `incomplete` otherwise. An incomplete basis is a result. The scope of a walled basis is `discrete`.

**Normalisation and sign.** Fields are recovered at a canonical unit seed, whatever the caller's amplitude, and
f = σ u / √⟨u, u⟩ with the principal root. Consequences: a lossless guided mode of positive power has the guided
profile; one of negative power is ±i times it; a lossless evanescent mode carries a common phase e^{∓iπ/4}; a
backward mode in a window is ±i M f. When |⟨u, u⟩| is below `self_pairing_rtol` of the field scale
(1/4Z0) ∫ (|E|² + |Z0 H|²) dz over the bounded regions, the mode is kept as `ill_conditioned` with its ratio and no
fields, and the basis is not complete. σ makes the reference ℓ = Ey + Z0 Hy at the first interface positive in its
real part, or in its imaginary part when the real part is zero to round-off (64 ε |ℓ|); when a wall lies on the first
interface, known from the input, ℓ = Ey + Z0 Hy + i(Z0 Hx − Ex), real for lossless modes and never structurally zero.
Each mode records `sign_margin` = |Re ℓ| / max |Ey + Z0 Hy|, flagged `sign_ambiguous` below 1e-6.

**Order and groups.** Modes are sorted by decreasing Re β², then Im β², then Re β, with exact comparisons: propagating
modes first by decreasing β, then lossless evanescent modes by increasing decay; a mirror pair splits forward first.
Modes within `degeneracy_gap_rad_per_m` (default 1e-6 k0, as in 0008) of each other are linked, and groups are the
connected components. A group's members become u G^(−1/2) with G their unconjugated Gram in key order and the
principal root, then the sign rule; members already bi-orthogonal are unchanged. The subspace is continuous in the
parameters; individual members are not promised to be.

**Determinism.** One platform gives the same modes, order, signs, groups and normalised fields bit for bit under
repeated solves, a reversed traversal of the cells, a rescaled recovery seed and equal inputs that canonicalise
identically. The gap is stored resolved in the input, so stating the default gives the same record and hash.
Rescaling every length and the wavelength by one factor holds to tolerance (n_eff to 1e-15 and fields to 1e-14 at a
factor 3.7e-4 in the tests), as residual and correction limits scale with k0.

**Basis pairing records.** Both matrices are rows first, row i the first basis's mode `rows[i]` and column j the
second's mode `columns[j]`, over each basis's normalised modes. A conjugated entry that does not exist is `null` with
its cause at the same place in `conjugated_absent`; between walls every entry exists. The record carries the singular
values, the condition number of every leading block, each basis's own max |G − I| on the same grid, the sample count
and the largest change of an entry at the last refinement. Every entry comes from one grid at one refinement level,
refined until each entry changes by less than 1e-9 of max(|entry|, 1 W/m).

**Additive schema.** `lamina.modal_basis_input`, `lamina.modal_basis`, `lamina.modal_profile` and
`lamina.basis_pairing` are new kinds at schema version 1 (0005). `CONVENTIONS` and every existing kind, its canonical
form and hash are unchanged. Profiles name the region of every sample: `layer`, `gap` or `exterior`.

**Defaults.** The 1e-6 k0 gap and the 1e-8 threshold are kept. On the walled slabs of the tests (TE and TM, PEC and
PMC), the smallest gap between distinct modes is 8.3e-2 k0 and the largest distance from a mode to its layered-box
root 1.5e-12 k0, six decades on either side of the gap; two cores 2.15 cm apart, whose supermodes split by 4.9e-7 k0,
form one group with an unchanged bi-orthogonal pair. The smallest self-pairing ratio of a normalised fixture mode is
0.27, and a box mode at β = 6.3e-4 k0 has ratio 6.3e-4: the ratio follows β/k0 near cutoff, so the threshold flags
modes within about 1e-8 k0 of the origin, where a forward window's mirror square reports `cutoff` first.

## Consequences

Bases are formed for stacks closed by walls on both sides; a wavenumber window on a stack with an open side is
refused. Fields are not projected onto a modal basis here, local coupling is not formed on one, and nothing is
differentiated with respect to a wall position.
