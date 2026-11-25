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
form and hash are unchanged. Profiles name the region of every sample: `layer`, `gap`, `pml` or `exterior`.

**Defaults.** The 1e-6 k0 gap and the 1e-8 threshold are kept. On the walled slabs of the tests (TE and TM, PEC and
PMC), the smallest gap between distinct modes is 8.3e-2 k0 and the largest distance from a mode to its layered-box
root 1.5e-12 k0, six decades on either side of the gap; two cores 2.15 cm apart, whose supermodes split by 4.9e-7 k0,
form one group with an unchanged bi-orthogonal pair. The smallest self-pairing ratio of a normalised fixture mode is
0.27, and a box mode at β = 6.3e-4 k0 has ratio 6.3e-4: the ratio follows β/k0 near cutoff, so the threshold flags
modes within about 1e-8 k0 of the origin, where a forward window's mirror square reports `cutoff` first.

**PMLs.** A side may be closed by a wall-backed PML (0009): the exterior material over
ζ ∈ [0, d] with stretch s(ζ) = 1 + c (ζ/d)^m, ζ̃(ζ) = ∫₀^ζ s and complex thickness τ = d (1 + c/(m+1)). In real
coordinates the stretch is the uniaxial medium εΛ, μΛ with Λ = diag(s, s, 1/s) in (x, y, z): complex symmetric,
hence reciprocal, and the backing wall removes the boundary term, so modes of a stack closed by walls and PMLs are
bi-orthogonal under ⟨·,·⟩ as between walls. Inside a PML the sampled fields are the exterior solution continued to
ζ̃; the equivalent-medium fields have the same tangential components and normal components Ê_z = s E_z,
Ĥ_z = s H_z. The densities there are

| pairing | density in real ζ |
| --- | --- |
| unconjugated | (Ê_a × Ĥ_b) · x = s (E_a × H_b) · x, so ∫ … dζ = ∫ … dζ̃ along the contour |
| conjugated | (Ê_a × Ĥ_b*) · x = s* E_y H_z* − s E_z H_y* |

The unconjugated integrand is analytic in ζ̃ (one homogeneous material), so a PML's contribution depends on τ alone:
two profiles of equal τ (orders 0, 2 and 4 in the tests) give the same modes and the same unconjugated contributions
to 1e-10, while their conjugated contributions differ by up to 18 times. The contribution agrees with the closed form
∫₀^τ e^{iqζ̃} dζ̃ of the fitted exponentials to 4e-11, and both pairings agree with Gauss–Legendre in every region
to 7e-11. PML intervals are split until each piece carries no more transverse phase |κ τ| than the most oscillatory
other interval. Two stacks pair when each side's PMLs agree in registered start, thickness, profile and backing;
the PML and gap materials may differ. A PML against a wall, or differing PMLs, is `Terminations`, naming the side.

**PML scale.** A PML side normalises the characteristic by exp(iκ_ref τ), leaving it of order exp(Σ [−Im(κ_ref τ) +
|Im(κ τ)|]); near the rotated continuum that is exp(Σ −Im(κ_ref τ)), e^−33 for PMLs 1.75 wavelengths thick on the
tests' slab. The basis search scales its near-zero threshold by that least scale, and root correction its residual
limit by the scale at the seed, so both mean what they mean between walls: unscaled, argloc refused 352 cells of the
tests' windows as boundary singularities, all then recovered by subdivision. A window whose scale leaves e^±300 is
refused (`Scale`, naming the side and the log-magnitude); a cell that does so during subdivision is unresolved with
cause `scale`.

**PML bases.** A basis counts every mode in its window: the modes converging to the open stack's guided and exposed
leaky modes (Im(κ (g + τ)) > 0) as the PML thickens and the modes near the rotated continuum arg κ ≈ −arg(g + τ).
For symmetric slabs each guided and exposed leaky ladder value lies within 10 k0 |e^{2iκ(g+τ)}| of exactly one basis
mode, with |Δβ| / (k0 |e^{2iκ(g+τ)}|) constant along the thickness sequence (0.07 to 5 in TE, 0.06 to 1 in TM), and
every other mode stays at least 0.07 k0 from every ladder value. Rotated-continuum modes high in Im β grow strongly
in thick PMLs; their self-pairing ratio falls below the 1e-8 threshold there, so they are reported ill-conditioned
and the window, the caller's choice, decides whether the basis is complete. Profiles name `pml` samples and carry
`stretched_z_m` and `stretch`, with the fixed statement of what the fields are there.

### Open exteriors

**Chart and window.** A stack with one or two open sides is searched on the conformal chart of its open exteriors:
with two, β² = k0² [s + e^{2z} + r² e^{−2z}], κ_L = i k0 (e^z − r e^{−z}), κ_R = i k0 (e^z + r e^{−z}),
r = (χ_L − χ_R)/4, s = (χ_L + χ_R)/2; with one (the other a wall or a PML), r = 0 and s = χ of the open side, so
κ = i k0 e^z. An open side's exterior field is exp(iκζ) at outward distance ζ: guided modes lie on the proper sheet,
Im κ > 0, leaky ones on the improper sheet, Im κ < 0. z ↦ (β², κ) uniformises the Riemann surface of the exterior
wavevectors with period 2πi, and the characteristic composed with it is entire in z, so the winding around a chart
rectangle counts zeros, with no poles and no mirror double count. The window is therefore `ModalWindow::Chart`, a
rectangle in z less than one period tall (`WindowError::Period` otherwise); a wavenumber window on such a stack is
refused (`OpenSide`), and so is a chart window holding a zero of β², a branch point of β
(`CoordinateBranchPoint`). β is continued from its principal root at the window's centre, and every cell's chart
continues it from there. Each mode records its sheet per open side. Open exteriors must be isotropic, so that
their channel is unscaled and χ = εμ in either polarisation; an anisotropic open exterior is refused
(`AnisotropicExterior`, naming the region); diagonal layers and closed sides are handled as before.

**Covering a β rectangle.** A β rectangle on the improper sheet near the real axis below a light line meets the
cut exactly where leaky modes live, while a chart rectangle reaches them continuously. `ModalWindow::covering` maps
a β rectangle on stated sheets (one per open side) to the bounding box of its boundary's image in z, padded by half
the largest sample step, and reports the extra area of the covering's β image (by the shoelace formula of its
boundary) over the rectangle's. A rectangle that meets a stated sheet's cut, κ² ∈ [0, ∞), is refused naming the
side; along each edge Im κ² is linear, so the test is exact. Equal exteriors (r = 0) need equal sheets.

**Branch points.** z is a local coordinate at the branch points too, so a zero there counts once; but the exterior
field is flat and the self-pairing tail i/(2κ) diverges. A root with |κ| ≤ 1e-4 k0 on an open side is unresolved
with cause `BranchPoint`, naming the side; with one open side, or equal exteriors, κ = 0 lies at Re z → −∞.

**Scope.** With an open side the transverse operator has a continuous spectrum, bounded non-normalisable radiation
solutions for every real κ ≥ 0, and a field's expansion is a sum over guided modes plus an integral over that
continuum. Deforming the integral into the improper sheet collects leaky poles but always leaves an integral along
the deformed cut. A basis of an open stack therefore has scope `DiscreteWithoutContinuum`: its guided and leaky
modes never span the fields, and "complete" means only that no discrete mode in the window is missing.

**Tails.** Beyond the partition, with each mode's κ on its own sheet and v₀ the density at the outermost sample,

| pairing | open tail |
| --- | --- |
| unconjugated | i v₀/(κ_a + κ_b), by analytic continuation, for guided and leaky modes alike |
| conjugated | i v₀/(κ_a − κ_b*) when Im κ_a + Im κ_b > 0; otherwise absent, `Unbounded`, naming the side |

For guided modes, κ = iα, both are the decaying tail v₀/(α_a + α_b): the conjugated pairings of open-slab guided
modes equal the guided power overlaps to 3e-11 W/m. For leaky modes the unconjugated tail is the integral along a
rotated ray on which both fields decay, the exterior regularisation of pole normalisation; it is what makes the
Lorentz boundary terms vanish, so guided and leaky modes of one open stack are bi-orthogonal under ⟨·,·⟩. A
divergent conjugated tail is never replaced by a regularised number. |κ_a + κ_b| (or |κ_a − κ_b*|) below 1e-8 k0
is singular (`TailSingular`). The field scale of the ill-conditioning test adds |F(edge)|²/(4 Z0 |κ|) per open side,
finite for leaky modes. Two stacks pair when a side is open in both; their exterior materials may differ; open
against closed is `Terminations`.

**Checks.** On a symmetric slab (core ε = 4, half-width 0.35 λ, cladding 2.25) in the window
Re z ∈ [−3, 0.35], Im z ∈ [−2.9, 0.4], TE and TM, the basis holds two guided modes and one leaky one, equal to the
even and odd ladders solved by Newton in z on their sheets to |Δz| ≤ 7e-15, with counts equal to an independent
argument-principle count; a slab on a PEC or PMC wall holds the odd or even modes of the mirror-doubled slab,
leaky ones included; the unconjugated Gram is the identity to 1e-11; positive-power guided modes equal the
signed-unit-power profiles to 2e-11; leaky conjugated self-products are absent. A slab with one open side and a
PML on the other counts and pairs likewise (6 TE, 7 TM modes). The covering of a β rectangle round the TE leaky mode
holds it on the improper sheet, and a rectangle across the real axis below the light line is refused.

### Projection and local coupling

**Projection.** A supplied field F is expanded in a basis's normalised modes by solving G a = b with
G_mn = ⟨f_m, f_n⟩ and b_m = ⟨f_m, F⟩, both unconjugated and both on one grid at one refinement level: the basis's
partition with the support ends and the field's discontinuities, PML intervals split as for pairings, dz̃ in a PML
and continued tails on open sides. G is the identity in theory; solving through it keeps its quadrature defect and
the accuracy of Löwdin groups out of the coefficients. Below full rank under `rank_rtol` there are no coefficients.
Inside a PML F is the continued field, as profiles give it (a field computed in the equivalent medium has its
normal components divided by s first). A support beyond a wall is refused. On an open side whose support edge lies
in the exterior, the field beyond it may be stated as exterior waves F_j exp(iκ_j ζ), each adding the continued
tail i ⟨⟨f_m(edge), F_j⟩⟩/(κ_m + κ_j) to b_m; without them F vanishes beyond its support. A field built from leaky
modes needs them: dropping the tails changed the coefficients of the tests' open-slab combination by more than
1e-3.

**Residual.** The result reports the field scale of F and of R = F − Σ a_m f_m over the bounded partition, their
ratio √(N(R)/N(F)), the equation residual |G a − b| and `residual_includes_continuum`. Between walls and PMLs the
residual is the content of F in modes outside the window plus numerical error. With an open side it also holds the
part of F on the radiation continuum, which no discrete basis represents: it is physical, not numerical. The
coefficients are still the discrete part of the expansion (the residues of the poles in the window), and
⟨f_m, R⟩ = 0 for every mode. The discrete part can be larger than F on a wide window: leaky terms grow outward and
only the continuum cancels them. A Gaussian of width 0.3 λ on the open test slab, over a support reaching 0.75 λ
into the cladding, leaves a relative residual of 5.0.

**Local coupling.** Lorentz reciprocity between the mirrored base mode M f_m and the mode f_n′ at p + δp, whose
boundary terms vanish at walls, behind PMLs and on open sides by continuation, gives

| quantity | value |
| --- | --- |
| K_mn | (ω ε0/4) [∫ Ẽ_m · ∂ε/∂p · E_n dz̃ + Σ_k v_k ((ε⁻ − ε⁺)_xx Ẽ_x E_x + (ε⁻ − ε⁺)_yy E_y E_y − (1/ε⁻_zz − 1/ε⁺_zz)(ε⁻_zz)² E_z E_z)] |
| dβ_n/dp | K_nn |
| C_mn = ⟨f_m, ∂f_n/∂p⟩ | K_mn / (β_n − β_m), row m receiving, column n differentiated |

with Ẽ = (−E_x, E_y, E_z) the electric part of M f, left-side limits at interface k moving at v_k, s Ẽ · δε E in a
PML (dz̃) and i v₀/(κ_m + κ_n) on an open side whose exterior rate is non-zero. For a lossless guided mode under the
guided phase rule E_x is imaginary and E_y, E_z real, so Ẽ_m · E_n = E_m* · E_n and the guided coupling of 0008 is
the special case; for evanescent, complex, PML and leaky modes only Ẽ is reciprocal. K is complex symmetric. The
self coefficient ½ ∂⟨f_n, f_n⟩ vanishes and is absent, as are coefficients between modes within the gap (default:
the basis's degeneracy gap), reported as groups whose block of K holds their first-order splitting. A direction
varying the wavelength or μ, or moving an interface across a jump in μ, is refused, as is one moving the first
interface of a stack with a wall or PML (walls are fixed in the stack frame) or an interface lying on a wall or PML
start.

**Checks.** Fields built from known coefficients (|c| ~ 1) are recovered within 1.1e-13 between walls (TE and TM, PEC and
PMC, 11–12 modes with real, imaginary and complex β), through PMLs and on the open slab with its leaky mode and
continuation, with Gram defects 5e-12 to 6e-11. The diagonal of K equals the implicit slope of the characteristic
along the core thickness to 6e-11 for every mode (walls, PMLs, open, and a lossy Drude core with complex modes),
the guided sensitivity along the core permittivity to 4e-11, and central differences of re-solved β along the core
and cladding permittivities to 4e-9. Central differences of `basis_pairing_with` between the basis and bases of
neighbouring stacks agree with C, relative to the largest coefficient, to 2e-8 along the permittivities between
walls and PMLs and 2e-6 on the open slab (h = 1e-4), and to 7e-6 (TE) and 5e-5 (TM) along the thickness at
h = 1e-4 t, falling with h as h² (TE) and as h (TM, where E_z jumps at the moving interface and the pairing is C¹
with a curvature jump at the base thickness). Two identical cores 2.15 λ apart
form one group whose block along one core's permittivity is ½ dβ/dε of a single core in every entry.

**Records.** `lamina.basis_projection` and `lamina.basis_local_coupling` are new kinds at schema version 1. They
carry the `definitions` above and a fixed kind `definition` (as in 0008), both checked on parsing; matrices
serialise rows first as `_real` and `_imag` arrays, absent coefficients as `null`.

## Consequences

Bases are formed for stacks closed by walls or PMLs on both sides in a β window, and for stacks with open sides in
a chart window, without the radiation continuum. Supplied fields are expanded in a basis, with a residual whose
meaning the result states, and local coupling coefficients are formed between its modes. Nothing is differentiated
with respect to a wall position or a PML's thickness, profile or position, and magnetic rates are not coupled.
