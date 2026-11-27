# Hybrid guided modes

A region whose optic axis lies off the stack's principal axes has ε or μ with
off-diagonal entries in the stack frame (x along propagation, y invariant, z
normal). With no `xy`, `yx`, `yz` or `zy` entry the region is *separable*: TE
`(E_y, H_x, H_z)` and TM `(H_y, E_x, E_z)` are still invariant subspaces. Any
other region couples them, and its guided modes are *hybrid*, carrying all six
field components. They are solved through the coupled channel of the 4×4
(Berreman) backends, Transfer4 and Scatter4, with the tangential state
`ψ = (E_x, Z₀H_y, E_y, −Z₀H_x)`.

## Families and labels

A `GuidedWindow` names a `ModeFamily`: `Polarised(TE)`, `Polarised(TM)` or
`All`. TE and TM are accepted on separable stacks only; a region that couples
them fails with `EnumerationError::FamilyNotInvariant { region }`. `All` solves
the coupled channel. Each mode carries a `ModeLabel`: on a separable stack TE or
TM, by the family that holds its field at the left interface (the other
family's relative part is recorded in `HybridCountEvidence::off_family`), and
`Hybrid` otherwise. TE and TM of stacks of isotropic and diagonal regions keep
the oscillation count; every other request is counted as below.
`enumerate_guided_modes` uses Scatter4 for them, `enumerate_guided_modes_with`
any backend.

## Eligibility and window

Every region must have Hermitian, positive-definite ε and μ at the operating
wavelength (`EnumerationError::Tensor`, naming the region and the cause).
Then no guided mode lies above

```text
β_max = k0 max over regions √(λmax(ε) λmax(μ)),
```

and none at or below an open exterior's light line, the largest real ξ = β/k0
at which its normal wavenumbers stop being non-real. Each open exterior's
normal wavenumbers n (units of k0) are the roots of

```text
p(n) = det [[K, −μ], [ε, K]],   K v = (ξ, 0, n) × v.
```

A separable exterior factorises it into one quadratic per family,
`a n² − b ξ n + c ξ² = v`, with `A` the inverse of the xz block of ε (TM) or μ
(TE), `a = A_xx`, `b = A_xz + A_zx`, `c = A_zz` and `v` = μ_yy (TM) or ε_yy
(TE); its light line is ξ² = 4av/(4ac − b²), ε_zz μ_yy for TM with a symmetric
block. Any other exterior's roots are found numerically. The outgoing pair of
the right exterior is the two roots with Im n > 0; on the left, the two with
Im n < 0, supplied outward as −n.

## The count

The eligible interval is covered by rectangles of the β chart, mirrored for
backward modes. Above a light line each rectangle is `[x, x + (x − line)]` in
width and `±0.45 (x − line)` in height, so it keeps the light-line branch point
at a fixed fraction of its size; between two walls, eight equal cells. On each
rectangle the zeros of the backend's coupled characteristic F are counted by
the argument principle and localised with argloc, and each counted root is
corrected in β. The count is accepted only where F is holomorphic and pole-free
on the rectangle, which is checked on samples of its boundary and midline and
recorded (`GuidedRegionEvidence`):

- each exterior keeps two roots strictly in each half plane (its outgoing pair
  is continuous over the rectangle);
- no outgoing root approaches an incoming one (no exterior branch point);
- the 4×4 backends' exterior reference basis stays conditioned (no spurious
  zero from a collapsed gauge).

A rectangle that fails is refused with its cause, and the count is unknown.
Scatter4 continues each layer's forward/backward split from the rectangle's
centre; a rectangle over which it cannot (a layer branch point) is counted and
corrected with Transfer4, recorded per rectangle. A root on an end of the
interval moves that end inside by at most the cutoff margin; one on an inner
side moves the side. A counted root that is not real is reported as
`NonRealRoot`, and a cell still counting two roots at the resolution limit
(10⁻⁹ k0) as `NearDegenerate { count, cell }`: a two-dimensional subspace, with
no labels assigned by order within it. On a separable stack the coupled count
is checked against the TE and TM counts of the same interval. The basis is
complete when the count is accepted, every counted root is accepted as real and
no cell is unresolved.

## Fields, power and phase

Fields come from the backend's modal null vector, with all six components and
only the outgoing waves in each exterior. They are scaled to

```text
P = ½ Re ∫ (E_y H_z* − E_z H_y*) dz = ±1 W per metre of invariant width,
```

with the sign of the mode's power flow. Power, pairings, overlaps and the
group index's energy are closed-form integrals of the mode (lamina-core's
`directional_forms`): a block matrix exponential per layer and wall gap, a
Sylvester equation per open exterior, exact at degenerate layer eigenvalues
and at cutoff. The phase rule makes `E_y + Z₀H_y` at the left interface real
and positive (for TE `H_y = 0`, for TM `E_y = 0`, so TE and TM modes keep their
scalar normalisation). Where that reference falls below 10⁻⁶ of the tangential
field there, `E_x + Z₀H_x` is used instead, and the reference is recorded. The
unconjugated norm `½ ∫ (E × H)·x̂ dz` is recorded; for hybrid modes it is
generally not real under this phase rule.

## Reciprocity, rates and tracking

For symmetric ε and μ the forward and backward ladders coincide, and the
unconjugated pairing `½ ∫ (E_m⁺ × H_n⁻ − E_n⁻ × H_m⁺)·x̂ dz` between forward and
backward modes is diagonal. The z-mirror of a stack (layers reversed, ε_xz,
ε_yz and their μ counterparts negated) has the same ladder.

β rates are `dβ/dp = −(∂F/∂p)/(∂F/∂β)` along a `HybridPerturbation`: layer
thicknesses, k0 (with every region's dispersion) and full stack-frame tensor
rates, which include rotations, `dε/dθ = Gε − εG` for the generator G of
rotations about an axis. The exterior pairs are continued along the direction.
The group index `dβ/dk0` is checked against

```text
n_g = ½ ∫ (E† ε_d E + η† μ_d η) dz / ∫ Re(E_y η_z* − E_z η_y*) dz,
ε_d = Herm(ε + k0 ∂ε/∂k0),  η = Z₀H.
```

Along a parameter path, bases are followed by their power overlaps; two modes
closer than a gap are followed together as a subspace with its 2×2 overlap
block, as at an avoided crossing that opens when an optic axis turns away from
a principal axis.

## Complex-plane searches

Leaky and quasi-normal searches accept the coupled channel with full-tensor
layers and isotropic exteriors, whose two outgoing waves share the branch of
the existing charts, `(κ, κ)`. Exteriors that are not isotropic are refused,
naming the region, off the guided sheet. The 4×4 exterior projector cannot be
evaluated at an exterior branch point κ = 0, so search windows must keep it off
their evaluation points.
