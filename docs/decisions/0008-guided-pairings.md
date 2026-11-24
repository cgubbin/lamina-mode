# 0008: Pairings of guided modes in the bindings

Status: accepted

## Context

lamina-mode pairs guided modes with each other and with supplied fields: the registered power overlap of two modes
and its derivative, the unconjugated reciprocity flux, the local perturbation matrix and coupling coefficients of a
basis, and the projection of a supplied field onto a basis. The bindings present them in SI with the evidence and
record contract of 0003–0006. Pairings bring choices those notes do not cover: which pairing a number is, how two
stacks share a frame, how matrices and absent entries serialise, and how a Python callable supplies a field.

## Decision

**Functions and methods.** Pairings of two modes are free functions over `(first, first_mode, second, second_mode)`,
because the modes may come from different bases (different stacks at one wavelength): `power_overlap`,
`power_overlap_sensitivity` and `reciprocity`. Results of one basis are `GuidedBasis` methods: `local_coupling` and
`project_field`. Deserialised bases work in all of them through the re-solve-and-compare path.

**Pairing definitions live in the record.** Each new record carries a fixed `definition` for its schema, and parsing
rejects any other value, as it does for conventions:

| Schema | `definition` (abridged) |
| --- | --- |
| `lamina.guided_power_overlap` | (1/4) ∫ (E1 × H2* + E2* × H1) · x dz, linear in the first mode |
| `lamina.guided_overlap_sensitivity` | d/dp of the power overlap, each mode along its own parameter direction |
| `lamina.guided_reciprocity` | ∫ (E1 × H2 − E2 × H1) · x dz, unconjugated |
| `lamina.guided_local_coupling` | K_ij = (ω ε0 / 4) [∫ (∂ε/∂p) E_i* · E_j dz + interface terms]; C_ij = s K_ij / (β_j − β_i) |
| `lamina.guided_projection` | G c = b, with G_ij the overlap of mode j with mode i and b_i that of the field with mode i |

Adding the definitions to `CONVENTIONS` instead would change the canonical form and hash of every existing record.
The five kinds are version 1 records of their own schema (0005); existing records, their hashes and `CONVENTIONS`
are unchanged.

**Records name every input.** Two-mode records carry `first_input_hash`, `second_input_hash` and both mode indices;
one-basis records carry `input_hash`. `assumption` is the dispersive declaration when either stack has a Sellmeier
material, else the constant one.

**Registration.** `origins_m = (z1, z2)` places each basis's first interface (its z = 0) in a common frame; the
default `(0, 0)` aligns them. The record stores the origins and the integration partition (the union of both
stacks' interfaces) in that frame. For a projection, `origin_m` places the basis's first interface in the supplied
field's coordinate.

**Overlap derivatives take one parameter per side.** `first_parameter` and `second_parameter` are each a
`Parameter` or `None` (that stack held fixed); at least one is set. The rate is per one shared unit, so two set
parameters must have the same `parameter_unit`, and the wavelength varies on both sides or on neither. Otherwise
`InvalidInput`. Each stack's first interface moves with its own parameter: `(None, translation())` is a relative
translation, `(translation(), translation())` a common one. Length parameters are passed to lamina-mode per cm,
the Sellmeier pole wavelength included, so a geometric and a material length can be paired.

**Derivative failures are results.** lamina-mode returns the base overlap when the derivative cannot be formed,
for example where coincident interfaces of the two stacks move at different rates. The record then has
`power_rate_w_per_m = null` and `derivative_failure` naming the cause (for separating interfaces, their position
in metres and both velocities); the base overlap and its evidence are always present.

**Matrices.** Matrices serialise rows first as nested `_real` and `_imag` arrays (`perturbation_matrix_real`,
`gram_real`, …), vectors as flat ones. Absent coupling coefficients (the diagonal, and pairs within an unresolved
group) are `null`; in Python they are `nan + nanj`, with a boolean `coefficient_resolved` mask beside them.
Projection coefficients, condition number and residual are `null` below full rank.

**Coupling units and gap.** The perturbation matrix is in rad/m per SI parameter unit
(`perturbation_matrix_rad_per_m`); on a forward basis its diagonal is each mode's dβ/dp. Coefficients are per
parameter unit, row the receiving mode and column the differentiated mode. Modes whose propagation constants differ
by no more than `gap_rad_per_m` form an unresolved group. Its default is `1e-6 · k0`, scale free like the residual
limits of 0003; the record stores the resolved value. An explicit value is used as given.

**Coupling eligibility.** The bindings name the cases lamina-mode refuses: a basis that is not complete, a region
with relative permeability other than one, a region whose permittivity is not positive at the operating
wavelength, and the wavelength or a permeability as the parameter. Each raises `InvalidInput` with its reason; any
remaining refusal from lamina-mode is `InvalidInput` too.

**Supplied fields are a per-point callable.** `field(z_m, side) -> (E, H)`, with E in V/m and H in A/m, each three
complex components (x, y, z). `side` is `"left"` (the limit from −z), `"right"` (the limit from +z) or
`"interior"`. The field vanishes outside `support_m`; `interfaces_m` lists the points strictly inside the support
where it may be discontinuous. lamina-mode samples one point at a time inside an adaptive composite quadrature per
mode, so the bindings memoise samples by (position, side) within one call, and pass the support edges and
interfaces back exactly as given. The callable runs with the GIL held. The first exception it raises stops the
projection and propagates unchanged; a wrong shape or a non-finite value raises `InvalidInput`. Each interval is
integrated with uniform panels, so a support far wider than the field's decay length costs samples and can exceed
the sampling budget (`IntegrationFailed`). The record stores `field_sample_count`, the number of distinct samples.

**Projection settings and identity.** Every mode-field recovery in a projection uses the bindings' k0-relative
field-recovery settings (0003); `project_guided_field` takes them as an argument. `field_id` (optional) names the
supplied field in the record, so it enters the hash; without it the record stores `null`, marking the field as
unidentified. The callable itself is never hashed.

**Rank deficiency is a result.** `rank` counts the singular values of the Gram matrix above `rank_rtol` (default
1e-8) times the largest; the singular values are always present.

**Errors.** Registration failures, different wavelengths, bad supports or interfaces, out-of-range mode indices,
mismatched parameters and ineligible coupling raise `InvalidInput`; field recovery raises `FieldsFailed`; a pairing
integral over its accuracy or sampling budget raises `IntegrationFailed`, a `LaminaError`.

## Reciprocity of forward and backward modes

Components are [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz] with phasors exp(+iβx − iωt), and a mode's signed power is
P = (1/2) ∫ (Ey Hz − Ez Hy) dz. A lossless backward mode has the transverse profile equation of its forward partner,
so after signed unit-power normalisation and the phase rule (left-interface Ey or Z0 Hy real and positive) its
primary field equals the forward one and only the β-odd component flips:

- TE: Z0 Hz = β Ey / k0, so Hz flips and flux(forward, backward) = −2 ∫ Ey Hz dz = −4 P_f = −4 W/m.
- TM: Ez = −β Z0 Hy / (k0 ε), so Ez flips and flux(forward, backward) = −2 ∫ Ez Hy dz = +4 P_f = +4 W/m.

Both bases are ordered by decreasing |β|, so index n of one is the partner of index n of the other.

The flux is unconjugated, so a residual phase in a recovered field enters it twice rather than cancelling as in
the power overlap. The bound on the pairing is therefore stated from the evidence each profile records:

    |flux − expected| ≤ 1e-8 W/m + C · 4 W/m · (d_m + d_n),   C = 1,

with d_m and d_n the two modes' relative boundary defects and `expected` ∓4 W/m for partners and zero otherwise
(forward–forward pairs included). On a three-layer slab every pair is within the 1e-8 W/m term alone. On the
four-region multilayer of the sensitivity tests (ε up to 6) the most confined TE and TM modes have defects of 2e-7
and 4e-8; their forward–backward fluxes carry imaginary parts of 1.8e-8 and 2.5e-8 W/m, while the real parts match
∓4 to 3e-11. Well-recovered pairs stay below 1.6e-10 W/m, the quadrature floor that the 1e-8 term covers; the
defect-dominated pairs reach at most 0.07 of 4 W/m · (d_m + d_n), so C = 1 leaves a margin of about 14.

## Defaults checked

The coupling gap and the projection rank threshold were checked on the multilayer fixtures of the test suite (a
four-region stack, two two-layer stacks, two coupled-core stacks and a Sellmeier slab), TE and TM, forward and
backward, at λ = 1 µm:

- The closest pair of propagation constants is 1.15e-5 k0 apart (coupled cores 1.5 µm apart, TE), eleven times the
  default gap; every other fixture's closest pair is at least 1.9e-3 k0 apart. No fixture forms an unresolved group.
- Every Gram matrix has full rank, with 1 − σ_min/σ_max at most 4.2e-8 against the threshold's 1 − 1e-8.

Both defaults stand.

## Alternatives

- **Definitions in `CONVENTIONS`:** changes every existing record's hash for no change in meaning.
- **One registration offset instead of two origins:** the absolute frame is needed for projection and for
  differentiating with moving first interfaces.
- **A vectorised field callable:** needs a batched sampler in lamina-mode; memoisation already removes repeat
  calls.
- **An absolute default gap:** its meaning would change with the wavelength, as lamina-mode's absolute residual
  limit did (0003).
