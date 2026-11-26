# lamina-py

Python bindings over the lamina-mode guided basis: enumerate the guided modes
of a planar multilayer, sample their unit-power fields and evaluate their local
derivatives, with numpy arrays and typed results that keep their evidence.

```python
import numpy as np
import lamina

stack = lamina.Stack(left=1.0, layers=[(4.0, 2e-2)], right=2.25)
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-2, family="TE"))

print(basis.completeness, basis.expected, basis.found)  # complete 6 6
print(basis.n_eff.real)

profile = basis.sample_fields(0, np.linspace(-1e-2, 3e-2, 401))
profile.electric_v_per_m  # (401, 3) complex: Ex, Ey, Ez
local = basis.local_derivatives(0, thickness_layers=[0])
local.dbeta_dthickness(0)  # rad/m per m

text = basis.to_json()                    # canonical, versioned
assert lamina.GuidedBasis.from_json(text) == basis
basis.input.canonical_hash()              # cache key for the input
```

## Sensitivities and the group index

```python
stack = lamina.Stack(
    left=1.0,
    layers=[(lamina.SellmeierMaterial(1.5, [(1.2, 0.7e-6)]), 1.2e-6)],
    right=lamina.SellmeierMaterial(1.6, [(0.3, 0.42e-6)]),
)
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6, family="TE"))

s = basis.sensitivity(0, lamina.Parameter.thickness(0), z_m=np.linspace(0, 1.2e-6, 50))
s.dbeta_rad_per_m          # rad/m per m of thickness
s.d_electric_v_per_m       # (50, 3): dE/dd of the unit-power field, material-attached frame
s.power_fractions          # left exterior, layers, right exterior; sums to one
s.power_fraction_rates     # per m; sums to zero

g = basis.group_index(0)   # certified: dβ/dk0 agrees with c W / P within g.tolerance
g.value, g.energy_estimate, g.relative_discrepancy
```

Parameters: `thickness`, `interface_position`, `translation`, `wavelength` (with dispersion), `permittivity` and
`permeability` (every diagonal entry at once, or one stack-frame `component=(row, column)`), `rotation(region, axis)`
(a turn of the region's material about a stack axis, per radian: dT/dθ = G T − T G with G v = axis × v), and the
Sellmeier `constant`, `coefficient` and `pole_wavelength` of a region's model, on an `axis` for a tensor region
("x", "y", "z" of a `Diagonal`, "a", "b", "c" of a principal medium). Each result names its `parameter_unit` and
carries the conditioning of the implicit derivative. A rate that is off-diagonal or complex in the stack frame
gives dβ only; a symmetric perturbation of ε_xz is the sum of the (0, 2) and (2, 0) components.

## Walls

Either side of a stack may be closed by a perfect electric (`"pec"`) or perfect magnetic (`"pmc"`) wall at an
absolute position in the stack frame, on or outside its outermost interface; the exterior material fills the gap:

```python
stack = lamina.Stack(
    left=2.25, layers=[(4.0, 0.7e-6)], right=2.25,
    left_termination=lamina.Wall("pec", -0.45e-6),
    right_termination=lamina.Wall("pec", 1.15e-6),
)
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6, family="TE"))
basis.light_line_rad_per_m   # 0.0: no open side
basis.n_eff.real             # every real-beta mode in the window, below the cladding index too
```

A PEC wall makes the tangential electric field vanish (Ey in TE, Ex in TM), a PMC wall the tangential magnetic field
(Hx in TE, Hy in TM). The light line and cutoff margin come from open sides; between two walls the basis holds every
real-β mode in the window. Fields are sampled up to the walls, and sensitivities hold the walls fixed, so
`translation` and `interface_position(0)` are refused on a walled stack. The record stores each wall as
`{"kind": "wall", "type": ..., "z_m": ...}` and omits open sides, so open stacks keep their hashes; decision note
0009 has the details.

## Perfectly matched layers

A side may instead be closed by a perfectly matched layer of its exterior material, starting at an absolute position
and extending outward by its thickness, with the stretch s(ζ) = 1 + strength (ζ/thickness)^order and a PEC or PMC
backing wall. Every field is required:

```python
stack = lamina.Stack(
    left=2.25, layers=[(4.0, 0.7e-6)], right=2.25,
    right_termination=lamina.Pml(start_z_m=1.0e-6, thickness_m=1.0e-6, order=2.0, strength=1 + 3j, backing="pec"),
)
```

The record stores it as `{"kind": "pml", "start_z_m": ..., "thickness_m": ..., "profile": {"kind": "polynomial",
"order": ..., "strength": [re, im]}, "backing": ...}`; a non-positive thickness, a negative order or a strength with
a negative real or non-positive imaginary part is `InvalidInput`. A PML-closed stack has no guided basis:
`enumerate_guided_modes` raises `UnsupportedTermination`, an `InvalidInput`; a stack closed on both sides has a
modal basis (below).

## Pairings

Modes pair with each other, from one basis or from two stacks at one wavelength, and with supplied fields:

```python
request = lamina.GuidedRequest(wavelength_m=1e-6, family="TE")
a = lamina.enumerate_guided_modes(lamina.Stack(1.0, [(4.0, 0.6e-6)], 1.44), request)
b = lamina.enumerate_guided_modes(lamina.Stack(1.0, [(4.0, 0.7e-6)], 1.44), request)

o = lamina.power_overlap(a, 0, b, 0, origins_m=(0.0, 0.07e-6))
o.power_w_per_m             # (1/4) ∫ (E1 × H2* + E2* × H1) · x dz, W/m

s = lamina.power_overlap_sensitivity(
    a, 0, b, 0, None, lamina.Parameter.thickness(0), origins_m=(0.0, 0.07e-6)
)
s.power_rate_w_per_m        # W/m per m of the second stack's layer, or None with s.derivative_failure

r = lamina.reciprocity(a, 0, b, 0)
r.flux_w_per_m              # ∫ (E1 × H2 − E2 × H1) · x dz, unconjugated

c = b.local_coupling(lamina.Parameter.permittivity(1))
c.perturbation_matrix_rad_per_m  # Hermitian, rad/m per unit permittivity
c.coefficients              # row receiving, column differentiated; nan where not c.coefficient_resolved

def field(z_m, side):       # side: "left", "right" or "interior"
    ey = np.exp(-(z_m / 0.3e-6) ** 2)
    return (0, ey, 0), (0, 0, ey / 300)  # E in V/m, H in A/m

p = b.project_field(field, support_m=(-2e-6, 2e-6), interfaces_m=(0.0, 0.7e-6), field_id="gaussian")
p.rank, p.singular_values_w_per_m, p.coefficients  # coefficients None below full rank
```

`origins_m` places each basis's first interface in a common frame. An overlap derivative takes one parameter per
stack (`None` holds it fixed), both per one unit. Each result records its pairing `definition`, the input hash of
every basis it used, and its quadrature evidence. A local coupling needs a complete basis of a nonmagnetic stack
and a fixed-wavelength parameter; modes within `gap_rad_per_m` (default 1e-6 k0) of each other form an unresolved
group without mutual coefficients. The supplied field must vanish outside its support and is sampled one point at
a time, once per position and side; an exception it raises propagates. Decision note 0008 has the details.

## Modal bases

A stack closed by walls or PMLs on both sides has a discrete spectrum. `modal_basis` returns every mode in a window of the
complex β plane, guided, radiation-like, evanescent and complex, as a complete modal basis in that window:

```python
stack = lamina.Stack(2.25, [(4.0, 0.7e-6)], 2.25, lamina.Wall("pec", -0.45e-6), lamina.Wall("pec", 1.15e-6))
k0 = 2 * math.pi / 1e-6
window = lamina.BetaWindow.forward(2.1 * k0, 3.0 * k0, 0.05 * k0)  # Re β ∈ [−m, β_max], Im β ∈ [−m, decay_max]
basis = lamina.modal_basis(stack, 1e-6, "TE", window)
basis.status, basis.expected, basis.found  # "complete", 11, 11
basis.beta_rad_per_m                       # real β first, then evanescent by increasing decay

profile = basis.profile(0, np.linspace(-0.45e-6, 1.15e-6, 201))
profile.region                             # "gap" or "layer" per sample

p = lamina.basis_pairing(basis, basis)
p.unconjugated_w_per_m                     # the identity, to the quadrature tolerance
```

`expected` counts the zeros in the window from the argument principle over its boundary, independently of the
search; `unresolved` lists every region that did not end as accepted modes, with its cause, and an incomplete basis
is a result. Each mode has unit unconjugated self-pairing ⟨f, f⟩ = ¼ ∫ (E × H + E × H) · x dz = 1 W/m, a sign
fixed at the first interface, and the order of `definitions["ordering"]`; modes within `degeneracy_gap_rad_per_m`
(default 1e-6 k0) of each other form a group with an unconjugated Löwdin basis. `basis_pairing` gives both pairing
matrices between two bases at one wavelength whose walls coincide once `origins_m` places each first interface,
with singular values, the condition number of every leading block and each basis's own Gram defect. Every record
carries the fixed `definitions`; decision note 0011 has the details.

A side may also be closed by a `Pml`; its basis holds the modes converging to the open stack's guided and exposed
leaky modes as the PML thickens and the modes near the rotated continuum, all counted:

```python
pml = lambda start: lamina.Pml(start, 0.75e-6, 2.0, 1 + 3j, "pec")
stack = lamina.Stack(2.25, [(4.0, 0.7e-6)], 2.25, pml(-0.1e-6), pml(0.8e-6))
basis = lamina.modal_basis(stack, 1e-6, "TE", lamina.BetaWindow((k0, 2.03 * k0), (-0.031 * k0, k0)))
profile = basis.profile(0, np.linspace(-0.85e-6, 1.55e-6, 241))
profile.region                             # "pml", "gap" or "layer" per sample
profile.stretched_z_m, profile.stretch     # z̃ and s inside a PML, nan elsewhere
```

Inside a PML the profile holds the fields continued to the stretched coordinate z̃; the equivalent medium in real z
has the same tangential components and normal components s E_z and s H_z (`profile.pml_fields`). Pairings integrate
the unconjugated pairing along z̃ and the conjugated product over the equivalent-medium fields, and two bases pair
when their PMLs agree in start, thickness, profile and backing; their materials may differ. A window on which a
PML's normalisation scale leaves the representable range raises `InvalidInput` naming the side and log-magnitude.

A stack with an open side has a radiation continuum besides its discrete modes. Its basis holds the guided and
leaky modes in a `ChartWindow`, a rectangle in the conformal coordinate z (κ = i k0 e^z with one open side or equal
exteriors), never the continuum, so it does not span the stack's fields (scope `"discrete_without_continuum"`).
`chart_window_covering` builds the window covering a β rectangle on stated sheets:

```python
stack = lamina.Stack(2.25, [(4.0, 0.7e-6)], 2.25)                      # open on both sides
window, extra = lamina.chart_window_covering(
    stack, 1e-6, ((1.45 * k0, 1.65 * k0), (0.03 * k0, 0.15 * k0)), ("improper", "improper")
)
basis = lamina.modal_basis(stack, 1e-6, "TE", window)
[(m.mode_type, m.sheet) for m in basis.modes]  # "guided" on the proper sheet, "leaky" on the improper one
```

A rectangle that meets a stated sheet's cut raises `InvalidInput` naming the side, and a `BetaWindow` on an open
stack raises `InvalidInput` naming `ChartWindow`. Pairings continue the unconjugated tail beyond an open side
analytically; a conjugated entry whose tail diverges (Im κ_a + Im κ_b ≤ 0, as between leaky modes) is nan with
cause `"unbounded"`. A root within 1e-4 k0 of a branch point (κ = 0) is unresolved with cause `"branch_point"`.

A supplied field F expands in a basis's normalised modes, and the modes couple along a parameter:

```python
p = basis.project_field(field, support_m=(-0.4e-6, 1.1e-6), exterior=(left_waves, right_waves))
p.coefficients, p.condition_number, p.gram_defect
p.relative_residual, p.residual_includes_continuum

c = basis.local_coupling(lamina.Parameter.thickness(0))
c.perturbation_matrix_rad_per_m.diagonal()  # each mode's dβ/dp
c.coefficients, c.groups                   # K[m, n] / (β_n − β_m); near-degenerate modes as groups
```

`project_field` solves G a = b with G[m, n] = ⟨f_m, f_n⟩ and b[m] = ⟨f_m, F⟩, unconjugated, on the basis's own
partition: along z̃ in a PML, where F is the continued field as `profile` gives it, and with continued tails on open
sides. F vanishes outside its support, which stays within the walls; on an open side the field beyond the support
may be stated as `ExteriorWave`s (κ, and E and H at the edge), as a field built from leaky modes needs. The record
has the reconstruction residual over the bounded partition. A basis of a stack with an open side never includes
the radiation continuum, so there the residual includes F's radiation content: it is physical, not numerical, and
`residual_includes_continuum` says so. `local_coupling` uses the mirrored field (−E_x, E_y, E_z) in place of a
conjugate, one-sided fields at moving interfaces, z̃ in a PML and continued tails; a wavelength or permeability
parameter, or one moving an interface against a wall or PML, raises `InvalidInput`.

## Scope

lamina-py presents lamina-mode's results in Python: SI quantities with the unit in every field name, typed
results with numpy arrays, and a versioned, canonical JSON form of every input and result with its SHA-256
identity. Every number comes from lamina-mode.

## Anisotropic regions

A region may hold a tensor medium: one scalar model per stack axis (`Diagonal`), one per principal axis
(`Biaxial`, or `Uniaxial` with the ordinary model on a and b), or constant complex tensors (`TensorMaterial`),
each oriented by a `Rotation` from its own frame to the stack frame (T_stack = R T Rᵀ). A scalar model is a
`Material` (constant), `SellmeierMaterial`, `Lorentz`, `DrudeLorentz` or `Tabulated`:

```python
k0 = 2 * math.pi / 1e-6
calcite = lamina.Uniaxial(lamina.SellmeierMaterial(1.73, [(0.98, 0.14e-6)]), 2.18, optic_axis=(0.6, 0.7))
metal = lamina.DrudeLorentz(1.0, plasma_rad_per_m=2.0 * k0, damping_rad_per_m=0.0)
rotated = lamina.Biaxial(2.0, 3.0, 4.0, lamina.Rotation.from_euler(0.4, 0.7, 0.2))
stack = lamina.Stack(1.0, [(calcite, 0.8e-6)], 2.1)
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6))
basis.regions[1].region_class, basis.regions[1].separable  # "full", False
basis.regions[1].relative_permittivity                      # 3×3, stack frame, at the wavelength
basis.labels                                                # ("hybrid", ...)
```

Rotations are recorded and hashed as matrices; `Rotation.from_euler` (z-y′-z″) and `Rotation.from_optic_axis(θ, φ)`
(the c axis at polar angle θ from z and azimuth φ from x) resolve to one, with exact entries at quarter turns. A
matrix that is not a proper rotation to 1e-12 is `InvalidInput`. The class of a region follows from its model and
rotation, not from its tag: a principal medium turned by a signed axis permutation is diagonal, and one with a
principal axis exactly on y is separable (TE and TM decouple). TE or TM of a stack with a non-separable region is
`InvalidInput`. Guided enumeration needs Hermitian, positive-definite ε and μ at the wavelength; a lossy region is
`InvalidInput`. Lorentz and Drude–Lorentz resonances, plasma and damping are angular wavenumbers k0 = ω/c in rad/m.
A `Tabulated` permittivity is interpolated linearly in k0 and held at its value at the operating wavelength, which
must lie inside the table; it has no wavelength dispersion, so a wavelength sensitivity or group index is
`InvalidInput`. A modal basis takes stacks of isotropic and diagonal regions.

## Families and labels

A request names a `family`: `"TE"` or `"TM"` for one polarisation of a separable stack, or `"all"`, the default,
for every mode. TE and TM of isotropic and diagonal regions are counted by their oscillation index; `"all"` is
counted by the argument principle on the coupled TE/TM channel, with the evidence in `basis.count_evidence`:

```python
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6))
basis.labels                  # ("TE", "TM", ...): TE or TM on a separable stack, "hybrid" otherwise
basis.count_method            # "argument_principle"
[m.degenerate_group for m in basis.modes]  # shared by modes within basis.degeneracy_gap_rad_per_m
```

An argument-principle count keeps its rectangles clear of the light-line branch point, so its default cutoff
margin is 1e-6 k0 (1e-5 rad/m for TE and TM). Modes so counted are solved through the 4×4 (Berreman)
characteristic: their profiles carry all six components, name their phase reference (`"Ey+Z0Hy"`, or `"Ex+Z0Hx"`
where that vanishes) and record the unconjugated norm (1/2) ∫ (E × H) · x dz; their sensitivities give dβ/dp with
its conditioning but no field derivatives or power fractions (request no `z_m`); their group index is certified
against the energy estimate of the full tensors, which for hybrid modes may need a `tolerance` near 1e-6. Pairings
and local derivatives take TE and TM modes of isotropic and diagonal regions; others raise `InvalidInput`.

## Conventions

Every result carries `conventions`, `schema` and `schema_version` (2); see
`lamina.CONVENTIONS`. In short:

- lengths in metres, wavenumbers in rad/m, E in V/m, H in A/m, power in W per
  metre of invariant width;
- z is the stack normal, with z = 0 at the first interface; x is the
  propagation axis; y is invariant;
- tensors are expressed in that frame: a region's rotation R maps its material
  frame to it, T = R T_material Rᵀ;
- polarisation is a property of each mode: TE has Ey, Hx, Hz and TM has Hy, Ex,
  Ez when every region leaves y decoupled; otherwise modes are hybrid;
- phasors use exp(+iβx − iωt);
- profiles carry unit signed longitudinal power, with Ey + Z₀Hy at the left
  interface real and positive, or Ex + Z₀Hx where that vanishes (named in the
  profile);
- a sample on an interface takes the limit from +z.

Every material and scalar model is tagged by `"kind"`; a version 1 record is
rejected with `SchemaError` (decision note 0010 gives the correspondence). The
materials make the analytic declarations lamina-mode requires true by
construction; each result records the declaration in `assumption`.

## Evidence and failures

A `GuidedBasis` records its resolved input and its hash, the light line, the
eligible window, the expected and found counts, the unresolved intervals with
their causes, and a `completeness` of `complete`, `partial` or
`count_unknown`. Each mode records its residual, correction and acceptance
limits. Partial coverage is a result, not an exception. Invalid input raises
`InvalidInput`; failed enumeration, field recovery or derivatives raise
`EnumerationFailed`, `FieldsFailed` or `DerivativesFailed`, and a pairing
integral over its budget raises `IntegrationFailed`, all `LaminaError`.

## Exterior fields

Exterior samples of an open side keep only the outgoing wave, so a guided field
decays monotonically however far from the stack it is sampled. Behind a wall the
field is a standing wave in the gap, integrated numerically. The unit-power
normalisation integrates the tails analytically and doesn't depend on the
samples.

## Serialisation and identity

`to_json()` gives the canonical form: fields in schema order, no whitespace and
shortest round-trip floats. `from_json()` checks the schema, its version and the
conventions. `canonical_hash()` is the SHA-256 of the canonical form, so equal
inputs or results give equal hashes. A deserialised basis can still sample
fields: it re-solves its stored input and refuses to continue unless the new
result is identical to the stored one.

## Build and test

From the lamina-mode dev shell (`nix develop`):

```sh
cd lamina-py
maturin build --release --zig --compatibility manylinux2014 --out ../target/wheels
python -m venv /tmp/lamina-venv
/tmp/lamina-venv/bin/pip install ../target/wheels/lamina_py-*.whl pytest
/tmp/lamina-venv/bin/python -m pytest tests
```

The wheel uses the stable ABI (abi3, Python 3.10 and later) and needs no Rust
toolchain to install.

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
