# lamina-py

Python bindings over the lamina-mode guided basis: enumerate the guided modes
of a planar multilayer, sample their unit-power fields and evaluate their local
derivatives, with numpy arrays and typed results that keep their evidence.

```python
import numpy as np
import lamina

stack = lamina.Stack(left=1.0, layers=[(4.0, 2e-2)], right=2.25)
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-2))

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
basis = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6))

s = basis.sensitivity(0, lamina.Parameter.thickness(0), z_m=np.linspace(0, 1.2e-6, 50))
s.dbeta_rad_per_m          # rad/m per m of thickness
s.d_electric_v_per_m       # (50, 3): dE/dd of the unit-power field, material-attached frame
s.power_fractions          # left exterior, layers, right exterior; sums to one
s.power_fraction_rates     # per m; sums to zero

g = basis.group_index(0)   # certified: dβ/dk0 agrees with c W / P within g.tolerance
g.value, g.energy_estimate, g.relative_discrepancy
```

Parameters: `thickness`, `interface_position`, `translation`, `wavelength` (with dispersion), `permittivity`,
`permeability`, and the Sellmeier `constant`, `coefficient` and `pole_wavelength` of a region's material. Each
result names its `parameter_unit` and carries the conditioning of the implicit derivative.

## Pairings

Modes pair with each other, from one basis or from two stacks at one wavelength, and with supplied fields:

```python
request = lamina.GuidedRequest(wavelength_m=1e-6)
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

## Scope

lamina-py presents lamina-mode's results in Python: SI quantities with the unit in every field name, typed
results with numpy arrays, and a versioned, canonical JSON form of every input and result with its SHA-256
identity. Every number comes from lamina-mode.

## Conventions

Every result carries `conventions`, `schema` and `schema_version`; see
`lamina.CONVENTIONS`. In short:

- lengths in metres, wavenumbers in rad/m, E in V/m, H in A/m, power in W per
  metre of invariant width;
- z is the stack normal, with z = 0 at the first interface; x is the
  propagation axis; y is invariant. TE has Ey, Hx, Hz and TM has Hy, Ex, Ez;
- phasors use exp(+iβx − iωt);
- profiles carry unit signed longitudinal power, with the left-interface Ey (TE)
  or Z₀Hy (TM) real and positive;
- a sample on an interface takes the limit from +z.

Materials are constant, real and positive. That makes the analytic
declarations lamina-mode requires true by construction; each result records
the declaration in `assumption`.

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

Exterior samples keep only the outgoing wave, so a guided field decays
monotonically however far from the stack it is sampled. The unit-power
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
