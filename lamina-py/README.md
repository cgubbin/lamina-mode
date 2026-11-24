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

## Boundary

lamina-py exposes lamina-mode's public results and nothing else. It owns the
conversion to SI at the boundary, the serialised form of inputs and results,
and their hashes. It owns no physics: every number comes from lamina-mode.

| lamina-py owns | lamina-py never owns |
| --- | --- |
| SI units at the boundary, named in every field | Mode searches, fields, derivatives (lamina-mode) |
| Typed Python results with numpy arrays | Geometry, layouts, processes, ports |
| The versioned JSON form of inputs and results, and their hashes | Choosing a mode or a design |

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
`EnumerationFailed`, `FieldsFailed` or `DerivativesFailed`, all `LaminaError`.

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
