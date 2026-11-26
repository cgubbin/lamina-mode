# 0010: Schema version 2

Status: accepted

## Context

Anisotropic regions change three conventions of the serialised records (0004): polarisation becomes a property of
each mode rather than of the request, a region's material may be a tensor with an orientation, and the phase rule of
a unit-power profile must cover modes with every field component. Under 0005 a convention change bumps the schema
version. No external records depend on version 1, so version 2 is a clean break.

## Decision

**Version policy.** Every record kind carries `"schema_version": 2`. A reader accepts version 2 only: any other
version is `SchemaError`, naming the version found and the version the build reads. Nothing converts version 1
records, and no hash is carried over.

**Tagged objects.** Every choice in a record is an object with an internal `"kind"` tag followed by the variant's
own fields: scalar models, region materials, terminations (0009), PML profiles, windows and parameters.

**Scalar models.** One relative permittivity and permeability as functions of wavelength:

```json
{"kind": "constant", "relative_permittivity": 2.25, "relative_permeability": 1.0}
{"kind": "sellmeier", "sellmeier": {"constant": 1.0, "terms": [{"coefficient": 1.2, "pole_wavelength_m": 7e-7}]},
 "relative_permeability": 1.0}
{"kind": "lorentz", "epsilon_infinity": 2.0,
 "oscillators": [{"delta_epsilon": 1.5, "resonance_rad_per_m": 3e6, "damping_rad_per_m": 0.0}],
 "relative_permeability": 1.0}
{"kind": "drude_lorentz", "epsilon_infinity": 1.0, "plasma_rad_per_m": 1.2e7, "damping_rad_per_m": 0.0,
 "oscillators": [], "relative_permeability": 1.0}
{"kind": "tabulated", "wavelength_m": [8e-7, 1.2e-6], "relative_permittivity": [[3.0, 0.0], [3.5, 0.0]],
 "relative_permeability": 1.0}
```

Resonances, plasma and damping are angular wavenumbers k0 = ω/c in rad/m. A tabulated model is interpolated
linearly in k0 and held at its value at the operating wavelength, inside the table; its record's `assumption` says
so, and no wavelength derivative is offered for it.

**Region materials.**

```json
{"kind": "isotropic", "model": M}
{"kind": "diagonal", "x": M, "y": M, "z": M}
{"kind": "full", "material": {"kind": "principal", "a": M, "b": M, "c": M}, "rotation": R}
{"kind": "full", "material": {"kind": "tensor", "relative_permittivity": T, "relative_permeability": T},
 "rotation": R}
```

`diagonal` places one scalar model on each stack axis. `full` gives a material in its own frame, either one scalar
model on each principal axis a, b, c or a constant complex 3×3 tensor pair (rows first, entries `[re, im]`), and the
rotation matrix R (rows first) that maps that frame to the stack frame: column k of R holds the stack-frame
components of material axis k, and T_stack = R T R^T. Only the matrix is recorded and hashed; constructors that take
Euler angles or an optic axis resolve to it first. The tag names the form of the input, not the structure of the
solved tensors: the class of a region (isotropic, diagonal or full) and whether it is separable (no `xy`, `yz`
coupling) are derived from the model and the exact entries of R, and recorded in the result. A principal model with
a signed axis permutation is diagonal; one with a material axis exactly on y is separable.

**Families.** A guided request names `"family"`: `"TE"` or `"TM"` of a separable stack, or `"all"`. An omitted
family is `"all"`, resolved before hashing. TE or TM of a stack with a non-separable region is `InvalidInput`. TE
and TM of isotropic and diagonal regions are counted by their oscillation index; every other request by the
argument principle on the coupled channel, with its evidence in `count_evidence`. A modal basis input names a TE or
TM family.

**Conventions.** Version 2 adds `tensor_frame` (tensors in the stack frame, T = R T_material R^T) and restates
`polarisation` (TE has Ey, Hx, Hz and TM has Hy, Ex, Ez when every region leaves y decoupled; otherwise modes are
hybrid) and `normalisation` (Ey + Z0 Hy at the left interface real and positive, or Ex + Z0 Hx where that vanishes,
named in each profile).

**Results.** A guided basis records `regions`, each with its derived `class`, `separable` and stack-frame
`relative_permittivity` and `relative_permeability` at the operating wavelength. Each mode of a guided basis carries its `label` (`"TE"`, `"TM"` or `"hybrid"`) and its
`degenerate_group`: modes whose |β| lie within `degeneracy_gap_rad_per_m` (1e-6 k0) of a neighbour share an index,
and a mode alone has none. Sensitivity and group-index records carry the basis's `assumption`.

**Version 1 inputs.** A version 1 input maps onto a version 2 input that solves to the same modes: each material
`{"relative_permittivity": …}` or `{"sellmeier": …}` becomes `{"kind": "isotropic", "model": {"kind": "constant" |
"sellmeier", …}}` with its fields unchanged, `"polarisation"` becomes `"family"` with the same value, and
`"schema_version"` becomes 2. Its hash differs.

## Consequences

- Records of isotropic stacks are longer by their tags; one rule covers every material.
- A reader built for version 1 rejects every version 2 record, and the reverse.
- 0004 (canonical form and hashes) and 0005 (additive changes keep the version) stand; this note is the version
  change 0005 anticipates.
