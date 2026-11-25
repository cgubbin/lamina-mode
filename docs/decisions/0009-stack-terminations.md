# 0009: Walls closing a stack

Status: accepted

## Context

A stack may be closed on either side by a perfect electric (PEC) or perfect magnetic (PMC) wall. lamina-core
places a wall at an absolute position in the stack frame and fills the gap between the outermost interface and the
wall with that side's exterior material. A walled side carries no exterior wavevector, so a stack closed on both
sides has a characteristic that is entire and even in β: its real spectrum is discrete and includes the
radiation-like modes below the exterior light line. The bindings must carry walls through the input record and its
hash (0004) without changing any open-stack record (0005), and say what each wall imposes.

## Decision

**Record fields.** The input stack gains two optional fields after `right`:

```json
"stack": {"left": M, "layers": [...], "right": M,
          "left_termination": {"kind": "wall", "type": "pec", "z_m": -1e-6},
          "right_termination": {"kind": "wall", "type": "pmc", "z_m": 3e-6}}
```

A termination is an internally tagged object: `"kind"` names the variant and the variant's own fields follow. A wall
has `"type"`, `"pec"` or `"pmc"`, and `"z_m"`. An open side is omitted. `{"kind": "open"}` is accepted on input and
canonicalises to omission, so it serialises and hashes as the same input without it. Unknown kinds and unknown or
missing fields are `SchemaError`.

**Absolute positions.** `z_m` is absolute in metres in the stack frame: z = 0 at the first interface, the layers
running towards +z and ending at L, the layer thicknesses summed left to right as in `Stack.interfaces_m`. A left
wall lies at z ≤ 0 and a right wall at z ≥ L; a wall exactly on its outermost interface is allowed. Gaps are
derived, never stored. Positions must be finite; −0 becomes +0. Changing a layer thickness keeps the walls where
they are: the right gap changes, and the record keeps the same positions.

**Wall conditions.** With ψ = E_y (TE) or H_y (TM):

| wall | TE | TM |
| --- | --- | --- |
| PEC | E_y = 0: ψ = 0 | E_x = 0: ∂ψ/∂z = 0 |
| PMC | H_x = 0: ∂ψ/∂z = 0 | H_y = 0: ψ = 0 |

**Enumeration.** The light line, and the cutoff margin above it, come from open sides only:
`light_line_rad_per_m` is the larger open-side light line, and zero when both sides are walled. A basis of a stack
walled on both sides holds every real-β mode in the window, below the exterior light line included. Evanescent and
complex modes are not enumerated.

**Fields and derivatives.** Samples run up to the walls; a z beyond a wall is `InvalidInput`. Unit-power profiles
integrate each gap. Sensitivities hold the walls fixed in the stack frame, whose origin is the first interface: a
parameter that moves that interface (`translation`, `interface_position(0)`) would move the stack against its walls
and is `InvalidInput`. With the right wall on the last interface, a thickness derivative that thickens the stack is
unavailable. Pairings take stacks open on both sides.

**Additive schema.** Open stacks serialise exactly as before, so every existing record keeps its form and hash
and the schema stays at version 1 (0005). Strict parsing makes a reader without these fields reject a walled record
rather than read it as an open stack.

## Consequences

The profile phase convention is unchanged: a PEC wall in TE (or PMC in TM) on the first interface makes the
left-interface primary field vanish, and such a profile is refused as ill-conditioned.
