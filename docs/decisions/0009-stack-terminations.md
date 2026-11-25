# 0009: Walls and PMLs closing a stack

Status: accepted

## Context

A stack may be closed on either side by a perfect electric (PEC) or perfect magnetic (PMC) wall, or by a
wall-backed perfectly matched layer (PML). lamina-core places a wall at an absolute position in the stack frame and
fills the gap between the outermost interface and the wall with that side's exterior material. A walled side carries no exterior wavevector, so a stack closed on both
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

**Perfectly matched layers.** A side may instead be closed by a perfectly matched layer of its exterior material:

```json
"right_termination": {"kind": "pml", "start_z_m": 1e-6, "thickness_m": 1e-6,
                      "profile": {"kind": "polynomial", "order": 2.0, "strength": [1.0, 3.0]},
                      "backing": "pec"}
```

`start_z_m` is the absolute position of its inner face, under the same rules as a wall's `z_m`; it extends outward,
away from the layers, by `thickness_m`, so its backing wall lies at `start_z_m ∓ thickness_m`. With ζ ∈ [0, d] the
outward distance into a PML of thickness d, the stretch is

```text
s(ζ) = 1 + c (ζ/d)^m,    ζ̃(ζ) = ζ + c d/(m+1) (ζ/d)^(m+1),
```

frequency independent, with order m ≥ 0 and strength c, Re c ≥ 0 and Im c > 0, both required; the profile is
internally tagged so that other profiles can be added. The backing wall imposes the wall conditions above. Fields in
the PML are the exterior solution continued to the stretched coordinate ζ̃.

**β-independent normalisation.** For the characteristic a PML is a layer of the exterior material with complex
thickness τ = d (1 + c/(m+1)), whose matrix grows like exp(|Im κτ|). Each PML side is normalised by
exp(iκ_ref τ), with κ_ref = k0 √(εμ) the PML material's wavevector at β = 0. The factor does not depend on β, so the
zeros are unchanged, a PML side takes no exterior wavevector, and a stack closed on both sides keeps an entire,
even characteristic searched on the direct β chart. Its scale is exp(Σ [−Im(κ_ref τ) + |Im(κτ)|]) over the PML
sides; a search rectangle on which it leaves e^±300 is refused, naming the side with the largest share.

**PML convergence.** The closed stack's mode differs from the open stack's by O(exp(2iκ_ℓ(g + τ))), g the gap from
the outermost interface to the PML. A guided mode (κ = iα) converges as the PML moves outward and as it thickens; a
leaky mode converges as the PML thickens at a fixed start when Im(κ_ℓ(g + τ)) > 0, that is
arg κ_ℓ > −arg(g + d s̄) with s̄ = 1 + c/(m+1), and moving the PML outward is not a direction of convergence for it.
The remaining modes lie near the rotated continuum, arg κ ≈ −arg(g + d s̄). docs/pml-convergence-report.md has the
measured rates.

**No gain or loss inferred.** A PML is an analytic continuation, not a passive medium, so the sign of Im β at the
level of its truncation error says nothing about gain or loss. A converged guided mode moves off the real axis by
the PML's perturbation, whose phase turns with d: at g = 0.1 cm, d = 0.75 cm (m = 2, c = 1 + 3i) the even TE mode
of a slab of permittivity 4 in 2.25 lies at 12.121063854 − 1.97 × 10⁻⁸ i cm⁻¹. A mode of a stack with a PML side
therefore carries `Evolution::PmlTruncation { truncation }`, the largest |exp(2iκ̃(g + τ))| over the PML sides,
instead of attenuation or amplification. Closed on both sides it is `ModeType::Closed`. Beside an open side its
type follows from the open side alone: decaying into it is guided, growing with outgoing flux is leaky, growing
with incoming flux is a source. Open and walled stacks are unchanged.

**Refusals.** A PML-closed stack has no guided basis and no modal basis in a β window: enumerating either is
`UnsupportedTermination`, an `InvalidInput`. Transverse integrals through a PML (unit power, stored energy, regional
power) and pairings are refused.

**Hashes.** The PML variant is additive: walled and open records keep their form and hash, and each single change
of start, thickness, order, strength or backing changes the hash.

## Consequences

The profile phase convention is unchanged: a PEC wall in TE (or PMC in TM) on the first interface makes the
left-interface primary field vanish, and such a profile is refused as ill-conditioned.
