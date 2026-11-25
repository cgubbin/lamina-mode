# PML convergence report

A stack closed by wall-backed PMLs has a discrete spectrum. As the PML is made more effective, the closed stack's
modes approach those of the open stack: this report checks the guided and leaky modes against independently solved
open-stack modes, and checks that the remaining modes keep clear of them.

## Setup

The vacuum wavelength is 1 cm (k0 = 2π cm⁻¹). A core of relative permittivity 4 and thickness 0.7 cm lies in a
cladding of relative permittivity 2.25, with relative permeability one throughout. Both sides are closed by equal
PMLs of the cladding a gap g beyond their interfaces, d thick, with the polynomial stretch
s(ζ) = 1 + c (ζ/d)^m, m = 2 and c = 1 + 3i, backed by PEC walls. The PML acts as a cladding layer of complex
thickness τ = d s̄ with s̄ = 1 + c/(m+1) = 4/3 + i. All results are TE.

The open-stack modes are the roots of the slab's even and odd conditions with fields ∝ exp(iκ|z − z_c|) outside,
solved without lamina: the guided modes by bracketing on the real axis (κ = iα, α > 0), the leaky modes by Newton's
method from a grid on the outgoing sheet (Im κ < 0), counted by the argument principle. The closed-stack modes are
searched on the direct β chart.

## Prediction

With g the gap and κ_ℓ the exterior wavevector of an open-stack mode, the closed stack's mode differs from it by
O(exp(2iκ_ℓ(g + d s̄))). At fixed profile:

- a guided mode converges as the PML moves outward at fixed thickness, at the rate 2α per unit gap, and as it
  thickens at a fixed start, at the rate 2α Re s̄ per unit thickness;
- a leaky mode converges as the PML thickens at a fixed start when it is exposed, Im(κ_ℓ(g + d s̄)) > 0, that is
  arg κ_ℓ > −arg(g + d s̄), at the rate 2 Im(κ_ℓ s̄) per unit thickness. Moving the PML outward multiplies the
  difference by exp(−2g Im κ_ℓ) > 1, so it is not a direction of convergence for leaky modes;
- the remaining modes lie near the rotated continuum, where κ(g + d s̄) is real, so arg κ ≈ −arg(g + d s̄).

The fitted rate is the least-squares slope of ln|β_closed − β_open|, over differences above 10⁻⁹ cm⁻¹, the root
accuracy. The test requires each fitted rate within 10 % of the predicted one.

## Guided modes

Outward at d = 0.5 cm, g = 0.1 to 0.5 cm in steps of 0.1 cm; thicker at g = 0.1 cm, d = 0.3 to 0.9 cm in steps of
0.15 cm.

| Mode | Open β (cm⁻¹) | Direction | \|Δβ\| first → last (cm⁻¹) | Fitted rate (cm⁻¹) | Predicted (cm⁻¹) | Ratio |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| even | 12.121064 | gap | 3.49e−6 → 7.86e−9 | 15.244 | 15.244 (2α) | 1.0000 |
| even | 12.121064 | thickness | 2.04e−4 → 1.03e−9 | 20.325 | 20.325 (2α Re s̄) | 1.0000 |
| odd | 10.789413 | gap | 3.44e−4 → 5.15e−6 | 10.504 | 10.504 (2α) | 1.0000 |
| odd | 10.789413 | thickness | 5.60e−3 → 1.27e−6 | 13.988 | 14.006 (2α Re s̄) | 0.9987 |

## Leaky modes

g = 0.1 cm; d = 0.75 to 1.75 cm in steps of 0.25 cm; arg(g + d s̄) runs from 0.598 to 0.625.

| Mode | Open β (cm⁻¹) | arg κ_ℓ | \|Δβ\| at d = 0.75, 1, 1.25, 1.5, 1.75 (cm⁻¹) | Fitted rate (cm⁻¹) | Predicted 2 Im(κ_ℓ s̄) (cm⁻¹) | Ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| even | 5.541181 + 12.387419i | −0.288 | 7.64e−5, 7.61e−7, 7.58e−9, 7.55e−11, 7.71e−13 | 18.435 | 18.436 | 1.0000 |
| odd | 6.418274 + 5.824667i | −0.371 | 3.16e−2, 3.07e−3, 2.92e−4, 2.75e−5, 2.60e−6 | 9.410 | 9.437 | 0.9972 |

The even fit uses the three differences above the root accuracy. The window also holds the open-stack root
9.736780 + 0.484874i cm⁻¹ above the cladding light line, with arg κ_ℓ = −1.059: it is not exposed, and no closed
mode approaches it.

## Remaining modes

The window is 0.77 k0 ≤ Re β ≤ 2.03 k0, −0.031 k0 ≤ Im β ≤ 2.07 k0. At each thickness exactly one closed mode is
nearest each exposed open mode; at d = 1.75 cm each lies within 10⁻⁴ cm⁻¹ of it. Every other closed mode stays more
than 0.05 k0 from every open-stack mode, the exposed and the unexposed, moves by more than 10⁻⁴ cm⁻¹ between
successive thicknesses, and lies near the rotated continuum:

| d (cm) | Closed modes | Remaining | Least distance to an open mode | Largest \|arg κ + arg(g + d s̄)\| |
| ---: | ---: | ---: | ---: | ---: |
| 0.75 | 19 | 15 | 0.074 k0 | 0.159 |
| 1.00 | 23 | 19 | 0.078 k0 | 0.133 |
| 1.25 | 29 | 25 | 0.073 k0 | 0.113 |
| 1.50 | 33 | 29 | 0.071 k0 | 0.098 |
| 1.75 | 39 | 35 | 0.074 k0 | 0.087 |

The test requires the angle below 0.25.

## Classification

Every closed mode classifies as `ModeType::Closed` with both sides closed, and no gain or loss is inferred from
Im β. A PML is an analytic continuation, not a passive medium, and the sign of Im β at the level of its truncation
says nothing. A converged guided mode moves off the real axis by the PML's perturbation O(exp(2iκ(g + d s̄))),
whose phase turns with d (by −2α Im s̄ per unit thickness for κ = iα), so it lies on either side of the axis by no
more than its distance from the open mode. Below the axis, where the sign alone would read as gain:

| d (cm) | Mode | β (cm⁻¹) | Truncation estimate |
| ---: | --- | --- | ---: |
| 0.75 | even | 12.121063854 − 1.97 × 10⁻⁸ i | 5.22e−8 |
| 1.00 | odd | 10.789412922 − 2.76 × 10⁻⁷ i | 2.89e−7 |
| 1.50 | odd | 10.789412775 − 1.4 × 10⁻¹¹ i | 2.63e−10 |
| 1.75 | odd | 10.789412775 − 3.9 × 10⁻¹² i | 7.93e−12 |

Each is reported with `Evolution::PmlTruncation`, whose `truncation` is the largest |exp(2iκ̃(g + τ))| over the PML
sides, κ̃ the root decaying over g + τ. It matches the predicted perturbation exp(−2α(g + d Re s̄)) of the guided
modes.

With one side open and the other a PML, the type follows from the open side alone. For the same slab with a PML
on the left (g = 0.1 cm, d = 1.75 cm) and the cladding open on the right, both guided modes classify as `Guided`
and the odd leaky mode, at 6.418273 + 5.824666i cm⁻¹ against the open slab's 6.418274 + 5.824667i cm⁻¹, as
`RightLeaky`, each with `Evolution::PmlTruncation`.

## Other checks

- A wall and a PML closing a box: on the direct chart, TE and TM with PMC/PEC and PEC/PMC, the six roots in
  1 k0 ≤ Re β ≤ 1.99 k0, −0.1 k0 ≤ Im β ≤ 0.6 k0 match the ladder of the box with a complex-thickness cladding,
  counted by the argument principle, to a relative error below 4 × 10⁻¹⁵.
- Fields beyond the interface follow the closed-side solution on the stretched coordinate, sin κ(L − Z̃)/sin κL
  behind a Dirichlet wall and cos κ(L − Z̃)/cos κL behind a Neumann one, with L = g + d s̄ and Z̃ = g + ζ̃(ζ) in the
  PML, to 2 × 10⁻¹¹ of the interface field.
- A 20 cm PML with c = 2.4i, |Im κτ| = 326 at the edge of 1.85 k0 ≤ Re β ≤ 3 k0, is searched and its root matches
  the complex-thickness box to 1.2 × 10⁻¹⁵. Out to 4 k0 the scale reaches exp(319) and the rectangle is refused,
  as is one with two such PMLs out to 3.3 k0 (exp(444)) and one whose PML scales by exp(−1885) everywhere.

```sh
cargo test --release --locked --offline --test pml -- --nocapture
```
