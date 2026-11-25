# Completeness and pairing across a step

Two symmetric slabs share perfect electric walls at ±0.8 cm about their common core centre: relative permittivity
4 in the core and 2.25 between core and walls, core half-widths 0.3 cm (left) and 0.2 cm (right), vacuum wavelength
1 cm. They are joined at x = 0. Each side's basis is solved in the forward window Re β ∈ [−0.05 k0, 2.1 k0],
Im β ∈ [−0.05 k0, 12 k0], and the right slab's backward partners in its mirror window; every basis is complete.

The fundamental even mode is incident from the left. With N even modes per side, pairing the fields at x = 0 with the
left modes f_m and their mirror images gives

```text
a = U t,   r = V t,   U_mn = ⟨f_m, g_n⟩,   V_mn = ⟨f_m, M g_n⟩,
```

where ⟨·,·⟩ is the unconjugated pairing and M the mirror x → −x. U comes from the forward bases; V from the
right slab's backward basis, whose members are i-multiples of M g_n (the factor is read from their fields). The
reference repeats the expansion with analytic layered-box modes and closed-form piecewise overlaps, and its value at
N = 120 stands for the converged limit. The truncation error is the reference's distance from that limit; the
predicted floor is κ_N δ, with κ_N the condition number of the N × N matrix and δ the larger of the quadrature
change and the bases' self-pairing defects.

The pairing matrices agree with their closed forms to 8.5e−12 W/m (TE) and 4.7e−12 W/m (TM), with δ = 1.4e−10 and
1.3e−10. The error is max(|r − r∞|, |t − t∞|) for the fundamental mode:

| N | TE error | TE truncation | TE κ_N δ | TM error | TM truncation | TM κ_N δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.265e−2 | 1.265e−2 | 1.4e−10 | 4.012e−3 | 4.012e−3 | 1.3e−10 |
| 4 | 4.703e−4 | 4.703e−4 | 1.6e−10 | 3.023e−3 | 3.023e−3 | 1.4e−10 |
| 6 | 2.498e−4 | 2.498e−4 | 1.6e−10 | 1.786e−3 | 1.786e−3 | 1.4e−10 |
| 8 | 5.384e−5 | 5.384e−5 | 1.6e−10 | 6.677e−4 | 6.677e−4 | 1.4e−10 |
| 10 | 1.264e−5 | 1.264e−5 | 1.6e−10 | 3.766e−4 | 3.766e−4 | 1.4e−10 |

The condition numbers stay between 1.0 and 1.15, so up to N = 10 the truncation error dominates the predicted floor
by five orders or more and the error follows the reference's convergence. At every N the expansion from the bases
differs from the reference's by at most 1.5e−13, below the floor. The check passes when the error lies within a
factor 10 of max(truncation, κ_N δ) at every N.

```sh
cargo test --release --test step_check -- --nocapture
```
