# Differentiable guided power overlap

Two fundamental TE modes at vacuum wavelength 1 cm are recovered from asymmetric slabs with relative permittivities 1, 4 and 1.44 from left to right. The core widths are 0.25 cm and 0.30 cm, and their left interfaces are registered at 0 cm and 0.07 cm. Both modes use signed unit power and the positive left-interface primary field phase.

The cross-section power overlap is **0.908555968 W/m**. Its derivative with respect to the second slab width, with both left interfaces fixed, is **−0.791081889 W/m per cm**. Imaginary components are below the displayed precision.

The base pairing used 773 samples. The derivative used 1541 samples. The last refinement changed the base integral by 1.527e−10 W/m and the derivative by 5.168e−11 W/m per cm. The complete integration includes analytic tails on both sides.

| Thickness step (cm) | Central difference (W/m per cm) | Absolute error (W/m per cm) |
| ---: | ---: | ---: |
| 1e−3 | −0.791077990 | 3.898e−6 |
| 1e−4 | −0.791081853 | 3.567e−8 |
| 1e−5 | −0.791081812 | 7.619e−8 |

Each central difference uses newly enumerated and normalized modes at both displaced widths. The regression also checks relative translation and shared wavelength against independent evaluations with a 1e−4 cm step and a 2e−6 absolute error bound. Common translation produces a zero overlap derivative within 1e−10 W/m per cm. Coincident interfaces that move together preserve the self-overlap derivative within 1e−8 W/m per cm. When they separate, the API returns a typed diagnostic alongside the base overlap.

```sh
cargo run --locked --offline --example guided_overlap
cargo test --locked --offline --test guided_basis
```
