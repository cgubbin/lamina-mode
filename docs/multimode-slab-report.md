# Differentiable multimode slab benchmark

Two forward TE modes propagate coherently in a symmetric slab with core index 2, cladding index 1, thickness 0.5 cm, and vacuum wavelength 1 cm. The supplied complex amplitudes are 1 and 0.3 + 0.6i in the signed unit-power, left-interface phase convention. The output is sampled at a quarter of the core thickness and at a propagation distance of 0.37 cm. The sample follows the core fraction as thickness changes. The propagation distance and modal coefficients stay fixed.

## Beating and output

The even and odd slab phase equations independently predict a beat length of **2.292837552 cm**. The two enumerated propagation constants agree with the phase-equation roots within 1e−8 cm⁻¹. The squared magnitude of the coherent electric field repeats after one beat length to relative tolerance 1e−8 and changes by more than 1% at the half beat. Here intensity means squared electric-field magnitude at the sample.

The complex output field is −166.519817 − 472.183015i.

| Perturbation | Complex field derivative per cm | Intensity derivative per cm |
| --- | ---: | ---: |
| Core thickness | 1184.879252 − 7.597998i | −387436.462 |
| Vacuum wavelength | −2345.661675 + 899.850114i | −68589.573 |

## Independent derivative checks

Each central difference enumerates and normalizes new modes on both sides of the perturbation before propagating them. Errors are absolute differences divided by the larger of one and the jet derivative magnitude.

| Perturbation | Step (cm) | Complex field error | Intensity error |
| --- | ---: | ---: | ---: |
| Thickness | 1e−3 | 8.63e−6 | 1.27e−5 |
| Thickness | 1e−4 | 8.79e−8 | 1.32e−7 |
| Thickness | 1e−5 | 5.61e−9 | 4.36e−9 |
| Wavelength | 1e−3 | 6.30e−6 | 4.71e−7 |
| Wavelength | 1e−4 | 6.31e−8 | 4.09e−9 |
| Wavelength | 1e−5 | 4.45e−9 | 1.18e−7 |

The regression requires both scaled errors below 2e−5 at the smallest step. The wavelength intensity comparison reaches its lowest error at the middle step.

```sh
cargo run --locked --offline --example multimode_slab
cargo test --locked --offline --example multimode_slab -- --nocapture
```
