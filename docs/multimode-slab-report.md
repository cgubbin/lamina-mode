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

## Uniaxial slab ladders

A symmetric slab of half-thickness 0.6 cm at a 1 cm wavelength, with a
uniaxial core (ε_o = 4, ε_e = 6.25) and a uniaxial cladding (ε_o = 2,
ε_e = 2.56) sharing an optic axis along x, y or z in turn, is enumerated for
TE and TM and compared with the closed-form conditions

```text
even: q tan(q a)/t_core = α/t_clad      odd: −q cot(q a)/t_core = α/t_clad
q² = s (v n k0² − β²) in the core,  α² = s (β² − v n k0²) in the cladding,  s = t/n
```

with (t, n, v) = (μ_xx, μ_zz, ε_yy) for TE and (ε_xx, ε_zz, μ_yy) for TM. All
six ladders are complete and their roots agree within 1e−8 cm⁻¹ (largest
difference 4.2e−12 cm⁻¹); the light line and upper bound of each are the
cladding and core lines k0 √(v n). With PEC walls 0.15 cm from each interface
(α → α coth(α g) for TE, α tanh(α g) for TM) the ladders above the cladding
lines agree to 1.2e−12 cm⁻¹. An isotropic two-layer guide, one layer magnetic,
rebuilt with every region a diagonal tensor of equal entries gives the
isotropic roots, counts and windows exactly and the isotropic unit-power
fields to 4e−16 of their peak.

```sh
cargo test --locked --release --test diagonal_modes -- --nocapture
```
