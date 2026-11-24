# Guided slab sensitivity report

The guided solver differentiates the propagation constant and signed unit-power electric and magnetic fields of a lossless planar mode. The checks below compare those derivatives with independently solved and normalized neighbouring modes.

## Two-layer slab benchmark

The base vacuum wavelength is 1 cm. The exterior relative permittivities are 1 and 1.44. The finite layers are 0.15 cm and 0.13 cm thick. Their relative permittivities at the base wavelength are 4 and 2.89. Relative permeability is one throughout. The first finite layer has a permittivity slope of 0.08 cm with respect to vacuum wavenumber.

The wavelength case varies the vacuum wavelength and evaluates the material dispersion at each perturbed wavelength. The index case varies the first layer index at fixed wavelength. The interface case moves the internal boundary while holding both exterior boundaries fixed. Each central difference independently enumerates and corrects the highest-β root on both sides and applies the existing signed unit-power and left-interface phase conventions.

Each error is the complex difference between the jet and central-difference derivatives, divided by the larger of one and the central-difference magnitude in the reported units. Field errors are maxima over all six components at the layer centres and sided interfaces. Steps are in centimetres for wavelength and interface motion and dimensionless for index.

| Polarization | Perturbation | β error at step 10⁻⁵ | Field error at 10⁻³ | Field error at 10⁻⁴ | Field error at 10⁻⁵ |
| --- | --- | ---: | ---: | ---: | ---: |
| TE | Dispersive wavelength | 3.96e−9 | 2.11e−5 | 2.11e−7 | 1.04e−6 |
| TE | Index | 1.22e−8 | 2.34e−6 | 1.79e−8 | 1.11e−7 |
| TE | Internal interface | 1.64e−9 | 9.75e−3 | 9.88e−5 | 2.39e−6 |
| TM | Dispersive wavelength | 1.69e−9 | 8.18e−5 | 5.97e−7 | 4.20e−6 |
| TM | Index | 6.93e−9 | 8.13e−5 | 9.47e−8 | 5.27e−6 |
| TM | Internal interface | 2.69e−9 | 1.08e−4 | 1.08e−6 | 5.66e−8 |

Both errors decrease from the first to the second step in all six cases. Some field errors rise at the smallest step, consistent with amplification of the finite accuracy of the root correction and independent power quadrature. The regression requires final scaled errors below 2e−7 for β and 2e−5 for the fields.

## Coordinate and normalization checks

- Under rigid translation, the β rate and all co-moving field tangents vanish within 1e−12. Fixed-lab tangents in a smooth layer agree with translated sampling to a scaled tolerance of 1e−8. Fixed-lab samples at moving material interfaces return a sided diagnostic and retain their base fields.
- The regional powers sum to unity and their rates sum to zero within 1e−12 in the forward normalization benchmark. Rescaling the backend seed by −2 + 3i preserves the normalized tangent to a scaled tolerance of 1e−8. The left-interface phase stays real to 1e−9 in the tested derivatives.
- The single-thickness β derivative agrees with the existing implicit slope within 1e−10 in both propagation directions. Independently corrected perturbed slab roots agree within 1e−6.

## Slab propagation example

The example enumerates six TE modes of an asymmetric slab and evaluates the outgoing field contribution of each mode with unit input amplitude over 0.5 cm. It applies the propagation phase and its derivative to the normalized centre field and its tangent. The first mode has β = 12.485931298 cm⁻¹ and dβ/dd = 0.072873 cm⁻², where d is the core thickness.

The field result owns the accepted point, the perturbation and conditioning evidence, the sided base fields and tangents, and the regional powers. These quantities remain available after enumeration and field recovery have finished. Excitation coefficients from a physical input field use the [guided overlaps](guided-overlap-report.md).

```sh
cargo run --locked --offline --example guided_basis
cargo test --locked --offline --test guided_basis -- --nocapture
```
