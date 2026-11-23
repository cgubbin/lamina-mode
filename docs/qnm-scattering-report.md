# Planar reflection and transmission from QNM poles

The two-port matrix uses outgoing rows and incoming columns, with left before right. Channel amplitudes use (E_y) for TE and impedance-scaled (H_y) for TM, in the exp(−iωt) convention. Reference planes are the two exterior interfaces. The pole contribution uses the unconjugated field product from the normalized QNM and the incident channel admittance. The residue is in inverse centimetres for vacuum wavenumber (k_0), with a separate angular-frequency conversion.

## Residues

For each case, the four predicted residues were compared with contours of direct Scatter2 reflection and transmission on the same exterior sheets. The contours used radii 0.03 and 0.015 cm⁻¹ and 64 and 128 samples, respectively. The largest absolute discrepancy across the symmetric and asymmetric cases was 1.76e−13 cm⁻¹. The tests enforce a 1e−7 cm⁻¹ bound and check rank one and reciprocity after accounting for channel admittance.

## Reconstruction over a real window

Both cases use a 0.3 cm slab of relative permittivity 4. The regular contribution is a degree-10 Chebyshev expansion fitted after subtracting the selected principal part from 22 direct scattering samples. The validation uses 103 separate points over (k_0 = 4.2) to (6.3 mathrm{cm}^{-1}). Both channels propagate throughout this window.

| Case | Pole k₀ (cm⁻¹) | Maximum held-out amplitude error, pole alone | Maximum held-out amplitude error, with regular term | Maximum flux-balance defect |
| --- | --- | ---: | ---: | ---: |
| Symmetric vacuum exteriors, TE, β = 0 | 5.235987756 − 1.831020481i | 1.214 | 5.54e−12 | 2.25e−12 |
| Exterior permittivities 1 and 2.25, TM, β = 1.3 cm⁻¹ | 5.296237177 − 2.542369821i | 2.045 | 1.30e−11 | 2.12e−11 |

At (k_0=5.25 mathrm{cm}^{-1}), the symmetric case gives (R=0.000039757) and (T=0.999960243). The asymmetric case gives (R=0.036693183) and (T=0.963306817) for left incidence. Power uses the real channel admittance ratio, while reconstruction and its error use complex field amplitudes.

A constant regular term fails the amplitude tolerance in both cases. These errors characterize the tested window and grid. The caller selects the poles and declares positive constant exterior parameters on the continuation to the window. A separate continuation API supports dispersive exterior models in a declared analytic region with a verified outgoing anchor. Real channels must be lossless and propagating.

```sh
cargo test --locked --offline --test pole_normalization -- --nocapture
```

## Dispersive and absorbing validation

For a right exterior with relative permittivity 2.1 + 0.025 cm × k₀ and β = 1.2 cm⁻¹, the maximum left-incidence amplitude errors against the analytic slab formula were 7.58e−12 for the real fit and 1.36e−14 for the 128-node contour. The continuation API rejected a flipped pole root, a declared threshold exclusion, and a contour outside the supplied region.

The absorbing slab with core relative permittivity 4 + 0.12i absorbed 10.94% of incident power at k₀ = 5.25 cm⁻¹, with zero measured passivity excess across the held-out window.

The [interactive HTML report](qnm-scattering-report.html) also includes the equal-budget comparison and the complete matrix diagnostics.
