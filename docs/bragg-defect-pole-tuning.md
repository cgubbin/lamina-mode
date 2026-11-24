# Bragg defect pole tuning

The example places one lossy defect between two quarter-wave mirrors. The mirror indices are 2.10 and 1.45, with four periods on the incident side and five on the exit side. The exterior media are vacuum and the defect relative permittivity is 2.25 + 0.035i. The quarter-wave reference wavelength is 1 cm. The objective is maximum left-incident TE transmission at vacuum wavenumber 6.36 cm⁻¹ and normal incidence. Both tuning methods use defect thicknesses from 0.30 to 0.36 cm.

| Design | Defect thickness (cm) | Defect pole (cm⁻¹) | R | T | A |
| --- | ---: | --- | ---: | ---: | ---: |
| Baseline | 0.333333333 | 6.283084826 − 0.037909140i | 0.799627802 | 0.079826100 | 0.120546097 |
| Target the pole real part | 0.317289634 | 6.360000000 − 0.037598878i | 0.003843897 | 0.420885466 | 0.575270637 |
| Maximize direct transmission | 0.317233649 | 6.360271282 − 0.037598228i | 0.004002312 | 0.420907382 | 0.575090305 |

Pole tuning captures 99.9948% of the direct optimum transmission. The optimized defect thickness differs by 5.60e−5 cm. At the pole-tuned design, reflection is 0.38%, transmission is 42.09%, and absorption is 57.53%. The stopband resonance gives a useful tuning coordinate, and the flux split identifies absorption as the main limit on transmission for this stack.

The baseline discovery reconciles one pole in the finite search region. Subsequent designs use 32 local corrections with no discovery refreshes. The local matches are conditional on the tracked branch. The direct maximum was checked against 101 thicknesses in the shared search interval. This benchmark compares design outcomes at the target for one isolated defect resonance.

```sh
cargo run --release --locked --offline --example bragg_defect_tuning
```
