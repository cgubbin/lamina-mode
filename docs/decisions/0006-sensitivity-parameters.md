# 0006: Sensitivity parameters, units and observation frames

Status: accepted (milestone 2)

## Decision

A parameter names what varies; every rate is per its SI unit, recorded in `parameter_unit`:

| Parameter | Meaning | Unit |
| --- | --- | --- |
| `thickness(layer)` | one layer's thickness, the interfaces to its left fixed | m |
| `interface_position(i)` | one interface (0..n from the left), its neighbours fixed | m |
| `translation()` | the whole stack along z | m |
| `wavelength()` | vacuum wavelength, every material re-evaluated (dispersion included) | m |
| `permittivity(region)`, `permeability(region)` | a region's relative ε or μ at the operating wavelength | 1 |
| `sellmeier_constant(region)`, `sellmeier_coefficient(region, j)` | a Sellmeier parameter of that region | 1 |
| `sellmeier_pole_wavelength(region, j)` | a Sellmeier pole wavelength | m |

Regions run left exterior (0), layers (1..n), right exterior (n + 1). dβ/dp is in rad/m per unit; field
derivatives in V/m and A/m per unit; fraction rates per unit.

Material-model parameters reach lamina-mode as constitutive rates: the model supplies ∂ε/∂p at the operating
wavelength (lamina-materials owns that), and lamina-mode differentiates the mode along it.

**Observation frames.** `material_attached` (the default) keeps each sample's offset from its region's left
interface; exterior samples keep their distance from their interface. `fixed_lab` keeps z fixed; a fixed-lab
sample on a moving interface has no derivative (the field is discontinuous there) and is reported in
`sample_failures`, while the other samples still return.

**Fractions** are each region's signed power over the total signed power, so they sum to one for forward and
backward modes alike; their rates sum to zero.
