# 0007: The group index is certified by two independent estimates

Status: accepted

## Context

The group index of a lossless guided mode is a real quantity. The complex local slopes lamina-mode already
returns neither show that it is real nor check its value.

## Decision

lamina-mode evaluates n_g = dβ/dk0 twice:

1. as the total derivative of β along the root when the vacuum wavelength changes, with every material's
   dispersion included through the materials' own derivatives;
2. as c W / P, the energy velocity's reciprocal, with the stored energy W weighted by ε + k0 ∂ε/∂k0 and
   μ + k0 ∂μ/∂k0. On the solver's own field gauge (E and Z0 H) this is W′/P′ with no unit factors. W′ and P′
   are closed-form integrals of the mode over every layer, wall gap and open exterior.

The result is certified only if the derivative is real (to round-off) and the two estimates agree within the
caller's tolerance (default 1e-8 in the bindings). Otherwise it fails with both values (`Uncertified`).
`value` is Re(dβ/dk0), signed like β. The result keeps the wavelength sensitivity's conditioning.

Eligibility is losslessness, not sign: every entry the polarisation sees must be real, finite and non-zero at
the mode's k0, so hyperbolic and negative-permittivity regions are certified. The identity n_g = c W / P
follows from the variational form of the characteristic and holds for any real dispersion. In a lossless
Lorentz medium the weight ∂(k0 ε)/∂k0 = ε∞ + Δε Ω² (Ω² + k0²)/(Ω² − k0²)² is the Brillouin (and, without
loss, Loudon) energy density, positive also where ε < 0. A mode whose power runs against β (a backward mode)
has a negative group index; a zero entry makes the characteristic singular and is refused, as is a complex
entry, named with its region.

## Alternatives

- **Certify by the imaginary part alone:** it detects a complex result, not a wrong one.
- **Energy estimate as the value:** it depends on quadrature; the derivative is exact up to root accuracy, so
  the derivative is reported and the integral certifies it.
