# 0007: The group index is certified by two independent estimates

Status: accepted (milestone 2)

## Context

The guide asks for the group index of lossless guided modes "as a certified real quantity, distinct from the
complex local slopes available now".

## Decision

lamina-mode evaluates n_g = dβ/dk0 twice:

1. as the total derivative of β along the root when the vacuum wavelength changes, with every material's
   dispersion included through the materials' own derivatives;
2. as c W / P, the energy velocity's reciprocal, with the stored energy W weighted by ε + k0 ∂ε/∂k0 and
   μ + k0 ∂μ/∂k0. On the solver's own field gauge (E and Z0 H) this is W′/P′ with no unit factors. Layers use
   composite Simpson with Richardson extrapolation and exterior tails are analytic.

The result is certified only if the derivative is real (to round-off) and the two estimates agree within the
caller's tolerance (default 1e-8 in the bindings). Otherwise it fails with both values (`Uncertified`).
`value` is Re(dβ/dk0), signed like β. The result keeps the wavelength sensitivity's conditioning.

## Alternatives

- **Certify by the imaginary part alone:** it detects a complex result, not a wrong one.
- **Energy estimate as the value:** it depends on quadrature; the derivative is exact up to root accuracy, so
  the derivative is reported and the integral certifies it.
