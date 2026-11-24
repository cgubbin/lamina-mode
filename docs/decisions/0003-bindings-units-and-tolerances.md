# 0003: SI at the bindings' boundary, with k0-relative residual limits

Status: accepted (milestone 1)

## Context

The result contract asks for SI at the boundary, with the unit named in every
field: lengths in metres or micrometres, chosen once; β and k0 in rad/m. The
internal cm⁻¹ stays internal.

## Decision

**Lengths are in metres.** Every quantity is then plain SI, consistent with β
and k0 in rad/m, fields in V/m and A/m, and power in W per metre of invariant
width. Field names carry their unit: `thickness_m`, `wavelength_m`,
`beta_rad_per_m`, `electric_v_per_m`, `magnetic_a_per_m`,
`signed_power_w_per_m`. The bindings convert to cm and cm⁻¹ at the call into
lamina-mode and back on return. Magnetic fields leave lamina-mode as Z₀H and are
divided by Z₀.

**Residual limits are relative to k0.** lamina-mode accepts a root when its
Newton correction in the native coordinate is small and its residual |F| is
within `residual_atol + residual_rtol × residual_scale`. Its default is an
absolute 1e-10. The characteristic is linear in wavenumber: across wavelengths
from 1 cm to 400 nm, |F| and |F′| at accepted roots both scale with k0 while
the native step |F|/|F′| stays fixed. An absolute limit suited to lamina-mode's
cm-scale fixtures (k0 ≈ 6 cm⁻¹) is below the round-off floor of micron-scale
photonics (k0 ≈ 6·10⁴ cm⁻¹); there, 4 of 6 modes of a multimode slab stopped
with `NoProgress` although the native corrections had converged.

The bindings therefore set `residual_scale = k0` (in lamina-mode's cm⁻¹ gauge)
and expose `residual_atol` (default 0) and `residual_rtol` (default 1e-11,
about lamina-mode's default ratio at its 1 cm reference wavelength). The field
recovery characteristic limit is scaled the same way: lamina-mode's default
1e-6 at 1 cm, proportional to k0. Root accuracy is still set by the native
correction tolerances, which keep lamina-mode's defaults; with them the
analytic slab ladders match to better than 1e-10 in n_eff at both scales.

## Consequences

- The serialised `refinement` has no `residual_scale`; it is implied by the
  wavelength. Each root still records the residual limit it was accepted
  against.
- A k0-relative default could move into lamina-mode itself; until then the
  bindings own it.
