# Changelog

The lamina crates (lamina-units, lamina-core, lamina-materials, lamina-mode) are
released in step until 1.0. Each entry names any shared convention it changes.

## 0.1.0 (unreleased)

First release on crates.io.

Conventions fixed by this release (see `docs/modal-fields.md` and the result
contract in `docs/target-development-guide.md`):

- the canonical spectral coordinate is vacuum angular wavenumber `k₀ = 2π/λ₀`
  in `cm⁻¹`; in-plane wavenumber β uses the same unit internally;
- the stack normal is z and the in-plane wavevector points along x; TE has
  Ey, Hx, Hz and TM has Hy, Ex, Ez;
- phasors use `exp(+iβx − iωt)`;
- guided modes are normalised to unit signed longitudinal power per metre of
  invariant width, with the left-interface primary field real and positive.
