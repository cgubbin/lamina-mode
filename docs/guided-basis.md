# Enumerating a guided basis

Run the example with `cargo run --locked --example guided_basis`. It builds an asymmetric dielectric slab at a fixed vacuum wavelength, enumerates six forward TE modes and samples each with unit longitudinal power per metre of invariant width.

Pass a stack and a `GuidedWindow` to `enumerate_guided_modes`. Set the wavelength with a typed length and specify the open interval of |β| and cutoff margin in cm⁻¹. Select TE or TM and forward or backward propagation. The caller's analytic declaration covers each local characteristic correction. Results use Scatter2; `enumerate_guided_modes_with` takes another 2×2 backend, Transfer2 for example.

Each region is isotropic or carries ε and μ diagonal in the stack frame, with principal axes along x, y and z (a uniaxial or biaxial medium whose axes lie on the stack axes). A polarisation sees three entries in each region: TE (Ey, Hx, Hz) sees μ_xx, μ_zz and ε_yy, TM (Hy, Ex, Ez) sees ε_xx, ε_zz and μ_yy. Writing them t, n and v, the primary field obeys −(ψ′/t)′ + (β²/n) ψ = k0² v ψ, so

```text
q² = s (line² − β²),   line = k0 √(v n),   s = t/n
TE: line = k0 √(ε_yy μ_zz),  s = μ_xx/μ_zz
TM: line = k0 √(ε_zz μ_yy),  s = ε_xx/ε_zz
```

and the exterior decay constant is α = √s √(β² − line²). A uniaxial cladding therefore has different TE and TM light lines. Materials may be dispersive; the three entries each polarisation sees must be finite, real and positive at the stated wavelength in every region, which keeps the oscillation count monotone. An entry that is not, such as a negative ε_zz in a hyperbolic layer for TM, is refused with `EnumerationError::Material`, naming the region and the entry; entries the polarisation does not see are not inspected. TE and TM of a stack with off-diagonal entries, and every mode at once (`ModeFamily::All`), are counted through the 4×4 backends' characteristic; see `docs/hybrid-modes.md`.

The eligible window starts above the light lines of the open exteriors by the requested positive margin and ends at or below the highest material light line. A side closed by a wall has no light line: its count starts or ends with the wall condition, ψ = 0 or ∂ψ/∂z = 0, and its gap counts as a layer of the exterior material. Between two walls the eligible window is the requested one below the upper bound, and the basis holds every real-β mode there, including the radiation-like modes below the exterior light lines; evanescent and complex modes are not enumerated. The report retains the request, exterior threshold, physical upper bound and eligible interval. `eligible == None` indicates an empty intersection. A light-line state with a nondecaying exterior tail falls outside the bound basis.

Read `expected`, `modes` and `unresolved` together. A complete result has one checked, decaying physical point per counted state in the eligible open interval. A complete zero count means the interval contains no states. An undecidable endpoint phase gives an unknown count. A work limit or failed optical correction yields partial coverage, with an interval and cause for each gap. Accepted points are ordered by decreasing |β| in either direction. Each owns its physical source and supports field sampling, unit-power normalization and local sensitivities.

The numerical count uses the unwrapped angle of the primary field and its continuous derivative weighted by 1/t across all layers, including the decaying half-space or wall boundary conditions. The endpoint floor difference counts interior crossings. `phase_tolerance` is in cycles of π; an endpoint close to an integer within the tolerance and roundoff allowance is unresolved. Narrowing the tolerance alone does not establish accuracy for poorly conditioned stacks. The completeness status applies to the recorded eligible window, polarization, numerical precision and material class. Cutoff guard bands and other propagation channels require separate treatment.
