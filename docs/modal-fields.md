# Fields at accepted mode points

An owned `ModePoint` can recover homogeneous electric and magnetic profiles with
an explicit supported core backend. It retains its source stack and physical
coordinates, so its report and problem can be dropped first:

```rust,ignore
let fields = point.recover_fields_with(
    lamina_core::Scatter2::new(),
    &sampling,
    lamina_mode::FieldRecoverySettings {
        characteristic_limit: 1e-5,
        ..Default::default()
    },
)?;
let positions = fields.fields.sampling().positions();
let electric = fields.fields.quantity().electric();
let magnetic = fields.fields.quantity().magnetic();
```

Use core `FieldSampling`, `LayerSampling` and `ExteriorSampling`. Layer offsets
and outward exterior distances are unit-aware. Samples retain request order and
region identity; opposite sides of the same interface are distinct. Note that
`LayerSampling::uniform(n)` excludes the right boundary by default: request
`uniform_with_boundaries(n, true, true)` when checking both boundaries.

The normal axis is z and the in-plane wavevector points along x. TE has Ey,
Hx, Hz; TM has Hy, Ex, Ez. Core uses impedance-scaled magnetic-field conventions:
these are arbitrary-amplitude relative fields, not calibrated SI source amplitudes.
At source-free interfaces compare tangential E/H and normal εE/μH. Normal E or H
alone can jump with constitutive contrast. Phasors use exp(+iβx−iωt).

## Gauge, evidence and failures

The core backend's seed supplies an arbitrary nonzero complex gauge. Transfer2
uses a unit right-outgoing primary amplitude; Scatter2 uses its projective
right-outgoing seed. Request `amplitude` to multiply E and H by one common complex
factor. The example explicitly chooses a common nonzero boundary primary field
for comparison; it does not infer a physical power or energy normalization.

`ModeFields` owns its values, resolved positions and source point, records source
and recovery backend names separately, and retains the requested gauge/settings,
original numerical evidence, fresh characteristic and seed boundary defect.
The relative boundary defect is |seed residual| divided by
|secondary|+|xi_left primary|; the ratio is dimensionless and independent of
requested amplitude. The characteristic limit uses the core characteristic gauge
and must suit the physical scale of the device. These are explicit postprocessing
limits; they do not change numerical acceptance.

`FieldRecoveryFailure` owns the unchanged accepted point and a structured cause.
Invalid settings, unavailable branch evidence, core seed/backend/sampling failures,
nonfinite output and excessive residuals remain distinguishable. Recovery uses the
stored ordered exterior roots and rejects detectable inconsistent/nonfinite data;
it does not rerun discovery/refinement, choose principal roots or switch backends.
Stable material behavior remains a caller obligation; arbitrary interior mutation
cannot be certified absent. Unsupported metallic named classification does not
block recovery. No new radiation qualification is inferred from field samples.

Outgoing complex-frequency modes can grow in exterior space. Finite growth is
retained; overflow fails rather than being clamped. No QNM normalization integral,
mode volume, field derivative or branch identity is supplied here. Existing core
normalized-mode consumers retain their normalization semantics.

## Reproduce physical profiles

```sh
cargo test --locked --test modal_fields
cargo run --release --locked --example modal_fields -- /tmp/modal-field-profiles
python scripts/plot_modal_fields.py /tmp/modal-field-profiles
```

Use a new output directory and the optional Matplotlib environment documented in
`plasmon-validation.md`. `fields.csv` records all Cartesian components and
`evidence.txt` records source, gauge and residual diagnostics. The example includes
both interface-plasmon cases, even-Hy and odd-Hy modes of a 40 nm Drude film, and an
outgoing dielectric cavity. PNG/PDF/SVG plots expose relative profiles and leaky
growth. The symmetric-film references independently solve
αd/εd + (αm/εm)tanh(αm d/2)=0 for even Hy and the corresponding coth equation
for odd Hy, with α=-iκ. Tests compare both backend profiles under one complex
scale, all interface conditions, film magnetic parity and decay.

## Unit longitudinal power for lossless guided modes

`ModePoint::guided_power_profile_with(backend, sampling, settings)` returns an owned
`ModalProfile`. A coefficient of magnitude |a| multiplying this profile carries
signed longitudinal power of ±|a|² W/m of invariant width. Take E in V/m and
the stored magnetic components as Z₀H in V/m. The calculation integrates
Re(E × (Z₀H)*)ₓ/(2 Z₀) through the stack in metres, in closed form: lamina-core's
`directional_forms` integrates the mode over each finite layer and wall gap by a
block matrix exponential and over each open exterior from its outgoing waves,
exactly, including at cutoff. A side closed by a PML is refused. The caller's
sampling controls only the returned samples. The backend must implement both
`ModalFieldBackend` and `ModalDerivativeBackend`, as every built-in one does.

The sign preserves forward or backward transport. A common complex multiplier makes
the left-interface primary (Ey for TE, Z₀Hy for TM) real and positive. The returned
`ProfileNormalization::GuidedPower` records that multiplier, phase reference,
seed power, both exterior contributions and signed normalized power. Phase-sensitive
parameter sweeps must maintain this reference or explicitly change gauge.

The supported material class has real positive relative permittivity and
permeability at the physical frequency, real propagation up to numerical tolerance,
and two evanescent exteriors. In a region whose ε and μ are diagonal in the stack
frame this applies to the entries the polarisation sees: μ_xx, μ_zz and ε_yy for TE,
ε_xx, ε_zz and μ_yy for TM. Each field component then carries its own entry: the
tails decay as exp(−α|z|) with α = √(t/n) √(β² − v n k0²) for the entries t, n, v
seen, and normal D = ε_zz E_z and B = μ_zz H_z are continuous across interfaces
with tangential E and H. The power integral uses the fields alone. Ineligible roots, cutoff, ill-conditioned phase or
power, integration budget exhaustion and field recovery failure are explicit errors.
Near cutoff, the analytic tails can dominate even when a finite plotted window
shows little field outside the guide.
