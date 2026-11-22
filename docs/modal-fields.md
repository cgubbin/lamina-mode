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
