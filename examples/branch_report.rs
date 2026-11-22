//! Continued dispersion and fields of passive planar waveguides.
#[path = "support/field_modes.rs"]
mod fixtures;
#[allow(dead_code)]
#[path = "support/plasmon.rs"]
mod plasmon;
use fixtures::{ASSUME, Error};
use lamina_core::{
    AnalyticalMaterialStack, Constant, EvaluateMeromorphicMaterial, ExteriorSampling,
    FieldSampling, LayerSampling, Polarisation, Scalar, Scatter2, Stack,
};
use lamina_materials::Drude;
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber as K};
use std::{fs::File, io::Write, path::Path};
const UM: f64 = 1e4;
const SPEED: f64 = 299.792458;
const LOCAL: LocalAnalyticityAssumption =
    LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness;
fn seed(
    stack: AnalyticalMaterialStack<Complex>,
    beta: f64,
    guess: Complex,
    pol: Polarisation,
) -> Result<ModePoint, Error> {
    let k = guess * UM;
    let width = 10.;
    let roots = [stack.left_exterior(), stack.right_exterior()].map(|m| {
        let r = (m.evaluate_relative_permittivity_complex(Scalar(k)) * k * k - (beta * UM).powi(2))
            .sqrt();
        if r.im < 0. { -r } else { r }
    });
    let p = FrequencyProblem::new(stack, beta * UM, pol)?;
    let d = p.domain(
        FrequencyDomainInput {
            bounds: FrequencyRegion::new(k.re - width, k.re + width, k.im - width, k.im + width)?,
            reference: FrequencyReference::Explicit(ExteriorReferenceValues {
                frequency: k,
                kappa: roots,
            }),
            continuation: Default::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        ASSUME,
    )?;
    let report = p.plan(vec![d], vec![])?.solve(
        Default::default(),
        RefinementConfig {
            residual_atol: 1e-6,
            ..Default::default()
        },
        Default::default(),
    )?;
    match report.observations() {
        [one] => Ok(one.clone()),
        _ => Err(format!(
            "seed {k} at {beta}: expected one root, found {}",
            report.observations().len()
        )
        .into()),
    }
}
fn profile(
    out: &Path,
    name: &str,
    id: usize,
    point: &ModePoint,
    widths: &[f64],
) -> Result<(), Error> {
    let length = |v: f64| Length::centimetres(v / UM);
    let mut sampling =
        FieldSampling::new().left_exterior(ExteriorSampling::uniform(81, length(0.5)));
    let mut positions: Vec<_> = (0..81).map(|i| -0.5 + i as f64 / 160.).collect();
    let mut start = 0.;
    for (layer, &width) in widths.iter().enumerate() {
        sampling = sampling.layer(
            layer,
            LayerSampling::uniform_with_boundaries(81, true, true),
        );
        positions.extend((0..81).map(|i| start + width * i as f64 / 80.));
        start += width;
    }
    sampling = sampling.right_exterior(ExteriorSampling::uniform(81, length(0.5)));
    positions.extend((0..81).map(|i| start + i as f64 / 160.));
    let settings = FieldRecoverySettings {
        characteristic_limit: 1e-5,
        ..Default::default()
    };
    let (fields, gauge) = match point.normalize_with(
        Scatter2::new(),
        PoleVariable::K0AtFixedBeta,
        ASSUME,
        Default::default(),
        settings,
    ) {
        Ok(mode) => (mode.fields(&sampling)?, "unit-delta pole".to_owned()),
        Err(e) => (
            point.recover_fields_with(Scatter2::new(), &sampling, settings)?,
            format!("unit amplitude, normalization unavailable: {:?}", e.cause),
        ),
    };
    let mut f = File::create(out.join(format!("{name}-{id}-fields.csv")))?;
    writeln!(
        f,
        "# {gauge}; {:?}",
        point.signature().map_err(Clone::clone)?
    )?;
    writeln!(
        f,
        "depth_um,Ex_re,Ex_im,Ey_re,Ey_im,Ez_re,Ez_im,Hx_re,Hx_im,Hy_re,Hy_im,Hz_re,Hz_im"
    )?;
    let a = fields.fields.quantity();
    for (i, z) in positions.iter().enumerate() {
        write!(f, "{z}")?;
        for v in [
            a.electric().x(),
            a.electric().y(),
            a.electric().z(),
            a.magnetic().x(),
            a.magnetic().y(),
            a.magnetic().z(),
        ] {
            write!(f, ",{},{}", v[i].re, v[i].im)?;
        }
        writeln!(f)?;
    }
    Ok(())
}
fn run_case(
    out: &Path,
    name: &str,
    seed: ModePoint,
    targets: Vec<f64>,
    widths: &[f64],
    samples: &[usize],
) -> Result<(), Error> {
    let spatial = seed.provenance().strategy() == Strategy::Propagation;
    let config = BranchSettings {
        local_assumption: LOCAL,
        domain_assumption: ASSUME,
        radius: if spatial { 0.02 } else { 30. },
        minimum_step: 1e-5,
        retries: 20,
        slope: Default::default(),
        refinement: seed.config(),
    };
    std::fs::write(
        out.join(format!("{name}-settings.txt")),
        format!(
            "{config:#?}\nstrategy: {:?}\nfinite layer widths (µm): {widths:?}\n",
            seed.provenance().strategy()
        ),
    )?;
    let trace = seed.continue_with(Scatter2::new(), &targets, config)?;
    if let Some(stop) = trace.stop {
        return Err(format!("{name} at {}: {:?}", stop.attempted, stop.cause).into());
    }
    let mut f = File::create(out.join(format!("{name}.csv")))?;
    writeln!(
        f,
        "beta_re_cm,beta_im_cm,k0_re_cm,k0_im_cm,residual,slope_re,slope_im,left_re,left_im,right_re,right_im,backend,reference_re_cm,reference_im_cm,relative_error,boundary_defect,backend_difference,residual_limit,correction,correction_limit"
    )?;
    for (i, p) in trace.points.iter().enumerate() {
        let s = p.signature().map_err(Clone::clone)?;
        let d = p
            .local_derivatives_with(
                Scatter2::new(),
                &[PhysicalParameter::K0, PhysicalParameter::Beta],
                LOCAL,
                Default::default(),
            )?
            .implicit_slope(
                if spatial {
                    ImplicitSlopeRequest::BetaPerK0
                } else {
                    ImplicitSlopeRequest::K0PerBeta
                },
                Default::default(),
            )?
            .value;
        let comparison = if name.starts_with("plasmon") {
            let (k, reference) = if spatial {
                (s.k_parallel, plasmon::beta_exact(s.k0.re / plasmon::KP))
            } else {
                (s.k0, plasmon::omega_exact(s.k_parallel.re / plasmon::KP))
            };
            let boundary =
                (s.left + s.right / plasmon::metal(s.k0 / plasmon::KP)).norm() / plasmon::KP;
            let transfer = p
                .local_derivatives_with(
                    lamina_core::Transfer2::new(),
                    &[PhysicalParameter::K0],
                    LOCAL,
                    Default::default(),
                )?
                .characteristic?;
            let backend = (transfer - p.evidence().value).norm() / plasmon::KP;
            plasmon::physical_checks(
                if spatial { "beta" } else { "omega" },
                k / plasmon::KP,
                reference,
                [s.left, s.right],
                boundary,
                backend,
            )?;
            format!(
                "{},{},{},{},{}",
                reference.re * plasmon::KP,
                reference.im * plasmon::KP,
                (k / plasmon::KP - reference).norm() / reference.norm(),
                boundary,
                backend
            )
        } else {
            ",,,,".to_string()
        };
        let ev = p.evidence();
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.k_parallel.re,
            s.k_parallel.im,
            s.k0.re,
            s.k0.im,
            ev.value.norm(),
            d.re,
            d.im,
            s.left.re,
            s.left.im,
            s.right.re,
            s.right.im,
            p.backend(),
            comparison,
            ev.residual_limit,
            ev.correction.ok_or("missing correction")?,
            ev.correction_limit
        )?;
        if samples.contains(&i) {
            profile(out, name, i, p, widths)?;
        }
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("supply a fresh output directory")?;
    let out = Path::new(&dir);
    std::fs::create_dir(out)?;
    for temporal in [false, true] {
        let start = if temporal { 1. } else { fixtures::U };
        let end = if temporal { 2.0 } else { 0.68 };
        run_case(
            out,
            if temporal {
                "plasmon-frequency"
            } else {
                "plasmon-propagation"
            },
            fixtures::interface(temporal)?,
            (1..=80)
                .map(|i| (start + (end - start) * i as f64 / 80.) * plasmon::KP)
                .collect(),
            &[],
            &[],
        )?;
    }
    let widths = [2.0_f64, 1.7].map(|n| {
        let q = ((5. * n).powi(2) - 49.).sqrt();
        2. * (24_f64.sqrt() / q).atan() / q
    });
    let um = |d: f64| Length::centimetres(d / UM);
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(4.), um(widths[0]))
        .analytical_layer(Constant::vacuum(), um(0.7))
        .analytical_layer(Constant::dielectric(1.7 * 1.7), um(widths[1]))
        .finalise();
    for (name, guess) in [("lower", 4.68315648), ("upper", 4.74670390)] {
        run_case(
            out,
            name,
            seed(
                stack.clone(),
                6.5,
                guess.into(),
                Polarisation::TransverseElectric,
            )?,
            (1..=100).map(|i| (6.5 + i as f64 / 100.) * UM).collect(),
            &[widths[0], 0.7, widths[1]],
            &[0, 50, 100],
        )?;
    }
    let ito = Drude::new(
        4.,
        K::inverse_centimetres(3130. * UM / SPEED),
        K::inverse_centimetres(10.7 * UM / SPEED),
    )?;
    let stack = Stack::from_analytical_materials(ito.clone(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(11.68), um(0.110))
        .analytical_layer(ito, um(0.050))
        .finalise();
    run_case(
        out,
        "page",
        seed(
            stack,
            4.,
            Complex::new(779.219, -3.637) / SPEED,
            Polarisation::TransverseMagnetic,
        )?,
        (1..=330).map(|i| (4. + i as f64 / 10.) * UM).collect(),
        &[0.11, 0.05],
        &[20, 34, 50, 220, 263, 300],
    )?;
    std::fs::write(
        out.join("page-model.json"),
        include_str!("support/page_stopped_light.json"),
    )?;
    std::fs::write(
        out.join("coupled_slabs.py"),
        include_str!("support/coupled_slabs.py"),
    )?;
    std::fs::write(out.join("schema.txt"), "modal-branch-report-v1\n")?;
    Ok(())
}
