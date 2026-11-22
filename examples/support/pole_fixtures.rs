#![allow(dead_code)]
#[path = "mod.rs"]
mod existing;
#[path = "field_modes.rs"]
pub mod fields;
use super::material;
use super::reference as green;
use fields::{ASSUME, c};
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, Constant, Layer, Polarisation, Stack,
};
use lamina_mode::*;
use lamina_units::Length;
pub type Error = Box<dyn std::error::Error>;
pub fn cavity(pol: Polarisation, direct: bool) -> Result<ModePoint, Error> {
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.), Length::centimetres(0.3))
        .finalise();
    let target = Complex::new(std::f64::consts::PI / 0.6, -21_f64.ln() / 1.2);
    if direct {
        return frequency(stack, target, pol);
    }
    let e = |v| ConstantExterior::new(c(v), c(1.)).unwrap();
    let p = ConformalSpectralProblem::new(
        e(1.),
        e(2.25),
        [Layer::new(
            AnalyticalMaterialHandle::new(Constant::dielectric(4.)),
            Length::centimetres(0.3),
        )],
        0.,
        pol,
    )?;
    let anchor = 5.5_f64.ln();
    let d = p.radiation_domain(
        ConformalRegion::new(anchor - 0.18, anchor + 0.18, -2.18, -1.45)?,
        RadiationReference {
            native_coordinate: Complex::new(anchor, -std::f64::consts::FRAC_PI_2),
            coordinate: 5.5,
            choices: [RadiationChoice::OutgoingOrDecaying; 2],
            tolerance: RadiationTolerance::default(),
        },
        ConformalChartConfig::default(),
        ASSUME,
    )?;
    let report = p.plan(vec![d], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig::default(),
        MatchConfig::default(),
    )?;
    let point = one(report)?;
    assert!((point.physical_coordinate() - target).norm() < 1e-8);
    Ok(point)
}
fn one(report: SolveReport) -> Result<ModePoint, Error> {
    if report.observations().len() != 1 {
        return Err(format!("expected one pole: {report:?}").into());
    }
    Ok(report.observations()[0].clone())
}
pub fn frequency(
    stack: AnalyticalMaterialStack<Complex>,
    target: Complex,
    pol: Polarisation,
) -> Result<ModePoint, Error> {
    use lamina_core::{EvaluateMeromorphicMaterial, Scalar};
    let anchor = target + Complex::new(0.03, 0.02);
    let kappa = [stack.left_exterior(), stack.right_exterior()].map(|m| {
        anchor
            * (m.evaluate_relative_permittivity_complex(Scalar(anchor))
                * m.evaluate_relative_permeability_complex(Scalar(anchor)))
            .sqrt()
    });
    let p = FrequencyProblem::new(stack, 0., pol)?;
    let d = p.domain(
        FrequencyDomainInput {
            bounds: FrequencyRegion::new(
                target.re - 0.12,
                target.re + 0.12,
                target.im - 0.12,
                target.im + 0.12,
            )?,
            reference: FrequencyReference::Explicit(ExteriorReferenceValues {
                frequency: anchor,
                kappa,
            }),
            continuation: ContinuationConfig::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        ASSUME,
    )?;
    one(p.plan(vec![d], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig::default(),
        MatchConfig::default(),
    )?)
}
pub fn dispersive(pol: Polarisation, magnetic: bool) -> Result<ModePoint, Error> {
    let layer = material::LinearDispersion::new(3.6, 0.03, 1., if magnetic { 0.015 } else { 0. });
    // Both finite-layer dispersion and a dispersive exterior are present.
    let exterior =
        material::LinearDispersion::new(2.25, 0.02, 1., if magnetic { 0.005 } else { 0. });
    let stack = Stack::from_analytical_materials(Constant::vacuum(), exterior)
        .analytical_layer(layer, Length::centimetres(0.3))
        .finalise();
    let mut target = Complex::new(5.4, -2.5);
    let reference = |k: Complex| {
        use lamina_core::{EvaluateMeromorphicMaterial, Scalar};
        let m = stack.right_exterior();
        let kr = k
            * (m.evaluate_relative_permittivity_complex(Scalar(k))
                * m.evaluate_relative_permeability_complex(Scalar(k)))
            .sqrt();
        let sig = BranchSignature {
            k0: k,
            k_parallel: c(0.),
            left: k,
            right: kr,
        };
        green::Green::new(&stack, sig, true, k, pol).wronskian()
    };
    for _ in 0..30 {
        let h = c(1e-5);
        let step = reference(target) / ((reference(target + h) - reference(target - h)) / (2. * h));
        target -= step;
        if step.norm() < 1e-12 {
            break;
        }
    }
    assert!(reference(target).norm() < 1e-9);
    let point = frequency(stack, target, pol)?;
    assert!((point.physical_coordinate() - target).norm() < 1e-8);
    Ok(point)
}
pub fn guided(pol: Polarisation) -> Result<ModePoint, Error> {
    if pol == Polarisation::TransverseElectric {
        return Ok(existing::guided()?.observations()[0].clone());
    }
    let (u, v) = (0.5_f64, 0.8_f64);
    let q = (4. * u * u - v * v).sqrt();
    let al = (v * v - u * u).sqrt();
    let ar = (v * v - 2.25 * u * u).sqrt();
    let d = ((4. * al / q).atan() + (4. * ar / (2.25 * q)).atan()) / q;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(
            Constant::dielectric(4.),
            Length::centimetres(d / fields::KP),
        )
        .finalise();
    fields::propagation(
        stack,
        u,
        c(v),
        [c(1.), c(2.25)],
        Polarisation::TransverseMagnetic,
    )
}
pub fn cases() -> Vec<(&'static str, Result<ModePoint, Error>)> {
    vec![
        ("guided_te", guided(Polarisation::TransverseElectric)),
        ("guided_tm", guided(Polarisation::TransverseMagnetic)),
        (
            "cavity_tm_direct",
            cavity(Polarisation::TransverseMagnetic, true),
        ),
        (
            "cavity_te_conformal",
            cavity(Polarisation::TransverseElectric, false),
        ),
        (
            "cavity_te_direct",
            cavity(Polarisation::TransverseElectric, true),
        ),
        (
            "cavity_tm_conformal",
            cavity(Polarisation::TransverseMagnetic, false),
        ),
        ("interface_spatial", fields::interface(false)),
        ("interface_temporal", fields::interface(true)),
        ("film_even_H", fields::film(true)),
        ("film_odd_H", fields::film(false)),
        (
            "dispersive_electric",
            dispersive(Polarisation::TransverseElectric, false),
        ),
        (
            "dispersive_magnetic",
            dispersive(Polarisation::TransverseMagnetic, true),
        ),
    ]
}
