#![allow(dead_code)]
use lamina_core::{AnalyticalMaterialHandle, Constant, Layer, Polarisation, Stack};
use lamina_mode::*;
use lamina_units::Length;
pub const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
pub fn c(v: f64) -> Complex {
    Complex::new(v, 0.0)
}
pub fn guided() -> Result<SolveReport, Diagnostic> {
    let (k0, beta) = (2.0_f64, 3.5_f64);
    let q = (4.0 * k0 * k0 - beta * beta).sqrt();
    let al = (beta * beta - k0 * k0).sqrt();
    let ar = (beta * beta - 2.25 * k0 * k0).sqrt();
    // Independent TE phase-matching thickness for the target guided mode.
    let thickness = ((al / q).atan() + (ar / q).atan()) / q;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(thickness))
        .finalise();
    let p = PropagationProblem::new(stack, k0, Polarisation::TransverseElectric)?;
    let z = ((al + ar) / (2.0 * k0)).ln();
    let d = p.radiation_domain(
        ConformalRegion::new(z - 0.08, z + 0.08, -0.06, 0.06).unwrap(),
        RadiationReference {
            native_coordinate: c(z),
            coordinate: beta,
            choices: [Some(RadiationChoice::OutgoingOrDecaying); 2],
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
    assert!((report.observations()[0].physical_coordinate() - c(beta)).norm() < 1e-8);
    assert_eq!(
        report.observations()[0]
            .describe(InterpretationConfig::default())?
            .interpretation
            .classification,
        Classification::Classified(ModeType::Guided)
    );
    Ok(report)
}
pub fn constant_cavity() -> Result<SolveReport, Diagnostic> {
    let e = |v| ConstantExterior::new(c(v), c(1.0)).unwrap();
    let p = ConformalSpectralProblem::new(
        e(1.0),
        e(2.25),
        [Layer::new(
            AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
            Length::centimetres(0.3),
        )],
        0.0,
        Polarisation::TransverseElectric,
    )?;
    let anchor = 5.5_f64.ln();
    let d = p.radiation_domain(
        ConformalRegion::new(anchor - 0.18, anchor + 0.18, -2.18, -1.45).unwrap(),
        RadiationReference {
            native_coordinate: Complex::new(anchor, -std::f64::consts::FRAC_PI_2),
            coordinate: 5.5,
            choices: [Some(RadiationChoice::OutgoingOrDecaying); 2],
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
    let expected = Complex::new(std::f64::consts::PI / 0.6, -21.0_f64.ln() / 1.2);
    assert!((report.observations()[0].physical_coordinate() - expected).norm() < 1e-8);
    assert_eq!(
        report.observations()[0]
            .describe(InterpretationConfig::default())?
            .interpretation
            .radiation,
        RadiationQualification::OutgoingContinuation
    );
    Ok(report)
}
pub fn print(report: &SolveReport) -> Result<(), Diagnostic> {
    for point in report.observations() {
        let details = point.describe(InterpretationConfig::default())?;
        println!(
            "k0={} cm^-1, beta={} cm^-1",
            details.data.k0, details.data.beta
        );
        println!(
            "residual={:.3e}, correction={:?} {}",
            point.evidence().value.norm(),
            point.evidence().correction,
            point.provenance().units()
        );
        println!(
            "classification={:?}, radiation={:?}",
            details.interpretation.classification, details.interpretation.radiation
        );
    }
    println!(
        "{} chart(s), {} observation(s), {} duplicate group(s), {} declared scope exclusion(s)",
        report.charts().len(),
        report.observations().len(),
        report.groups().len(),
        report.exclusions().len()
    );
    println!(
        "Coverage is conditional on the finite domains, branch references and supplied analytic assumptions; no global completeness or QNM normalization is claimed."
    );
    Ok(())
}
