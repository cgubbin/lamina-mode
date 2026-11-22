//! Local outgoing frequency pole with an actual Lorentz substrate.
mod support;
use lamina_core::{Constant, Polarisation, Stack};
use lamina_materials::{Lorentz, LorentzOscillator};
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber as K};
fn characteristic(k: Complex) -> Complex {
    let e = support::c(2.25) + 100.0 / (support::c(400.0) - k * k);
    let (yc, yl, yr) = (2.0 * k, k, k * e.sqrt());
    support::c(1.0) - (yc - yl) / (yc + yl) * (yc - yr) / (yc + yr) * (Complex::i() * k * 1.2).exp()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Delta-epsilon=0.25 at resonance 20 cm^-1 gives strength 100 cm^-2.
    let oscillator = LorentzOscillator::from_delta_epsilon(
        0.25,
        K::inverse_centimetres(20.0),
        K::inverse_centimetres(0.0),
    )?;
    let substrate = Lorentz::new(2.25, vec![oscillator])?;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), substrate)
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise();
    let problem = FrequencyProblem::new(stack, 0.0, Polarisation::TransverseElectric)?;
    let domain = problem.domain(
        FrequencyDomainInput {
            bounds: FrequencyRegion::new(4.6, 6.0, -3.2, 0.1)?,
            reference: FrequencyReference::RealRadiation {
                frequency: 5.5,
                choices: [RadiationChoice::OutgoingOrDecaying; 2],
                tolerance: RadiationTolerance::default(),
            },
            continuation: ContinuationConfig::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![
                FrequencyExclusion {
                    region: FrequencyRegion::new(19.9, 20.1, -0.1, 0.1)?,
                    reason: ExclusionReason::MaterialSingularity,
                },
                FrequencyExclusion {
                    region: FrequencyRegion::new(-20.1, -19.9, -0.1, 0.1)?,
                    reason: ExclusionReason::MaterialSingularity,
                },
            ],
        },
        support::ASSUME,
    )?;
    let report = problem.plan(vec![domain], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig::default(),
        MatchConfig::default(),
    )?;
    let mut independent = Complex::new(std::f64::consts::PI / 0.6, -21.0_f64.ln() / 1.2);
    let h = support::c(1e-5);
    for _ in 0..10 {
        independent -= characteristic(independent)
            / ((characteristic(independent + h) - characteristic(independent - h)) / (2e-5));
    }
    assert!((report.observations()[0].physical_coordinate() - independent).norm() < 1e-8);
    let e: f64 = 2.25 + 100.0 / (400.0 - 5.5 * 5.5);
    let reflection = (2.0 - 1.0) / (2.0 + 1.0) * (2.0 - e.sqrt()) / (2.0 + e.sqrt());
    let frozen = Complex::new(std::f64::consts::PI / 0.6, reflection.ln() / 1.2);
    assert!((independent - frozen).norm() > 1e-3);
    assert_eq!(
        report.observations()[0]
            .describe(InterpretationConfig::default())?
            .interpretation
            .radiation,
        RadiationQualification::OutgoingContinuation
    );
    support::print(&report)?;
    Ok(())
}
