//! Searches on polar films that used to end silently incomplete.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, Polarisation, Rotation,
    Stack,
};
use lamina_materials::{Lorentz, LorentzOscillator, PrincipalAxes};
use lamina_mode::*;
use lamina_units::{Length, VacuumWavenumber};
use std::f64::consts::TAU;

const TM: Polarisation = Polarisation::TransverseMagnetic;
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;

/// ε(ν) = ε∞ (1 + (ν_LO² − ν_TO²)/(ν_TO² − ν² − iγν)) at k0 = 2πν.
fn phonon(eps_inf: f64, to: f64, lo: f64, gamma: f64) -> Lorentz<f64> {
    let oscillator = LorentzOscillator::from_delta_epsilon(
        eps_inf * (lo * lo / (to * to) - 1.0),
        VacuumWavenumber::inverse_centimetres(to),
        VacuumWavenumber::inverse_centimetres(gamma),
    )
    .unwrap();
    Lorentz::new(eps_inf, vec![oscillator]).unwrap()
}
/// The c-axis (ε_zz) oscillator of hBN: TO 780, LO 830, γ 4 cm⁻¹.
fn hbn_par() -> Lorentz<f64> {
    phonon(2.95, 780.0, 830.0, 4.0)
}
/// The in-plane oscillator of hBN: TO 1370, LO 1610, γ 5 cm⁻¹.
fn hbn_perp() -> Lorentz<f64> {
    phonon(4.87, 1370.0, 1610.0, 5.0)
}
fn air() -> Region {
    AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into()
}
/// A film in air: isotropic with the c-axis oscillator, or uniaxial hBN.
fn film(uniaxial: bool, nm: f64) -> ModeStack {
    let region: Region = if uniaxial {
        Region::oriented(
            AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(hbn_perp(), hbn_par())),
            Rotation::identity(),
        )
    } else {
        AnalyticalMaterialHandle::new(hbn_par()).into()
    };
    Stack::builder(air(), air())
        .layer(region, Length::nanometres(nm))
        .finalise()
}
/// A ν box (cm⁻¹) in k0, referenced at its centre on the decaying sheet.
fn frequency_input(beta: f64, nu: Complex, half: f64) -> FrequencyDomainInput {
    let centre = nu * TAU;
    let kappa = {
        let r = (centre * centre - beta * beta).sqrt();
        if r.im < 0.0 { -r } else { r }
    };
    FrequencyDomainInput {
        bounds: FrequencyRegion::new(
            (nu.re - half) * TAU,
            (nu.re + half) * TAU,
            (nu.im - half) * TAU,
            (nu.im + half) * TAU,
        )
        .unwrap(),
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: centre,
            kappa: [kappa; 2],
        }),
        continuation: Default::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    }
}

/// ν* = √(ν_LO² − γ²/4) − iγ/2, where the c-axis permittivity vanishes.
fn enz_zero() -> Complex {
    Complex::new((830.0f64 * 830.0 - 4.0).sqrt(), -2.0)
}
fn enz_beta() -> f64 {
    1.4 * TAU * 830.0
}

/// An isotropic polar film searched in complex k0 around ε = 0, where its TM
/// characteristic has a pole: the negative winding there is a violated
/// assumption, never an empty, complete box.
#[test]
fn a_negative_winding_violates_the_pole_free_assumption() {
    let problem = FrequencyProblem::new(film(false, 50.0), enz_beta(), TM).unwrap();
    let domain = problem
        .domain(frequency_input(enz_beta(), enz_zero(), 0.2), ASSUME)
        .unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    let entry = report.discovery().entries().next().unwrap().unwrap();
    assert_eq!(*entry.termination(), Termination::Converged);
    assert!(report.observations().is_empty());
    let negative: Vec<_> = entry
        .leaves()
        .iter()
        .filter(|leaf| leaf.data.root_count < 0)
        .collect();
    assert!(!negative.is_empty());
    let violations: Vec<_> = report
        .unresolved()
        .into_iter()
        .filter_map(|region| match region.cause {
            UnresolvedCause::AssumptionViolated(violation) => Some((region.bounds, violation)),
            _ => None,
        })
        .collect();
    assert_eq!(violations.len(), negative.len());
    for ((bounds, violation), leaf) in violations.iter().zip(&negative) {
        assert_eq!(*bounds, leaf.bounds);
        assert_eq!(
            *violation,
            WindingViolation::Negative {
                count: leaf.data.root_count
            }
        );
    }
    assert!(!report.is_complete());
}
