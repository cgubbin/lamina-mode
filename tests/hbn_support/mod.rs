//! Polar dielectrics shared by the tests: hBN (uniaxial, optic axis along z)
//! and the isotropic Lorentz of one phonon axis, in films in air.
//! Spectroscopic wavenumbers ν̃ in cm⁻¹; k0 = 2πν̃.
#![allow(dead_code)]
use lamina_core::{Constant, Polarisation, Stack};
use lamina_mode::*;
use lamina_units::Length;

pub fn k0_of(nu: f64) -> f64 {
    std::f64::consts::TAU * nu
}

/// ε(ν̃) = ε∞ (1 + (ν̃_LO² − ν̃_TO²)/(ν̃_TO² − ν̃² − iγν̃)).
pub fn phonon(eps_inf: f64, to: f64, lo: f64, gamma: f64) -> lamina_materials::Lorentz<f64> {
    use lamina_materials::{Lorentz, LorentzOscillator};
    use lamina_units::VacuumWavenumber;
    let oscillator = LorentzOscillator::from_delta_epsilon(
        eps_inf * (lo * lo / (to * to) - 1.0),
        VacuumWavenumber::inverse_centimetres(to),
        VacuumWavenumber::inverse_centimetres(gamma),
    )
    .unwrap();
    Lorentz::new(eps_inf, vec![oscillator]).unwrap()
}

/// hBN with in-plane (⊥) and out-of-plane (∥) damping rates in cm⁻¹.
pub fn hbn(gamma_perp: f64, gamma_par: f64) -> Region {
    use lamina_core::{AnalyticalTensorMaterialHandle, Rotation};
    use lamina_materials::PrincipalAxes;
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            phonon(4.87, 1370.0, 1610.0, gamma_perp),
            phonon(2.95, 780.0, 830.0, gamma_par),
        )),
        Rotation::identity(),
    )
}

pub fn air() -> Region {
    lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into()
}

pub fn hbn_film(film: Region, nm: f64) -> ModeStack {
    Stack::builder(air(), air())
        .layer(film, Length::nanometres(nm))
        .finalise()
}

/// The TM root nearest `beta` at ν̃ of a stack in air, from one
/// argument-principle box of half-width `radius` about it on the in-plane
/// chart β² = k0² (s + t + r²/t), t = e^{2z}, s = (χ_L + χ_R)/2,
/// r = (χ_L − χ_R)/4 with χ_L = 1 and χ_R = `right` at ν̃.
pub fn root_on(stack: &ModeStack, nu: f64, beta: f64, radius: f64, right: f64) -> ModePoint {
    let k0 = k0_of(nu);
    let (s, r) = ((1.0 + right) / 2.0, (1.0 - right) / 4.0);
    // t² + (s − n²) t + r² = 0; the larger root keeps both exteriors bound.
    let b = s - (beta * beta) / (k0 * k0);
    let z = 0.5 * ((-b + (b * b - 4.0 * r * r).sqrt()) / 2.0).ln();
    let problem =
        PropagationProblem::new(stack.clone(), k0, Polarisation::TransverseMagnetic).unwrap();
    let domain = problem
        .domain(
            ConformalRegion::new(z - radius, z + radius, -radius, radius).unwrap(),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    report
        .observations()
        .iter()
        .min_by(|a, b| {
            let d = |p: &ModePoint| (p.physical_coordinate().re - beta).abs();
            d(a).total_cmp(&d(b))
        })
        .unwrap_or_else(|| panic!("no root near {beta} at {nu}: {:?}", report.unresolved()))
        .clone()
}

pub fn root(stack: &ModeStack, nu: f64, beta: f64, radius: f64) -> ModePoint {
    root_on(stack, nu, beta, radius, 1.0)
}

/// An isotropic lossless polar dielectric.
pub fn isotropic(eps_inf: f64, to: f64, lo: f64) -> Region {
    lamina_core::AnalyticalMaterialHandle::new(phonon(eps_inf, to, lo, 0.0)).into()
}
