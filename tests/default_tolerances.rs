//! Default discovery and refinement tolerances scale with the wavenumber, so
//! a mid-infrared search needs no hand-scaled settings.
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

fn phonon(eps_inf: f64, to: f64, lo: f64, gamma: f64) -> Lorentz<f64> {
    let oscillator = LorentzOscillator::from_delta_epsilon(
        eps_inf * (lo * lo / (to * to) - 1.0),
        VacuumWavenumber::inverse_centimetres(to),
        VacuumWavenumber::inverse_centimetres(gamma),
    )
    .unwrap();
    Lorentz::new(eps_inf, vec![oscillator]).unwrap()
}
/// A 100 nm hBN film in air (Caldwell et al. 2014 phonons).
fn hbn() -> ModeStack {
    let air = || -> Region { AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into() };
    let film = Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            phonon(4.87, 1370.0, 1610.0, 5.0),
            phonon(2.95, 780.0, 830.0, 4.0),
        )),
        Rotation::identity(),
    );
    Stack::builder(air(), air())
        .layer(film, Length::nanometres(100.0))
        .finalise()
}

fn solve(
    k0: f64,
    discovery: DiscoveryConfig,
    refinement: RefinementConfig,
) -> (SolveReport, Vec<String>) {
    let problem = PropagationProblem::new(hbn(), k0, TM).unwrap();
    // The first hyperbolic branch of the type II band, β ≈ 26 k0.
    let domain = problem
        .domain(
            ConformalRegion::new(3.0, 3.5, -0.6, 0.6).unwrap(),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(discovery, refinement, MatchConfig::default())
        .unwrap();
    let statuses = report.charts()[0]
        .candidates
        .iter()
        .flatten()
        .map(|c| format!("{:?}", c.status))
        .collect();
    (report, statuses)
}

#[test]
fn default_tolerances_resolve_a_mid_infrared_root() {
    let k0 = TAU * 1500.0;
    // With the old absolute defaults the residual at the root is below the
    // limit 1e-10 only to round-off: the root is accepted against its
    // round-off floor (before that was judged, it stagnated).
    let absolute = (
        DiscoveryConfig {
            zero_tolerance: Some(DEFAULT_ZERO_TOLERANCE),
            ..Default::default()
        },
        RefinementConfig {
            residual_atol: Some(DEFAULT_RESIDUAL_ATOL),
            ..Default::default()
        },
    );
    let (old, statuses) = solve(k0, absolute.0, absolute.1);
    assert_eq!(statuses, ["Accepted"]);
    let [at_floor] = old.observations() else {
        panic!()
    };
    let evidence = at_floor.evidence();
    let floor = evidence.residual_floor.unwrap();
    assert!(floor > DEFAULT_RESIDUAL_ATOL && evidence.residual_limit == floor);
    assert!(evidence.value.norm() > DEFAULT_RESIDUAL_ATOL);
    let (report, statuses) = solve(k0, Default::default(), Default::default());
    assert_eq!(statuses, ["Accepted"]);
    assert!(report.is_complete());
    let [point] = report.observations() else {
        panic!()
    };
    assert_eq!(
        point.evidence().residual_limit,
        DEFAULT_RESIDUAL_ATOL * k0 / TAU
    );
    assert_eq!(point.evidence().residual_floor, None);
    let (a, b) = (at_floor.physical_coordinate(), point.physical_coordinate());
    assert!((a - b).norm() < 1e-12 * b.norm(), "{a} {b}");
    assert_eq!(
        point.config().residual_atol,
        Some(DEFAULT_RESIDUAL_ATOL * k0 / TAU)
    );
}

#[test]
fn scaled_defaults_are_the_old_values_at_a_one_centimetre_wavelength() {
    assert_eq!(
        DiscoveryConfig::default().zero_tolerance_at(TAU),
        DEFAULT_ZERO_TOLERANCE
    );
    assert_eq!(
        RefinementConfig::default().residual_atol_at(TAU),
        DEFAULT_RESIDUAL_ATOL
    );
    assert_eq!(
        RefinementConfig::default().residual_atol_at(1e4 * TAU),
        1e4 * DEFAULT_RESIDUAL_ATOL
    );
    // A given value is used as given.
    let given = RefinementConfig {
        residual_atol: Some(3e-7),
        ..Default::default()
    };
    assert_eq!(given.residual_atol_at(1e4 * TAU), 3e-7);
}

/// The bound ENZ mode of a 50 nm Lorentz film in complex k0, with default
/// settings, where the residual limit used to have to be scaled by hand.
#[test]
fn frequency_defaults_scale_with_k0() {
    let air = || -> Region { AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into() };
    let film: Region = AnalyticalMaterialHandle::new(phonon(2.95, 780.0, 830.0, 4.0)).into();
    let stack = Stack::builder(air(), air())
        .layer(film, Length::nanometres(50.0))
        .finalise();
    let beta = 1.4 * TAU * 830.0;
    let problem = FrequencyProblem::new(stack, beta, TM).unwrap();
    let centre = Complex::new(829.565, -2.0) * TAU;
    let half = 0.3 * TAU;
    let kappa = {
        let r = (centre * centre - beta * beta).sqrt();
        if r.im < 0.0 { -r } else { r }
    };
    let input = FrequencyDomainInput {
        bounds: FrequencyRegion::new(
            centre.re - half,
            centre.re + half,
            centre.im - half,
            centre.im + half,
        )
        .unwrap(),
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: centre,
            kappa: [kappa; 2],
        }),
        continuation: Default::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    };
    let domain = problem.domain(input, ASSUME).unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    assert!(report.is_complete(), "{:?}", report.unresolved());
    let [point] = report.observations() else {
        panic!()
    };
    assert!((point.physical_coordinate() / TAU - Complex::new(829.567, -1.999)).norm() < 1e-2);
}
