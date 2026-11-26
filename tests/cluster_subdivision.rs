//! Plan discovery splits converged leaves that hold several roots, so one
//! chart over a crowded hyperbolic ladder resolves every root.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, Polarisation, Rotation,
    Stack,
};
use lamina_materials::{Lorentz, LorentzOscillator, PrincipalAxes};
use lamina_mode::*;
use lamina_units::{Length, VacuumWavenumber};
use std::f64::consts::TAU;

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

/// The type II band at 1500 cm⁻¹ over β/k0 ≈ 1.9 … 245, one chart.
fn solve(max_subdivisions: usize) -> SolveReport {
    let problem =
        PropagationProblem::new(hbn(), TAU * 1500.0, Polarisation::TransverseMagnetic).unwrap();
    let domain = problem
        .domain(
            ConformalRegion::new(0.5, 5.5, -0.6, 0.6).unwrap(),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(
            DiscoveryConfig {
                max_subdivisions,
                ..Default::default()
            },
            Default::default(),
            Default::default(),
        )
        .unwrap()
}

#[test]
fn one_chart_resolves_a_crowded_ladder() {
    let report = solve(DiscoveryConfig::default().max_subdivisions);
    assert!(report.is_complete(), "{:?}", report.unresolved());
    // Forty 0.125-wide tiles of the same window find these ten roots.
    assert_eq!(report.observations().len(), 10);
    assert_eq!(report.groups().len(), 10);
    let entry = report.discovery().entries().next().unwrap().unwrap();
    assert!(entry.subdivisions() > 0);
    assert!(entry.estimates().all(|e| e.raw().multiplicity == 1));
}

#[test]
fn without_a_split_budget_clusters_stay_unresolved() {
    let report = solve(0);
    let clusters = report
        .unresolved()
        .into_iter()
        .filter(|u| {
            matches!(
                u.cause,
                UnresolvedCause::Candidate {
                    status: CandidateStatus::UnresolvedCluster,
                    ..
                }
            )
        })
        .count();
    assert!(clusters > 0);
    assert!(!report.is_complete());
}
