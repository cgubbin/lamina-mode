//! A lossy mode that crosses the light line leaves the proper sheet; the
//! full-period plan covers the improper sheets and points name their sector.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, Polarisation, Rotation,
    Stack,
};
use lamina_materials::{Lorentz, LorentzOscillator, PrincipalAxes};
use lamina_mode::*;
use lamina_units::{Length, VacuumWavenumber};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

fn phonon(eps_inf: f64, to: f64, lo: f64, gamma: f64) -> Lorentz<f64> {
    let oscillator = LorentzOscillator::from_delta_epsilon(
        eps_inf * (lo * lo / (to * to) - 1.0),
        VacuumWavenumber::inverse_centimetres(to),
        VacuumWavenumber::inverse_centimetres(gamma),
    )
    .unwrap();
    Lorentz::new(eps_inf, vec![oscillator]).unwrap()
}
/// A 50 nm hBN film in air (Caldwell et al. 2014 phonons).
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
        .layer(film, Length::nanometres(50.0))
        .finalise()
}

/// The fundamental TM mode just above the ε_zz = 0 point, past the light line.
const MODE: Complex = Complex::new(-2.6976, 2.2945);

fn near(report: &SolveReport) -> Vec<ModePoint> {
    report
        .observations()
        .iter()
        .filter(|p| (p.native_coordinate() - MODE).norm() < 1e-3)
        .cloned()
        .collect()
}

#[test]
fn full_period_domains_find_a_mode_off_the_proper_sheet() {
    let k0 = TAU * 832.125;
    let problem = PropagationProblem::new(hbn(), k0, Polarisation::TransverseMagnetic).unwrap();
    let reference = ReferenceSelection::Sheet(CoordinateSheet::Principal);
    let config = ConformalChartConfig::default();
    let assume = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
    let solve = |domains: Vec<Domain>| {
        problem
            .plan(domains, vec![])
            .unwrap()
            .solve(Default::default(), Default::default(), Default::default())
            .unwrap()
    };
    let domains = problem
        .full_period_domains([-4.0, 1.0], 0.05, reference, config, assume)
        .unwrap();
    let (low, high) = domains[1]
        .provenance()
        .requested_bounds()
        .imaginary_bounds();
    assert!((low - (FRAC_PI_2 + 0.05)).abs() < 1e-12);
    assert!((high - (PI + FRAC_PI_2 - 0.05)).abs() < 1e-12);
    // The proper strip alone misses the mode.
    let proper = solve(domains[..1].to_vec());
    assert!(near(&proper).is_empty());
    let full = solve(domains);
    assert!(full.is_complete(), "{:?}", full.unresolved());
    let [point] = near(&full).try_into().unwrap();
    assert_eq!(point.chart(), 1);
    assert_eq!(
        point.signature().unwrap().sectors(),
        [Some(ExteriorSector::IncomingImproper); 2]
    );
    assert_eq!(
        ExteriorSector::IncomingImproper.sheet(),
        ExteriorSheet::Improper
    );
    // Every root of the proper strip is found again, on the proper sheet.
    for root in proper.observations() {
        assert_eq!(
            root.signature().unwrap().sectors()[0].map(ExteriorSector::sheet),
            Some(ExteriorSheet::Proper)
        );
        let k = root.physical_coordinate();
        assert!(
            full.observations()
                .iter()
                .any(|p| (p.physical_coordinate() - k).norm() < 1e-8 * k.norm())
        );
    }
}

/// Different claddings: the canonical plan's automatic truncation applies.
#[test]
fn the_full_period_plan_adds_the_shifted_strip_to_the_canonical_one() {
    let k0 = TAU * 832.125;
    let air: Region = AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into();
    let glass: Region = AnalyticalMaterialHandle::new(Constant::dielectric(2.0)).into();
    let film: Region = AnalyticalMaterialHandle::new(Constant::dielectric(4.0)).into();
    let stack = Stack::builder(air, glass)
        .layer(film, Length::nanometres(5000.0))
        .finalise();
    let problem = PropagationProblem::new(stack, k0, Polarisation::TransverseMagnetic).unwrap();
    let extent = InPlaneSearchExtent {
        max_parallel_wavenumber: 2.5 * k0,
        imaginary_margin: 0.05,
    };
    let reference = ReferenceSelection::Sheet(CoordinateSheet::Principal);
    let config = ConformalChartConfig::default();
    let assume = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
    let canonical = problem
        .canonical_plan(extent, reference, config, assume)
        .unwrap();
    let plan = problem
        .full_period_plan(extent, reference, config, assume)
        .unwrap();
    assert!(matches!(
        plan.geometry(),
        PlanGeometry::FullPeriodInPlane(_)
    ));
    let [first, second] = plan.domains().try_into().unwrap();
    let base = canonical.domains()[0].requested_bounds();
    assert_eq!(first.requested_bounds(), base);
    let (y0, y1) = base.imaginary_bounds();
    assert_eq!(
        second.requested_bounds().imaginary_bounds(),
        (y0 + PI, y1 + PI)
    );
    assert!(
        !plan
            .exclusions()
            .iter()
            .any(|e| matches!(e, CoverageExclusion::OmittedExteriorSheetSectors { .. }))
    );
    let report = plan
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    assert!(!report.observations().is_empty());
}
