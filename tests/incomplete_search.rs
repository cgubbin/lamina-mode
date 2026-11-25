//! A solve whose discovery fails on part of its region is incomplete.
use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Stack, Wall, WallKind};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::TAU;

const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
const CORE: f64 = 4.;
const CLAD: f64 = 2.25;
const K: f64 = TAU;

/// A core 0.7 cm thick in cladding, closed by PEC walls 0.45 cm beyond it.
fn walled_slab() -> AnalyticalMaterialStack<Complex> {
    let cm = Length::centimetres;
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(0.7))
        .left_termination(Wall::new(WallKind::PerfectElectric, cm(-0.45)))
        .right_termination(Wall::new(WallKind::PerfectElectric, cm(1.15)))
        .finalise()
}

#[test]
fn a_failed_discovery_on_the_core_light_line_leaves_the_solve_incomplete() {
    let problem =
        PropagationProblem::new(walled_slab(), K, Polarisation::TransverseElectric).unwrap();
    // The upper edge lies exactly on the core light line, β = n_core k0.
    let bounds = ConformalRegion::new(1.5 * K, CORE.sqrt() * K, -0.5, 0.5).unwrap();
    let domain = problem.direct_domain(bounds, ASSUME).unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    let failure = report.discovery().entries().next().unwrap().unwrap_err();
    assert!(failure.to_string().contains("NonFiniteOutput"), "{failure}");
    assert!(report.observations().is_empty());
    assert!(!report.is_complete());
    let unresolved = report.unresolved();
    let [region] = unresolved.as_slice() else {
        panic!("{unresolved:?}")
    };
    assert_eq!(region.chart, 0);
    assert_eq!(
        (region.bounds.x_min, region.bounds.x_max),
        bounds.real_bounds()
    );
    assert_eq!(
        (region.bounds.y_min, region.bounds.y_max),
        bounds.imaginary_bounds()
    );
    let UnresolvedCause::Discovery(cause) = &region.cause else {
        panic!("{:?}", region.cause)
    };
    assert_eq!(cause.to_string(), failure.to_string());
}

#[test]
fn a_resolved_walled_solve_is_complete() {
    let problem =
        PropagationProblem::new(walled_slab(), K, Polarisation::TransverseElectric).unwrap();
    // Away from both light lines, around one isolated real root.
    let bounds = ConformalRegion::new(1.6 * K, 1.9 * K, -0.5, 0.5).unwrap();
    let domain = problem.direct_domain(bounds, ASSUME).unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    assert!(report.unresolved().is_empty(), "{:?}", report.unresolved());
    assert!(report.is_complete());
    assert!(!report.observations().is_empty());
}

#[test]
fn a_stopped_walled_solve_lists_its_unresolved_leaves() {
    let problem =
        PropagationProblem::new(walled_slab(), K, Polarisation::TransverseElectric).unwrap();
    let bounds = ConformalRegion::new(0.4 * K, 1.9 * K, -0.5, 0.5).unwrap();
    let domain = problem.direct_domain(bounds, ASSUME).unwrap();
    // No refinement step, and a score tolerance below the counted leaves' scores.
    let discovery = DiscoveryConfig {
        max_iterations: 0,
        score_tolerance: 1e-15,
        ..Default::default()
    };
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(discovery, Default::default(), Default::default())
        .unwrap();
    assert!(!report.is_complete());
    let unresolved = report.unresolved();
    assert!(
        unresolved
            .iter()
            .any(|r| matches!(r.cause, UnresolvedCause::Stopped(_)))
    );
    let (x_min, x_max) = bounds.real_bounds();
    let leaves: Vec<_> = unresolved
        .iter()
        .filter(|r| matches!(r.cause, UnresolvedCause::Leaf { score } if score > 1e-15))
        .collect();
    assert!(!leaves.is_empty());
    for leaf in leaves {
        assert!(leaf.bounds.x_min >= x_min && leaf.bounds.x_max <= x_max);
    }
}
