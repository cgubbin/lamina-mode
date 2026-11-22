#[path = "../examples/support/pole_fixtures.rs"]
mod fixtures;
#[path = "../examples/support/pole_material.rs"]
mod material;
#[path = "../examples/support/pole_green.rs"]
mod reference;
use lamina_core::{Polarisation, Scatter2, Transfer2};
use lamina_mode::*;
fn settings() -> BranchSettings {
    BranchSettings {
        local_assumption:
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
        domain_assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        radius: 0.01,
        minimum_step: 1e-7,
        retries: 40,
        refinement: Default::default(),
        slope: Default::default(),
    }
}
#[test]
fn validates_before_advancing() {
    let seed = fixtures::cavity(Polarisation::TransverseElectric, true).unwrap();
    for path in [
        vec![],
        vec![f64::NAN],
        vec![0.],
        vec![0.1, 0.1],
        vec![0.1, -0.1],
    ] {
        assert!(matches!(
            seed.continue_with(Scatter2::new(), &path, settings()),
            Err(BranchFailure::InvalidRequest)
        ));
    }
    assert!(
        seed.continue_with(Transfer2::new(), &[0.1], settings())
            .is_err()
    );
    assert!(
        seed.continue_with(
            Scatter2::new(),
            &[0.1],
            BranchSettings {
                minimum_step: 0.,
                ..settings()
            }
        )
        .is_err()
    );
}
#[test]
fn spectral_strategies_agree_and_keep_own_domains() {
    let mut results = Vec::new();
    for direct in [false, true] {
        let seed = fixtures::cavity(Polarisation::TransverseElectric, direct).unwrap();
        let trace = seed
            .continue_with(Scatter2::new(), &[0.1, 0.2], settings())
            .unwrap();
        assert!(trace.stop.is_none(), "{:?}", trace.stop);
        let end = trace.points.last().unwrap();
        assert_eq!(end.provenance().strategy(), seed.provenance().strategy());
        assert!(!seed.provenance().same_domain(&end.provenance()));
        results.push(end.physical_coordinate());
    }
    assert!((results[0] - results[1]).norm() < 1e-8);
}
#[test]
fn subdivides_and_returns_prefix_on_failure() {
    let seed = fixtures::cavity(Polarisation::TransverseElectric, true).unwrap();
    let policy = BranchSettings {
        radius: 0.0002,
        ..settings()
    };
    let trace = seed
        .continue_with(Scatter2::new(), &[0.25], policy)
        .unwrap();
    assert!(trace.stop.is_none(), "{:?}", trace.stop);
    assert!(trace.points.len() > 2);
    assert!(trace.destinations.iter().all(|target| *target == 0.25));
    let failed = seed
        .continue_with(
            Scatter2::new(),
            &[0.001, 0.25],
            BranchSettings {
                retries: 0,
                ..policy
            },
        )
        .unwrap();
    assert_eq!(failed.points.len(), 2);
    assert_eq!(failed.stop.unwrap().requested, 0.25);
}
#[test]
fn records_degenerate_slope_and_exterior_stop() {
    let seed = fixtures::cavity(Polarisation::TransverseElectric, true).unwrap();
    let mut policy = settings();
    policy.retries = 0;
    policy.slope.denominator_atol = 1e50;
    let trace = seed
        .continue_with(Scatter2::new(), &[0.01], policy)
        .unwrap();
    assert!(matches!(
        trace.stop.unwrap().cause,
        BranchFailure::Slope(ImplicitSlopeFailure::DegenerateDenominator { .. })
    ));
    let trace = seed
        .continue_with(
            Scatter2::new(),
            &[1e5],
            BranchSettings {
                retries: 0,
                ..settings()
            },
        )
        .unwrap();
    assert!(matches!(trace.stop.unwrap().cause, BranchFailure::Exterior));
}
