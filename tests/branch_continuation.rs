#[path = "../examples/support/pole_fixtures.rs"]
mod fixtures;
mod hbn_support;
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

/// Continue a TM mode of a lossless 100 nm hBN film in air through its
/// crowded type II ladder just above ν̃_TO⊥ = 1370 cm⁻¹, from ν̃ = 1371 to
/// 1372 cm⁻¹, with chart half-width `radius` and `steps` equal steps.
fn hbn_ladder(radius: f64, steps: usize) -> (BranchTrace, Complex) {
    use hbn_support::*;
    let stack = hbn_film(hbn(0.0, 0.0), 100.0);
    let seed = root(&stack, 1371.0, 15.44 * k0_of(1371.0), 0.01);
    let targets: Vec<f64> = (1..=steps)
        .map(|i| k0_of(1371.0 + i as f64 / steps as f64))
        .collect();
    let trace = seed
        .continue_with(
            Scatter2::new(),
            &targets,
            BranchSettings {
                radius,
                minimum_step: 1e-3,
                retries: 60,
                refinement: seed.config(),
                ..settings()
            },
        )
        .unwrap();
    let end = trace.points.last().unwrap().physical_coordinate();
    (trace, end)
}

/// Neighbouring roots of the ladder are about 0.1 apart on the chart. A
/// corrector region holding several of them is refused and the step
/// subdivided, so a coarse continuation ends on the same root as a fine one
/// instead of on its neighbour, and reports the crowded regions.
#[test]
fn refuses_a_corrector_region_holding_neighbouring_roots() {
    let (fine, reference) = hbn_ladder(0.002, 100);
    assert!(fine.stop.is_none(), "{:?}", fine.stop);
    assert!(fine.near_degenerate().is_empty());
    let (coarse, end) = hbn_ladder(0.1, 1);
    assert!(coarse.stop.is_none(), "{:?}", coarse.stop);
    assert!(
        (end - reference).norm() < 1e-9 * reference.norm(),
        "continued to {end}, the branch is at {reference}"
    );
    let crowded = coarse.near_degenerate();
    assert!(!crowded.is_empty());
    assert!(crowded.iter().all(|&(_, zeros, _)| zeros > 1));
}

/// A chart radius wider than the ladder spacing at every step: the
/// continuation stops on a region holding several roots rather than
/// accepting any of them.
#[test]
fn stops_when_every_corrector_region_is_crowded() {
    let (trace, _) = hbn_ladder(0.5, 1);
    assert_eq!(trace.points.len(), 1);
    assert!(!trace.near_degenerate().is_empty());
    let stop = trace.stop.unwrap();
    assert!(
        matches!(stop.cause, BranchFailure::RootCount { zeros, .. } if zeros > 1),
        "{:?}",
        stop.cause
    );
}
