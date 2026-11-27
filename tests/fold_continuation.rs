//! Branches followed through their folds (zero-group-velocity points) by
//! switching the control coordinate between k0 and β.
use lamina_core::Scatter2;
use lamina_mode::*;

mod hbn_support;
use hbn_support::*;

fn settings(k0_min: f64, step: f64) -> ArclengthSettings {
    ArclengthSettings {
        branch: BranchSettings {
            local_assumption:
                LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            domain_assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
            radius: 0.01,
            minimum_step: 1e-3,
            retries: 20,
            refinement: Default::default(),
            slope: Default::default(),
        },
        step,
        frequency_radius: 0.05,
        frequency: FrequencyStation {
            continuation: Default::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        k0_bounds: [k0_min, f64::MAX],
        beta_bounds: [0.0, f64::MAX],
        max_points: 2000,
        fold_tolerance: 1e-12,
    }
}

/// From the forward (light-line) piece of the fundamental TM mode of a
/// lossless 100 nm film in air at ν̃0, follow the branch with k0 increasing
/// until it returns below ν̃0. It passes one fold at `fold` cm⁻¹ (the root of
/// F = ∂F/∂β = 0 of the exact characteristic) and ends on the backward piece,
/// whose effective index at ν̃0 is `backward`.
fn passes_fold(film: Region, nu0: f64, forward: f64, backward: f64, fold: f64, step: f64) {
    let stack = hbn_film(film, 100.0);
    let k0 = k0_of(nu0);
    let seed = root(&stack, nu0, forward * k0, 0.05);
    // Continuation in k0 alone cannot pass the fold: it stops short of it.
    let targets: Vec<f64> = (1..=40)
        .map(|i| k0_of(nu0 + (fold + 0.1 - nu0) * i as f64 / 40.))
        .collect();
    let plain = seed
        .continue_with(Scatter2::new(), &targets, settings(k0, step).branch)
        .unwrap();
    assert!(plain.stop.is_some());
    let reached = plain.points.last().unwrap().signature().unwrap().k0.re;
    assert!(reached < k0_of(fold));
    let trace = seed
        .continue_through_folds_with(Scatter2::new(), 1.0, &settings(k0, step))
        .unwrap();
    assert!(
        matches!(trace.end, ArclengthEnd::Bounds),
        "{:?} after {} points",
        trace.end,
        trace.points.len()
    );
    let [found] = &trace.folds[..] else {
        panic!("{} folds", trace.folds.len())
    };
    let nu = found.k0.re / k0_of(1.0);
    assert!((nu - fold).abs() < 2e-6, "fold at {nu}, expected {fold}");
    assert!(found.k0.im.abs() < 1e-9 * found.k0.re);
    // Opposite group velocities on the two sides; dk0/dβ vanishes at the fold.
    let [before, after] = found.side_slopes;
    assert!(before.re > 0.0 && after.re < 0.0, "{before} {after}");
    assert!(found.slope.re.abs() < 1e-6 * before.re.abs().max(after.re.abs()));
    for side in &found.sides {
        let k = side.signature().unwrap().k0.re;
        assert!(k < found.k0.re);
    }
    // The end: the backward piece, just below ν̃0, as found independently.
    let end = trace.points.last().unwrap().signature().unwrap();
    assert!(end.k0.re < k0);
    let independent = root(&stack, end.k0.re / k0_of(1.0), end.k_parallel.re, 0.01);
    let beta = independent.physical_coordinate();
    assert!((end.k_parallel - beta).norm() < 1e-9 * beta.norm());
    // That piece at ν̃0 itself is the backward root.
    let back = root(&stack, nu0, end.k_parallel.re, 0.01).physical_coordinate();
    assert!(
        (back.re / k0 - backward).abs() < 1e-4,
        "n_eff {}",
        back.re / k0
    );
    // k0 rises to the fold and falls after it.
    let k0s: Vec<f64> = trace
        .points
        .iter()
        .map(|p| p.signature().unwrap().k0.re)
        .collect();
    let peak = k0s.iter().cloned().fold(f64::MIN, f64::max);
    assert!(peak <= found.k0.re * (1. + 1e-12));
}

#[test]
fn hbn_type_one_fold() {
    passes_fold(hbn(0.0, 0.0), 829.0, 1.1934, 1.7409, 829.076764, 20.0);
}

#[test]
fn isotropic_lower_band_fold() {
    passes_fold(
        isotropic(2.95, 780.0, 830.0),
        829.0,
        1.1625,
        2.0536,
        829.136626,
        20.0,
    );
}

#[test]
fn isotropic_upper_band_fold() {
    passes_fold(
        isotropic(4.87, 1370.0, 1610.0),
        1605.0,
        1.2555,
        1.7787,
        1605.267328,
        40.0,
    );
}

#[test]
fn validates_the_request() {
    let stack = hbn_film(hbn(0.0, 0.0), 100.0);
    let seed = root(&stack, 829.0, 1.1934 * k0_of(829.0), 0.05);
    let base = settings(0.0, 20.0);
    for (direction, settings) in [
        (0.0, base.clone()),
        (
            1.0,
            ArclengthSettings {
                step: 0.0,
                ..base.clone()
            },
        ),
        (
            1.0,
            ArclengthSettings {
                frequency_radius: f64::NAN,
                ..base.clone()
            },
        ),
        (
            1.0,
            ArclengthSettings {
                k0_bounds: [1.0, 0.0],
                ..base.clone()
            },
        ),
        (
            1.0,
            ArclengthSettings {
                fold_tolerance: 0.0,
                ..base.clone()
            },
        ),
    ] {
        assert!(matches!(
            seed.continue_through_folds_with(Scatter2::new(), direction, &settings),
            Err(BranchFailure::InvalidRequest)
        ));
    }
}
