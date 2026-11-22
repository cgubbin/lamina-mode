use super::*;
use crate::{
    conformal::{
        ConformalRegion,
        chart::{ConformalChartConfig, CoordinateSheet},
    },
    problem::ReferenceSelection,
    problem::{AnalyticAssumption, InPlaneProblem},
};
use lamina_core::{Constant, DeterminantEvaluationError, Scatter2, Stack};
use std::{cell::RefCell, error::Error};
fn c(v: f64) -> C {
    C::new(v, 0.0)
}
fn provenance() -> DomainProvenance {
    let stack =
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25)).finalise();
    let p = InPlaneProblem::in_plane(
        stack,
        2.0,
        lamina_core::Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    p.domain(
        ConformalRegion::new(-0.3, 0.3, -0.2, 0.2).unwrap(),
        ReferenceSelection::Sheet(CoordinateSheet::Principal),
        ConformalChartConfig::default(),
        AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    )
    .unwrap()
    .provenance()
    .clone()
}
fn estimate(z: C) -> SingularPointEstimate<C> {
    SingularPointEstimate {
        location: z,
        multiplicity: 1,
        enclosure: Rect {
            x_min: -0.3,
            x_max: 0.3,
            y_min: -0.2,
            y_max: 0.2,
        },
        kind: SingularPointEstimateKind::SinglePoint,
        target: SearchTarget::Zeros,
    }
}
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("injected backend fault")]
struct Fault;
type Failure = DeterminantEvaluationError<Fault>;
fn bundle(value: C, derivative: C) -> FunctionBundle<C> {
    FunctionBundle::new(value, derivative)
}
fn run(
    e: &SingularPointEstimate<C>,
    cfg: RefinementConfig,
    mut eval: impl FnMut(C) -> FunctionBundle<C>,
) -> CandidateOutcome<Failure> {
    refine_candidate(
        e,
        0,
        &provenance(),
        cfg.validate().unwrap(),
        |z| Ok(eval(z)),
        Ok,
    )
}
#[test]
fn configuration_checks_tolerances_scales_thresholds_and_zero_budgets() {
    let base = RefinementConfig::default();
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        for cfg in [
            RefinementConfig {
                residual_atol: value,
                ..base
            },
            RefinementConfig {
                residual_rtol: value,
                ..base
            },
            RefinementConfig {
                z_atol: value,
                ..base
            },
            RefinementConfig {
                z_rtol: value,
                ..base
            },
            RefinementConfig {
                residual_scale: value,
                ..base
            },
        ] {
            assert!(cfg.validate().is_err());
        }
    }
    for cfg in [
        RefinementConfig {
            residual_scale: 0.0,
            ..base
        },
        RefinementConfig {
            residual_atol: 0.0,
            residual_rtol: 0.0,
            ..base
        },
        RefinementConfig {
            z_atol: 0.0,
            z_rtol: 0.0,
            ..base
        },
        RefinementConfig {
            residual_rtol: f64::MAX,
            residual_scale: f64::MAX,
            ..base
        },
        RefinementConfig {
            z_atol: f64::MAX,
            z_rtol: f64::MAX,
            ..base
        },
    ] {
        assert!(cfg.validate().is_err());
    }
    for cfg in [
        RefinementConfig {
            max_updates: 0,
            ..base
        },
        RefinementConfig {
            max_backtracks: 0,
            ..base
        },
        RefinementConfig {
            max_evaluations: 0,
            ..base
        },
    ] {
        let out = run(&estimate(c(0.0)), cfg, |_| {
            panic!("zero budget must not evaluate")
        });
        assert!(matches!(out.status(), CandidateStatus::BudgetExceeded));
        assert_eq!(out.work().evaluations, 0);
    }
}
#[test]
fn eligibility_does_not_evaluate_clusters_or_invalid_metadata() {
    let e = estimate(c(0.0));
    for (kind, n) in [
        (SingularPointEstimateKind::ClusterCentroid, 2),
        (SingularPointEstimateKind::ClusterCentroid, 3),
    ] {
        let e = SingularPointEstimate {
            kind,
            multiplicity: n,
            ..e
        };
        let out = run(&e, RefinementConfig::default(), |_| panic!());
        assert!(matches!(out.status(), CandidateStatus::UnresolvedCluster));
    }
    for e in [
        SingularPointEstimate {
            multiplicity: 2,
            ..e
        },
        SingularPointEstimate {
            multiplicity: 0,
            ..e
        },
        SingularPointEstimate {
            kind: SingularPointEstimateKind::ClusterCentroid,
            ..e
        },
        SingularPointEstimate {
            target: SearchTarget::Poles,
            ..e
        },
    ] {
        assert!(matches!(
            run(&e, RefinementConfig::default(), |_| panic!()).status(),
            CandidateStatus::IneligibleEstimate
        ));
    }
    for r in [
        Rect {
            x_min: -0.4,
            ..e.enclosure
        },
        Rect {
            x_min: f64::NAN,
            ..e.enclosure
        },
        Rect {
            x_max: -0.3,
            ..e.enclosure
        },
    ] {
        assert!(matches!(
            run(
                &SingularPointEstimate { enclosure: r, ..e },
                RefinementConfig::default(),
                |_| panic!()
            )
            .status(),
            CandidateStatus::InvalidEnclosure
        ));
    }
    for z in [c(0.3), c(0.3 + 1e-15), c(f64::NAN)] {
        assert!(matches!(
            run(&estimate(z), RefinementConfig::default(), |_| panic!()).status(),
            CandidateStatus::BoundaryLimited
        ));
    }
}
#[test]
fn simple_complex_polynomial_converges_with_fresh_evidence() {
    let root = C::new(0.1, 0.04);
    let calls = RefCell::new(Vec::new());
    let e = estimate(C::new(0.02, -0.03));
    let out = run(&e, RefinementConfig::default(), |z| {
        calls.borrow_mut().push(z);
        bundle((z - root) * (z + c(1.0)), 2.0 * z + c(1.0) - root)
    });
    let CandidateStatus::Accepted(root) = out.status() else {
        panic!("{out:?}")
    };
    assert!((root.evidence().z - C::new(0.1, 0.04)).norm() < 1e-10);
    assert!(passes(root.evidence()));
    assert_eq!(root.coordinate(), root.evidence().z);
    assert_eq!(root.work(), out.work());
    assert_eq!(root.index(), 0);
    assert_eq!(root.config(), RefinementConfig::default());
    let calls = calls.borrow();
    assert_eq!(calls.len(), out.work().evaluations);
    assert_eq!(calls[calls.len() - 1], calls[calls.len() - 2]);
    assert!(calls.iter().all(|z| inside(e.enclosure, *z)));
}
#[test]
fn bounded_backtracking_rejects_outward_steps_without_evaluation() {
    // Full Newton from .02 for z^2-.01 goes to .26, outside this enclosure;
    // a halved step stays inside and subsequently converges to +.1.
    let e = SingularPointEstimate {
        enclosure: Rect {
            x_min: 0.0,
            x_max: 0.2,
            y_min: -0.1,
            y_max: 0.1,
        },
        ..estimate(c(0.02))
    };
    let calls = RefCell::new(Vec::new());
    let out = run(&e, RefinementConfig::default(), |z| {
        assert!(inside(e.enclosure, z));
        calls.borrow_mut().push(z);
        bundle(z * z - c(0.01), 2.0 * z)
    });
    assert!(
        matches!(out.status(), CandidateStatus::Accepted(_)),
        "{out:?}"
    );
    assert!(out.work().trials > out.work().updates);
    assert!(!calls.borrow().contains(&c(0.26)));
}
#[test]
fn tiny_residual_and_damped_motion_do_not_replace_correction_test() {
    let cfg = RefinementConfig {
        max_updates: 1,
        ..RefinementConfig::default()
    };
    let out = run(&estimate(c(0.0)), cfg, |z| {
        bundle(1e-100 * (z - c(0.2)), c(1e-100))
    });
    if let CandidateStatus::Accepted(root) = out.status() {
        assert!((root.evidence().z - c(0.2)).norm() < 1e-12)
    } else {
        panic!("{out:?}")
    }
    // Bad derivative forces very small admissible moves but the undamped
    // correction remains huge. Small damped updates cannot be accepted.
    let out = run(
        &estimate(c(0.0)),
        RefinementConfig {
            max_updates: 1,
            max_backtracks: 20,
            ..cfg
        },
        |z| bundle(1e-100 * (z - c(0.2)), c(1e-104)),
    );
    assert!(
        !matches!(out.status(), CandidateStatus::Accepted(_)),
        "{out:?}"
    );
    let zero = run(&estimate(c(0.0)), cfg, |_| bundle(c(0.0), c(0.0)));
    assert!(matches!(zero.status(), CandidateStatus::ZeroDerivative));
}
#[test]
fn work_limits_and_numerical_failure_reasons_are_explicit() {
    let base = RefinementConfig::default();
    let e = estimate(c(0.0));
    for limit in [1, 2] {
        let out = run(
            &e,
            RefinementConfig {
                max_evaluations: limit,
                ..base
            },
            |z| bundle(z - c(0.1), c(1.0)),
        );
        assert!(matches!(out.status(), CandidateStatus::BudgetExceeded));
        assert_eq!(out.work().evaluations, limit);
    }
    let out = run(
        &e,
        RefinementConfig {
            max_evaluations: 1,
            ..base
        },
        |_| bundle(c(0.0), c(1.0)),
    );
    assert!(matches!(out.status(), CandidateStatus::BudgetExceeded));
    assert_eq!(out.work().evaluations, 1);
    let out = run(&e, base, |_| bundle(c(1.0), c(1e-6)));
    assert!(matches!(out.status(), CandidateStatus::BoundaryLimited));
    assert_eq!(out.work().evaluations, 1);
    let out = run(&e, base, |_| bundle(c(1.0), c(1.0)));
    assert!(matches!(out.status(), CandidateStatus::NoProgress));
    let out = run(&estimate(c(0.1)), base, |_| bundle(c(1.0), c(1e300)));
    assert!(matches!(out.status(), CandidateStatus::Stagnated));
    let out = run(&e, base, |_| bundle(c(1e308), c(1e-308)));
    assert!(matches!(out.status(), CandidateStatus::NonFiniteCorrection));
    let out = run(&e, base, |_| bundle(c(f64::INFINITY), c(1.0)));
    assert!(matches!(out.status(), CandidateStatus::NonFiniteEvaluation));
}
fn contains_source<T: Error + 'static>(mut e: &(dyn Error + 'static)) -> bool {
    loop {
        if e.is::<T>() {
            return true;
        }
        match e.source() {
            Some(next) => e = next,
            None => return false,
        }
    }
}
#[test]
fn checked_failures_at_initial_trial_final_and_projection_remain_identifiable() {
    for failure in [
        Failure::NonFiniteInput,
        Failure::NonFiniteOutput,
        Failure::Backend { source: Fault },
    ] {
        for failing_call in [1, 2, 3] {
            let mut count = 0;
            let out = refine_candidate(
                &estimate(c(0.0)),
                0,
                &provenance(),
                RefinementConfig::default().validate().unwrap(),
                |z| {
                    count += 1;
                    if count == failing_call {
                        Err(failure.clone())
                    } else {
                        Ok(bundle(z - c(0.1), c(1.0)))
                    }
                },
                Ok,
            );
            let CandidateStatus::EvaluationFailed(error) = out.status() else {
                panic!("{out:?}")
            };
            assert!(contains_source::<Failure>(error));
            if matches!(failure, Failure::Backend { .. }) {
                assert!(contains_source::<Fault>(error));
            }
            assert_eq!(out.work().evaluations, failing_call);
            assert_eq!(out.last_valid().is_some(), failing_call > 1);
        }
    }
    let out = refine_candidate(
        &estimate(c(0.1)),
        0,
        &provenance(),
        RefinementConfig::default().validate().unwrap(),
        |_| Ok(bundle(c(0.0), c(1.0))),
        |_| Err(Failure::Backend { source: Fault }),
    );
    assert!(matches!(out.status(), CandidateStatus::ProjectionFailed(_)));
    assert_eq!(out.work().evaluations, 2);
    assert!(out.last_valid().is_some());
}
#[test]
fn exact_simple_zero_and_dynamic_threshold_overflow_are_checked() {
    let out = run(&estimate(c(0.1)), RefinementConfig::default(), |z| {
        bundle(z - c(0.1), c(1.0))
    });
    assert!(matches!(out.status(), CandidateStatus::Accepted(_)));
    assert_eq!(out.work().updates, 0);
    assert_eq!(out.work().evaluations, 2);
    let stack =
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25)).finalise();
    let p = InPlaneProblem::in_plane(
        stack,
        2.0,
        lamina_core::Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let domain = p
        .domain(
            ConformalRegion::new(1.0, 2.0, -0.1, 0.1).unwrap(),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    let e = SingularPointEstimate {
        location: c(1.5),
        enclosure: Rect {
            x_min: 1.0,
            x_max: 2.0,
            y_min: -0.1,
            y_max: 0.1,
        },
        ..estimate(c(0.0))
    };
    let cfg = RefinementConfig {
        z_rtol: f64::MAX,
        z_atol: 0.0,
        ..RefinementConfig::default()
    }
    .validate()
    .unwrap();
    let out: CandidateOutcome<Failure> = refine_candidate(
        &e,
        0,
        domain.provenance(),
        cfg,
        |_| Ok(bundle(c(1.0), c(1.0))),
        Ok,
    );
    assert!(matches!(out.status(), CandidateStatus::NonFiniteThreshold));
    assert_eq!(out.work().evaluations, 1);
}

#[test]
fn fresh_finite_check_can_veto_acceptance() {
    let mut calls = 0;
    let out = run(&estimate(c(0.0)), RefinementConfig::default(), |_| {
        calls += 1;
        bundle(if calls == 1 { c(0.0) } else { c(1.0) }, c(1.0))
    });
    assert!(matches!(out.status(), CandidateStatus::NoProgress));
    assert_eq!(out.work().evaluations, 2);
    let mut calls = 0;
    let out = run(&estimate(c(0.0)), RefinementConfig::default(), |_| {
        calls += 1;
        bundle(c(0.0), if calls == 1 { c(1.0) } else { c(0.0) })
    });
    assert!(matches!(out.status(), CandidateStatus::ZeroDerivative));
    assert_eq!(out.work().evaluations, 2);
}
