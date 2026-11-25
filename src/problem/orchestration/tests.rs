use super::*;
use crate::conformal::RawConformalMap;
use crate::problem::physical::{
    Classification, InputError, InterpretationConfig, ModeType, PointError, Unsupported,
};
use lamina_core::{
    CanonicalCoordinates, Constant, DeterminantEvaluationError, ExteriorWavevectors, ScalarAlgebra,
    Scatter2,
};
use std::{
    error::Error,
    sync::atomic::{AtomicUsize, Ordering},
};
fn c(v: f64) -> C {
    C::new(v, 0.0)
}
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn selection() -> ReferenceSelection {
    ReferenceSelection::Sheet(CoordinateSheet::Principal)
}
fn bounds() -> ConformalRegion<f64> {
    ConformalRegion::new(-0.3, 0.3, -0.2, 0.2).unwrap()
}
fn problem() -> InPlaneProblem<Scatter2> {
    InPlaneProblem::in_plane(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25)).finalise(),
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap()
}
fn domain<M, S>(p: &ModalProblem<M, S>, r: ConformalRegion<f64>) -> ProblemDomain<M, S>
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    RegionError<M>: std::fmt::Debug,
    S: ModeDeterminantSolver<J> + Clone,
{
    p.domain(r, selection(), ConformalChartConfig::default(), ASSUME)
        .unwrap()
}
fn exterior(e: f64) -> ConstantExterior {
    ConstantExterior::new(c(e), c(1.0)).unwrap()
}
fn source_contains<T: Error + 'static>(mut e: &(dyn Error + 'static)) -> bool {
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
fn plans_preserve_domain_identity_and_reject_invalid_combinations() {
    let p = problem();
    let d = domain(&p, bounds());
    let before = d.provenance().evaluation();
    assert!(matches!(
        ChartPlan::<_, _>::explicit(p.clone(), vec![], vec![]),
        Err(PlanError::Empty)
    ));
    assert!(matches!(
        ChartPlan::explicit(p.clone(), vec![d.clone(), d.clone()], vec![]),
        Err(PlanError::DuplicateDomain)
    ));
    assert!(matches!(
        ChartPlan::explicit(problem(), vec![d.clone()], vec![]),
        Err(PlanError::ForeignDomain)
    ));
    let other = domain(&p, bounds());
    let plan = ChartPlan::explicit(p, vec![d.clone(), other], vec![]).unwrap();
    let clone = plan.clone();
    assert!(clone.domains()[0].provenance().same_domain(d.provenance()));
    assert!(
        !clone.domains()[0]
            .provenance()
            .same_domain(clone.domains()[1].provenance())
    );
    assert_eq!(clone.domains()[0].provenance().evaluation(), before);
    assert!(Arc::ptr_eq(&plan.identity, &clone.identity));
}
fn assert_sheet_omissions(exclusions: &[CoverageExclusion]) {
    let half = std::f64::consts::FRAC_PI_2;
    let pi = std::f64::consts::PI;
    assert_eq!(
        exclusions[3],
        CoverageExclusion::OmittedExteriorSheetSectors {
            selected_min: -half,
            selected_max: half,
            exterior_period: 2.0 * pi,
            omitted: [(-pi, -half), (half, pi)],
        }
    );
}
#[test]
fn canonical_plans_match_existing_geometry_and_record_exclusions() {
    let p = problem();
    let extent = InPlaneSearchExtent {
        max_parallel_wavenumber: 4.0,
        imaginary_margin: 0.05,
    };
    let expected = p.map.search_region(extent).unwrap().into_inner();
    let plan = p
        .canonical_in_plane_plan(extent, selection(), ConformalChartConfig::default(), ASSUME)
        .unwrap();
    assert_eq!(plan.domains().len(), 1);
    assert_eq!(plan.domains()[0].provenance().requested(), expected);
    assert_eq!(*plan.geometry(), PlanGeometry::CanonicalInPlane(extent));
    assert_eq!(plan.exclusions().len(), 4);
    assert_sheet_omissions(plan.exclusions());
    let p = SpectralProblem::spectral(
        exterior(1.0),
        exterior(2.25),
        [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
        1.5,
        Polarisation::TransverseMagnetic,
        Scatter2::new(),
    )
    .unwrap();
    let extent = WavenumberSearchExtent {
        max_vacuum_wavenumber: 3.0,
        imaginary_margin: 0.05,
    };
    let expected = p.map.search_regions(extent).unwrap().into_regions();
    let plan = p
        .canonical_spectral_plan(
            extent,
            [selection(), selection()],
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    assert_eq!(plan.domains().len(), 2);
    for (i, r) in expected.into_iter().enumerate() {
        assert_eq!(plan.domains()[i].provenance().requested(), r);
    }
    assert_eq!(*plan.geometry(), PlanGeometry::CanonicalSpectral(extent));
    assert_sheet_omissions(plan.exclusions());
    assert!(
        matches!(plan.exclusions().last(),Some(CoverageExclusion::ImaginaryStrip{min,max}) if *min == -0.05&&*max == 0.05)
    );
    let error = p
        .canonical_spectral_plan(
            WavenumberSearchExtent {
                imaginary_margin: 0.0,
                ..extent
            },
            [selection(), selection()],
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap_err();
    assert!(source_contains::<WavenumberSearchError>(&error));
}
#[test]
fn degenerate_canonical_geometry_keeps_typed_cause_and_accepts_explicit_plans() {
    for fixed in [0.0, 1.5] {
        let p = SpectralProblem::spectral(
            exterior(1.0),
            exterior(1.0),
            [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
            fixed,
            Polarisation::TransverseElectric,
            Scatter2::new(),
        )
        .unwrap();
        let error = p
            .canonical_spectral_plan(
                WavenumberSearchExtent {
                    max_vacuum_wavenumber: 3.0,
                    imaginary_margin: 0.05,
                },
                [selection(), selection()],
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CanonicalPlanError::Geometry(WavenumberSearchError::DegenerateExterior)
        ));
        let d = domain(&p, ConformalRegion::new(-0.1, 0.1, 0.3, 0.5).unwrap());
        let exclusions = vec![CoverageExclusion::Declared {
            description: "explicit finite scope only".into(),
            region: None,
        }];
        let plan = ChartPlan::explicit(p, vec![d], exclusions.clone()).unwrap();
        assert_eq!(plan.exclusions(), exclusions);
        assert_eq!(*plan.geometry(), PlanGeometry::Explicit);
    }
    let p = InPlaneProblem::in_plane(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum()).finalise(),
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let error = p
        .canonical_in_plane_plan(
            InPlaneSearchExtent {
                max_parallel_wavenumber: 4.0,
                imaginary_margin: 0.05,
            },
            selection(),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CanonicalPlanError::Geometry(InPlaneSearchError::SymmetricExterior)
    ));
    let lossy = ConstantExterior::new(C::new(1.0, 0.1), c(1.0)).unwrap();
    let stack = Stack::builder(lossy.region(), exterior(2.25).region()).finalise();
    let p = InPlaneProblem::in_plane(
        stack,
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let error = p
        .canonical_in_plane_plan(
            InPlaneSearchExtent {
                max_parallel_wavenumber: 4.0,
                imaginary_margin: 0.05,
            },
            selection(),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CanonicalPlanError::Geometry(InPlaneSearchError::UnsupportedExterior)
    ));
    let d = domain(&p, bounds());
    assert!(ChartPlan::explicit(p, vec![d], vec![]).is_ok());
}
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("injected chart failure")]
struct Fault;
#[derive(Clone, Debug)]
struct Solver {
    calls: Arc<AtomicUsize>,
    fail_band: Option<(f64, f64)>,
    roots: Vec<C>,
}
impl ModeDeterminantSolver<J> for Solver {
    type Error = DeterminantEvaluationError<Fault>;
    fn determinant(
        &self,
        coords: &CanonicalCoordinates<J>,
        exterior: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // For this map kappa_L/k0=i*(exp(z)-r exp(-z)); use exterior values
        // only to distinguish disjoint domains in this synthetic injected fault.
        let value = exterior.left().unwrap().scalar().unwrap().value()[()].im;
        if self
            .fail_band
            .is_some_and(|(lo, hi)| value > lo && value < hi)
        {
            return Err(DeterminantEvaluationError::Backend { source: Fault });
        }
        let k = coords.parallel_angular_wavenumber();
        Ok(self.roots.iter().fold(k.constant(c(1.0)), |product, root| {
            product.multiply(&k.subtract(&k.constant(*root)))
        }))
    }
}
fn mock_problem(roots: Vec<C>) -> ModalProblem<ComplexInPlaneMap<J>, Solver> {
    let p = problem();
    ModalProblem {
        context: p.context,
        map: p.map,
        solver: Solver {
            calls: Arc::new(AtomicUsize::new(0)),
            fail_band: None,
            roots,
        },
    }
}
#[test]
fn invalid_stage_configuration_and_foreign_batches_fail_before_evaluation() {
    let p = mock_problem(vec![]);
    let d = domain(&p, bounds());
    let plan = ChartPlan::explicit(p.clone(), vec![d.clone()], vec![]).unwrap();
    let other = ChartPlan::explicit(p.clone(), vec![d], vec![]).unwrap();
    let base = DiscoveryConfig::default();
    for cfg in [
        DiscoveryConfig {
            score_tolerance: f64::NAN,
            ..base
        },
        DiscoveryConfig {
            zero_tolerance: 0.0,
            ..base
        },
        DiscoveryConfig {
            boundary_shift_fraction: 0.5,
            ..base
        },
        DiscoveryConfig {
            absolute_tolerance: f64::INFINITY,
            ..base
        },
        DiscoveryConfig {
            max_leaves: 0,
            ..base
        },
        DiscoveryConfig {
            max_function_evaluations: 0,
            ..base
        },
    ] {
        assert!(matches!(plan.discover(cfg), Err(PlanError::InvalidConfig)));
    }
    assert_eq!(p.solver.calls.load(Ordering::SeqCst), 0);
    let batch = plan.discover(base).unwrap();
    let before = p.solver.calls.load(Ordering::SeqCst);
    assert!(matches!(
        other.refine(&batch, RefinementConfig::default(), MatchConfig::default()),
        Err(BatchRefinementError::Plan(PlanError::ForeignBatch))
    ));
    assert!(
        plan.refine(
            &batch,
            RefinementConfig {
                residual_scale: 0.0,
                ..RefinementConfig::default()
            },
            MatchConfig::default()
        )
        .is_err()
    );
    assert!(
        plan.refine(
            &batch,
            RefinementConfig::default(),
            MatchConfig {
                physical: MatchTolerance {
                    absolute: f64::NAN,
                    relative: 0.0
                },
                ..MatchConfig::default()
            }
        )
        .is_err()
    );
    assert_eq!(p.solver.calls.load(Ordering::SeqCst), before);
    let clone = plan.clone();
    let report = clone
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert!(report.observations().is_empty());
    assert!(std::ptr::eq(report.discovery(), &batch));
    assert_eq!(report.charts().len(), 1);
    assert!(
        report.charts()[0]
            .refinement()
            .unwrap()
            .outcomes()
            .is_empty()
    );
}

#[test]
fn independent_chart_failures_stay_in_coverage_ledger() {
    let mut p = mock_problem(vec![]);
    p.solver.fail_band = Some((2.60, 2.65));
    let ranges = [(-0.25, -0.15), (-0.18, 0.025), (0.15, 0.25)];
    let domains = ranges
        .into_iter()
        .map(|(lo, hi)| domain(&p, ConformalRegion::new(lo, hi, -0.03, 0.03).unwrap()))
        .collect();
    let exclusions = vec![CoverageExclusion::Declared {
        description: "unsearched finite extension".into(),
        region: Some(ConformalRegion::new(0.5, 0.6, -0.1, 0.1).unwrap()),
    }];
    let plan = ChartPlan::explicit(p, domains, exclusions.clone()).unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    assert!(batch.entries()[0].is_ok());
    assert!(batch.entries()[1].is_err());
    assert!(batch.entries()[2].is_ok());
    assert!(source_contains::<Fault>(
        batch.entries()[1].as_ref().unwrap_err()
    ));
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert_eq!(report.exclusions(), exclusions);
    assert_eq!(report.charts().len(), 3);
    assert!(report.charts()[1].refinement().is_none());
    assert!(report.charts()[1].discovery().is_err());
    assert!(report.groups().is_empty());
    assert!(overlapping(
        report.charts()[0].provenance().requested(),
        report.charts()[1].provenance().requested()
    ));
    for (i, chart) in report.charts().iter().enumerate() {
        assert_eq!(chart.index(), i);
        assert!(std::ptr::eq(chart.discovery(), &batch.entries()[i]));
        assert!(
            chart
                .provenance()
                .same_domain(plan.domains()[i].provenance())
        );
        if let Some(refinement) = chart.refinement() {
            assert!(std::ptr::eq(
                refinement.discovery(),
                batch.entries()[i].as_ref().unwrap()
            ));
        }
    }
}
#[test]
fn budget_clusters_and_candidate_failures_remain_visible() {
    let p = mock_problem(vec![]);
    let chart = domain(&p, bounds());
    let zs = [C::new(-0.15, -0.1), C::new(-0.12, -0.08), C::new(0.17, 0.1)];
    let roots = zs.map(|z| chart.search.dispersion().chart().coordinate(z).unwrap());
    let p = ModalProblem {
        context: p.context,
        map: p.map,
        solver: Solver {
            roots: roots.to_vec(),
            ..p.solver
        },
    };
    let d = domain(&p, bounds());
    let plan = ChartPlan::explicit(p, vec![d], vec![]).unwrap();
    let batch = plan
        .discover(DiscoveryConfig {
            max_iterations: 0,
            ..DiscoveryConfig::default()
        })
        .unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    let descriptions = report.describe_points(InterpretationConfig::default());
    assert_eq!(descriptions.len(), report.observations().len());
    assert!(std::ptr::eq(report.discovery(), &batch));
    let coverage = &report.charts()[0];
    let raw = coverage.discovery().as_ref().unwrap();
    assert_eq!(
        *raw.result().termination(),
        argloc::Termination::ExceededMaxIterations
    );
    assert!(!raw.result().leaves().is_empty());
    let refinement = coverage.refinement().unwrap();
    assert!(std::ptr::eq(refinement.discovery(), raw));
    assert!(
        refinement
            .outcomes()
            .iter()
            .any(|o| matches!(o.status(), CandidateStatus::UnresolvedCluster))
    );
    assert!(
        refinement
            .outcomes()
            .iter()
            .any(|o| matches!(o.status(), CandidateStatus::Accepted(_))),
        "{:?}",
        refinement.outcomes()
    );
    let stopped = plan
        .refine(
            &batch,
            RefinementConfig {
                max_updates: 0,
                ..RefinementConfig::default()
            },
            MatchConfig::default(),
        )
        .unwrap();
    assert!(
        stopped.charts()[0]
            .refinement()
            .unwrap()
            .outcomes()
            .iter()
            .any(|o| matches!(o.status(), CandidateStatus::BudgetExceeded))
    );
    assert!(std::ptr::eq(stopped.discovery(), &batch));
    assert!(stopped.observations().is_empty());
}
fn observation(
    chart: usize,
    z: C,
    physical: C,
    signature: Result<BranchSignature, DeterminantEvaluationError<Fault>>,
) -> AcceptedObservation<DeterminantEvaluationError<Fault>> {
    let p = problem();
    let d = domain(&p, bounds());
    let e = argloc::SingularPointEstimate {
        location: z,
        enclosure: bounds().try_into().unwrap(),
        multiplicity: 1,
        kind: argloc::SingularPointEstimateKind::SinglePoint,
        target: argloc::SearchTarget::Zeros,
    };
    let outcome = crate::refinement::refine_candidate(
        &e,
        0,
        d.provenance(),
        RefinementConfig::default().validate().unwrap(),
        |_| Ok::<_, DeterminantEvaluationError<Fault>>(argloc::FunctionBundle::new(c(0.0), c(1.0))),
        |_| Ok(physical),
    );
    let CandidateStatus::Accepted(root) = outcome.status() else {
        panic!("{outcome:?}")
    };
    AcceptedObservation {
        chart,
        candidate: 0,
        root: (**root).clone(),
        enclosure: e.enclosure,
        signature,
    }
}
fn signature() -> BranchSignature {
    BranchSignature {
        k0: c(2.0),
        k_parallel: c(3.5),
        left: Some(C::new(0.0, 2.0)),
        right: Some(C::new(0.0, 1.0)),
    }
}
#[test]
fn matching_preserves_radiation_branches_order_and_signature_failures() {
    let cfg = MatchConfig::default();
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            MatchConfig {
                conformal: MatchTolerance {
                    absolute: bad,
                    relative: 0.0
                },
                ..cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            MatchConfig {
                exterior: MatchTolerance {
                    absolute: 0.0,
                    relative: bad
                },
                ..cfg
            }
            .validate()
            .is_err()
        );
    }
    let first = observation(0, c(0.0), c(3.5), Ok(signature()));
    let second = observation(1, c(0.0), c(3.5), Ok(signature()));
    assert_eq!(
        compatibility(&first, &second, cfg),
        Compatibility::Compatible
    );
    for sign in [
        BranchSignature {
            left: signature().left.map(|k| -k),
            right: signature().right.map(|k| -k),
            ..signature()
        },
        BranchSignature {
            left: signature().left.map(|k| -k),
            ..signature()
        },
        BranchSignature {
            right: signature().left,
            left: signature().right,
            ..signature()
        },
        BranchSignature {
            k0: c(-2.0),
            ..signature()
        },
    ] {
        let other = observation(1, c(0.0), c(3.5), Ok(sign));
        assert_eq!(
            compatibility(&first, &other, cfg),
            Compatibility::Ambiguous(Ambiguity::IncompatibleBranch)
        );
        let (groups, _) = reconcile(
            &[observation(0, c(0.0), c(3.5), Ok(signature())), other],
            cfg,
        );
        assert_eq!(groups.len(), 2);
    }
    let near = BranchSignature {
        left: Some(c(1e-12)),
        ..signature()
    };
    let a = observation(0, c(0.0), c(3.5), Ok(near));
    let b = observation(1, c(0.0), c(3.5), Ok(near));
    assert_eq!(
        compatibility(&a, &b, cfg),
        Compatibility::Ambiguous(Ambiguity::ExteriorSignAmbiguous)
    );
    let failed = observation(
        2,
        c(0.0),
        c(3.5),
        Err(DeterminantEvaluationError::Backend { source: Fault }),
    );
    assert!(source_contains::<Fault>(failed.signature().unwrap_err()));
    let (g, d) = reconcile(&[first, second, failed], cfg);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0].members(), [0, 1]);
    assert!(d.iter().any(|d| d.reason == Ambiguity::SignatureFailed));
}
#[test]
fn grouping_is_deterministic_nontransitive_and_enclosure_conservative() {
    let tol = MatchTolerance {
        absolute: 0.1,
        relative: 0.0,
    };
    let cfg = MatchConfig {
        conformal: tol,
        physical: tol,
        exterior: tol,
    };
    let obs = vec![
        observation(0, c(0.0), c(3.5), Ok(signature())),
        observation(1, c(0.075), c(3.5), Ok(signature())),
        observation(2, c(0.15), c(3.5), Ok(signature())),
    ];
    let (groups, _) = reconcile(&obs, cfg);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].members(), [0, 1]);
    assert_eq!(groups[0].representative(), 0);
    assert_eq!(groups[1].members(), [2]);
    assert_eq!(reconcile(&obs, cfg).0, groups);
    let mut second = observation(1, c(0.075), c(3.5), Ok(signature()));
    second.enclosure = Rect {
        x_min: 0.05,
        x_max: 0.1,
        y_min: -0.1,
        y_max: 0.1,
    };
    assert_eq!(
        compatibility(&obs[0], &second, cfg),
        Compatibility::Ambiguous(Ambiguity::IncompatibleEnclosure)
    );
    let same_chart = observation(0, c(0.0), c(3.5), Ok(signature()));
    assert_eq!(
        compatibility(&obs[0], &same_chart, cfg),
        Compatibility::Separate
    );
    let periodic = observation(3, c(0.2), c(3.5), Ok(signature()));
    assert_eq!(
        compatibility(&obs[0], &periodic, MatchConfig::default()),
        Compatibility::Ambiguous(Ambiguity::UnsupportedOverlap)
    );
}

fn check_overlapping_physical<M>(p: ModalProblem<M, Compiled<Scatter2>>, z: C, expected: C)
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone + std::fmt::Debug,
    RegionError<M>: std::fmt::Debug,
{
    let r = ConformalRegion::new(z.re - 0.06, z.re + 0.06, z.im - 0.05, z.im + 0.05).unwrap();
    let first = domain(&p, r);
    let reference = z + C::new(0.01, 0.01);
    let coordinate = first
        .search
        .dispersion()
        .chart()
        .coordinate(reference)
        .unwrap();
    let second = p
        .domain(
            r,
            ReferenceSelection::Coordinate {
                reference,
                coordinate,
            },
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let plan = ChartPlan::explicit(p.clone(), vec![first, second], vec![]).unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert_eq!(report.charts().len(), 2);
    assert_eq!(report.observations().len(), 2, "{report:?}");
    assert_eq!(report.groups().len(), 1, "{report:?}");
    assert_eq!(report.groups()[0].members(), [0, 1]);
    assert_eq!(report.groups()[0].representative(), 0);
    for observation in report.observations() {
        assert!((observation.root().coordinate() - expected).norm() < 1e-7);
        assert_eq!(observation.candidate(), 0);
        assert!(observation.signature().is_ok());
        let evidence = observation.root().evidence();
        assert!(evidence.value.norm() <= evidence.residual_limit);
        assert!(inside(observation.enclosure(), evidence.z));
        assert!(
            observation
                .root()
                .provenance()
                .same_domain(plan.domains()[observation.chart()].provenance())
        );
    }
    let descriptions = report.describe_points(InterpretationConfig::default());
    assert_eq!(descriptions.len(), report.observations().len());
    for (observation, description) in report.observations().iter().zip(&descriptions) {
        assert!(std::ptr::eq(description.root, observation.root()));
        assert_eq!(
            description.observation,
            Some((observation.chart(), observation.candidate()))
        );
        let single = p.describe_root(
            &plan.domains()[observation.chart()],
            observation.root(),
            InterpretationConfig::default(),
        );
        assert_eq!(
            single.result.as_ref().unwrap(),
            description.result.as_ref().unwrap()
        );
        match p.context().coordinate() {
            SearchCoordinate::InPlane { .. } => assert_eq!(
                description
                    .result
                    .as_ref()
                    .unwrap()
                    .interpretation
                    .classification,
                Classification::Classified(ModeType::Guided)
            ),
            SearchCoordinate::Spectral { .. } => assert_eq!(
                description
                    .result
                    .as_ref()
                    .unwrap()
                    .interpretation
                    .classification,
                Classification::Unsupported(Unsupported::SolveDirection)
            ),
        }
        let other = domain(&p, r);
        assert!(matches!(
            p.describe_root(&other, observation.root(), InterpretationConfig::default())
                .result,
            Err(PointError::Input(InputError::ForeignDomain))
        ));
        let foreign = self::observation(0, c(0.0), c(3.5), Ok(signature()));
        assert!(matches!(
            p.describe_root(
                &plan.domains()[0],
                foreign.root(),
                InterpretationConfig::default()
            )
            .result,
            Err(PointError::Input(InputError::ForeignProblem))
        ));
    }
    for chart in report.charts() {
        let raw = chart.discovery().as_ref().unwrap();
        assert!(std::ptr::eq(chart.refinement().unwrap().discovery(), raw));
    }
}
#[test]
fn overlapping_physical_te_tm_guided_and_normal_spectral_roots() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let k0: f64 = 2.0;
        let beta: f64 = 3.5;
        let q = (4.0 * k0 * k0 - beta * beta).sqrt();
        let al = (beta * beta - k0 * k0).sqrt();
        let ar = (beta * beta - 2.25 * k0 * k0).sqrt();
        let (fc, fr) = match pol {
            Polarisation::TransverseElectric => (1.0, 1.0),
            Polarisation::TransverseMagnetic => (4.0, 2.25),
        };
        let d = ((al * fc / q).atan() + (ar * fc / (q * fr)).atan()) / q;
        let stack =
            Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
                .analytical_layer(
                    Constant::dielectric(4.0),
                    lamina_core::units::Length::centimetres(d),
                )
                .finalise();
        let p = InPlaneProblem::in_plane(stack, k0, pol, Scatter2::new()).unwrap();
        let z = p
            .map
            .physical_preimages(beta)
            .unwrap()
            .into_iter()
            .flatten()
            .max_by(|a, b| a.re.total_cmp(&b.re))
            .unwrap();
        check_overlapping_physical(p, z, c(beta));
        let thickness = 0.3;
        let expected = C::new(std::f64::consts::PI, -3.0_f64.ln()) / (2.0 * thickness);
        let z = (C::new(0.0, -1.0) * expected).ln();
        // This all-outgoing pole lies beyond the canonical width-pi selection.
        assert!(z.im < -std::f64::consts::FRAC_PI_2);
        let p = SpectralProblem::spectral(
            exterior(1.0),
            exterior(1.0),
            [Layer::new(
                lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                lamina_core::units::Length::centimetres(thickness),
            )],
            0.0,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        // The half-period copy has the same physical coordinate but incoming
        // exterior signs, and is not a zero of this outgoing cavity problem.
        let shifted = z + C::new(0.0, std::f64::consts::PI);
        let incoming = p
            .domain(
                ConformalRegion::new(
                    shifted.re - 0.02,
                    shifted.re + 0.02,
                    shifted.im - 0.02,
                    shifted.im + 0.02,
                )
                .unwrap(),
                ReferenceSelection::Coordinate {
                    reference: shifted,
                    coordinate: expected,
                },
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let first = p.map.map_raw(J::lift_constant(ndarray::arr0(z)));
        let opposite = p.map.map_raw(J::lift_constant(ndarray::arr0(shifted)));
        for (out, inc) in [
            (
                first
                    .exterior_wavevectors()
                    .left()
                    .unwrap()
                    .scalar()
                    .unwrap(),
                opposite
                    .exterior_wavevectors()
                    .left()
                    .unwrap()
                    .scalar()
                    .unwrap(),
            ),
            (
                first
                    .exterior_wavevectors()
                    .right()
                    .unwrap()
                    .scalar()
                    .unwrap(),
                opposite
                    .exterior_wavevectors()
                    .right()
                    .unwrap()
                    .scalar()
                    .unwrap(),
            ),
        ] {
            assert!((out.value()[()] - expected).norm() < 1e-10);
            assert!((inc.value()[()] + expected).norm() < 1e-10);
        }
        assert!(p.evaluate(&incoming, shifted).unwrap().value().norm() > 1e-3);
        check_overlapping_physical(p, z, expected);
    }
}
// Independent scalar round-trip characteristic on the selected lower chart:
// left kappa is the negative root and right kappa is the positive root.
// Its upper-chart counterpart has conjugate k0 and reversed exterior signs.
// The all-outgoing cavity reference is tested separately on explicit charts.
fn scalar_slab(k: C, pol: Polarisation) -> C {
    let beta = 1.5;
    let d = 0.3;
    let q = (4.0 * k * k - c(beta * beta)).sqrt();
    let l = -(k * k - c(beta * beta)).sqrt();
    let r = (2.25 * k * k - c(beta * beta)).sqrt();
    let (yc, yl, yr) = match pol {
        Polarisation::TransverseElectric => (q, l, r),
        Polarisation::TransverseMagnetic => (q / 4.0, l, r / 2.25),
    };
    c(1.0) - (yc - yl) / (yc + yl) * (yc - yr) / (yc + yr) * (C::new(0.0, 2.0) * q * d).exp()
}
#[test]
fn canonical_spectral_lower_upper_keep_physical_te_tm_evidence() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let mut expected = C::new(std::f64::consts::PI, -0.5 * (7.0_f64 / 3.0).ln()) / 0.6;
        for _ in 0..20 {
            let h = 1e-5;
            let f = scalar_slab(expected, pol);
            let derivative =
                (scalar_slab(expected + h, pol) - scalar_slab(expected - h, pol)) / (2.0 * h);
            expected -= f / derivative;
        }
        assert!(scalar_slab(expected, pol).norm() < 1e-12);
        let p = SpectralProblem::spectral(
            exterior(1.0),
            exterior(2.25),
            [Layer::new(
                lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                lamina_core::units::Length::centimetres(0.3),
            )],
            1.5,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        let plan = p
            .canonical_spectral_plan(
                WavenumberSearchExtent {
                    max_vacuum_wavenumber: 8.0,
                    imaginary_margin: 0.05,
                },
                [selection(), selection()],
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let batch = plan
            .discover(DiscoveryConfig {
                score_tolerance: 1e-5,
                ..DiscoveryConfig::default()
            })
            .unwrap();
        let report = plan
            .refine(&batch, RefinementConfig::default(), MatchConfig::default())
            .unwrap();
        assert_eq!(report.charts().len(), 2);
        assert!(
            report
                .charts()
                .iter()
                .all(|entry| entry.discovery().is_ok()),
            "{report:?}"
        );
        assert!(
            report
                .observations()
                .iter()
                .any(|o| (o.root().coordinate() - expected).norm() < 1e-6),
            "expected {expected:?}, {report:?}"
        );
        assert_eq!(report.exclusions().len(), 5);
        assert_eq!(report.exclusions(), plan.exclusions());
        assert_sheet_omissions(report.exclusions());
        for (index, chart) in report.charts().iter().enumerate() {
            let target = if index == 0 {
                expected
            } else {
                expected.conj()
            };
            let observation = report
                .observations()
                .iter()
                .find(|o| o.chart() == index && (o.root().coordinate() - target).norm() < 1e-6)
                .expect("selected-sheet scalar root missing");
            let signature = observation.signature().unwrap();
            if index == 0 {
                assert!(signature.left.unwrap().re < 0.0 && signature.right.unwrap().re > 0.0)
            } else {
                assert!(signature.left.unwrap().re > 0.0 && signature.right.unwrap().re < 0.0)
            }
            assert_eq!(chart.index(), index);
            assert!(chart.refinement().is_some());
            assert!(
                chart
                    .provenance()
                    .same_domain(plan.domains()[index].provenance())
            );
        }
    }
}

#[test]
fn physical_interpretation_keeps_original_signature_errors_and_batch_evidence() {
    let p = mock_problem(vec![]);
    let d = domain(&p, bounds());
    let roots = [C::new(-0.12, -0.05), C::new(0.15, 0.06)]
        .map(|z| d.search.dispersion().chart().coordinate(z).unwrap());
    let p = ModalProblem {
        solver: Solver {
            roots: roots.to_vec(),
            ..p.solver
        },
        context: p.context,
        map: p.map,
    };
    let plan = ChartPlan::explicit(
        p.clone(),
        vec![domain(&p, bounds())],
        vec![CoverageExclusion::Declared {
            description: "unsearched extension".into(),
            region: None,
        }],
    )
    .unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    let mut report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert_eq!(report.observations().len(), 2);
    let groups: Vec<_> = report
        .groups()
        .iter()
        .map(|g| (g.members().to_vec(), g.representative()))
        .collect();
    let exclusions = report.exclusions().to_vec();
    report.observations[0].signature = Err(ExecutionError::Optical(Box::new(
        DispersionError::Solver(DeterminantEvaluationError::Backend { source: Fault }),
    )));
    let descriptions = report.describe_points(InterpretationConfig::default());
    assert_eq!(descriptions.len(), 2);
    let failure = descriptions[0].result.as_ref().unwrap_err();
    let original = *failure.evaluation_error().unwrap();
    assert!(std::ptr::eq(
        original,
        report.observations()[0].signature().unwrap_err()
    ));
    assert!(source_contains::<Fault>(original));
    assert!(descriptions[1].result.is_ok());
    for (desc, obs) in descriptions.iter().zip(report.observations()) {
        assert!(std::ptr::eq(desc.root, obs.root()));
    }
    assert_eq!(exclusions, report.exclusions());
    assert_eq!(
        groups,
        report
            .groups()
            .iter()
            .map(|g| (g.members().to_vec(), g.representative()))
            .collect::<Vec<_>>()
    );
    assert!(std::ptr::eq(report.discovery(), &batch));
    let bad = report.describe_points(InterpretationConfig {
        relative: f64::NAN,
        ..InterpretationConfig::default()
    });
    assert!(
        bad.iter()
            .all(|d| matches!(d.result, Err(PointError::Input(InputError::InvalidConfig))))
    );
}
