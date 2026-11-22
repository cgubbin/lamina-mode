use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, Constant, Layer, Polarisation, Stack,
    Transfer2,
};
use lamina_mode::*;
use lamina_units::Length;
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn c(v: f64) -> Complex {
    Complex::new(v, 0.0)
}
fn stack() -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise()
}
fn exterior(v: f64) -> ConstantExterior {
    ConstantExterior::new(c(v), c(1.0)).unwrap()
}
fn frequency_input() -> FrequencyDomainInput {
    let k = c(5.5);
    FrequencyDomainInput {
        bounds: FrequencyRegion::new(4.6, 6.2, -3.2, 0.1).unwrap(),
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: k,
            kappa: [k, 1.5 * k],
        }),
        continuation: ContinuationConfig::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    }
}
fn conformal() -> ConformalSpectralProblem {
    ConformalSpectralProblem::new(
        exterior(1.0),
        exterior(2.25),
        [Layer::new(
            AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
            Length::centimetres(0.3),
        )],
        0.0,
        Polarisation::TransverseElectric,
    )
    .unwrap()
}
fn expected() -> Complex {
    Complex::new(std::f64::consts::PI / 0.6, -21.0_f64.ln() / 1.2)
}
fn cavity_domain(p: &ConformalSpectralProblem) -> Domain {
    let k = expected();
    let z = (-Complex::i() * k).ln();
    p.domain(
        ConformalRegion::new(z.re - 0.12, z.re + 0.12, z.im - 0.12, z.im + 0.12).unwrap(),
        ReferenceSelection::Coordinate {
            reference: z,
            coordinate: k,
        },
        ConformalChartConfig::default(),
        ASSUME,
    )
    .unwrap()
}
#[test]
fn external_construction_all_strategies_and_backends() {
    let _ = PropagationProblem::new(stack(), 2.0, Polarisation::TransverseElectric).unwrap();
    let _ = PropagationProblem::with_backend(
        stack(),
        2.0,
        Polarisation::TransverseMagnetic,
        Transfer2::new(),
    )
    .unwrap();
    let _ = conformal();
    let _ = ConformalSpectralProblem::with_backend(
        exterior(1.0),
        exterior(2.25),
        [],
        1.0,
        Polarisation::TransverseElectric,
        Transfer2::new(),
    )
    .unwrap();
    let p = FrequencyProblem::new(stack(), 0.0, Polarisation::TransverseElectric).unwrap();
    let d = p.domain(frequency_input(), ASSUME).unwrap();
    let t = FrequencyProblem::with_backend(
        stack(),
        0.0,
        Polarisation::TransverseElectric,
        Transfer2::new(),
    )
    .unwrap();
    let td = t.domain(frequency_input(), ASSUME).unwrap();
    let k = expected();
    let a = p.evaluate(&d, k).unwrap();
    let b = t.evaluate(&td, k).unwrap();
    assert!((a.value - b.value).norm() < 1e-10);
    assert!((a.derivative - b.derivative).norm() < 1e-8);
    assert_eq!(d.provenance().units(), "cm^-1");
    assert_eq!(a.coordinate_kind, CoordinateKind::Frequency);
    assert!(matches!(
        d.provenance().radiation(),
        RadiationProvenance::Unspecified
    ));
    let mut real = frequency_input();
    real.reference = FrequencyReference::RealRadiation {
        frequency: 5.5,
        choices: [RadiationChoice::OutgoingOrDecaying; 2],
        tolerance: RadiationTolerance::default(),
    };
    assert!(
        matches!(p.domain(real,ASSUME).unwrap().provenance().radiation(),RadiationProvenance::Verified(v) if v.all_outgoing())
    );
}
#[test]
fn external_domain_units_derivatives_and_ownership() {
    let p = conformal();
    let d = cavity_domain(&p);
    let k = expected();
    let z = (-Complex::i() * k).ln();
    let f = FrequencyProblem::new(stack(), 0.0, Polarisation::TransverseElectric).unwrap();
    let fd = f.domain(frequency_input(), ASSUME).unwrap();
    let a = p.evaluate(&d, z).unwrap();
    let b = f.evaluate(&fd, k).unwrap();
    assert_eq!(d.provenance().units(), "dimensionless");
    assert!((a.derivative / k - b.derivative).norm() < 1e-8);
    assert!(d.provenance().same_domain(&d.clone().provenance()));
    assert!(
        conformal()
            .evaluate(&d, z)
            .unwrap_err()
            .is_foreign_association()
    );
    assert!(p.plan(vec![], vec![]).is_err());
    assert!(p.plan(vec![d.clone(), d.clone()], vec![]).is_err());
    assert!(conformal().plan(vec![d.clone()], vec![]).is_err());
    assert!(
        p.plan(vec![fd], vec![])
            .unwrap_err()
            .find_source::<StrategyMismatch>()
            .is_some()
    );
    let before = d.provenance().requested_bounds();
    let plan = p.plan(vec![d], vec![]).unwrap();
    assert_eq!(plan.domains()[0].requested_bounds(), before);
    let invalid = plan
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig {
                native_atol: f64::NAN,
                ..Default::default()
            },
            MatchConfig::default(),
        )
        .unwrap_err();
    assert_eq!(invalid.stage(), Stage::Refinement);
    assert!(
        p.canonical_plan(
            WavenumberSearchExtent {
                max_vacuum_wavenumber: 8.0,
                imaginary_margin: 0.01
            },
            [ReferenceSelection::Sheet(CoordinateSheet::Principal); 2],
            ConformalChartConfig::default(),
            ASSUME
        )
        .is_err()
    );
}
fn returned_report() -> SolveReport {
    let p = conformal();
    let d = cavity_domain(&p);
    let plan = p.plan(vec![d.clone(), cavity_domain(&p)], vec![]).unwrap();
    let discovery = plan.discover(DiscoveryConfig::default()).unwrap();
    let first = discovery.entries().next().unwrap().unwrap();
    assert_eq!(first.estimates().len(), 1);
    assert_eq!(first.estimates().next().unwrap().raw().multiplicity, 1);
    let report = plan
        .refine(
            &discovery,
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    assert_eq!(
        report.discovery().entries().len(),
        discovery.entries().len()
    );
    for (a, b) in report.discovery().entries().zip(discovery.entries()) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!(a.termination(), b.termination());
        assert_eq!(a.leaves().len(), b.leaves().len());
        assert_eq!(a.summary().iterations, b.summary().iterations);
    }
    assert!(
        conformal()
            .plan(vec![cavity_domain(&conformal())], vec![])
            .is_err()
    );
    let other = p.plan(vec![cavity_domain(&p)], vec![]).unwrap();
    assert!(
        other
            .refine(
                &discovery,
                RefinementConfig::default(),
                MatchConfig::default()
            )
            .is_err()
    );
    report
}
#[test]
fn owned_reports_points_and_overlap_survive_execution_locals() {
    let report = returned_report();
    assert!(report.backend().contains("Scatter2"));
    assert_eq!(report.observations()[0].backend(), report.backend());
    assert_eq!(
        report.observations()[0].provenance().strategy(),
        Strategy::ConformalSpectral
    );
    assert_eq!(report.observations().len(), 2);
    assert_eq!(report.groups().len(), 1);
    assert_eq!(report.groups()[0].members(), &[0, 1]);
    assert_eq!(report.charts().len(), 2);
    for (i, point) in report.observations().iter().enumerate() {
        assert!((point.physical_coordinate() - expected()).norm() < 1e-8);
        assert_eq!(point.chart(), i);
        assert!(matches!(
            report.charts()[i].candidates.as_ref().unwrap()[point.candidate()].status,
            CandidateStatus::Accepted
        ));
        let evidence = point.evidence();
        assert!(evidence.value.norm() <= evidence.residual_limit);
        assert!(evidence.correction.unwrap() <= evidence.correction_limit);
        assert!(
            point
                .provenance()
                .same_domain(&report.charts()[i].provenance)
        );
        let description = point.describe(InterpretationConfig::default()).unwrap();
        assert_eq!(
            description.interpretation.classification,
            Classification::Unsupported(Unsupported::SolveDirection)
        );
        assert_eq!(
            description.interpretation.radiation,
            RadiationQualification::Unspecified
        );
    }
    let point = report.observations()[0].clone();
    drop(report);
    assert!((point.physical_coordinate() - expected()).norm() < 1e-8);
    assert!(point.describe(InterpretationConfig::default()).is_ok());
}
#[test]
fn public_frequency_errors_keep_exclusions_and_side_work() {
    let p = FrequencyProblem::new(stack(), 0.0, Polarisation::TransverseElectric).unwrap();
    let mut invalid = frequency_input();
    invalid.continuation.integral_tolerance = f64::NAN;
    assert!(matches!(
        p.domain(invalid, ASSUME).unwrap_err().frequency_error(),
        Some(FrequencyError::InvalidConfig)
    ));
    let mut i = frequency_input();
    i.exclusions.push(FrequencyExclusion {
        region: FrequencyRegion::new(6.2, 6.3, -0.1, 0.0).unwrap(),
        reason: ExclusionReason::ExteriorThreshold { side: 1 },
    });
    let error = p.domain(i, ASSUME).unwrap_err();
    assert!(matches!(
        error.frequency_error(),
        Some(FrequencyError::Excluded(_))
    ));
    let mut i = frequency_input();
    i.continuation.max_evaluations = 1;
    let d = p.domain(i, ASSUME).unwrap();
    let error = p.evaluate(&d, expected()).unwrap_err();
    assert!(matches!(
        error.frequency_error(),
        Some(FrequencyError::Exterior {
            side: 0,
            source: ExteriorFailure::EvaluationBudget,
            evaluations: 1
        })
    ));
    let plan = p.plan(vec![d], vec![]).unwrap();
    let report = plan
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    let discovery = report.discovery().entries().next().unwrap().unwrap_err();
    assert!(discovery.frequency_error().is_some());
    assert!(report.observations().is_empty());
    assert!(report.charts()[0].candidates.is_none());
    assert!(!report.exclusions().is_empty());
}
#[test]
fn frequency_owned_coverage_keeps_overlapping_failure_and_all_outcomes() {
    let p = FrequencyProblem::new(stack(), 0.0, Polarisation::TransverseElectric).unwrap();
    let k = expected();
    let anchor = k + c(0.05);
    let mut good = frequency_input();
    good.bounds = FrequencyRegion::new(k.re - 0.15, k.re + 0.15, k.im - 0.15, k.im + 0.15).unwrap();
    good.reference = FrequencyReference::Explicit(ExteriorReferenceValues {
        frequency: anchor,
        kappa: [anchor, 1.5 * anchor],
    });
    let mut bad = good.clone();
    bad.continuation.max_evaluations = 1;
    let d = p.domain(good, ASSUME).unwrap();
    let before = d.provenance();
    let plan = p
        .plan(vec![d, p.domain(bad, ASSUME).unwrap()], vec![])
        .unwrap();
    let report = plan
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    assert_eq!(report.observations().len(), 1);
    assert!(report.observations()[0].provenance().same_domain(&before));
    let entries: Vec<_> = report.discovery().entries().collect();
    assert!(entries[0].is_ok());
    assert!(entries[1].is_err());
    assert!(matches!(
        entries[1].unwrap_err().frequency_error(),
        Some(FrequencyError::Exterior {
            source: ExteriorFailure::EvaluationBudget,
            ..
        })
    ));
    assert!(report.charts()[0].candidates.is_some());
    assert!(report.charts()[1].candidates.is_none());
    assert_eq!(report.groups()[0].members(), &[0]);
    let candidate = &report.charts()[0].candidates.as_ref().unwrap()[0];
    assert!(candidate.work.evaluations > 0);
    assert!(candidate.last_valid.is_some());
}
#[test]
fn canonical_plans_and_budget_clusters_keep_public_diagnostics() {
    let propagation =
        PropagationProblem::new(stack(), 2.0, Polarisation::TransverseElectric).unwrap();
    let canonical = propagation
        .canonical_plan(
            InPlaneSearchExtent {
                max_parallel_wavenumber: 4.0,
                imaginary_margin: 0.05,
            },
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    assert_eq!(canonical.domains().len(), 1);
    let p = ConformalSpectralProblem::new(
        exterior(1.0),
        exterior(2.25),
        [],
        1.0,
        Polarisation::TransverseElectric,
    )
    .unwrap();
    let plan = p
        .canonical_plan(
            WavenumberSearchExtent {
                max_vacuum_wavenumber: 8.0,
                imaginary_margin: 0.05,
            },
            [ReferenceSelection::Sheet(CoordinateSheet::Principal); 2],
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    assert_eq!(plan.domains().len(), 2);
    assert!(
        plan.exclusions()
            .iter()
            .any(|e| matches!(e, CoverageExclusion::OmittedExteriorSheetSectors { .. }))
    );
    let p = conformal();
    let k = expected();
    let z = (-Complex::i() * k).ln();
    let bounds = ConformalRegion::new(
        (k.norm() * 0.7).ln(),
        (k.norm() * 6.0).ln(),
        z.im - 0.12,
        -1.45,
    )
    .unwrap();
    let d = p
        .domain(
            bounds,
            ReferenceSelection::Coordinate {
                reference: z,
                coordinate: k,
            },
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let report = p
        .plan(vec![d], vec![])
        .unwrap()
        .solve(
            DiscoveryConfig {
                max_iterations: 0,
                ..Default::default()
            },
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    let discovery = report.discovery().entries().next().unwrap().unwrap();
    assert_eq!(*discovery.termination(), Termination::ExceededMaxIterations);
    assert!(!discovery.leaves().is_empty());
    assert!(discovery.estimates().any(|e| e.raw().multiplicity > 1));
    assert!(
        report.charts()[0]
            .candidates
            .as_ref()
            .unwrap()
            .iter()
            .any(|o| matches!(o.status, CandidateStatus::UnresolvedCluster))
    );
}
// Interior mutability is used only to inject post-acceptance failures. Real
// callers must keep analytical material behaviour stable for a problem's lifetime.
#[derive(Clone, Debug)]
struct FaultyMaterial(std::sync::Arc<std::sync::atomic::AtomicBool>);
impl FaultyMaterial {
    fn value(&self, good: f64) -> f64 {
        if self.0.load(std::sync::atomic::Ordering::SeqCst) {
            f64::NAN
        } else {
            good
        }
    }
}
macro_rules! material_values {
    ($trait:ident,$eps:ident,$mu:ident,$element:ty) => {
        impl lamina_core::$trait for FaultyMaterial {
            fn $eps<I, T>(&self, k: I) -> I::Mapped<T>
            where
                T: lamina_core::ComplexScalar<RealField = f64>,
                I: lamina_core::Sampled<Elem = $element>,
            {
                k.map(|_| T::from_real(self.value(2.25)))
            }
            fn $mu<I, T>(&self, k: I) -> I::Mapped<T>
            where
                T: lamina_core::ComplexScalar<RealField = f64>,
                I: lamina_core::Sampled<Elem = $element>,
            {
                k.map(|_| T::from_real(self.value(1.0)))
            }
        }
    };
}
// Material's real type needs its own implementation.
impl lamina_core::Material for FaultyMaterial {
    type Real = f64;
    fn relative_permittivity<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: lamina_core::ComplexScalar<RealField = f64>,
        I: lamina_core::Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(self.value(2.25)))
    }
    fn relative_permeability<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: lamina_core::ComplexScalar<RealField = f64>,
        I: lamina_core::Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(self.value(1.0)))
    }
}
material_values!(
    MeromorphicMaterial,
    relative_permittivity_complex,
    relative_permeability_complex,
    T
);
macro_rules! material_derivatives {
    ($trait:ident,$eps:ident,$mu:ident,$element:ty) => {
        impl lamina_core::$trait for FaultyMaterial {
            fn $eps<I, T>(&self, k: I, _: lamina_core::DerivativeOrder) -> I::Mapped<T>
            where
                T: lamina_core::ComplexScalar<RealField = f64>,
                I: lamina_core::Sampled<Elem = $element>,
            {
                k.map(|_| T::from_real(0.0))
            }
            fn $mu<I, T>(&self, k: I, _: lamina_core::DerivativeOrder) -> I::Mapped<T>
            where
                T: lamina_core::ComplexScalar<RealField = f64>,
                I: lamina_core::Sampled<Elem = $element>,
            {
                k.map(|_| T::from_real(0.0))
            }
        }
    };
}
material_derivatives!(
    DifferentiableMaterial,
    relative_permittivity_derivative,
    relative_permeability_derivative,
    f64
);
material_derivatives!(
    DifferentiableMeromorphicMaterial,
    relative_permittivity_complex_derivative,
    relative_permeability_complex_derivative,
    T
);
#[test]
fn accepted_guided_point_survives_unavailable_interpretation_and_checked_failure() {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (k0, beta) = (2.0_f64, 3.5_f64);
    let q = (4.0 * k0 * k0 - beta * beta).sqrt();
    let al = (beta * beta - k0 * k0).sqrt();
    let ar = (beta * beta - 2.25 * k0 * k0).sqrt();
    let thickness = ((al / q).atan() + (ar / q).atan()) / q;
    let s = Stack::from_analytical_materials(Constant::vacuum(), FaultyMaterial(flag.clone()))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(thickness))
        .finalise();
    let p = PropagationProblem::new(s, k0, Polarisation::TransverseElectric).unwrap();
    let z = ((al + ar) / (2.0 * k0)).ln();
    let d = p
        .radiation_domain(
            ConformalRegion::new(z - 0.08, z + 0.08, -0.06, 0.06).unwrap(),
            RadiationReference {
                native_coordinate: c(z),
                coordinate: beta,
                choices: [RadiationChoice::OutgoingOrDecaying; 2],
                tolerance: RadiationTolerance::default(),
            },
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let plan = p.plan(vec![d.clone()], vec![]).unwrap();
    let report = plan
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    let point = &report.observations()[0];
    assert_eq!(
        point
            .describe(InterpretationConfig::default())
            .unwrap()
            .interpretation
            .classification,
        Classification::Classified(ModeType::Guided)
    );
    let sampling = lamina_core::FieldSampling::new()
        .left_exterior(lamina_core::ExteriorSampling::point(Length::zero()));
    point
        .recover_fields_with(
            lamina_core::Scatter2::new(),
            &sampling,
            FieldRecoverySettings::default(),
        )
        .unwrap();
    let normalized = point
        .normalize_with(
            lamina_core::Transfer2::new(),
            PoleVariable::BetaAtFixedK0,
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
            PoleNormalizationSettings::default(),
            FieldRecoverySettings::default(),
        )
        .unwrap();
    flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let normalization_failure = normalized.fields(&sampling).unwrap_err();
    assert!(matches!(
        normalization_failure.cause,
        FieldRecoveryError::Normalization(_)
    ));
    assert_eq!(
        normalization_failure.point.physical_coordinate(),
        point.physical_coordinate()
    );

    let field_failure = point
        .recover_fields_with(
            lamina_core::Scatter2::new(),
            &sampling,
            FieldRecoverySettings::default(),
        )
        .unwrap_err();
    assert!(matches!(
        field_failure.cause,
        FieldRecoveryError::Core(lamina_core::UnscaledFieldError::InvalidInput)
    ));
    assert_eq!(
        field_failure.point.physical_coordinate(),
        point.physical_coordinate()
    );
    let error = point.describe(InterpretationConfig::default()).unwrap_err();
    assert_eq!(error.stage(), Stage::Interpretation);
    assert!(matches!(
        error.find_source::<InterpretationError>(),
        Some(InterpretationError::NonFinite)
    ));
    assert!((point.physical_coordinate() - c(beta)).norm() < 1e-8);
    assert_eq!(report.observations().len(), 1);
    assert_eq!(report.groups().len(), 1);
    let error = p.evaluate(&d, c(z)).unwrap_err();
    assert_eq!(error.optical_failure(), Some(OpticalFailure::Output));
    // Refinement errors keep discovery and do not revoke previous acceptance.
    let failed = plan
        .refine(
            report.discovery(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    assert!(failed.observations().is_empty());
    assert!(
        matches!(&failed.charts()[0].candidates.as_ref().unwrap()[0].status,CandidateStatus::EvaluationFailed(e) if e.optical_failure()==Some(OpticalFailure::Output))
    );
}
#[test]
fn public_backend_failure_retains_original_core_source() {
    let huge = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(1e6))
        .finalise();
    let p = FrequencyProblem::with_backend(
        huge,
        0.0,
        Polarisation::TransverseElectric,
        Transfer2::new(),
    )
    .unwrap();
    let d = p.domain(frequency_input(), ASSUME).unwrap();
    let error = p.evaluate(&d, expected()).unwrap_err();
    assert_eq!(error.optical_failure(), Some(OpticalFailure::Backend));
    assert!(
        error
            .find_source::<lamina_core::backend::Transfer2Error>()
            .is_some()
    );
    let report = p
        .plan(vec![d], vec![])
        .unwrap()
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    let error = report.discovery().entries().next().unwrap().unwrap_err();
    assert_eq!(error.optical_failure(), Some(OpticalFailure::Backend));
    assert!(
        error
            .find_source::<lamina_core::backend::Transfer2Error>()
            .is_some()
    );
}
#[test]
fn opposite_frequency_branches_keep_separate_public_observations() {
    let p = FrequencyProblem::new(stack(), 0.0, Polarisation::TransverseElectric).unwrap();
    let mut domains = vec![];
    for sign in [1.0, -1.0] {
        let k = expected();
        let target = if sign > 0.0 { k } else { k.conj() };
        let anchor = target + c(0.04);
        let mut i = frequency_input();
        i.bounds =
            FrequencyRegion::new(k.re - 0.12, k.re + 0.12, target.im - 0.12, target.im + 0.12)
                .unwrap();
        i.reference = FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: anchor,
            kappa: [sign * anchor, sign * 1.5 * anchor],
        });
        domains.push(p.domain(i, ASSUME).unwrap());
    }
    let report = p
        .plan(domains, vec![])
        .unwrap()
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    assert_eq!(report.observations().len(), 2);
    assert_eq!(report.groups().len(), 2);
    let a = report.observations()[0].signature().unwrap();
    let b = report.observations()[1].signature().unwrap();
    assert!(a.left.re > 0.0 && b.left.re < 0.0);
    assert!(a.right.re > 0.0 && b.right.re < 0.0);
}
