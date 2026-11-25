use super::*;
fn c(x: f64) -> C {
    C::new(x, 0.0)
}
fn data(beta: C, signs: [f64; 2], eps: [f64; 2]) -> PhysicalData {
    PhysicalData {
        k0: c(1.0),
        beta,
        kappa: [0, 1].map(|j| Some(C::new(0.0, 1.0) * (beta * beta - c(eps[j])).sqrt() * signs[j])),
        epsilon: eps.map(c),
        mu: [c(1.0); 2],
    }
}
fn describe(d: PhysicalData) -> PhysicalDetails {
    interpret(
        d,
        SearchCoordinate::InPlane { fixed_k0: 1.0 },
        Polarisation::TransverseElectric,
        &RadiationProvenance::Unspecified,
        InterpretationConfig::default(),
    )
    .unwrap()
}
#[test]
fn all_eight_dielectric_sectors_follow_independent_alpha_signs() {
    use ModeType::*;
    // Each principal alpha solves alpha²=beta²-epsilon*k0². Choosing its sign
    // independently gives the reference paper's four sheets for each beta sign.
    let cases = [
        (0.2, [1.0, 1.0], Absorptive),
        (0.2, [-1.0, 1.0], LeftLeaky),
        (0.2, [1.0, -1.0], RightLeaky),
        (0.2, [-1.0, -1.0], TwoSidedLeaky),
        (-0.2, [1.0, 1.0], Gain),
        (-0.2, [-1.0, 1.0], LeftSource),
        (-0.2, [1.0, -1.0], RightSource),
        (-0.2, [-1.0, -1.0], TwoSidedSource),
    ];
    for (imag, signs, expected) in cases {
        for eps in [[1.0, 2.25], [2.25, 1.0]] {
            for pol in [
                Polarisation::TransverseElectric,
                Polarisation::TransverseMagnetic,
            ] {
                let d = data(C::new(2.0, imag), signs, eps);
                for j in 0..2 {
                    let alpha = d.alpha()[j].unwrap();
                    assert!((alpha * alpha - d.beta * d.beta + c(eps[j])).norm() < 1e-12);
                    assert_eq!(alpha.re.signum(), signs[j]);
                    assert_eq!(alpha.im.signum(), imag.signum() * signs[j]);
                }
                let result = interpret(
                    d,
                    SearchCoordinate::InPlane { fixed_k0: 1.0 },
                    pol,
                    &RadiationProvenance::Unspecified,
                    InterpretationConfig::default(),
                )
                .unwrap();
                assert_eq!(
                    result.interpretation.classification,
                    Classification::Classified(expected)
                );
            }
        }
    }
    for (imag, s, expected) in [
        (0.2, 1.0, Absorptive),
        (0.2, -1.0, TwoSidedLeaky),
        (-0.2, 1.0, Gain),
        (-0.2, -1.0, TwoSidedSource),
    ] {
        assert_eq!(
            describe(data(C::new(2.0, imag), [s; 2], [1.0; 2]))
                .interpretation
                .classification,
            Classification::Classified(expected)
        );
    }
}
#[test]
fn guided_cutoff_improper_and_unsupported_states_remain_explicit() {
    use Classification::*;
    let guided = data(c(2.0), [1.0; 2], [1.0, 2.25]);
    assert_eq!(
        describe(guided).interpretation.classification,
        Classified(ModeType::Guided)
    );
    assert_eq!(
        describe(data(c(2.0), [-1.0; 2], [1.0, 2.25]))
            .interpretation
            .classification,
        Boundary(BoundaryReason::ImproperRealAxis)
    );
    assert_eq!(
        describe(data(c(1.5), [1.0; 2], [1.0, 2.25]))
            .interpretation
            .classification,
        Boundary(BoundaryReason::Cutoff)
    );
    let mut inconsistent = guided;
    inconsistent.kappa[0] = inconsistent.kappa[0].map(|k| k + c(0.1));
    assert_eq!(
        describe(inconsistent).interpretation.classification,
        Boundary(BoundaryReason::InconsistentEvidence)
    );
    for beta in [c(-2.0), c(0.0)] {
        assert_eq!(
            describe(data(beta, [1.0; 2], [1.0, 2.25]))
                .interpretation
                .classification,
            Unsupported(super::Unsupported::PropagationDirection)
        );
    }
    for (e, m) in [(C::new(1.0, 0.1), c(1.0)), (c(1.0), c(2.0))] {
        let mut d = guided;
        d.epsilon[0] = e;
        d.mu[0] = m;
        let result = describe(d);
        assert_eq!(
            result.interpretation.classification,
            Unsupported(super::Unsupported::ExteriorMaterial)
        );
        assert_eq!(
            result.interpretation.exterior,
            [ExteriorBehaviour::Decay; 2]
        );
    }
    let d = data(C::new(2.0, 1e-5), [1.0; 2], [1.0, 2.25]);
    let cfg = InterpretationConfig {
        flux_absolute: 0.1,
        ..InterpretationConfig::default()
    };
    let p = interpret(
        d,
        SearchCoordinate::InPlane { fixed_k0: 1.0 },
        Polarisation::TransverseElectric,
        &RadiationProvenance::Unspecified,
        cfg,
    )
    .unwrap();
    assert_eq!(
        p.interpretation.classification,
        Boundary(BoundaryReason::NearSectorBoundary)
    );
}
#[test]
fn component_bands_scale_dimensionally_and_validate_inputs() {
    let d = data(C::new(2.0, 0.2), [-1.0, 1.0], [1.0, 2.25]);
    let cfg = InterpretationConfig {
        absolute: 0.0,
        ..InterpretationConfig::default()
    };
    let run = |d| {
        interpret(
            d,
            SearchCoordinate::InPlane { fixed_k0: d.k0.re },
            Polarisation::TransverseElectric,
            &RadiationProvenance::Unspecified,
            cfg,
        )
        .unwrap()
    };
    let first = run(d);
    let scale = 1e-4;
    let scaled = run(PhysicalData {
        k0: d.k0 * scale,
        beta: d.beta * scale,
        kappa: d.kappa.map(|v| v.map(|v| v * scale)),
        ..d
    });
    assert!(
        (scaled.interpretation.wavenumber_threshold
            - first.interpretation.wavenumber_threshold * scale)
            .abs()
            < 1e-20
    );
    assert_eq!(
        first.interpretation.exterior,
        [ExteriorBehaviour::Growth, ExteriorBehaviour::Decay]
    );
    assert_eq!(first.interpretation.evolution, Evolution::Attenuation);
    assert_eq!(
        first.interpretation.classification,
        scaled.interpretation.classification
    );
    for j in 0..2 {
        let (Flux::Supported { coefficient: a, .. }, Flux::Supported { coefficient: b, .. }) =
            (first.interpretation.flux[j], scaled.interpretation.flux[j])
        else {
            panic!()
        };
        assert!((a - b).abs() < 1e-14);
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let bad = InterpretationConfig {
            relative: value,
            ..cfg
        };
        assert_eq!(
            interpret(
                d,
                SearchCoordinate::InPlane { fixed_k0: 1.0 },
                Polarisation::TransverseElectric,
                &RadiationProvenance::Unspecified,
                bad
            ),
            Err(InputError::InvalidConfig)
        );
    }
    assert!(
        InterpretationConfig { cutoff: 0.0, ..cfg }
            .validate()
            .is_err()
    );
    let mut bad = d;
    bad.k0 = C::new(f64::NAN, 0.0);
    assert_eq!(
        interpret(
            bad,
            SearchCoordinate::InPlane { fixed_k0: 1.0 },
            Polarisation::TransverseElectric,
            &RadiationProvenance::Unspecified,
            cfg
        ),
        Err(InputError::NonFinite)
    );
    let huge = InterpretationConfig {
        relative: f64::MAX,
        ..cfg
    };
    assert_eq!(
        interpret(
            d,
            SearchCoordinate::InPlane { fixed_k0: 1.0 },
            Polarisation::TransverseElectric,
            &RadiationProvenance::Unspecified,
            huge
        ),
        Err(InputError::InvalidConfig)
    );
}
#[test]
fn spectral_temporal_components_do_not_assign_paper_labels_or_flux() {
    for (imag, evolution) in [
        (-0.2, Evolution::TemporalDecay),
        (0.2, Evolution::TemporalGrowth),
        (0.0, Evolution::Boundary),
    ] {
        let d = PhysicalData {
            k0: C::new(2.0, imag),
            beta: c(0.0),
            kappa: [Some(C::new(2.0, imag)); 2],
            epsilon: [c(1.0); 2],
            mu: [c(1.0); 2],
        };
        let result = interpret(
            d,
            SearchCoordinate::Spectral {
                fixed_k_parallel: 0.0,
            },
            Polarisation::TransverseElectric,
            &RadiationProvenance::Unspecified,
            InterpretationConfig::default(),
        )
        .unwrap();
        assert_eq!(result.interpretation.evolution, evolution);
        assert_eq!(
            result.interpretation.classification,
            Classification::Unsupported(Unsupported::SolveDirection)
        );
        assert_eq!(
            result.interpretation.radiation,
            RadiationQualification::Unspecified
        );
        assert_eq!(
            result.interpretation.flux,
            [Flux::Unsupported(Unsupported::SolveDirection); 2]
        );
    }
}

#[derive(Clone, Debug)]
struct FragileMap {
    inner: ComplexInPlaneMap<J>,
    fail: Arc<std::sync::atomic::AtomicBool>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl ValidateConformalRegion<f64> for FragileMap {
    type Error = <ComplexInPlaneMap<J> as ValidateConformalRegion<f64>>::Error;
    fn validate_region(&self, r: &ConformalRegion<f64>) -> Result<(), Self::Error> {
        self.inner.validate_region(r)
    }
}
impl LinearCoordinateMap<J> for FragileMap {
    fn coordinate_squared_scalar(&self, z: C) -> C {
        self.inner.coordinate_squared_scalar(z)
    }
    fn coordinate_squared_log_derivative_scalar(&self, z: C) -> C {
        use std::sync::atomic::Ordering;
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail.load(Ordering::SeqCst) {
            C::new(f64::NAN, 0.0)
        } else {
            self.inner.coordinate_squared_log_derivative_scalar(z)
        }
    }
    fn map_with_coordinate_sign(&self, z: J, sign: C) -> crate::conformal::ConformalPoint<J> {
        self.inner.map_with_coordinate_sign(z, sign)
    }
}
#[test]
fn single_point_failure_retains_root_and_ownership_checks_precede_evaluation() {
    use crate::refinement::{CandidateStatus, RefinementConfig, refine_candidate};
    use lamina_core::{Constant, Scatter2};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let base = InPlaneProblem::in_plane(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25)).finalise(),
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let fail = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let p = ModalProblem {
        context: base.context,
        map: FragileMap {
            inner: base.map,
            fail: fail.clone(),
            calls: calls.clone(),
        },
        solver: base.solver,
    };
    let region = ConformalRegion::new(-0.2, 0.2, -0.1, 0.1).unwrap();
    let domain = p
        .domain(
            region,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    let z = C::new(0.1, 0.03);
    let coordinate = domain.search.dispersion().chart().coordinate(z).unwrap();
    // Only this test oracle is synthetic; it isolates post-acceptance evaluation.
    let estimate = argloc::SingularPointEstimate {
        location: z,
        enclosure: region.try_into().unwrap(),
        multiplicity: 1,
        kind: argloc::SingularPointEstimateKind::SinglePoint,
        target: argloc::SearchTarget::Zeros,
    };
    let outcome = refine_candidate(
        &estimate,
        0,
        domain.provenance(),
        RefinementConfig::default().validate().unwrap(),
        |_| Ok::<_, std::convert::Infallible>(FunctionBundle::new(c(0.0), c(1.0))),
        |_| Ok(coordinate),
    );
    let CandidateStatus::Accepted(root) = outcome.status() else {
        panic!("{outcome:?}")
    };
    assert!(
        p.describe_root(&domain, root, InterpretationConfig::default())
            .result
            .is_ok()
    );
    let foreign_domain = p
        .domain(
            region,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    calls.store(0, Ordering::SeqCst);
    fail.store(true, Ordering::SeqCst);
    assert!(matches!(
        p.describe_root(&foreign_domain, root, InterpretationConfig::default())
            .result,
        Err(PointError::Input(InputError::ForeignDomain))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let unavailable = p.describe_root(&domain, root, InterpretationConfig::default());
    assert!(std::ptr::eq(unavailable.root, root.as_ref()));
    let error = unavailable
        .result
        .as_ref()
        .unwrap_err()
        .evaluation_error()
        .unwrap();
    assert!(
        matches!(error,ExecutionError::Optical(source) if matches!(source.as_ref(),DispersionError::Chart(_)))
    );
    assert!(calls.load(Ordering::SeqCst) > 0);
}
