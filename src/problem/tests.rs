use super::*;
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, CanonicalCoordinates, ComplexScalar,
    Constant, DerivativeOrder, DeterminantEvaluationError, DifferentiableMaterial,
    DifferentiableMeromorphicMaterial, EvaluateMeromorphicMaterial, ExteriorWavevectors, Material,
    MeromorphicMaterial, Sampled, Scalar, ScalarAlgebra, Scatter2, SeedJet,
};
use lamina_units::Length;
use ndarray::arr0;
use std::{
    error::Error,
    sync::atomic::{AtomicUsize, Ordering},
};
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn c(x: f64) -> C {
    C::new(x, 0.0)
}
fn region() -> ConformalRegion<f64> {
    ConformalRegion::new(-0.3, 0.3, -0.2, 0.2).unwrap()
}
fn exterior(e: f64) -> ConstantExterior {
    ConstantExterior::new(c(e), c(1.0)).unwrap()
}
fn stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise()
}
fn problem() -> InPlaneProblem<Scatter2> {
    InPlaneProblem::in_plane(
        stack(),
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap()
}
fn domain<M, S>(p: &ModalProblem<M, S>) -> ProblemDomain<M, S>
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    RegionError<M>: std::fmt::Debug,
    S: ModeDeterminantSolver<J> + Clone,
{
    p.domain(
        region(),
        ReferenceSelection::Sheet(CoordinateSheet::Principal),
        ConformalChartConfig::default(),
        ASSUME,
    )
    .unwrap()
}
// A meromorphic finite layer and exterior with nonzero spectral derivative.
#[derive(Clone, Copy, Debug)]
struct Linear;
impl Material for Linear {
    type Real = f64;
    fn relative_permittivity<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|v| T::from_real(3.0 + 0.1 * v))
    }
    fn relative_permeability<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|v| T::from_real(1.2 + 0.03 * v))
    }
}
impl DifferentiableMaterial for Linear {
    fn relative_permittivity_derivative<I, T>(&self, k: I, o: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| {
            T::from_real(if o == DerivativeOrder::First {
                0.1
            } else {
                0.0
            })
        })
    }
    fn relative_permeability_derivative<I, T>(&self, k: I, o: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| {
            T::from_real(if o == DerivativeOrder::First {
                0.03
            } else {
                0.0
            })
        })
    }
}
impl MeromorphicMaterial for Linear {
    fn relative_permittivity_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|v| T::from_real(3.0) + v * T::from_real(0.1))
    }
    fn relative_permeability_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|v| T::from_real(1.2) + v * T::from_real(0.03))
    }
}
impl DifferentiableMeromorphicMaterial for Linear {
    fn relative_permittivity_complex_derivative<I, T>(
        &self,
        k: I,
        o: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| {
            T::from_real(if o == DerivativeOrder::First {
                0.1
            } else {
                0.0
            })
        })
    }
    fn relative_permeability_complex_derivative<I, T>(
        &self,
        k: I,
        o: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| {
            T::from_real(if o == DerivativeOrder::First {
                0.03
            } else {
                0.0
            })
        })
    }
}
fn compare<M>(
    p: &ModalProblem<M, Compiled<Scatter2>>,
    d: &ProblemDomain<M, Compiled<Scatter2>>,
    z: C,
) where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    RegionError<M>: std::fmt::Debug,
{
    let point = d
        .search
        .dispersion()
        .chart()
        .map(J::variable(arr0(z), 0).unwrap())
        .unwrap();
    let solver = Compiled::compile(
        p.context().stack(),
        Scatter2::new(),
        p.context().polarisation(),
    )
    .unwrap();
    let expected = solver
        .determinant(point.coordinates(), point.exterior_wavevectors())
        .unwrap();
    let actual = p.evaluate(d, z).unwrap();
    assert!((*actual.value() - expected.value()[()]).norm() < 1e-11);
    assert!((*actual.derivative() - expected.first()[()]).norm() < 1e-11);
    let h = 1e-6;
    let fd = (*p.evaluate(d, z + h).unwrap().value() - *p.evaluate(d, z - h).unwrap().value())
        / (2.0 * h);
    assert!((fd - *actual.derivative()).norm() < 1e-5 * (1.0 + fd.norm()));
}
#[test]
fn in_plane_uses_stack_exteriors_at_fixed_spectral_point() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let stack = Stack::from_analytical_materials(Linear, Constant::dielectric(2.25))
            .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
            .finalise();
        let p = InPlaneProblem::in_plane(stack, 2.0, pol, Scatter2::new()).unwrap();
        let d = domain(&p);
        let z = C::new(0.04, 0.03);
        let map = ComplexInPlaneMap::<J>::new(2.0, c(3.2), c(1.26), c(2.25), c(1.0)).unwrap();
        let chart = ConformalChart::new(
            map,
            region(),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();
        let a = d
            .search
            .dispersion()
            .chart()
            .map(J::variable(arr0(z), 0).unwrap())
            .unwrap();
        let b = chart.map(J::variable(arr0(z), 0).unwrap()).unwrap();
        for (a, b) in [
            (
                a.exterior_wavevectors().left().unwrap(),
                b.exterior_wavevectors().left().unwrap(),
            ),
            (
                a.exterior_wavevectors().right().unwrap(),
                b.exterior_wavevectors().right().unwrap(),
            ),
            (
                a.coordinates().parallel_angular_wavenumber(),
                b.coordinates().parallel_angular_wavenumber(),
            ),
        ] {
            assert_eq!(a.value(), b.value());
            assert_eq!(a.first(), b.first());
        }
        compare(&p, &d, z);
    }
    for k in [0.0, f64::NAN, f64::INFINITY] {
        assert!(
            InPlaneProblem::in_plane(
                stack(),
                k,
                Polarisation::TransverseElectric,
                Scatter2::new()
            )
            .is_err()
        );
    }
}
#[test]
fn complex_constant_material_preserves_individual_factors_and_zero_derivatives() {
    let e = C::new(2.0, 0.2);
    let m = C::new(1.3, -0.1);
    use material::Principal::{Diagonal, Isotropic};
    let isotropic = ConstantExterior::new(e, m).unwrap().region();
    let diagonal = ConstantExterior::diagonal([e, 2.0 * e, 3.0 * e], [m, 2.0 * m, 3.0 * m])
        .unwrap()
        .region();
    assert_eq!(diagonal.class(), lamina_core::ConstitutiveClass::Diagonal);
    let region = lamina_core::StackRegion::LeftExterior;
    for k in [c(2.0), C::new(1.0, 0.4)] {
        let handle = isotropic.scalar().unwrap();
        assert_eq!(handle.evaluate_relative_permittivity_complex(Scalar(k)), e);
        assert_eq!(handle.evaluate_relative_permeability_complex(Scalar(k)), m);
        assert_eq!(
            material::tensors_and_rates(region, &isotropic, k).unwrap(),
            [[Isotropic(e), Isotropic(m)], [Isotropic(c(0.0)); 2]]
        );
        assert_eq!(
            material::tensors_and_rates(region, &diagonal, k).unwrap(),
            [
                [
                    Diagonal([e, 2.0 * e, 3.0 * e]),
                    Diagonal([m, 2.0 * m, 3.0 * m])
                ],
                [Diagonal([c(0.0); 3]); 2]
            ]
        );
    }
    for invalid in [c(0.0), c(f64::NAN), C::new(1.0, f64::INFINITY)] {
        assert!(ConstantExterior::new(invalid, m).is_err());
        assert!(ConstantExterior::new(e, invalid).is_err());
        assert!(ConstantExterior::diagonal([e, e, invalid], [m; 3]).is_err());
    }
}
#[test]
fn spectral_keeps_dispersive_layers_and_shared_complex_exterior_materials() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let left = ConstantExterior::new(C::new(1.2, 0.03), C::new(1.1, 0.02)).unwrap();
        let right = exterior(2.25);
        let p = SpectralProblem::spectral(
            left,
            right,
            [Layer::new(
                AnalyticalMaterialHandle::new(Linear),
                Length::centimetres(0.3),
            )],
            1.5,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        let d = p
            .domain(
                ConformalRegion::new(-0.2, 0.2, 0.2, 0.8).unwrap(),
                ReferenceSelection::Sheet(CoordinateSheet::Opposite),
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let k = C::new(1.7, 0.3);
        let l = p.context().stack().left_exterior().scalar().unwrap();
        assert_eq!(
            l.evaluate_relative_permittivity_complex(Scalar(k)),
            left.epsilon()[0]
        );
        assert_eq!(
            l.evaluate_relative_permeability_complex(Scalar(k)),
            left.mu()[0]
        );
        compare(&p, &d, C::new(0.05, 0.4));
    }
    for fixed in [f64::NAN, f64::INFINITY] {
        assert!(
            SpectralProblem::spectral(
                exterior(1.0),
                exterior(2.25),
                [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
                fixed,
                Polarisation::TransverseElectric,
                Scatter2::new()
            )
            .is_err()
        );
    }
}

fn verify_domain<M>(p: &ModalProblem<M, Compiled<Scatter2>>, requested: ConformalRegion<f64>)
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    RegionError<M>: std::fmt::Debug,
{
    for sheet in [CoordinateSheet::Principal, CoordinateSheet::Opposite] {
        let selection = ReferenceSelection::Sheet(sheet);
        let d = p
            .domain(
                requested,
                selection,
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let provenance = d.provenance();
        let z = provenance.reference();
        let coordinate = provenance.reference_coordinate();
        assert_eq!(provenance.requested(), requested);
        assert_eq!(provenance.selection(), selection);
        assert_eq!(provenance.assumption(), ASSUME);
        assert!(provenance.belongs_to(p));
        let explicit = p
            .domain(
                requested,
                ReferenceSelection::Coordinate {
                    reference: z,
                    coordinate,
                },
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        assert_eq!(explicit.provenance().reference_coordinate(), coordinate);
        let a = d
            .search
            .dispersion()
            .chart()
            .map(J::variable(arr0(z), 0).unwrap())
            .unwrap();
        let b = explicit
            .search
            .dispersion()
            .chart()
            .map(J::variable(arr0(z), 0).unwrap())
            .unwrap();
        for (a, b) in [
            (
                a.exterior_wavevectors().left().unwrap(),
                b.exterior_wavevectors().left().unwrap(),
            ),
            (
                a.exterior_wavevectors().right().unwrap(),
                b.exterior_wavevectors().right().unwrap(),
            ),
        ] {
            assert_eq!(a.value(), b.value());
            assert_eq!(a.first(), b.first());
        }
        assert_eq!(*d.search.search_region(), requested);
        assert_eq!(
            p.clone().evaluate(&d, z).unwrap().value(),
            p.evaluate(&explicit, z).unwrap().value()
        );
        compare(p, &d, z);
    }
}
#[test]
fn domains_preserve_both_map_families_and_reference_forms() {
    verify_domain(&problem(), region());
    let spectral = SpectralProblem::spectral(
        exterior(1.0),
        exterior(2.25),
        [Layer::new(
            AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
            Length::centimetres(0.3),
        )],
        1.5,
        Polarisation::TransverseMagnetic,
        Scatter2::new(),
    )
    .unwrap();
    verify_domain(
        &spectral,
        ConformalRegion::new(-0.2, 0.2, 0.2, 0.8).unwrap(),
    );
}
fn contains_source<E: Error + 'static>(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if error.is::<E>() {
            return true;
        }
        match error.source() {
            Some(next) => error = next,
            None => return false,
        }
    }
}
#[test]
fn invalid_construction_preserves_sources_before_search() {
    let bad = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(-0.3))
        .finalise();
    let error =
        InPlaneProblem::in_plane(bad, 2.0, Polarisation::TransverseElectric, Scatter2::new())
            .unwrap_err();
    assert!(contains_source::<StackCompileError<f64>>(&error));
    let invalid =
        Stack::from_analytical_materials(Constant::dielectric(f64::NAN), Constant::vacuum())
            .finalise();
    let error = InPlaneProblem::in_plane(
        invalid,
        2.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap_err();
    assert!(contains_source::<InPlaneMapError>(&error));
    let p = problem();
    let branch = p.map.landmarks().unwrap().coordinate_branch_points()[0];
    let requested = ConformalRegion::new(
        branch.re - 0.1,
        branch.re + 0.1,
        branch.im - 0.2,
        branch.im - 1e-15,
    )
    .unwrap();
    assert!(matches!(
        p.domain(
            requested,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME
        ),
        Err(ConformalChartError::InvalidRegion(_))
    ));
    assert!(
        p.domain(
            region(),
            ReferenceSelection::Coordinate {
                reference: c(10.0),
                coordinate: c(1.0)
            },
            ConformalChartConfig::default(),
            ASSUME
        )
        .is_err()
    );
}
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("injected optical failure")]
struct Fault;
#[derive(Clone, Debug)]
struct Failing {
    failure: DeterminantEvaluationError<Fault>,
    calls: Arc<AtomicUsize>,
}
impl ModeDeterminantSolver<J> for Failing {
    type Error = DeterminantEvaluationError<Fault>;
    fn determinant(
        &self,
        _: &CanonicalCoordinates<J>,
        _: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(self.failure.clone())
    }
}
#[test]
fn foreign_domains_fail_before_evaluation_and_optical_sources_survive() {
    for failure in [
        DeterminantEvaluationError::NonFiniteInput,
        DeterminantEvaluationError::NonFiniteOutput,
        DeterminantEvaluationError::Backend { source: Fault },
    ] {
        let base = problem();
        let calls = Arc::new(AtomicUsize::new(0));
        let p = ModalProblem {
            context: base.context,
            map: base.map,
            solver: Failing {
                failure,
                calls: calls.clone(),
            },
        };
        let d = domain(&p);
        let other = ModalProblem {
            context: problem().context,
            map: p.map.clone(),
            solver: p.solver.clone(),
        };
        assert!(matches!(
            other.search(&d, ArgumentConfig::new(1e-6)),
            Err(ExecutionError::ForeignDomain)
        ));
        assert!(matches!(
            other.evaluate(&d, c(0.0)),
            Err(ExecutionError::ForeignDomain)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let error = p.clone().search(&d, ArgumentConfig::new(1e-6)).unwrap_err();
        assert!(contains_source::<DeterminantEvaluationError<Fault>>(&error));
        assert!(calls.load(Ordering::SeqCst) > 0);
        if matches!(p.solver.failure, DeterminantEvaluationError::Backend { .. }) {
            assert!(contains_source::<Fault>(&error));
        }
    }
}
#[derive(Clone, Debug)]
struct TwoRoots(C, C);
impl ModeDeterminantSolver<J> for TwoRoots {
    type Error = std::convert::Infallible;
    fn determinant(
        &self,
        coords: &CanonicalCoordinates<J>,
        _: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        let k = coords.parallel_angular_wavenumber();
        Ok(k.subtract(&k.constant(self.0))
            .multiply(&k.subtract(&k.constant(self.1))))
    }
}
#[test]
fn incomplete_results_preserve_provenance_clusters_and_domain_reuse() {
    let base = problem();
    let chart = ConformalChart::new(
        base.map.clone(),
        region(),
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap();
    let roots = TwoRoots(
        chart.coordinate(C::new(-0.15, -0.1)).unwrap(),
        chart.coordinate(C::new(-0.12, -0.08)).unwrap(),
    );
    let p = ModalProblem {
        context: base.context,
        map: base.map,
        solver: roots,
    };
    let d = domain(&p);
    let before = d.provenance().evaluation();
    let mut config = ArgumentConfig::new(1e-12);
    config.quad_tree.max_iter = 0;
    let raw = d.search.search(config.clone()).unwrap();
    for _ in 0..2 {
        let bound = p.clone().search(&d, config.clone()).unwrap();
        let result = bound.result();
        let provenance = bound.provenance();
        assert_eq!(provenance.requested(), region());
        assert_eq!(provenance.evaluation(), before);
        assert!(provenance.belongs_to(&p));
        assert_eq!(
            provenance.problem().polarisation(),
            Polarisation::TransverseElectric
        );
        assert_eq!(
            provenance.problem().coordinate(),
            SearchCoordinate::InPlane { fixed_k0: 2.0 }
        );
        assert_eq!(provenance.reference(), d.provenance().reference());
        assert_eq!(
            provenance.reference_coordinate(),
            d.provenance().reference_coordinate()
        );
        assert_eq!(provenance.assumption(), ASSUME);
        assert_eq!(provenance.selection(), d.provenance().selection());
        assert_eq!(
            *result.termination(),
            argloc::Termination::ExceededMaxIterations
        );
        assert_eq!(result.termination(), raw.termination());
        assert_eq!(result.summary().iterations, raw.summary().iterations);
        assert_eq!(result.summary().best_measure, raw.summary().best_measure);
        assert!(result.summary().elapsed.as_nanos() > 0);
        assert!(!result.leaves().is_empty());
        assert_eq!(result.leaves().len(), raw.leaves().len());
        for (a, b) in result.leaves().iter().zip(raw.leaves()) {
            assert_eq!(a.bounds, b.bounds);
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.data.root_count, b.data.root_count);
        }
        assert_eq!(result.estimates().len(), raw.estimates().len());
        for (a, b) in result.estimates().iter().zip(raw.estimates()) {
            assert_eq!(a.estimate(), b.estimate());
            assert_eq!(
                a.estimate().kind,
                argloc::SingularPointEstimateKind::ClusterCentroid
            );
            assert_eq!(a.estimate().multiplicity, 2);
        }
        let (result, provenance) = bound.into_parts();
        assert_eq!(result.len(), raw.len());
        assert!(provenance.belongs_to(&p));
        assert_eq!(d.provenance().evaluation(), before);
    }
}

fn check_physical<M>(p: &ModalProblem<M, Compiled<Scatter2>>, z: C, expected: C)
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    RegionError<M>: std::fmt::Debug,
{
    let requested =
        ConformalRegion::new(z.re - 0.06, z.re + 0.06, z.im - 0.05, z.im + 0.05).unwrap();
    let d = p
        .domain(
            requested,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    compare(p, &d, z);
    assert!(p.evaluate(&d, z).unwrap().value().norm() < 1e-10);
    let result = p.search(&d, ArgumentConfig::new(1e-7)).unwrap();
    assert_eq!(result.result().len(), 1);
    assert!(result.provenance().belongs_to(p));
    assert_eq!(
        result.provenance().problem().coordinate(),
        p.context().coordinate()
    );
    assert_eq!(
        result.provenance().problem().polarisation(),
        p.context().polarisation()
    );
    let estimate = &result.result().estimates()[0];
    assert_eq!(estimate.estimate().multiplicity, 1);
    let mode = estimate.mode();
    assert!((mode.coordinate() - expected).norm() < 1e-5, "{mode:?}");
    assert!(mode.residual().norm() < 1e-6);
    // Keep the discovery checks above, then exercise additive numerical
    // acceptance with both supported reference-selection forms.
    for selection in [
        ReferenceSelection::Sheet(CoordinateSheet::Principal),
        ReferenceSelection::Coordinate {
            reference: d.provenance().reference(),
            coordinate: d.provenance().reference_coordinate(),
        },
    ] {
        let refined_domain = p
            .domain(
                requested,
                selection,
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let discovery = p
            .search(&refined_domain, ArgumentConfig::new(1e-7))
            .unwrap();
        let cfg = crate::refinement::RefinementConfig::default();
        let report = p.refine(&refined_domain, &discovery, cfg).unwrap();
        assert_eq!(report.outcomes().len(), 1);
        let crate::refinement::CandidateStatus::Accepted(root) = report.outcomes()[0].status()
        else {
            panic!("{:?}", report.outcomes())
        };
        let evidence = root.evidence();
        assert!(evidence.value.norm() <= evidence.residual_limit);
        assert!(evidence.correction.unwrap() <= evidence.correction_limit);
        assert!((root.coordinate() - expected).norm() < 1e-7);
        let enclosure = discovery.result().estimates()[0].estimate().enclosure;
        assert!(
            evidence.z.re > enclosure.x_min
                && evidence.z.re < enclosure.x_max
                && evidence.z.im > enclosure.y_min
                && evidence.z.im < enclosure.y_max
        );
        assert!(root.provenance().same_domain(refined_domain.provenance()));
        let fresh = p.evaluate(&refined_domain, evidence.z).unwrap();
        assert_eq!(*fresh.value(), evidence.value);
        assert_eq!(*fresh.derivative(), evidence.derivative);
        assert!(
            (refined_domain
                .search
                .dispersion()
                .chart()
                .coordinate(evidence.z)
                .unwrap()
                - root.coordinate())
            .norm()
                < 1e-12
        );
    }

    let point = d
        .search
        .dispersion()
        .chart()
        .map(J::variable(arr0(mode.z()), 0).unwrap())
        .unwrap();
    let varying = match p.context().coordinate() {
        SearchCoordinate::InPlane { .. } => point.coordinates().parallel_angular_wavenumber(),
        SearchCoordinate::Spectral { .. } => point.coordinates().vacuum_angular_wavenumber(),
    };
    assert!((varying.value()[()] - mode.coordinate()).norm() < 1e-10);
    let k0 = point.coordinates().vacuum_angular_wavenumber().value()[()];
    let kp = point.coordinates().parallel_angular_wavenumber().value()[()];
    for (material, kappa) in [
        (
            p.context().stack().left_exterior(),
            point.exterior_wavevectors().left().unwrap(),
        ),
        (
            p.context().stack().right_exterior(),
            point.exterior_wavevectors().right().unwrap(),
        ),
    ] {
        let material = material.scalar().unwrap();
        let chi = material.evaluate_relative_permittivity_complex(Scalar(k0))
            * material.evaluate_relative_permeability_complex(Scalar(k0));
        assert!((kappa.value()[()].powi(2) + kp * kp - chi * k0 * k0).norm() < 1e-9);
    }
}
#[test]
fn bound_physical_in_plane_te_and_tm() {
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
        let thickness = ((al * fc / q).atan() + (ar * fc / (q * fr)).atan()) / q;
        let stack =
            Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
                .analytical_layer(Constant::dielectric(4.0), Length::centimetres(thickness))
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
        check_physical(&p, z, c(beta));
    }
}
#[test]
fn bound_physical_spectral_te_and_tm() {
    let thickness = 0.3;
    let expected = C::new(std::f64::consts::PI, -3.0_f64.ln()) / (2.0 * thickness);
    let z = (C::new(0.0, -1.0) * expected).ln();
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let p = SpectralProblem::spectral(
            exterior(1.0),
            exterior(1.0),
            [Layer::new(
                AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                Length::centimetres(thickness),
            )],
            0.0,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        check_physical(&p, z, expected);
    }
}

#[test]
fn chart_and_spectral_construction_keep_original_causes() {
    let p = problem();
    let branch = p.map.landmarks().unwrap().coordinate_branch_points()[0];
    let requested = ConformalRegion::new(
        branch.re - 0.1,
        branch.re + 0.1,
        branch.im - 0.2,
        branch.im - 1e-15,
    )
    .unwrap();
    let error = p
        .domain(
            requested,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap_err();
    assert!(error.source().is_some());
    assert!(contains_source::<
        crate::conformal::in_plane::region::InPlaneRegionError,
    >(&error));
    let tiny = ConstantExterior::new(c(1e-300), c(1e-300)).unwrap();
    let error = SpectralProblem::spectral(
        tiny,
        exterior(1.0),
        [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
        1.5,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap_err();
    assert!(contains_source::<WavenumberMapError>(&error));
}

#[test]
fn refinement_rejects_foreign_problem_and_exact_domain_before_evaluation() {
    use crate::refinement::{RefinementConfig, RefinementInputError};
    let base = problem();
    let calls = Arc::new(AtomicUsize::new(0));
    let p = ModalProblem {
        context: base.context,
        map: base.map,
        solver: Failing {
            failure: DeterminantEvaluationError::NonFiniteInput,
            calls: calls.clone(),
        },
    };
    let d = domain(&p);
    let discovery = ProblemSearchResult {
        result: ModalSearchResult::new(
            vec![],
            vec![],
            argloc::RunSummary {
                iterations: 3,
                elapsed: std::time::Duration::from_secs(2),
                best_measure: Some(0.0),
            },
            argloc::Termination::ExceededMaxIterations,
        ),
        provenance: d.provenance.clone(),
    };
    let other = ModalProblem {
        context: problem().context,
        map: p.map.clone(),
        solver: p.solver.clone(),
    };
    assert!(matches!(
        other.refine(&d, &discovery, RefinementConfig::default()),
        Err(RefinementInputError::ForeignProblem)
    ));
    let another = domain(&p);
    assert!(matches!(
        p.refine(&another, &discovery, RefinementConfig::default()),
        Err(RefinementInputError::ForeignDomain)
    ));
    let report = p
        .clone()
        .refine(&d.clone(), &discovery, RefinementConfig::default())
        .unwrap();
    assert!(report.outcomes().is_empty());
    assert!(std::ptr::eq(report.discovery(), &discovery));
    assert_eq!(
        report.discovery().result().summary().elapsed,
        std::time::Duration::from_secs(2)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let invalid = RefinementConfig {
        residual_scale: 0.0,
        ..RefinementConfig::default()
    };
    assert!(matches!(
        p.refine(&d, &discovery, invalid),
        Err(RefinementInputError::InvalidConfig)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
#[derive(Clone, Debug)]
struct RefinementSolver {
    roots: Vec<C>,
    fail_at: Option<C>,
}
impl ModeDeterminantSolver<J> for RefinementSolver {
    type Error = DeterminantEvaluationError<Fault>;
    fn determinant(
        &self,
        coords: &CanonicalCoordinates<J>,
        _: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        let k = coords.parallel_angular_wavenumber();
        if self
            .fail_at
            .is_some_and(|root| (root - k.value()[()]).norm() < 1e-8)
        {
            return Err(DeterminantEvaluationError::Backend { source: Fault });
        }
        Ok(self.roots.iter().fold(k.constant(c(1.0)), |product, root| {
            product.multiply(&k.subtract(&k.constant(*root)))
        }))
    }
}
#[test]
fn refinement_mixed_outcomes_preserve_complete_discovery_and_cluster_metadata() {
    use crate::refinement::{CandidateStatus, RefinementConfig};
    let base = problem();
    let chart = ConformalChart::new(
        base.map.clone(),
        region(),
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap();
    let zs = [C::new(-0.1, -0.05), C::new(0.1, 0.05)];
    let roots = zs.map(|z| chart.coordinate(z).unwrap());
    let p = ModalProblem {
        context: base.context,
        map: base.map,
        solver: RefinementSolver {
            roots: roots.to_vec(),
            fail_at: Some(roots[1]),
        },
    };
    let d = domain(&p);
    let estimates = zs
        .into_iter()
        .chain([c(0.0)])
        .enumerate()
        .map(|(index, z)| {
            let estimate = argloc::SingularPointEstimate {
                location: z,
                multiplicity: if index == 2 { 2 } else { 1 },
                enclosure: (*d.search.search_region()).try_into().unwrap(),
                kind: if index == 2 {
                    argloc::SingularPointEstimateKind::ClusterCentroid
                } else {
                    argloc::SingularPointEstimateKind::SinglePoint
                },
                target: argloc::SearchTarget::Zeros,
            };
            crate::search::ModalEstimate::new(
                estimate,
                crate::mode::ConformalMode::new(z, chart.coordinate(z).unwrap(), c(0.0)),
            )
        })
        .collect();
    let discovery = ProblemSearchResult {
        result: ModalSearchResult::new(
            estimates,
            vec![],
            argloc::RunSummary {
                iterations: 7,
                elapsed: std::time::Duration::from_nanos(1234567),
                best_measure: Some(0.2),
            },
            argloc::Termination::ExceededMaxIterations,
        ),
        provenance: d.provenance.clone(),
    };
    let report = p
        .refine(&d, &discovery, RefinementConfig::default())
        .unwrap();
    assert!(std::ptr::eq(report.discovery(), &discovery));
    assert_eq!(report.outcomes().len(), 3);
    assert!(
        matches!(report.outcomes()[0].status(), CandidateStatus::Accepted(_)),
        "{:?}",
        report.outcomes()
    );
    assert!(matches!(
        report.outcomes()[1].status(),
        CandidateStatus::EvaluationFailed(_)
    ));
    assert!(contains_source::<Fault>(
        report.outcomes()[1].error().unwrap()
    ));
    assert!(matches!(
        report.outcomes()[2].status(),
        CandidateStatus::UnresolvedCluster
    ));
    for (index, outcome) in report.outcomes().iter().enumerate() {
        assert_eq!(outcome.index(), index)
    }
    assert_eq!(
        report.discovery().result().estimates()[2]
            .estimate()
            .multiplicity,
        2
    );
    assert_eq!(
        *report.discovery().result().termination(),
        argloc::Termination::ExceededMaxIterations
    );
    assert_eq!(
        report.discovery().result().summary().elapsed,
        std::time::Duration::from_nanos(1234567)
    );
    assert!(report.discovery().provenance().same_domain(d.provenance()));
    // Also exercise a real successful incomplete search and retain its leaves.
    let base = problem();
    let p = ModalProblem {
        context: base.context,
        map: base.map,
        solver: RefinementSolver {
            roots: roots.to_vec(),
            fail_at: None,
        },
    };
    let d = domain(&p);
    let mut cfg = ArgumentConfig::new(1e-12);
    cfg.quad_tree.max_iter = 0;
    let discovery = p.search(&d, cfg).unwrap();
    let report = p
        .refine(&d, &discovery, RefinementConfig::default())
        .unwrap();
    assert!(std::ptr::eq(report.discovery(), &discovery));
    assert!(!report.discovery().result().leaves().is_empty());
    assert_eq!(
        *report.discovery().result().termination(),
        argloc::Termination::ExceededMaxIterations
    );
    assert_eq!(report.outcomes().len(), discovery.result().len());
    assert!(!report.outcomes().is_empty());
    assert!(
        report
            .outcomes()
            .iter()
            .any(|outcome| matches!(outcome.status(), CandidateStatus::Accepted(_))),
        "{:?}",
        report.outcomes()
    );
}

#[test]
fn refinement_preserves_opposite_coordinate_sheet() {
    use crate::refinement::{CandidateStatus, RefinementConfig};
    let base = problem();
    let z = C::new(-0.1, -0.05);
    let principal = ConformalChart::new(
        base.map.clone(),
        region(),
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap();
    let target = -principal.coordinate(z).unwrap();
    let p = ModalProblem {
        context: base.context,
        map: base.map,
        solver: RefinementSolver {
            roots: vec![target],
            fail_at: None,
        },
    };
    let d = p
        .domain(
            region(),
            ReferenceSelection::Sheet(CoordinateSheet::Opposite),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let discovery = p.search(&d, ArgumentConfig::new(1e-6)).unwrap();
    let report = p
        .refine(&d, &discovery, RefinementConfig::default())
        .unwrap();
    assert_eq!(report.outcomes().len(), 1);
    let CandidateStatus::Accepted(root) = report.outcomes()[0].status() else {
        panic!("{:?}", report.outcomes())
    };
    assert!((root.coordinate() - target).norm() < 1e-9);
    assert!((root.coordinate() + target).norm() > 1.0);
    assert_eq!(
        root.provenance().selection(),
        ReferenceSelection::Sheet(CoordinateSheet::Opposite)
    );
    assert_eq!(
        root.provenance().reference_coordinate(),
        d.provenance().reference_coordinate()
    );
}
#[test]
fn diagonal_exteriors_scale_kappa_and_keep_branch_points_at_their_lines() {
    use crate::conformal::RawConformalMap;
    let k0 = 2.0;
    let left =
        ConstantExterior::diagonal([c(2.0), c(2.5), c(3.0)], [c(1.1), c(1.0), c(0.8)]).unwrap();
    let right =
        ConstantExterior::diagonal([c(1.5), c(1.2), c(1.0)], [c(1.0), c(1.3), c(1.2)]).unwrap();
    let core = || {
        Layer::new(
            Region::from(AnalyticalMaterialHandle::new(Constant::dielectric(4.0))),
            Length::centimetres(0.3),
        )
    };
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let channels = [left.channel(pol), right.channel(pol)];
        assert!(channels.iter().all(|c| c.scale().is_some()));
        let stack = Stack::builder(left.region(), right.region())
            .layer(core().into_parts().0, Length::centimetres(0.3))
            .finalise();
        let in_plane = InPlaneProblem::in_plane(stack, k0, pol, Scatter2::new()).unwrap();
        let spectral =
            SpectralProblem::spectral(left, right, [core()], 3.0, pol, Scatter2::new()).unwrap();
        // κ² = s (χ k0² − kx²) on each side, wherever the charts map.
        for z in [C::new(0.3, 0.1), C::new(-0.2, 0.4), C::new(0.7, -0.3)] {
            let z = J::lift_constant(arr0(z));
            for point in [
                in_plane.map.map_raw(z.clone()),
                spectral.map.map_raw(z.clone()),
            ] {
                let k = point.coordinates().vacuum_angular_wavenumber().value()[()];
                let b = point.coordinates().parallel_angular_wavenumber().value()[()];
                let exterior = point.exterior_wavevectors();
                for (kappa, channel) in [exterior.left(), exterior.right()]
                    .into_iter()
                    .zip(channels)
                {
                    let kappa = kappa.unwrap().value()[()];
                    let expected = channel.kappa_squared(k, b);
                    assert!((kappa * kappa - expected).norm() < 1e-12 * expected.norm().max(1.0));
                }
            }
        }
        // The guided sheet: real β above both lines decays on both sides, and
        // the outer line, where one κ vanishes, is a branch point of the chart.
        let lines = channels.map(|c| k0 * c.chi().re.sqrt());
        let (outer, inner) = if lines[0] > lines[1] { (0, 1) } else { (1, 0) };
        for beta in [lines[outer] * 1.3, lines[outer]] {
            let alpha = |line: f64| (beta * beta - line * line).sqrt();
            let z = ((alpha(lines[0]) + alpha(lines[1])) / (2.0 * k0)).ln();
            let point = in_plane.map.map_raw(J::lift_constant(arr0(c(z))));
            let b = point.coordinates().parallel_angular_wavenumber().value()[()];
            assert!((b - beta).norm() < 1e-12 * beta);
            let exterior = point.exterior_wavevectors();
            let kappa = [exterior.left(), exterior.right()].map(|k| k.unwrap().value()[()]);
            let s = channels[inner].scale().unwrap().re;
            assert!((kappa[inner] - C::new(0.0, s.sqrt() * alpha(lines[inner]))).norm() < 1e-12);
            if beta == lines[outer] {
                assert!(kappa[outer].norm() < 1e-6);
            } else {
                assert!(kappa[outer].im > 0.0 && kappa[outer].re.abs() < 1e-12);
            }
        }
    }
}
