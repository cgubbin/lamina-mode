use crate::{
    conformal::{
        ConformalRegion, ValidateConformalRegion,
        chart::{ConformalChart, ConformalChartConfig, CoordinateSheet, LinearCoordinateMap},
        in_plane::ComplexInPlaneMap,
        wavenumber::ComplexWavenumberMap,
    },
    dispersion::{CompiledPlaneWaveModeSolver, ConformalDispersion, ModeDeterminantSolver},
    search::SingleChartModalSearch,
};
use argloc::{ArgumentConfig, SingularPointEstimateKind, Termination, find_zeros};
use lamina_core::{
    CanonicalCoordinates, Constant, DeterminantEvaluationError, ExteriorWavevectors, ModeJet1,
    Polarisation, ScalarAlgebra, Scatter2, SeedJet, Stack, units::Length,
};
use ndarray::{Ix0, arr0};
use num_complex::Complex64 as C;
use std::{
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
type J = ModeJet1<C, Ix0>;
fn c(v: f64) -> C {
    C::new(v, 0.0)
}
fn chart() -> ConformalChart<ComplexInPlaneMap<J>, J> {
    ConformalChart::new(
        ComplexInPlaneMap::new(2.0, c(1.0), c(1.0), c(2.25), c(1.0)).unwrap(),
        ConformalRegion::new(-0.3, 0.3, -0.2, 0.2).unwrap(),
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap()
}
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("injected backend failure")]
struct Fault;
#[derive(Clone, Debug)]
struct Failing(DeterminantEvaluationError<Fault>);
impl ModeDeterminantSolver<J> for Failing {
    type Error = DeterminantEvaluationError<Fault>;
    fn determinant(
        &self,
        _: &CanonicalCoordinates<J>,
        _: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        Err(self.0.clone())
    }
}
fn source_contains<E: Error + 'static>(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if error.downcast_ref::<E>().is_some() {
            return true;
        }
        match error.source() {
            Some(next) => error = next,
            None => return false,
        }
    }
}
#[test]
fn checked_failures_survive_dispersion_and_adaptive_search() {
    for failure in [
        DeterminantEvaluationError::NonFiniteInput,
        DeterminantEvaluationError::NonFiniteOutput,
        DeterminantEvaluationError::Backend { source: Fault },
    ] {
        let dispersion = ConformalDispersion::new(Failing(failure.clone()), chart());
        let error = dispersion.evaluate(C::new(0.0, 0.05)).err().unwrap();
        assert!(source_contains::<DeterminantEvaluationError<Fault>>(&error));
        let search = SingleChartModalSearch::new(dispersion).unwrap();
        let error = search
            .search(ArgumentConfig::new(1e-6))
            .expect_err("evaluation failure must not become an empty success");
        assert!(
            source_contains::<DeterminantEvaluationError<Fault>>(&error),
            "{error:?}"
        );
        if matches!(failure, DeterminantEvaluationError::Backend { .. }) {
            assert!(source_contains::<Fault>(&error));
        }
    }
}

#[derive(Clone, Debug)]
struct Counting<S> {
    inner: S,
    count: Arc<AtomicUsize>,
    inputs: Arc<Mutex<Option<[C; 8]>>>,
}
impl<S: ModeDeterminantSolver<J>> ModeDeterminantSolver<J> for Counting<S> {
    type Error = S::Error;
    fn determinant(
        &self,
        coords: &CanonicalCoordinates<J>,
        exterior: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        self.count.fetch_add(1, Ordering::SeqCst);
        let [a, b, l, r] = [
            coords.vacuum_angular_wavenumber(),
            coords.parallel_angular_wavenumber(),
            exterior.left().unwrap().scalar().unwrap(),
            exterior.right().unwrap().scalar().unwrap(),
        ];
        *self.inputs.lock().unwrap() = Some([
            a.value()[()],
            a.first()[()],
            b.value()[()],
            b.first()[()],
            l.value()[()],
            l.first()[()],
            r.value()[()],
            r.first()[()],
        ]);
        self.inner.determinant(coords, exterior)
    }
}
fn check_adapter<M>(map: M, region: ConformalRegion<f64>, z: C)
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    <M as ValidateConformalRegion<f64>>::Error: std::fmt::Debug,
{
    let stack = Stack::builder(Constant::vacuum(), Constant::dielectric(2.25))
        .layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise();
    let chart = ConformalChart::new(
        map.clone(),
        region,
        CoordinateSheet::Opposite,
        ConformalChartConfig::default(),
    )
    .unwrap();
    let principal = ConformalChart::new(
        map,
        region,
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap();
    assert!((chart.coordinate(z).unwrap() + principal.coordinate(z).unwrap()).norm() < 1e-11);
    let point = chart.map(J::variable(arr0(z), 0).unwrap()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let inputs = Arc::new(Mutex::new(None));
    let solver = CompiledPlaneWaveModeSolver::<J, _, _>::compile(
        &stack,
        Scatter2::new(),
        Polarisation::TransverseMagnetic,
    )
    .unwrap();
    let expected = solver
        .determinant(point.coordinates(), point.exterior_wavevectors())
        .unwrap();
    let dispersion = ConformalDispersion::new(
        Counting {
            inner: solver,
            count: count.clone(),
            inputs: inputs.clone(),
        },
        chart,
    );
    let actual = dispersion.evaluate(z).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let [a, b, l, r] = [
        point.coordinates().vacuum_angular_wavenumber(),
        point.coordinates().parallel_angular_wavenumber(),
        point
            .exterior_wavevectors()
            .left()
            .unwrap()
            .scalar()
            .unwrap(),
        point
            .exterior_wavevectors()
            .right()
            .unwrap()
            .scalar()
            .unwrap(),
    ];
    assert_eq!(
        inputs.lock().unwrap().unwrap(),
        [
            a.value()[()],
            a.first()[()],
            b.value()[()],
            b.first()[()],
            l.value()[()],
            l.first()[()],
            r.value()[()],
            r.first()[()]
        ]
    );
    assert!((*actual.value() - expected.value()[()]).norm() < 1e-12);
    assert!((*actual.derivative() - expected.first()[()]).norm() < 1e-12);
    let h = 1e-5;
    let numerical =
        (dispersion.value(z + h).unwrap() - dispersion.value(z - h).unwrap()) / (2.0 * h);
    assert!((numerical - *actual.derivative()).norm() < 1e-6);
}
#[test]
fn checked_adapter_preserves_both_map_families_and_nonprincipal_sheet() {
    check_adapter(
        ComplexInPlaneMap::new(2.0, c(1.0), c(1.0), c(2.25), c(1.0)).unwrap(),
        ConformalRegion::new(-0.2, 0.2, -0.2, 0.2).unwrap(),
        C::new(0.05, 0.1),
    );
    check_adapter(
        ComplexWavenumberMap::new(1.5, c(1.0), c(1.0), c(2.25), c(1.0)).unwrap(),
        ConformalRegion::new(-0.2, 0.2, 0.2, 0.8).unwrap(),
        C::new(0.05, 0.4),
    );
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
fn budget_limited_search_preserves_cluster_and_unresolved_diagnostics() {
    let chart = chart();
    let a = chart.coordinate(C::new(-0.15, -0.1)).unwrap();
    let b = chart.coordinate(C::new(-0.12, -0.08)).unwrap();
    let search =
        SingleChartModalSearch::new(ConformalDispersion::new(TwoRoots(a, b), chart)).unwrap();
    let mut config = ArgumentConfig::new(1e-12);
    config.quad_tree.max_iter = 0;
    let raw = find_zeros(
        search.dispersion().clone(),
        (*search.search_region()).try_into().unwrap(),
        config.clone(),
    )
    .unwrap();
    let result = search.search(config).unwrap();
    assert_eq!(*result.termination(), Termination::ExceededMaxIterations);
    assert_eq!(*result.termination(), raw.termination);
    assert_eq!(result.summary().iterations, raw.summary.iterations);
    assert_eq!(result.summary().best_measure, raw.summary.best_measure);
    assert!(result.summary().elapsed.as_nanos() > 0);
    assert_eq!(result.leaves().len(), raw.leaves.len());
    assert!(!result.leaves().is_empty());
    assert_eq!(result.estimates().len(), raw.points.len());
    assert_eq!(
        result
            .estimates()
            .iter()
            .map(|e| e.estimate().multiplicity)
            .sum::<usize>(),
        2
    );
    for (a, b) in result.estimates().iter().zip(raw.points) {
        assert_eq!(*a.estimate(), b);
        assert_eq!(
            a.estimate().kind,
            SingularPointEstimateKind::ClusterCentroid
        );
    }
    for (a, b) in result.leaves().iter().zip(raw.leaves) {
        assert_eq!(a.bounds, b.bounds);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.data.root_count, b.data.root_count);
    }
}

// Independent phase-matching thickness for the fundamental guided slab mode.
#[test]
fn physical_in_plane_te_and_tm_searches() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let k0: f64 = 2.0;
        let beta: f64 = 3.5;
        let q = (4.0 * k0 * k0 - beta * beta).sqrt();
        let al = (beta * beta - k0 * k0).sqrt();
        let ar = (beta * beta - 2.25 * k0 * k0).sqrt();
        let (fc, fr) = match polarisation {
            Polarisation::TransverseElectric => (1.0, 1.0),
            Polarisation::TransverseMagnetic => (4.0, 2.25),
        };
        let d = ((al * fc / q).atan() + (ar * fc / (q * fr)).atan()) / q;
        let stack = Stack::builder(Constant::vacuum(), Constant::dielectric(2.25))
            .layer(Constant::dielectric(4.0), Length::centimetres(d))
            .finalise();
        let map = ComplexInPlaneMap::<J>::new(k0, c(1.0), c(1.0), c(2.25), c(1.0)).unwrap();
        let z = map
            .physical_preimages(beta)
            .unwrap()
            .into_iter()
            .flatten()
            .max_by(|a, b| a.re.total_cmp(&b.re))
            .unwrap();
        check_physical(stack, map, z, c(beta), polarisation);
    }
}

// For n=2 between vacuum half spaces, r^2 exp(2 i n k0 d)=1,
// r^2=1/9, gives this outgoing Fabry-Perot pole for both polarisations.
#[test]
fn physical_spectral_te_and_tm_searches() {
    let d = 0.3;
    let expected = C::new(std::f64::consts::PI, -3.0_f64.ln()) / (2.0 * d);
    let z = (C::new(0.0, -1.0) * expected).ln();
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let stack = Stack::builder(Constant::vacuum(), Constant::vacuum())
            .layer(Constant::dielectric(4.0), Length::centimetres(d))
            .finalise();
        let map = ComplexWavenumberMap::<J>::new(0.0, c(1.0), c(1.0), c(1.0), c(1.0)).unwrap();
        check_physical(stack, map, z, expected, polarisation);
    }
}
fn check_physical<M>(
    stack: Stack<Constant<f64>, f64>,
    map: M,
    z: C,
    expected: C,
    polarisation: Polarisation,
) where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    <M as ValidateConformalRegion<f64>>::Error: std::fmt::Debug,
{
    let chart = ConformalChart::new(
        map,
        ConformalRegion::new(z.re - 0.06, z.re + 0.06, z.im - 0.05, z.im + 0.05).unwrap(),
        CoordinateSheet::Principal,
        ConformalChartConfig::default(),
    )
    .unwrap();
    assert!((chart.coordinate(z).unwrap() - expected).norm() < 1e-11);
    let point = chart.map(J::variable(arr0(z), 0).unwrap()).unwrap();
    let coordinate = chart.coordinate(z).unwrap();
    let coords = point.coordinates();
    let ext = point.exterior_wavevectors();
    // The exterior jets belong to this exact sheet and satisfy their dispersion.
    for (kappa, eps) in [
        (ext.left().unwrap().scalar().unwrap(), 1.0),
        (
            ext.right().unwrap().scalar().unwrap(),
            if expected.im == 0.0 { 2.25 } else { 1.0 },
        ),
    ] {
        let identity = kappa.value()[()] * kappa.value()[()]
            + coords.parallel_angular_wavenumber().value()[()].powi(2)
            - eps * coords.vacuum_angular_wavenumber().value()[()].powi(2);
        assert!(identity.norm() < 1e-10);
    }
    assert!((coordinate - expected).norm() < 1e-11);
    let solver =
        CompiledPlaneWaveModeSolver::<J, _, _>::compile(&stack, Scatter2::new(), polarisation)
            .unwrap();
    let dispersion = ConformalDispersion::new(solver, chart);
    let value = dispersion.evaluate(z).unwrap();
    assert!(value.value().norm() < 1e-10, "{value:?}");
    let h = 1e-6;
    let fd = (dispersion.value(z + h).unwrap() - dispersion.value(z - h).unwrap()) / (2.0 * h);
    assert!((fd - *value.derivative()).norm() < 1e-6);
    let search = SingleChartModalSearch::new(dispersion).unwrap();
    let result = search.search(ArgumentConfig::new(1e-7)).unwrap();
    assert_eq!(result.len(), 1);
    let estimate = &result.estimates()[0];
    assert_eq!(estimate.estimate().multiplicity, 1);
    let mode = estimate.mode();
    assert!((mode.coordinate() - expected).norm() < 1e-5, "{mode:?}");
    assert!(mode.residual().norm() < 1e-6);
    assert!(
        (search.dispersion().chart().coordinate(mode.z()).unwrap() - mode.coordinate()).norm()
            < 1e-11
    );
}
