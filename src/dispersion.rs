//! Analytic modal dispersion over one fallible conformal or frequency chart.
//! The legacy type names remain private. Native derivatives are dF/dz for
//! conformal charts and dF/dk0 for direct frequency charts; the latter use core
//! constitutive jets and independently continued exterior branch values.
//!
//! This module joins the conformal-coordinate layer to a precompiled
//! `lamina-core` mode solver.
//!
//! The conformal layer provides an analytic map
//!
//! ```text
//! z
//! |
//! +--> k0(z)
//! +--> kx(z)
//! +--> kappa_L(z)
//! `--> kappa_R(z)
//! ```
//!
//! on a validated [`ConformalChart`].
//!
//! The core solver consumes these quantities and evaluates an outgoing-wave
//! modal determinant
//!
//! ```text
//! F(z) = D(
//!     k0(z),
//!     kx(z),
//!     kappa_L(z),
//!     kappa_R(z)
//! ).
//! ```
//!
//! A first-order jet is seeded directly in the conformal coordinate `z`.
//! Consequently all coordinate transformations, exterior-wavevector formulas,
//! material evaluation, propagation, boundary conditions, and determinant
//! construction participate in the same automatic-differentiation chain.
//!
//! The resulting first jet coefficient is therefore
//!
//! ```text
//! dF/dz,
//! ```
//!
//! with no explicit chain rule required in this crate.
//!
//! # Analyticity
//!
//! [`ConformalDispersion`] operates over exactly one [`ConformalChart`].
//! The chart guarantees that the varying *linear* physical coordinate (`kx`
//! for an in-plane map or `k0` for a wavenumber map) is represented by one
//! analytic square-root branch throughout its region.
//!
//! This restriction is important for argument-principle methods: the
//! determinant supplied to the APM must be analytic on and inside its contour,
//! with zeros allowed but no poles anywhere in the closed search domain or its
//! surrounding analytic neighborhood. Finite samples and a valid coordinate
//! chart do not establish that material-domain obligation.
//!
//! A physical search may compile into more than one dispersion function. For
//! example, the canonical lossless complex-wavenumber search uses separate
//! upper and lower conformal charts.
//!
//! # Solver abstraction
//!
//! [`ModeDeterminantSolver`] is intentionally small. It isolates this crate
//! from the concrete type of the precompiled `lamina-core` solver while
//! expressing the only operation required here: evaluate the outgoing modal
//! determinant from canonical coordinates and explicitly supplied exterior
//! wavevectors.
//!
//! The concrete adapter forwards to checked_determinant and carries its input,
//! backend, or output failure through DispersionError::Solver. This is a
//! pointwise numerical check, not a pole-free material-domain certificate.
//! Exterior wavevector jets are passed through unchanged; one conformal seed
//! produces both F and F_prime without recomputing exterior principal roots.

use crate::analytic::AnalyticChart;
#[cfg(test)]
use crate::conformal::{ValidateConformalRegion, chart::LinearCoordinateMap};
use lamina_core::{
    Backend, CanonicalCoordinates, Channel, ComplexPlane, ComplexPlaneEvaluator, ComplexScalar,
    DeterminantEvaluationError, ExteriorBranch, ExteriorWavevectors, Jet, ModeJet1,
    PlaneWaveDeterminant, ProjectPlaneWaveModeDeterminant, ScalarAlgebra, SeedJet, Stack,
    StackCompileError, StackThicknessJet, TensorConstitutiveLift,
};

use argloc::FunctionBundle;
use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0, arr0};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::{
    conformal::chart::{ConformalChart, ConformalChartError},
    mode::ConformalMode,
};

/// Minimal interface required from a precompiled optical mode solver.
///
/// Implementations should evaluate the outgoing-wave modal determinant for
/// the supplied canonical coordinates and exterior transverse wavevectors.
///
/// The conformal layer has already selected the required analytic branches.
/// Implementations must therefore use the supplied exterior wavevectors
/// directly rather than recomputing them from `k0` and `kx`.
pub trait ModeDeterminantSolver<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
{
    type Error: Send + Sync + std::fmt::Display + 'static + Clone + std::error::Error;

    /// Evaluate the outgoing modal determinant.
    ///
    /// The returned algebra value must carry the derivative information from
    /// `coordinates` and `exterior`.
    fn determinant(
        &self,
        coordinates: &CanonicalCoordinates<A>,
        exterior: &ExteriorWavevectors<A>,
    ) -> Result<A, Self::Error>;
}

/// Failure while evaluating a charted modal dispersion function.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ChartDispersionError<ChartError, SolverError> {
    /// The analytic chart could not evaluate the requested point.
    #[error("analytic chart evaluation failed: {0}")]
    Chart(#[source] ChartError),

    /// The precompiled optical solver failed.
    #[error("modal determinant evaluation failed: {0}")]
    Solver(#[source] SolverError),
}

pub type DispersionError<R, E> = ChartDispersionError<ConformalChartError<R>, E>;

/// An analytic modal dispersion function over one fallible chart.
///
/// `S` is a precompiled optical mode solver.
///
/// `M` is the raw conformal map for the default chart; direct frequency uses ().
/// `H` supplies the fallible native-coordinate chart contract.
///
/// `C` is the complex scalar type. Internally the dispersion function uses
/// [`ModeJet1`] with scalar dimension [`Ix0`], because the first modal
/// implementation requires only `F(z)` and `dF/dz`.
#[derive(Clone, Debug)]
pub struct ConformalDispersion<S, M, C, H = ConformalChart<M, ModeJet1<C, Ix0>>>
where
    C: ComplexField + Copy,
{
    solver: S,
    chart: H,
    marker: std::marker::PhantomData<(M, C)>,
}

impl<S, M, C> ConformalDispersion<S, M, C>
where
    C: ComplexField + Copy,
{
    pub fn new(solver: S, chart: ConformalChart<M, ModeJet1<C, Ix0>>) -> Self {
        Self {
            solver,
            chart,
            marker: std::marker::PhantomData,
        }
    }
}
impl<S, M, C, H> ConformalDispersion<S, M, C, H>
where
    C: ComplexField + Copy,
{
    pub fn with_chart(solver: S, chart: H) -> Self {
        Self {
            solver,
            chart,
            marker: std::marker::PhantomData,
        }
    }
    pub fn solver(&self) -> &S {
        &self.solver
    }
    pub fn chart(&self) -> &H {
        &self.chart
    }
    pub fn into_parts(self) -> (S, H) {
        (self.solver, self.chart)
    }
}

impl<S, M, C, R, H> ConformalDispersion<S, M, C, H>
where
    C: ComplexScalar<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
    H: AnalyticChart<ModeJet1<C, Ix0>>,
    S: ModeDeterminantSolver<ModeJet1<C, Ix0>>,
{
    /// Evaluate the characteristic and its native derivative (dF/dz or dF/dk0).
    ///
    /// `z` is seeded as the sole independent variable of a first-order mode
    /// jet. The chart then maps that jet into canonical optical coordinates
    /// and branch-preserving exterior transverse wavevectors. The core solver
    /// consumes those values and returns the outgoing modal determinant jet.
    pub fn evaluate(
        &self,
        z: C,
    ) -> Result<FunctionBundle<C>, ChartDispersionError<H::Error, S::Error>> {
        // ModeJet1 has exactly one independent variable, so seed index zero
        // is structurally valid.
        let z = ModeJet1::variable(arr0(z), 0)
            .expect("ModeJet1 must accept its sole variable at index zero");

        let point = self.chart.map(z).map_err(ChartDispersionError::Chart)?;

        let determinant = self
            .solver
            .determinant(point.coordinates(), point.exterior_wavevectors())
            .map_err(ChartDispersionError::Solver)?;

        Ok(FunctionBundle::new(
            determinant.value()[()],
            determinant.first()[()],
        ))
    }

    /// Evaluate only the determinant value.
    ///
    /// This currently performs the same first-order solve as [`Self::evaluate`].
    /// Keeping this method nevertheless gives the APM-facing API the shape it
    /// requires today without committing the implementation to separate value
    /// and derivative execution paths.
    pub fn value(&self, z: C) -> Result<C, ChartDispersionError<H::Error, S::Error>> {
        Ok(*self.evaluate(z)?.value())
    }

    /// Evaluate only the conformal derivative `dF/dz`.
    ///
    /// This currently performs the same first-order solve as [`Self::evaluate`].
    pub fn derivative(&self, z: C) -> Result<C, ChartDispersionError<H::Error, S::Error>> {
        Ok(*self.evaluate(z)?.derivative())
    }

    /// Convert a conformal root into a physical modal result.
    pub fn mode_at(
        &self,
        z: C,
    ) -> Result<ConformalMode<C>, ChartDispersionError<H::Error, S::Error>> {
        let coordinate = self
            .chart()
            .coordinate(z)
            .map_err(ChartDispersionError::Chart)?;

        let residual = self.value(z)?;

        Ok(ConformalMode::new(z, coordinate, residual))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledPlaneWaveModeSolver<J, M, B> {
    evaluator: ComplexPlaneEvaluator<J, M, B>,
    channel: Channel,
}

impl<J, M, B> CompiledPlaneWaveModeSolver<J, M, B> {
    pub(crate) fn compile(
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        backend: B,
        channel: impl Into<Channel>,
    ) -> Result<Self, StackCompileError<<J::Scalar as ComplexField>::RealField>>
    where
        J: Jet<Dimension = Ix0> + StackThicknessJet,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive + Copy + Debug,
        J::Dimension: Dimension + Clone,
        M: Clone,
    {
        let evaluator = ComplexPlaneEvaluator::compile(stack, backend)?;

        Ok(Self {
            evaluator,
            channel: channel.into(),
        })
    }

    fn evaluator(&self) -> &ComplexPlaneEvaluator<J, M, B> {
        &self.evaluator
    }

    pub(crate) fn into_parts(self) -> (ComplexPlaneEvaluator<J, M, B>, Channel) {
        (self.evaluator, self.channel)
    }
}

/// Evaluate the outgoing mode determinant using the caller-supplied
/// canonical coordinates and exterior transverse wavevectors.
///
/// In particular, the exterior wavevectors are not recomputed from
/// `k0` and `kx`: their conformal branch has already been selected by
/// the modal layer.
impl<J, M, B> ModeDeterminantSolver<J> for CompiledPlaneWaveModeSolver<J, M, B>
where
    J: ScalarAlgebra<Dimension = Ix0> + TensorConstitutiveLift<ComplexPlane, M>,
    J::Scalar: ComplexScalar,
    M: Clone,
    B: Backend<J, ComplexPlane>,
    <B as Backend<J, ComplexPlane>>::Entries:
        ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
{
    type Error = DeterminantEvaluationError<<B as Backend<J, ComplexPlane>>::Error>;

    fn determinant(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        exterior: &ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        // An isotropic exterior's two outgoing waves of the coupled channel
        // share the branch the chart continues.
        let paired;
        let exterior = match self.channel {
            Channel::Coupled => {
                let pair = |side: Option<&ExteriorBranch<J>>| {
                    side.map(|branch| match branch {
                        ExteriorBranch::Scalar(kappa) => {
                            ExteriorBranch::Pair([kappa.clone(), kappa.clone()])
                        }
                        pair => pair.clone(),
                    })
                };
                paired = ExteriorWavevectors::from_branches(
                    pair(exterior.left()),
                    pair(exterior.right()),
                );
                &paired
            }
            _ => exterior,
        };
        self.evaluator()
            .checked_determinant(coordinates, exterior, self.channel)
            .map(|determinant| determinant.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use lamina_core::ScalarAlgebra;
    use num_complex::Complex64;

    use crate::conformal::{
        ConformalRegion,
        chart::{ConformalChartConfig, CoordinateSheet},
        in_plane::ComplexInPlaneMap,
    };

    type C = Complex64;
    type J = ModeJet1<C, Ix0>;
    type Map = ComplexInPlaneMap<J>;

    const ABS_TOL: f64 = 1.0e-10;
    const REL_TOL: f64 = 1.0e-10;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    fn chart() -> ConformalChart<Map, J> {
        let map = ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))
            .unwrap();

        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap()
    }

    /// Fake determinant:
    ///
    ///     F = k0 + 2 kx.
    ///
    /// This is intentionally simple. Its purpose is to check that the
    /// conformal jet propagated by the chart reaches the solver unchanged.
    #[derive(Clone, Copy, Debug)]
    struct LinearSolver;

    impl ModeDeterminantSolver<J> for LinearSolver {
        type Error = std::convert::Infallible;

        fn determinant(
            &self,
            coordinates: &CanonicalCoordinates<J>,
            _exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            let two = C::new(2.0, 0.0);

            Ok(coordinates
                .vacuum_angular_wavenumber()
                .add(&coordinates.parallel_angular_wavenumber().scale(two)))
        }
    }

    #[test]
    fn evaluate_returns_value_and_conformal_derivative() {
        let dispersion = ConformalDispersion::new(LinearSolver, chart());

        let z = c(0.25, 0.30);

        let result = dispersion.evaluate(z).unwrap();

        // k0 is fixed at 2, so
        //
        //     F = 2 + 2 kx
        //
        // and
        //
        //     F' = 2 kx'.
        let mapped = dispersion
            .chart()
            .map(J::variable(arr0(z), 0).unwrap())
            .unwrap();

        let kx = mapped.coordinates().parallel_angular_wavenumber();

        let expected_value = c(2.0, 0.0) + c(2.0, 0.0) * kx.value()[()];

        let expected_derivative = c(2.0, 0.0) * kx.first()[()];

        assert_complex_close(*result.value(), expected_value);

        assert_complex_close(*result.derivative(), expected_derivative);
    }

    #[test]
    fn value_matches_combined_evaluation() {
        let dispersion = ConformalDispersion::new(LinearSolver, chart());

        let z = c(0.20, -0.15);

        assert_complex_close(
            dispersion.value(z).unwrap(),
            *dispersion.evaluate(z).unwrap().value(),
        );
    }

    #[test]
    fn derivative_matches_combined_evaluation() {
        let dispersion = ConformalDispersion::new(LinearSolver, chart());

        let z = c(-0.20, 0.25);

        assert_complex_close(
            dispersion.derivative(z).unwrap(),
            *dispersion.evaluate(z).unwrap().derivative(),
        );
    }
}

#[cfg(test)]
mod mock_solver_tests {
    use super::*;

    use approx::assert_relative_eq;
    use lamina_core::{
        CanonicalCoordinates, ExteriorWavevectors, ModeJet1, ScalarAlgebra, SeedJet,
    };
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;
    use std::convert::Infallible;

    use crate::conformal::{
        ConformalRegion,
        chart::{ConformalChart, ConformalChartConfig, CoordinateSheet},
        in_plane::ComplexInPlaneMap,
    };

    type C = Complex64;
    type J = ModeJet1<C, Ix0>;
    type Map = ComplexInPlaneMap<J>;
    type Chart = ConformalChart<Map, J>;

    const ABS_TOL: f64 = 1.0e-10;
    const REL_TOL: f64 = 1.0e-10;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    fn chart() -> Chart {
        let map = ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))
            .unwrap();

        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap()
    }

    fn mapped_point(
        dispersion: &ConformalDispersion<impl ModeDeterminantSolver<J>, Map, C>,
        z: C,
    ) -> crate::conformal::ConformalPoint<J> {
        let z = J::variable(arr0(z), 0).unwrap();

        dispersion.chart().map(z).unwrap()
    }

    // -------------------------------------------------------------------------
    // Coordinate-dependent fake solver
    // -------------------------------------------------------------------------

    /// Fake determinant
    ///
    ///     F = k0 + 2 kx.
    ///
    /// This checks that the complete canonical-coordinate jet produced by the
    /// chart reaches the solver unchanged.
    #[derive(Clone, Copy, Debug)]
    struct CoordinateSolver;

    impl ModeDeterminantSolver<J> for CoordinateSolver {
        type Error = Infallible;

        fn determinant(
            &self,
            coordinates: &CanonicalCoordinates<J>,
            _exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            Ok(coordinates
                .vacuum_angular_wavenumber()
                .add(&coordinates.parallel_angular_wavenumber().scale(c(2.0, 0.0))))
        }
    }

    #[test]
    fn coordinate_solver_returns_expected_value_and_derivative() {
        let dispersion = ConformalDispersion::new(CoordinateSolver, chart());

        let z = c(0.25, 0.30);

        let evaluation = dispersion.evaluate(z).unwrap();

        let point = mapped_point(&dispersion, z);

        let k0 = point.coordinates().vacuum_angular_wavenumber();

        let kx = point.coordinates().parallel_angular_wavenumber();

        let expected_value = k0.value()[()] + c(2.0, 0.0) * kx.value()[()];

        let expected_derivative = k0.first()[()] + c(2.0, 0.0) * kx.first()[()];

        assert_complex_close(*evaluation.value(), expected_value);

        assert_complex_close(*evaluation.derivative(), expected_derivative);
    }

    #[test]
    fn fixed_coordinate_contributes_zero_derivative() {
        let dispersion = ConformalDispersion::new(CoordinateSolver, chart());

        let z = c(-0.20, 0.35);

        let point = mapped_point(&dispersion, z);

        // ComplexInPlaneMap holds k0 fixed.
        assert_complex_close(
            point.coordinates().vacuum_angular_wavenumber().first()[()],
            c(0.0, 0.0),
        );

        let evaluation = dispersion.evaluate(z).unwrap();

        let kx_derivative = point.coordinates().parallel_angular_wavenumber().first()[()];

        assert_complex_close(*evaluation.derivative(), c(2.0, 0.0) * kx_derivative);
    }

    // -------------------------------------------------------------------------
    // Exterior-wavevector-dependent fake solver
    // -------------------------------------------------------------------------

    /// Fake determinant
    ///
    ///     F = kappa_L + 3 kappa_R.
    ///
    /// This checks the critical branch-preserving path from the conformal map
    /// to the explicitly supplied exterior wavevectors consumed by the solver.
    #[derive(Clone, Copy, Debug)]
    struct ExteriorSolver;

    impl ModeDeterminantSolver<J> for ExteriorSolver {
        type Error = Infallible;

        fn determinant(
            &self,
            _coordinates: &CanonicalCoordinates<J>,
            exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            Ok(exterior.left().unwrap().scalar().unwrap().add(
                &exterior
                    .right()
                    .unwrap()
                    .scalar()
                    .unwrap()
                    .scale(c(3.0, 0.0)),
            ))
        }
    }

    #[test]
    fn exterior_solver_returns_expected_value_and_derivative() {
        let dispersion = ConformalDispersion::new(ExteriorSolver, chart());

        let z = c(0.30, -0.25);

        let evaluation = dispersion.evaluate(z).unwrap();

        let point = mapped_point(&dispersion, z);

        let left = point
            .exterior_wavevectors()
            .left()
            .unwrap()
            .scalar()
            .unwrap();

        let right = point
            .exterior_wavevectors()
            .right()
            .unwrap()
            .scalar()
            .unwrap();

        let expected_value = left.value()[()] + c(3.0, 0.0) * right.value()[()];

        let expected_derivative = left.first()[()] + c(3.0, 0.0) * right.first()[()];

        assert_complex_close(*evaluation.value(), expected_value);

        assert_complex_close(*evaluation.derivative(), expected_derivative);
    }

    // -------------------------------------------------------------------------
    // Combined fake solver
    // -------------------------------------------------------------------------

    /// Deliberately depends on every mapped quantity:
    ///
    ///     F =
    ///         k0
    ///       + 2 kx
    ///       + 3 kappa_L
    ///       - 4 kappa_R.
    ///
    /// This is the most useful plumbing regression test because any dropped
    /// derivative or accidentally recomputed exterior quantity changes F'.
    #[derive(Clone, Copy, Debug)]
    struct CombinedSolver;

    impl ModeDeterminantSolver<J> for CombinedSolver {
        type Error = Infallible;

        fn determinant(
            &self,
            coordinates: &CanonicalCoordinates<J>,
            exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            let coordinate_part = coordinates
                .vacuum_angular_wavenumber()
                .add(&coordinates.parallel_angular_wavenumber().scale(c(2.0, 0.0)));

            let exterior_part = exterior
                .left()
                .unwrap()
                .scalar()
                .unwrap()
                .scale(c(3.0, 0.0))
                .subtract(
                    &exterior
                        .right()
                        .unwrap()
                        .scalar()
                        .unwrap()
                        .scale(c(4.0, 0.0)),
                );

            Ok(coordinate_part.add(&exterior_part))
        }
    }

    #[test]
    fn combined_solver_preserves_complete_conformal_derivative() {
        let dispersion = ConformalDispersion::new(CombinedSolver, chart());

        let z = c(-0.15, 0.40);

        let evaluation = dispersion.evaluate(z).unwrap();

        let point = mapped_point(&dispersion, z);

        let k0 = point.coordinates().vacuum_angular_wavenumber();

        let kx = point.coordinates().parallel_angular_wavenumber();

        let left = point
            .exterior_wavevectors()
            .left()
            .unwrap()
            .scalar()
            .unwrap();

        let right = point
            .exterior_wavevectors()
            .right()
            .unwrap()
            .scalar()
            .unwrap();

        let expected_value =
            k0.value()[()] + c(2.0, 0.0) * kx.value()[()] + c(3.0, 0.0) * left.value()[()]
                - c(4.0, 0.0) * right.value()[()];

        let expected_derivative =
            k0.first()[()] + c(2.0, 0.0) * kx.first()[()] + c(3.0, 0.0) * left.first()[()]
                - c(4.0, 0.0) * right.first()[()];

        assert_complex_close(*evaluation.value(), expected_value);

        assert_complex_close(*evaluation.derivative(), expected_derivative);
    }

    // -------------------------------------------------------------------------
    // Convenience evaluation methods
    // -------------------------------------------------------------------------

    #[test]
    fn value_matches_combined_evaluation() {
        let dispersion = ConformalDispersion::new(CombinedSolver, chart());

        let z = c(0.20, -0.15);

        assert_complex_close(
            dispersion.value(z).unwrap(),
            *dispersion.evaluate(z).unwrap().value(),
        );
    }

    #[test]
    fn derivative_matches_combined_evaluation() {
        let dispersion = ConformalDispersion::new(CombinedSolver, chart());

        let z = c(-0.20, 0.25);

        assert_complex_close(
            dispersion.derivative(z).unwrap(),
            *dispersion.evaluate(z).unwrap().derivative(),
        );
    }

    // -------------------------------------------------------------------------
    // Error propagation
    // -------------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
    #[error("fake solver failure")]
    struct FakeSolverError;

    #[derive(Clone, Copy, Debug)]
    struct FailingSolver;

    impl ModeDeterminantSolver<J> for FailingSolver {
        type Error = FakeSolverError;

        fn determinant(
            &self,
            _coordinates: &CanonicalCoordinates<J>,
            _exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            Err(FakeSolverError)
        }
    }

    #[test]
    fn solver_error_is_propagated() {
        let dispersion = ConformalDispersion::new(FailingSolver, chart());

        let error = dispersion.evaluate(c(0.0, 0.0)).unwrap_err();

        assert_eq!(error, DispersionError::Solver(FakeSolverError,),);
    }

    #[test]
    fn chart_error_is_propagated_before_solver_evaluation() {
        let dispersion = ConformalDispersion::new(CombinedSolver, chart());

        // Outside [-1,1] x [-1,1].
        let error = dispersion.evaluate(c(2.0, 0.0)).unwrap_err();

        assert!(
            matches!(error, DispersionError::Chart(_),),
            "expected chart error, got {error:?}",
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    use approx::assert_relative_eq;
    use argloc::ArgumentConfig;
    use lamina_core::{
        AnalyticalMaterialHandle, AnalyticalMaterialStack, Constant, EvaluateMaterial, ModeJet1,
        Polarisation, Scatter2,
    };
    use lamina_materials::database::SiliconCarbideWang2012;
    use lamina_units::Length;
    use ndarray::Ix0;
    use num_complex::Complex64;

    use crate::conformal::{
        ConformalRegion,
        chart::{ConformalChart, ConformalChartConfig, CoordinateSheet},
        in_plane::{ComplexInPlaneMap, region::InPlaneSearchExtent},
        wavenumber::ComplexWavenumberMap,
    };

    type C = Complex64;
    type J = ModeJet1<C, Ix0>;

    const ABS_TOL: f64 = 1.0e-8;
    const REL_TOL: f64 = 1.0e-7;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    // =========================================================================
    // Physical fixtures
    // =========================================================================

    /// Simple asymmetric lossless dielectric slab.
    fn lossless_stack() -> Stack<Constant<f64>, f64> {
        Stack::builder(Constant::vacuum(), Constant::dielectric(2.25))
            .layer(Constant::dielectric(4.0), Length::centimetres(0.5))
            .finalise()
    }

    /// Same geometry, but with a lossy finite layer.
    ///
    /// Exterior media should remain lossless constants so that we can use the
    /// canonical conformal search-region constructors.
    fn lossy_interior_stack() -> AnalyticalMaterialStack<C> {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
            .analytical_layer(
                SiliconCarbideWang2012::new().model(),
                Length::micrometres(1.0),
            )
            .finalise()
    }

    /// Return the concrete isotropic backend currently used for mode solves.
    fn backend() -> Scatter2 {
        Scatter2::new()
    }

    // =========================================================================
    // Numerical helpers
    // =========================================================================

    /// Central finite difference along the real conformal direction.
    fn finite_difference<S, M>(
        dispersion: &ConformalDispersion<S, M, C>,
        z: C,
        h: f64,
    ) -> Result<C, DispersionError<<M as ValidateConformalRegion<f64>>::Error, S::Error>>
    where
        M: LinearCoordinateMap<ModeJet1<C, Ix0>> + ValidateConformalRegion<f64> + Clone,
        S: ModeDeterminantSolver<ModeJet1<C, Ix0>>,
    {
        let delta = c(h, 0.0);

        let plus = dispersion.value(z + delta)?;

        let minus = dispersion.value(z - delta)?;

        Ok((plus - minus) / (2.0 * h))
    }
    /// Use several finite-difference step sizes rather than relying on one
    /// arbitrarily chosen `h`.
    ///
    /// This protects the test against both truncation error at large `h` and
    /// cancellation at very small `h`.
    fn assert_derivative_matches_finite_difference<S, M>(
        dispersion: &ConformalDispersion<S, M, C>,
        z: C,
    ) -> Result<(), DispersionError<<M as ValidateConformalRegion<f64>>::Error, S::Error>>
    where
        M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
        S: ModeDeterminantSolver<J>,
        C: ComplexField<RealField = f64> + Copy,
        J: ScalarAlgebra<Scalar = C, Dimension = Ix0> + SeedJet + Clone,
    {
        let analytic = dispersion.derivative(z)?;

        let candidates = [1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6];

        let best_error = candidates
            .into_iter()
            .map(|h| {
                finite_difference(dispersion, z, h)
                    .map(|numerical| ((numerical - analytic).norm(), numerical))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();

        let numerical = best_error.1;

        assert_complex_close(analytic, numerical);

        Ok(())
    }

    // =========================================================================
    // Complex-kx dispersion
    // =========================================================================

    fn in_plane_dispersion(
        polarisation: Polarisation,
    ) -> ConformalDispersion<
        CompiledPlaneWaveModeSolver<J, Constant<f64>, Scatter2>,
        ComplexInPlaneMap<J>,
        C,
    > {
        let stack = lossless_stack();

        let solver = CompiledPlaneWaveModeSolver::compile(&stack, backend(), polarisation).unwrap();

        // Use a fixed real k0 that is safely away from any material
        // singularity.
        let map =
            ComplexInPlaneMap::<J>::new(2.0, c(1.0, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
                .unwrap();

        // For this test the explicit rectangle is preferable to making the
        // derivative test depend on search-region construction as well.
        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        let chart = ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        ConformalDispersion::new(solver, chart)
    }

    #[test]
    fn in_plane_te_dispersion_derivative_matches_finite_difference() {
        let dispersion = in_plane_dispersion(Polarisation::TransverseElectric);

        for z in [c(-0.45, 0.20), c(0.00, 0.35), c(0.40, -0.25)] {
            assert_derivative_matches_finite_difference(&dispersion, z).unwrap();
        }
    }

    #[test]
    fn in_plane_tm_dispersion_derivative_matches_finite_difference() {
        let dispersion = in_plane_dispersion(Polarisation::TransverseMagnetic);

        for z in [c(-0.40, -0.20), c(0.10, 0.30), c(0.45, -0.15)] {
            assert_derivative_matches_finite_difference(&dispersion, z).unwrap();
        }
    }

    // =========================================================================
    // Complex-k0 dispersion
    // =========================================================================

    fn wavenumber_dispersion(
        polarisation: Polarisation,
    ) -> ConformalDispersion<
        CompiledPlaneWaveModeSolver<J, Constant<f64>, Scatter2>,
        ComplexWavenumberMap<J>,
        C,
    > {
        let stack = lossless_stack();

        let solver = CompiledPlaneWaveModeSolver::compile(&stack, backend(), polarisation).unwrap();

        // Fixed real kx.
        let map = ComplexWavenumberMap::<J>::new(
            1.5,
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(2.25, 0.0),
            c(1.0, 0.0),
        )
        .unwrap();

        // Use the upper chart. The real axis contains the linear-k0 branch
        // points, so deliberately stay clear of it.
        let region = ConformalRegion::new(-1.0, 1.0, 0.15, 1.20).unwrap();

        let chart = ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        ConformalDispersion::new(solver, chart)
    }

    #[test]
    fn wavenumber_te_dispersion_derivative_matches_finite_difference() {
        let dispersion = wavenumber_dispersion(Polarisation::TransverseElectric);

        for z in [c(-0.45, 0.30), c(0.00, 0.55), c(0.40, 0.80)] {
            assert_derivative_matches_finite_difference(&dispersion, z).unwrap();
        }
    }

    #[test]
    fn wavenumber_tm_dispersion_derivative_matches_finite_difference() {
        let dispersion = wavenumber_dispersion(Polarisation::TransverseMagnetic);

        for z in [c(-0.35, 0.35), c(0.10, 0.60), c(0.35, 0.90)] {
            assert_derivative_matches_finite_difference(&dispersion, z).unwrap();
        }
    }

    // =========================================================================
    // Lossy interior
    // =========================================================================

    fn lossy_interior_in_plane_dispersion() -> ConformalDispersion<
        CompiledPlaneWaveModeSolver<J, AnalyticalMaterialHandle<C>, Scatter2>,
        ComplexInPlaneMap<J>,
        C,
    > {
        let stack = lossy_interior_stack();

        let solver = CompiledPlaneWaveModeSolver::compile(
            &stack,
            backend(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

        let map =
            ComplexInPlaneMap::<J>::new(2.0, c(1.0, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
                .unwrap();

        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        let chart = ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        ConformalDispersion::new(solver, chart)
    }

    #[test]
    fn lossy_interior_dispersion_derivative_matches_finite_difference() {
        let dispersion = lossy_interior_in_plane_dispersion();

        for z in [c(-0.35, 0.20), c(0.05, -0.30), c(0.40, 0.25)] {
            assert_derivative_matches_finite_difference(&dispersion, z).unwrap();
        }
    }

    #[test]
    fn apm_finds_real_stack_mode_in_single_in_plane_chart() {
        type C = Complex64;
        type J = ModeJet1<C, Ix0>;

        // ---------------------------------------------------------------------
        // 1. Physical stack
        // ---------------------------------------------------------------------

        let stack = lossless_stack();

        let solver = CompiledPlaneWaveModeSolver::<J, _, _>::compile(
            &stack,
            Scatter2::new(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

        // ---------------------------------------------------------------------
        // 2. Conformal map and canonical search region
        // ---------------------------------------------------------------------

        let map =
            ComplexInPlaneMap::<J>::new(2.0, c(1.0, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
                .unwrap();

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: 0.05,
            })
            .unwrap();

        let chart = ConformalChart::new(
            map,
            region.clone().into_inner(),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        let dispersion = ConformalDispersion::new(solver, chart);

        // ---------------------------------------------------------------------
        // 3. Run APM over this one conformal chart
        // ---------------------------------------------------------------------

        let result = argloc::find_zeros(
            dispersion.clone(),
            region.try_into().unwrap(),
            ArgumentConfig::new(1e-5),
        )
        .unwrap();

        dbg!(&result);

        // ---------------------------------------------------------------------
        // 4. A real optical mode should have been found
        // ---------------------------------------------------------------------

        assert!(
            !result.points.is_empty(),
            "expected at least one mode in the conformal search region",
        );

        // ---------------------------------------------------------------------
        // 5. Every reported zero must actually be a determinant zero
        // ---------------------------------------------------------------------

        for zero in &result.points {
            let z = zero.location;

            let residual = dispersion.value(z).unwrap();

            assert!(
                residual.modulus() < 1.0e-7,
                "APM root does not satisfy the modal determinant: \
             z={z:?}, F(z)={residual:?}",
            );
        }
    }

    const WAVELENGTH_UM: f64 = 1.0;
    const CORE_THICKNESS_UM: f64 = 0.283916;

    fn known_te_mode_stack() -> AnalyticalMaterialStack<Complex64> {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
            .analytical_layer(
                Constant::dielectric(4.0),
                Length::micrometres(CORE_THICKNESS_UM),
            )
            .finalise()
    }

    #[test]
    fn apm_recovers_known_asymmetric_te_slab_mode() {
        type C = Complex64;
        type J = ModeJet1<C, Ix0>;

        let wavelength_cm = WAVELENGTH_UM * 1e-4;
        let k0 = 2.0 * std::f64::consts::PI / wavelength_cm;
        let expected_kx = 1.75 * k0;

        // ---------------------------------------------------------------------
        // 1. Stack engineered to have a TE mode at kx = 3.5.
        // ---------------------------------------------------------------------

        let stack = known_te_mode_stack();

        let solver = CompiledPlaneWaveModeSolver::<J, _, _>::compile(
            &stack,
            Scatter2::new(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

        // ---------------------------------------------------------------------
        // 2. Matching conformal map.
        // ---------------------------------------------------------------------

        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();

        let map = ComplexInPlaneMap::<J>::new(
            k0,
            left_exterior.evaluate_relative_permittivity(arr0(k0))[()],
            left_exterior.evaluate_relative_permeability(arr0(k0))[()],
            right_exterior.evaluate_relative_permittivity(arr0(k0))[()],
            right_exterior.evaluate_relative_permeability(arr0(k0))[()],
        )
        .unwrap();

        let region = map
            .search_region(InPlaneSearchExtent {
                // Above the core light line as well as both exterior
                // light lines, so the expected guided root lies safely
                // inside the finite conformal truncation.
                max_parallel_wavenumber: 1.95 * k0,
                imaginary_margin: 0.05,
            })
            .unwrap();

        let chart = ConformalChart::new(
            map,
            region.clone().into_inner(),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        let dispersion = ConformalDispersion::new(solver, chart);

        // ---------------------------------------------------------------------
        // 3. Run APM.
        // ---------------------------------------------------------------------

        let result = argloc::find_zeros(
            dispersion.clone(),
            region.try_into().unwrap(),
            ArgumentConfig::new(1.0e-6),
        )
        .unwrap();

        assert!(
            !result.points.is_empty(),
            "expected the engineered TE slab mode to be found",
        );

        // ---------------------------------------------------------------------
        // 4. Convert every conformal root back to the physical kx sheet.
        // ---------------------------------------------------------------------

        let modes: Vec<_> = result
            .points
            .iter()
            .map(|point| dispersion.mode_at(point.location).unwrap())
            .collect();

        // ---------------------------------------------------------------------
        // 5. Find the mode corresponding to the known positive-kx solution.
        // ---------------------------------------------------------------------

        let mode = modes
            .iter()
            .min_by(|a, b| {
                let da = (a.coordinate() - c(expected_kx, 0.0)).modulus();

                let db = (b.coordinate() - c(expected_kx, 0.0)).modulus();

                da.partial_cmp(&db).unwrap()
            })
            .expect("APM returned no modes");

        // ---------------------------------------------------------------------
        // 6. Independent physical assertion.
        // ---------------------------------------------------------------------

        assert_relative_eq!(
            mode.coordinate().real(),
            expected_kx,
            epsilon = 1.0e-5,
            max_relative = 1.0e-5,
        );

        assert_relative_eq!(mode.coordinate().imaginary(), 0.0, epsilon = 1.0e-6,);

        // ---------------------------------------------------------------------
        // 7. It must also genuinely be a determinant zero.
        // ---------------------------------------------------------------------

        assert!(
            mode.residual().modulus() < 1.0e-7,
            "known slab mode has non-small determinant residual: \
         z={:?}, kx={:?}, F={:?}",
            mode.z(),
            mode.coordinate(),
            mode.residual(),
        );
    }
}
