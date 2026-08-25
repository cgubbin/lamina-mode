//! Modal roots expressed in conformal and physical coordinates.
//!
//! The argument-principle solver locates zeros in the conformal coordinate
//! `z`. Users, however, normally want the corresponding physical vacuum
//! wavenumber or in-plane wavenumber.
//!
//! A [`ConformalMode`] retains both representations.
//!
//! Retaining `z` is important even after conversion to physical coordinates:
//! it provides the natural coordinate for later mode continuation and avoids
//! repeatedly inverting the conformal map.

/// A modal root represented in both conformal and physical coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalMode<C> {
    z: C,
    coordinate: C,
    residual: C,
}

impl<C> ConformalMode<C>
where
    C: Copy,
{
    pub(crate) fn new(z: C, coordinate: C, residual: C) -> Self {
        Self {
            z,
            coordinate,
            residual,
        }
    }

    /// Root position in the conformal plane.
    pub fn z(&self) -> C {
        self.z
    }

    /// Physical varying coordinate corresponding to this root.
    ///
    /// This is:
    ///
    /// - `kx` for a complex-in-plane search;
    /// - `k0` for a complex-wavenumber search.
    pub fn coordinate(&self) -> C {
        self.coordinate
    }

    /// Modal determinant evaluated at the reported root.
    pub fn residual(&self) -> C {
        self.residual
    }

    pub fn into_parts(self) -> (C, C, C) {
        (self.z, self.coordinate, self.residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::convert::Infallible;

    use lamina_core::{CanonicalCoordinates, ExteriorWavevectors, ModeJet1, ScalarAlgebra};
    use ndarray::Ix0;
    use num_complex::Complex64;

    use crate::{
        conformal::{
            ConformalRegion,
            chart::{ConformalChart, ConformalChartConfig, CoordinateSheet},
            in_plane::ComplexInPlaneMap,
        },
        dispersion::{ConformalDispersion, DispersionError, ModeDeterminantSolver},
    };

    type C = Complex64;
    type J = ModeJet1<C, Ix0>;
    type Map = ComplexInPlaneMap<J>;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    /// Fake determinant
    ///
    ///     F = kx.
    ///
    /// This keeps the fixture simple while ensuring that the returned residual
    /// depends on the branch-consistent physical coordinate produced by the
    /// chart.
    #[derive(Clone, Copy, Debug)]
    struct CoordinateSolver;

    impl ModeDeterminantSolver<J> for CoordinateSolver {
        type Error = Infallible;

        fn determinant(
            &self,
            coordinates: &CanonicalCoordinates<J>,
            _exterior: &ExteriorWavevectors<J>,
        ) -> Result<J, Self::Error> {
            Ok(coordinates.parallel_angular_wavenumber().clone())
        }
    }

    fn fixture_dispersion() -> ConformalDispersion<CoordinateSolver, Map, C> {
        let map =
            ComplexInPlaneMap::<J>::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))
                .unwrap();

        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        let chart = ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        ConformalDispersion::new(CoordinateSolver, chart)
    }

    #[test]
    fn mode_retains_conformal_coordinate() {
        let dispersion = fixture_dispersion();

        let z = c(0.25, 0.30);

        let mode = dispersion.mode_at(z).unwrap();

        assert_eq!(mode.z(), z,);
    }

    #[test]
    fn mode_uses_branch_consistent_chart_coordinate() {
        let dispersion = fixture_dispersion();

        let z = c(0.25, 0.30);

        let expected = dispersion.chart().coordinate(z).unwrap();

        let mode = dispersion.mode_at(z).unwrap();

        assert_eq!(mode.coordinate(), expected,);
    }

    #[test]
    fn mode_residual_matches_dispersion_value() {
        let dispersion = fixture_dispersion();

        let z = c(0.25, 0.30);

        let expected = dispersion.value(z).unwrap();

        let mode = dispersion.mode_at(z).unwrap();

        assert_eq!(mode.residual(), expected,);
    }

    #[test]
    fn mode_residual_matches_branch_consistent_coordinate_for_coordinate_solver() {
        let dispersion = fixture_dispersion();

        let z = c(-0.20, 0.35);

        let mode = dispersion.mode_at(z).unwrap();

        // CoordinateSolver returns F = kx, so the determinant residual should
        // equal the branch-consistent physical coordinate selected by the
        // chart.
        assert_eq!(mode.residual(), mode.coordinate(),);
    }

    #[test]
    fn mode_into_parts_preserves_all_values() {
        let dispersion = fixture_dispersion();

        let z = c(0.10, -0.25);

        let mode = dispersion.mode_at(z).unwrap();

        let expected_z = mode.z();

        let expected_coordinate = mode.coordinate();

        let expected_residual = mode.residual();

        let (actual_z, actual_coordinate, actual_residual) = mode.into_parts();

        assert_eq!(actual_z, expected_z,);

        assert_eq!(actual_coordinate, expected_coordinate,);

        assert_eq!(actual_residual, expected_residual,);
    }

    #[test]
    fn mode_outside_chart_region_propagates_chart_error() {
        let dispersion = fixture_dispersion();

        let error = dispersion.mode_at(c(2.0, 0.0)).unwrap_err();

        assert!(
            matches!(error, DispersionError::Chart(_),),
            "expected chart error, got {error:?}",
        );
    }

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

    fn failing_dispersion() -> ConformalDispersion<FailingSolver, Map, C> {
        let map =
            ComplexInPlaneMap::<J>::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))
                .unwrap();

        let region = ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap();

        let chart = ConformalChart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        ConformalDispersion::new(FailingSolver, chart)
    }

    #[test]
    fn mode_propagates_solver_error_when_evaluating_residual() {
        let dispersion = failing_dispersion();

        let error = dispersion.mode_at(c(0.0, 0.0)).unwrap_err();

        assert_eq!(error, DispersionError::Solver(FakeSolverError,),);
    }
}
