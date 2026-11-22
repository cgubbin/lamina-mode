use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FloatConst, float::FloatCore};

use crate::conformal::landmarks::{ExteriorSide, finite_t_preimage, require_finite_pair};

use super::{ComplexWavenumberMap, WavenumberSearchError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavenumberLandmarks<C> {
    lower_light_line: [C; 2],
    upper_light_line: [C; 2],

    lower_exterior: ExteriorSide,
    upper_exterior: ExteriorSide,

    coordinate_branch_points: [C; 2],
}

impl<C> WavenumberLandmarks<C>
where
    C: Copy,
{
    pub fn lower_light_line(&self) -> [C; 2] {
        self.lower_light_line
    }

    pub fn upper_light_line(&self) -> [C; 2] {
        self.upper_light_line
    }

    pub fn lower_exterior(&self) -> ExteriorSide {
        self.lower_exterior
    }

    pub fn upper_exterior(&self) -> ExteriorSide {
        self.upper_exterior
    }

    pub fn coordinate_branch_points(&self) -> [C; 2] {
        self.coordinate_branch_points
    }
}

impl<A, R> ComplexWavenumberMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexScalar<RealField = R> + Copy,
    R: Float + FloatConst + FloatCore + Copy,
{
    pub fn landmarks(&self) -> Result<WavenumberLandmarks<A::Scalar>, WavenumberSearchError> {
        let (chi_left, chi_right) = self.real_positive_exterior_chi()?;

        if self.r_scalar().modulus_squared() == R::zero() {
            return Err(WavenumberSearchError::DegenerateExterior);
        }

        let kx = Float::abs(self.kx_scalar().real());

        let left_light_line = kx / Float::sqrt(chi_left);

        let right_light_line = kx / Float::sqrt(chi_right);

        let (lower_k0, lower_exterior, upper_k0, upper_exterior) =
            if left_light_line <= right_light_line {
                (
                    left_light_line,
                    ExteriorSide::Left,
                    right_light_line,
                    ExteriorSide::Right,
                )
            } else {
                (
                    right_light_line,
                    ExteriorSide::Right,
                    left_light_line,
                    ExteriorSide::Left,
                )
            };

        let lower_light_line = require_finite_pair(self.physical_preimages(lower_k0)?)
            .ok_or(WavenumberSearchError::DegenerateExterior)?;

        let upper_light_line = require_finite_pair(self.physical_preimages(upper_k0)?)
            .ok_or(WavenumberSearchError::DegenerateExterior)?;

        let coordinate_branch_points =
            require_finite_pair(self.branch_roots.map(finite_t_preimage))
                .ok_or(WavenumberSearchError::DegenerateExterior)?;

        Ok(WavenumberLandmarks {
            lower_light_line,
            upper_light_line,

            lower_exterior,
            upper_exterior,

            coordinate_branch_points,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use lamina_core::ModeJet1;
    use ndarray::Ix0;
    use num_complex::Complex64;

    const ABS_TOL: f64 = 1.0e-11;
    const REL_TOL: f64 = 1.0e-11;

    type J = ModeJet1<Complex64, Ix0>;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
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

    fn lossless_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn lossy_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
    }

    fn symmetric_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(2.25, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
            .unwrap()
    }

    #[test]
    fn landmarks_identify_lower_and_upper_exteriors() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        // kx = 1.5
        //
        // left:  chi=4 -> light line k0=0.75
        // right: chi=1 -> light line k0=1.5
        assert_eq!(landmarks.lower_exterior(), ExteriorSide::Left,);

        assert_eq!(landmarks.upper_exterior(), ExteriorSide::Right,);
    }

    #[test]
    fn lower_light_line_landmark_lies_on_real_axis() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        let [first, second] = landmarks.lower_light_line();

        assert_complex_close(first, second);

        assert_relative_eq!(first.im, 0.0, epsilon = ABS_TOL,);

        assert_complex_close(map.k0_squared_scalar(first), c(0.75_f64.powi(2), 0.0));
    }

    #[test]
    fn upper_light_line_landmarks_have_equal_squared_coordinates() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        let [first, second] = landmarks.upper_light_line();

        assert_relative_eq!(
            first.re,
            second.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            first.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
        );

        assert_relative_eq!(
            second.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
        );

        assert_relative_eq!(
            (first.im - second.im).abs(),
            std::f64::consts::PI,
            epsilon = ABS_TOL,
        );

        assert_complex_close(map.k0_squared_scalar(first), c(1.5_f64.powi(2), 0.0));

        assert_complex_close(map.k0_squared_scalar(second), c(1.5_f64.powi(2), 0.0));
    }

    #[test]
    fn coordinate_branch_points_lie_on_real_axis() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        for z in landmarks.coordinate_branch_points() {
            assert_relative_eq!(z.im, 0.0, epsilon = ABS_TOL,);

            assert_complex_close(map.k0_squared_scalar(z), c(0.0, 0.0));
        }
    }

    #[test]
    fn coordinate_branch_points_are_distinct_for_asymmetric_exteriors() {
        let map = lossless_map();

        let [first, second] = map.landmarks().unwrap().coordinate_branch_points();

        assert!((first - second).modulus() > 1.0e-8);
    }

    #[test]
    fn coordinate_branch_points_match_compiled_t_roots() {
        let map = lossless_map();

        for point in map.landmarks().unwrap().coordinate_branch_points() {
            let t = (c(2.0, 0.0) * point).exp();

            assert!(
                map.branch_roots()
                    .into_iter()
                    .any(|root| { (root - t).modulus() < 1.0e-10 }),
                "no branch root matched z={point:?}",
            );
        }
    }

    #[test]
    fn canonical_landmarks_reject_lossy_exteriors() {
        let map = lossy_map();

        assert_eq!(
            map.landmarks(),
            Err(WavenumberSearchError::UnsupportedExterior,),
        );
    }

    #[test]
    fn canonical_landmarks_reject_symmetric_exteriors() {
        let map = symmetric_map();

        assert_eq!(
            map.landmarks(),
            Err(WavenumberSearchError::DegenerateExterior,),
        );
    }
}
