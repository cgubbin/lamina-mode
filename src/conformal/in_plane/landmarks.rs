use crate::conformal::landmarks::{ExteriorSide, finite_t_preimage, require_finite_pair};

use super::{ComplexInPlaneMap, InPlaneSearchError};

use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FloatConst};

/// Distinguished conformal landmarks for the canonical lossless in-plane map.
///
/// These landmarks are available when both exterior constitutive products
/// ```text
///     chi = epsilon * mu
/// ```
/// are real and strictly positive.
///
/// `inner_light_line` and `outer_light_line` each contain the two conformal
/// preimages returned within one closed representative of the fundamental
/// period. Depending on the light line these may either coincide or lie on
/// periodically identified boundaries.
///
/// The `coordinate_branch_points` are the finite zeros of the linear
/// coordinate `kx(z)`. Each entry is represented on one of the boundaries of
/// the chosen fundamental period; periodic copies differ by integer multiples
/// of `i*pi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InPlaneLandmarks<C> {
    inner_light_line: [C; 2],
    outer_light_line: [C; 2],

    inner_exterior: ExteriorSide,
    outer_exterior: ExteriorSide,

    coordinate_branch_points: [C; 2],
}

impl<C> InPlaneLandmarks<C>
where
    C: Copy,
{
    pub fn inner_light_line(&self) -> [C; 2] {
        self.inner_light_line
    }

    pub fn outer_light_line(&self) -> [C; 2] {
        self.outer_light_line
    }

    pub fn inner_exterior(&self) -> ExteriorSide {
        self.inner_exterior
    }

    pub fn outer_exterior(&self) -> ExteriorSide {
        self.outer_exterior
    }

    pub fn coordinate_branch_points(&self) -> [C; 2] {
        self.coordinate_branch_points
    }
}

impl<A, R> ComplexInPlaneMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexScalar<RealField = R> + Copy,
    R: Float + FloatConst + Copy,
{
    /// Return the canonical physical landmarks of the asymmetric,
    /// lossless in-plane map.
    ///
    /// This operation requires the two exterior constitutive products
    /// ```text
    ///     chi_j = epsilon_j * mu_j
    /// ```
    /// to be real, strictly positive, and unequal.
    ///
    /// Equal exterior products produce a degenerate conformal geometry and are
    /// not represented by `InPlaneLandmarks`.
    pub fn landmarks(&self) -> Result<InPlaneLandmarks<A::Scalar>, InPlaneSearchError> {
        let (chi_left, chi_right) = self.real_positive_exterior_chi()?;

        let k0 = Float::abs(self.k0_scalar().real());

        let left_light_line = k0 * Float::sqrt(chi_left);

        let right_light_line = k0 * Float::sqrt(chi_right);

        let (inner_kx, inner_exterior, outer_kx, outer_exterior) =
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

        let inner_light_line = self.physical_preimages(inner_kx)?;

        let outer_light_line = self.physical_preimages(outer_kx)?;

        let inner_light_line =
            require_finite_pair(inner_light_line).ok_or(InPlaneSearchError::SymmetricExterior)?;

        let outer_light_line =
            require_finite_pair(outer_light_line).ok_or(InPlaneSearchError::SymmetricExterior)?;

        let coordinate_branch_points = self.branch_roots.map(finite_t_preimage);

        let coordinate_branch_points = require_finite_pair(coordinate_branch_points)
            .ok_or(InPlaneSearchError::SymmetricExterior)?;

        Ok(InPlaneLandmarks {
            inner_light_line,
            outer_light_line,

            inner_exterior,
            outer_exterior,

            coordinate_branch_points,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use lamina_core::ModeJet1;
    use num_complex::Complex64;

    use super::*;

    type J = ModeJet1<Complex64, Ix0>;

    const ABS_TOL: f64 = 1.0e-11;
    const REL_TOL: f64 = 1.0e-11;

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

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Lossless, non-magnetic asymmetric exterior.
    ///
    /// k0 = 2
    /// chi_L = epsilon_L mu_L = 4
    /// chi_R = epsilon_R mu_R = 1
    ///
    /// r = 3/4
    /// s = 5/2
    fn lossless_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    /// Lossy, non-magnetic asymmetric exterior.
    fn lossy_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
    }

    fn symmetric_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(2.25, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0)).unwrap()
    }

    #[test]
    fn landmarks_identify_inner_and_outer_exteriors() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        // chi_L = 4, chi_R = 1.
        //
        // k0 = 2, so
        //
        // left light line  = 4
        // right light line = 2.
        assert_eq!(landmarks.inner_exterior(), ExteriorSide::Right,);

        assert_eq!(landmarks.outer_exterior(), ExteriorSide::Left,);
    }

    #[test]
    fn outer_light_line_landmark_lies_on_real_axis() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        let [first, second] = landmarks.outer_light_line();

        assert_complex_close(first, second);

        assert_relative_eq!(first.im, 0.0, epsilon = ABS_TOL,);

        assert_complex_close(map.kx_squared_scalar(first), c(16.0, 0.0));
    }

    #[test]
    fn inner_light_line_landmarks_are_periodically_equivalent() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        let [first, second] = landmarks.inner_light_line();

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

        assert_complex_close(map.kx_squared_scalar(first), c(4.0, 0.0));

        assert_complex_close(map.kx_squared_scalar(second), c(4.0, 0.0));
    }

    #[test]
    fn coordinate_branch_points_lie_on_fundamental_boundaries() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        for z in landmarks.coordinate_branch_points() {
            assert_relative_eq!(
                z.im.abs(),
                std::f64::consts::FRAC_PI_2,
                epsilon = ABS_TOL,
                max_relative = REL_TOL,
            );

            assert_complex_close(map.kx_squared_scalar(z), c(0.0, 0.0));
        }
    }

    #[test]
    fn coordinate_branch_points_match_compiled_branch_roots() {
        let map = lossless_map();

        let landmarks = map.landmarks().unwrap();

        let points = landmarks.coordinate_branch_points();

        for point in points {
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
            Err(InPlaneSearchError::UnsupportedExterior),
        );
    }

    #[test]
    fn canonical_landmarks_reject_symmetric_exteriors() {
        let map = symmetric_map();

        assert_eq!(map.landmarks(), Err(InPlaneSearchError::SymmetricExterior),);
    }
}
