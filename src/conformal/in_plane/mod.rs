mod chart;

use lamina_core::{CanonicalCoordinates, ComplexScalar, ExteriorWavevectors, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::{Ix0, arr0};
use num_traits::{Float, FloatConst, One, Zero};

use super::{ConformalMap, ConformalPoint, ConformalRegion, ValidateConformalRegion};

#[derive(Clone, Debug, PartialEq)]
pub struct ComplexInPlaneMap<A: ScalarAlgebra> {
    k0: A,
    r: A,
    s: A,

    // Ordinary scalar data used for domain validation.
    branch_roots: [A::Scalar; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InPlaneMapError {
    #[error("vacuum wavenumber must be finite and non-zero")]
    InvalidWavenumber,

    #[error("exterior constitutive parameters must be finite")]
    InvalidExteriorMaterial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InPlaneRegionError {
    #[error("search region contains or touches a branch point of the linear in-plane coordinate")]
    CoordinateBranchPoint,
}

impl<A> ComplexInPlaneMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField + Copy,
{
    pub fn new(
        k0: <A::Scalar as ComplexField>::RealField,
        epsilon_left: A::Scalar,
        mu_left: A::Scalar,
        epsilon_right: A::Scalar,
        mu_right: A::Scalar,
    ) -> Result<Self, InPlaneMapError> {
        if !k0.is_finite() || k0 == <A::Scalar as Zero>::zero().real() {
            return Err(InPlaneMapError::InvalidWavenumber);
        }

        if !epsilon_left.is_finite()
            || !mu_left.is_finite()
            || !epsilon_right.is_finite()
            || !mu_right.is_finite()
        {
            return Err(InPlaneMapError::InvalidExteriorMaterial);
        }

        let k0 = arr0(<A::Scalar as ComplexField>::from_real(k0));

        let chi_left = epsilon_left * mu_left;
        let chi_right = epsilon_right * mu_right;

        let two = A::Scalar::one() + A::Scalar::one();
        let four = two + two;

        let r = (chi_left - chi_right) / four;
        let s = (chi_left + chi_right) / two;

        let branch_roots = coordinate_branch_roots(s, r);

        Ok(Self {
            k0: A::lift_constant(k0),
            r: A::lift_constant(arr0(r)),
            s: A::lift_constant(arr0(s)),
            branch_roots,
        })
    }

    /// Evaluate q(z) = kx²(z) using ordinary scalar arithmetic.
    pub(crate) fn kx_squared_scalar(&self, z: A::Scalar) -> A::Scalar {
        let two = A::Scalar::one() + A::Scalar::one();

        let exp_2z = (z * two).exp();
        let exp_minus_2z = (-z * two).exp();

        let k0_scalar = self.k0().value()[()];
        let s_scalar = self.s().value()[()];
        let r_scalar = self.r().value()[()];

        k0_scalar * k0_scalar * (s_scalar + exp_2z + r_scalar * r_scalar * exp_minus_2z)
    }

    /// Evaluate q'(z) / q(z), where q(z) = kx²(z).
    ///
    /// The common factor k0² cancels:
    ///
    /// q'/q =
    ///
    ///     2 [exp(2z) - r² exp(-2z)]
    ///     ---------------------------------
    ///       s + exp(2z) + r² exp(-2z).
    pub(crate) fn kx_squared_log_derivative_scalar(&self, z: A::Scalar) -> A::Scalar {
        let two = A::Scalar::one() + A::Scalar::one();

        let exp_2z = (z * two).exp();
        let exp_minus_2z = (-z * two).exp();

        let r_scalar = self.r().value()[()];
        let s_scalar = self.s().value()[()];

        let r_squared = r_scalar * r_scalar;

        let numerator = two * (exp_2z - r_squared * exp_minus_2z);

        let denominator = s_scalar + exp_2z + r_squared * exp_minus_2z;

        numerator / denominator
    }
}

impl<A> ComplexInPlaneMap<A>
where
    A: ScalarAlgebra,
{
    pub fn k0(&self) -> &A {
        &self.k0
    }

    pub fn r(&self) -> &A {
        &self.r
    }

    pub fn s(&self) -> &A {
        &self.s
    }

    #[cfg(test)]
    fn branch_roots(&self) -> [A::Scalar; 2]
    where
        A::Scalar: Copy,
    {
        self.branch_roots
    }
    pub(crate) fn map_with_kx_sign(&self, z: A, kx_sign: A::Scalar) -> ConformalPoint<A>
    where
        A: Clone,
        A::Scalar: ComplexScalar,
    {
        let i = <A::Scalar as ComplexScalar>::i();

        let exp_z = z.exp();
        let exp_minus_z = z.scale(-A::Scalar::one()).exp();

        let exp_2z = exp_z.multiply(&exp_z);
        let exp_minus_2z = exp_minus_z.multiply(&exp_minus_z);

        let r_squared = self.r.square();
        let k0_squared = self.k0.square();

        let kx_squared = self
            .s
            .add(&exp_2z)
            .add(&r_squared.multiply(&exp_minus_2z))
            .multiply(&k0_squared);

        let kx = kx_squared.sqrt().scale(kx_sign);

        let r_exp_minus_z = self.r.multiply(&exp_minus_z);

        let kappa_left = exp_z.subtract(&r_exp_minus_z).multiply(&self.k0).scale(i);

        let kappa_right = exp_z.add(&r_exp_minus_z).multiply(&self.k0).scale(i);

        ConformalPoint::new(
            CanonicalCoordinates::new(self.k0.clone(), kx),
            ExteriorWavevectors::new(kappa_left, kappa_right),
        )
    }
}

impl<A> ConformalMap<A> for ComplexInPlaneMap<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
{
    fn map(&self, z: A) -> ConformalPoint<A> {
        self.map_with_kx_sign(z, A::Scalar::one())
    }
}

/// Roots in the t = exp(2z) plane at which kx² vanishes.
///
/// A zero root corresponds to z -> -infinity and is therefore not a
/// finite branch point of the conformal plane.
pub(crate) fn coordinate_branch_roots<C>(s: C, r: C) -> [C; 2]
where
    C: ComplexField + Copy,
{
    stable_quadratic_roots(s, r * r)
}

/// Roots of x² + s x + product = 0.
///
/// The second argument is the product of the roots.  Computing the
/// smaller root from Vieta's relation avoids catastrophic cancellation
/// when the two exterior media are nearly identical.
fn stable_quadratic_roots<C>(s: C, product: C) -> [C; 2]
where
    C: ComplexField + Copy,
{
    let two = C::one() + C::one();
    let four = two + two;

    let discriminant = s * s - product * four;
    let root_discriminant = discriminant.sqrt();

    let numerator_plus = -s + root_discriminant;
    let numerator_minus = -s - root_discriminant;

    let large_numerator = if numerator_plus.modulus_squared() >= numerator_minus.modulus_squared() {
        numerator_plus
    } else {
        numerator_minus
    };

    let large = large_numerator / two;

    if large.modulus_squared() == C::zero().real() {
        // Degenerate polynomial x² = 0.
        return [C::zero(); 2];
    }

    let small = product / large;

    [large, small]
}

fn branch_root_intersects_region<C>(t: C, region: &ConformalRegion<C::RealField>) -> bool
where
    C: ComplexField + Copy,
    C::RealField: Float + FloatConst + Copy,
{
    // exp(2z) is never zero at finite z.
    if t.modulus_squared() == C::zero().real() {
        return false;
    }

    let two = <C::RealField as One>::one() + <C::RealField as One>::one();
    let pi = <C::RealField as FloatConst>::PI();

    let branch_re = Float::ln(t.modulus()) / two;

    let (re_min, re_max) = region.real_bounds();

    // Closed comparison: a branch point on the contour is invalid too.
    if branch_re < re_min || branch_re > re_max {
        return false;
    }

    let base_im = t.argument() / two;
    let (im_min, im_max) = region.imaginary_bounds();

    // We need an integer n satisfying
    //
    // im_min <= base_im + n*pi <= im_max.
    //
    // Therefore
    //
    // ceil((im_min - base_im)/pi)
    //     <=
    // floor((im_max - base_im)/pi).
    let n_min = Float::ceil((im_min - base_im) / pi);
    let n_max = Float::floor((im_max - base_im) / pi);

    n_min <= n_max
}

impl<A, R> ValidateConformalRegion<R> for ComplexInPlaneMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField<RealField = R> + Copy,
    R: Float + FloatConst + Copy,
{
    type Error = InPlaneRegionError;

    fn validate_region(&self, region: &ConformalRegion<R>) -> Result<(), Self::Error> {
        if self
            .branch_roots
            .iter()
            .copied()
            .any(|root| branch_root_intersects_region(root, region))
        {
            Err(InPlaneRegionError::CoordinateBranchPoint)
        } else {
            Ok(())
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     use approx::assert_relative_eq;
//     use lamina_core::UnivariateVariableAlgebra;
//     use nalgebra::ComplexField;
//     use ndarray::arr0;
//     use num_complex::Complex64;
//     use std::f64::consts::{FRAC_PI_2, PI};

//     // Replace this alias with the actual exported first-order univariate jet.
//     //
//     // It must satisfy:
//     //
//     // J: ScalarAlgebra<Scalar = Complex64, Dimension = Ix0>
//     //  + UnivariateVariableAlgebra
//     //
//     type J = FirstJet<Complex64, Ix0>;

//     const ABS_TOL: f64 = 1.0e-11;
//     const REL_TOL: f64 = 1.0e-11;

//     fn c(re: f64, im: f64) -> Complex64 {
//         Complex64::new(re, im)
//     }

//     fn assert_complex_close(actual: Complex64, expected: Complex64) {
//         assert_relative_eq!(
//             actual.re,
//             expected.re,
//             epsilon = ABS_TOL,
//             max_relative = REL_TOL,
//         );

//         assert_relative_eq!(
//             actual.im,
//             expected.im,
//             epsilon = ABS_TOL,
//             max_relative = REL_TOL,
//         );
//     }

//     // -------------------------------------------------------------------------
//     // Jet helpers
//     // -------------------------------------------------------------------------

//     fn constant(value: Complex64) -> J {
//         J::lift_constant(arr0(value))
//     }

//     fn variable(value: Complex64) -> J {
//         J::variable(arr0(value))
//     }

//     // Adapt only these two helpers if your first-order jet exposes its
//     // coefficients under different names.
//     fn jet_value(value: &J) -> Complex64 {
//         value.value()[()]
//     }

//     fn jet_derivative(value: &J) -> Complex64 {
//         value.derivative()[()]
//     }

//     // -------------------------------------------------------------------------
//     // Fixtures
//     // -------------------------------------------------------------------------

//     /// Lossless, non-magnetic asymmetric exterior.
//     ///
//     /// k0 = 2
//     /// chi_L = epsilon_L mu_L = 4
//     /// chi_R = epsilon_R mu_R = 1
//     ///
//     /// r = 3/4
//     /// s = 5/2
//     fn lossless_map() -> ComplexInPlaneMap<J> {
//         ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
//     }

//     /// Lossy, non-magnetic asymmetric exterior.
//     fn lossy_map() -> ComplexInPlaneMap<J> {
//         ComplexInPlaneMap::new(2.0, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
//     }

//     /// Lossy and magnetic exterior.
//     ///
//     /// This is intentionally not reducible to epsilon alone, so it catches
//     /// accidental reintroduction of a mu = 1 assumption.
//     fn magnetic_lossy_map() -> ComplexInPlaneMap<J> {
//         ComplexInPlaneMap::new(1.7, c(3.2, 0.4), c(1.15, 0.08), c(1.4, 0.25), c(0.92, 0.06))
//             .unwrap()
//     }

//     fn symmetric_map() -> ComplexInPlaneMap<J> {
//         ComplexInPlaneMap::new(2.0, c(2.25, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0)).unwrap()
//     }

//     fn nearly_symmetric_map() -> ComplexInPlaneMap<J> {
//         ComplexInPlaneMap::new(
//             2.0,
//             c(2.25, 0.0),
//             c(1.0, 0.0),
//             c(2.25 + 1.0e-10, 0.0),
//             c(1.0, 0.0),
//         )
//         .unwrap()
//     }

//     fn test_points() -> [Complex64; 5] {
//         [
//             c(0.0, 0.0),
//             c(0.30, 0.20),
//             c(-0.70, 0.40),
//             c(0.60, -0.50),
//             c(-0.20, -0.70),
//         ]
//     }

//     // -------------------------------------------------------------------------
//     // Physical/scalar fixture quantities
//     // -------------------------------------------------------------------------

//     fn lossless_chi_left() -> Complex64 {
//         c(4.0, 0.0)
//     }

//     fn lossless_chi_right() -> Complex64 {
//         c(1.0, 0.0)
//     }

//     fn lossy_chi_left() -> Complex64 {
//         c(4.0, 0.8)
//     }

//     fn lossy_chi_right() -> Complex64 {
//         c(1.5, 0.35)
//     }

//     fn magnetic_lossy_chi_left() -> Complex64 {
//         c(3.2, 0.4) * c(1.15, 0.08)
//     }

//     fn magnetic_lossy_chi_right() -> Complex64 {
//         c(1.4, 0.25) * c(0.92, 0.06)
//     }

//     fn r_from_chi(left: Complex64, right: Complex64) -> Complex64 {
//         (left - right) / 4.0
//     }

//     fn s_from_chi(left: Complex64, right: Complex64) -> Complex64 {
//         (left + right) / 2.0
//     }

//     fn expected_kx_squared(k0: f64, r: Complex64, s: Complex64, z: Complex64) -> Complex64 {
//         k0 * k0 * (s + (2.0 * z).exp() + r * r * (-2.0 * z).exp())
//     }

//     fn expected_kappa_left(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
//         c(0.0, 1.0) * k0 * (z.exp() - r * (-z).exp())
//     }

//     fn expected_kappa_right(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
//         c(0.0, 1.0) * k0 * (z.exp() + r * (-z).exp())
//     }

//     fn expected_kx_derivative(k0: f64, r: Complex64, z: Complex64, kx: Complex64) -> Complex64 {
//         k0 * k0 * ((2.0 * z).exp() - r * r * (-2.0 * z).exp()) / kx
//     }

//     fn expected_kappa_left_derivative(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
//         c(0.0, 1.0) * k0 * (z.exp() + r * (-z).exp())
//     }

//     fn expected_kappa_right_derivative(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
//         c(0.0, 1.0) * k0 * (z.exp() - r * (-z).exp())
//     }

//     // -------------------------------------------------------------------------
//     // Conformal-region helpers
//     // -------------------------------------------------------------------------

//     fn finite_branch_point(t: Complex64, period: i32) -> Option<Complex64> {
//         if t.modulus_squared() == 0.0 {
//             return None;
//         }

//         Some(c(
//             0.5 * t.modulus().ln(),
//             0.5 * t.argument() + f64::from(period) * PI,
//         ))
//     }

//     fn region_around(z: Complex64, half_width: f64) -> ConformalRegion<f64> {
//         ConformalRegion::new(
//             z.re - half_width,
//             z.re + half_width,
//             z.im - half_width,
//             z.im + half_width,
//         )
//         .unwrap()
//     }

//     // =========================================================================
//     // Construction
//     // =========================================================================

//     #[test]
//     fn rejects_zero_wavenumber() {
//         let result =
//             ComplexInPlaneMap::<J>::new(0.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0));

//         assert_eq!(result.unwrap_err(), InPlaneMapError::InvalidWavenumber);
//     }

//     #[test]
//     fn rejects_non_finite_wavenumber() {
//         let result = ComplexInPlaneMap::<J>::new(
//             f64::NAN,
//             c(4.0, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//         );

//         assert_eq!(result.unwrap_err(), InPlaneMapError::InvalidWavenumber);
//     }

//     #[test]
//     fn rejects_non_finite_left_permittivity() {
//         let result = ComplexInPlaneMap::<J>::new(
//             1.0,
//             c(f64::NAN, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//         );

//         assert_eq!(
//             result.unwrap_err(),
//             InPlaneMapError::InvalidExteriorMaterial
//         );
//     }

//     #[test]
//     fn rejects_non_finite_left_permeability() {
//         let result = ComplexInPlaneMap::<J>::new(
//             1.0,
//             c(4.0, 0.0),
//             c(f64::INFINITY, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//         );

//         assert_eq!(
//             result.unwrap_err(),
//             InPlaneMapError::InvalidExteriorMaterial
//         );
//     }

//     #[test]
//     fn rejects_non_finite_right_permittivity() {
//         let result = ComplexInPlaneMap::<J>::new(
//             1.0,
//             c(4.0, 0.0),
//             c(1.0, 0.0),
//             c(1.0, f64::NAN),
//             c(1.0, 0.0),
//         );

//         assert_eq!(
//             result.unwrap_err(),
//             InPlaneMapError::InvalidExteriorMaterial
//         );
//     }

//     #[test]
//     fn rejects_non_finite_right_permeability() {
//         let result = ComplexInPlaneMap::<J>::new(
//             1.0,
//             c(4.0, 0.0),
//             c(1.0, 0.0),
//             c(1.0, 0.0),
//             c(0.0, f64::INFINITY),
//         );

//         assert_eq!(
//             result.unwrap_err(),
//             InPlaneMapError::InvalidExteriorMaterial
//         );
//     }

//     #[test]
//     fn constructor_uses_epsilon_times_mu() {
//         let map = magnetic_lossy_map();

//         let expected_r = r_from_chi(magnetic_lossy_chi_left(), magnetic_lossy_chi_right());

//         let expected_s = s_from_chi(magnetic_lossy_chi_left(), magnetic_lossy_chi_right());

//         assert_complex_close(jet_value(map.r()), expected_r);
//         assert_complex_close(jet_value(map.s()), expected_s);

//         assert_complex_close(jet_derivative(map.r()), Complex64::new(0.0, 0.0));

//         assert_complex_close(jet_derivative(map.s()), Complex64::new(0.0, 0.0));
//     }

//     #[test]
//     fn compiled_parameters_are_derivative_free() {
//         let map = lossy_map();

//         assert_complex_close(jet_derivative(map.k0()), Complex64::new(0.0, 0.0));

//         assert_complex_close(jet_derivative(map.r()), Complex64::new(0.0, 0.0));

//         assert_complex_close(jet_derivative(map.s()), Complex64::new(0.0, 0.0));
//     }

//     // =========================================================================
//     // Mapping identities
//     // =========================================================================

//     #[test]
//     fn mapped_kx_satisfies_uniformisation_for_lossless_exterior() {
//         let map = lossless_map();

//         let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

//         let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             let kx = jet_value(point.coordinates().kx());

//             assert_complex_close(kx * kx, expected_kx_squared(2.0, r, s, z0));
//         }
//     }

//     #[test]
//     fn mapped_kx_satisfies_uniformisation_for_lossy_exterior() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             let kx = jet_value(point.coordinates().kx());

//             assert_complex_close(kx * kx, expected_kx_squared(2.0, r, s, z0));
//         }
//     }

//     #[test]
//     fn exterior_wavevectors_satisfy_core_dispersion_relation_lossless() {
//         let map = lossless_map();

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             let k0 = jet_value(point.coordinates().k0());
//             let kx = jet_value(point.coordinates().kx());

//             let left = jet_value(point.exterior_wavevectors().left());

//             let right = jet_value(point.exterior_wavevectors().right());

//             assert_complex_close(left * left, lossless_chi_left() * k0 * k0 - kx * kx);

//             assert_complex_close(right * right, lossless_chi_right() * k0 * k0 - kx * kx);
//         }
//     }

//     #[test]
//     fn exterior_wavevectors_satisfy_core_dispersion_relation_lossy() {
//         let map = lossy_map();

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             let k0 = jet_value(point.coordinates().k0());
//             let kx = jet_value(point.coordinates().kx());

//             let left = jet_value(point.exterior_wavevectors().left());

//             let right = jet_value(point.exterior_wavevectors().right());

//             assert_complex_close(left * left, lossy_chi_left() * k0 * k0 - kx * kx);

//             assert_complex_close(right * right, lossy_chi_right() * k0 * k0 - kx * kx);
//         }
//     }

//     #[test]
//     fn exterior_wavevectors_satisfy_core_dispersion_relation_magnetic_lossy() {
//         let map = magnetic_lossy_map();

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             let k0 = jet_value(point.coordinates().k0());
//             let kx = jet_value(point.coordinates().kx());

//             let left = jet_value(point.exterior_wavevectors().left());

//             let right = jet_value(point.exterior_wavevectors().right());

//             assert_complex_close(left * left, magnetic_lossy_chi_left() * k0 * k0 - kx * kx);

//             assert_complex_close(
//                 right * right,
//                 magnetic_lossy_chi_right() * k0 * k0 - kx * kx,
//             );
//         }
//     }

//     #[test]
//     fn exterior_wavevectors_match_selected_conformal_sheet() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         for z0 in test_points() {
//             let point = map.map(constant(z0));

//             assert_complex_close(
//                 jet_value(point.exterior_wavevectors().left()),
//                 expected_kappa_left(2.0, r, z0),
//             );

//             assert_complex_close(
//                 jet_value(point.exterior_wavevectors().right()),
//                 expected_kappa_right(2.0, r, z0),
//             );
//         }
//     }

//     // =========================================================================
//     // Constant versus variable evaluation
//     // =========================================================================

//     #[test]
//     fn constant_and_variable_z_have_identical_mapped_values() {
//         let map = lossy_map();
//         let z0 = c(0.31, -0.27);

//         let constant_point = map.map(constant(z0));
//         let variable_point = map.map(variable(z0));

//         assert_complex_close(
//             jet_value(constant_point.coordinates().k0()),
//             jet_value(variable_point.coordinates().k0()),
//         );

//         assert_complex_close(
//             jet_value(constant_point.coordinates().kx()),
//             jet_value(variable_point.coordinates().kx()),
//         );

//         assert_complex_close(
//             jet_value(constant_point.exterior_wavevectors().left()),
//             jet_value(variable_point.exterior_wavevectors().left()),
//         );

//         assert_complex_close(
//             jet_value(constant_point.exterior_wavevectors().right()),
//             jet_value(variable_point.exterior_wavevectors().right()),
//         );
//     }

//     #[test]
//     fn fixed_k0_has_zero_conformal_derivative() {
//         let map = lossless_map();
//         let point = map.map(variable(c(0.31, 0.27)));

//         assert_complex_close(jet_derivative(point.coordinates().k0()), c(0.0, 0.0));
//     }

//     // =========================================================================
//     // Analytic z derivatives
//     // =========================================================================

//     #[test]
//     fn mapped_kx_derivative_matches_closed_form() {
//         let map = lossless_map();

//         let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

//         for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
//             let point = map.map(variable(z0));

//             let kx = jet_value(point.coordinates().kx());

//             assert_complex_close(
//                 jet_derivative(point.coordinates().kx()),
//                 expected_kx_derivative(2.0, r, z0, kx),
//             );
//         }
//     }

//     #[test]
//     fn left_kappa_derivative_matches_closed_form() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
//             let point = map.map(variable(z0));

//             assert_complex_close(
//                 jet_derivative(point.exterior_wavevectors().left()),
//                 expected_kappa_left_derivative(2.0, r, z0),
//             );
//         }
//     }

//     #[test]
//     fn right_kappa_derivative_matches_closed_form() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
//             let point = map.map(variable(z0));

//             assert_complex_close(
//                 jet_derivative(point.exterior_wavevectors().right()),
//                 expected_kappa_right_derivative(2.0, r, z0),
//             );
//         }
//     }

//     // =========================================================================
//     // Periodicity / sheet topology
//     // =========================================================================

//     #[test]
//     fn squared_coordinate_is_periodic_under_i_pi_shift() {
//         let map = lossy_map();
//         let z0 = c(0.37, 0.21);

//         let first = map.map(constant(z0));
//         let second = map.map(constant(z0 + c(0.0, PI)));

//         let first_kx = jet_value(first.coordinates().kx());
//         let second_kx = jet_value(second.coordinates().kx());

//         assert_complex_close(second_kx * second_kx, first_kx * first_kx);
//     }

//     #[test]
//     fn exterior_wavevectors_change_sign_under_i_pi_shift() {
//         let map = lossy_map();
//         let z0 = c(0.37, 0.21);

//         let first = map.map(constant(z0));
//         let second = map.map(constant(z0 + c(0.0, PI)));

//         assert_complex_close(
//             jet_value(second.exterior_wavevectors().left()),
//             -jet_value(first.exterior_wavevectors().left()),
//         );

//         assert_complex_close(
//             jet_value(second.exterior_wavevectors().right()),
//             -jet_value(first.exterior_wavevectors().right()),
//         );
//     }

//     // =========================================================================
//     // Branch-root algebra
//     // =========================================================================

//     #[test]
//     fn branch_roots_satisfy_quadratic_lossless() {
//         let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

//         let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

//         let roots = coordinate_branch_roots(s, r);

//         for root in roots {
//             let residual = root * root + s * root + r * r;
//             assert!(residual.modulus() < 1.0e-12);
//         }
//     }

//     #[test]
//     fn branch_roots_satisfy_quadratic_lossy() {
//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let roots = coordinate_branch_roots(s, r);

//         for root in roots {
//             let residual = root * root + s * root + r * r;
//             assert!(residual.modulus() < 1.0e-11);
//         }
//     }

//     #[test]
//     fn branch_root_product_is_r_squared() {
//         for (left, right) in [
//             (lossless_chi_left(), lossless_chi_right()),
//             (lossy_chi_left(), lossy_chi_right()),
//             (magnetic_lossy_chi_left(), magnetic_lossy_chi_right()),
//         ] {
//             let r = r_from_chi(left, right);
//             let s = s_from_chi(left, right);

//             let [first, second] = coordinate_branch_roots(s, r);

//             assert_complex_close(first * second, r * r);
//         }
//     }

//     #[test]
//     fn lossless_branch_roots_are_negative_real() {
//         let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

//         let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

//         let roots = coordinate_branch_roots(s, r);

//         for root in roots {
//             assert!(root.re < 0.0);
//             assert_relative_eq!(root.im, 0.0, epsilon = ABS_TOL);
//         }
//     }

//     #[test]
//     fn lossless_branch_points_lie_at_half_pi_mod_pi() {
//         let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

//         let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

//         for root in coordinate_branch_roots(s, r) {
//             let z = finite_branch_point(root, 0).unwrap();

//             assert_relative_eq!(z.im.abs(), FRAC_PI_2, epsilon = ABS_TOL,);
//         }
//     }

//     #[test]
//     fn lossy_media_move_branch_points_off_lossless_boundaries() {
//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let roots = coordinate_branch_roots(s, r);

//         assert!(roots.into_iter().any(|root| {
//             let z = finite_branch_point(root, 0).unwrap();
//             (z.im.abs() - FRAC_PI_2).abs() > 1.0e-4
//         }));
//     }

//     #[test]
//     fn symmetric_exterior_has_zero_t_root() {
//         let chi = c(2.25, 0.0);
//         let r = r_from_chi(chi, chi);
//         let s = s_from_chi(chi, chi);

//         let [first, second] = coordinate_branch_roots(s, r);

//         assert!(first.modulus_squared() == 0.0 || second.modulus_squared() == 0.0);
//     }

//     #[test]
//     fn nearly_symmetric_branch_roots_remain_accurate() {
//         let left = c(2.25, 0.0);
//         let right = c(2.25 + 1.0e-10, 0.0);

//         let r = r_from_chi(left, right);
//         let s = s_from_chi(left, right);

//         let [first, second] = coordinate_branch_roots(s, r);

//         let residual_first = first * first + s * first + r * r;

//         let residual_second = second * second + s * second + r * r;

//         assert!(residual_first.modulus() < 1.0e-12);
//         assert!(residual_second.modulus() < 1.0e-12);

//         assert_complex_close(first * second, r * r);
//     }

//     // =========================================================================
//     // Region validation
//     // =========================================================================

//     #[test]
//     fn lossless_region_inside_standard_strip_is_valid() {
//         let map = lossless_map();

//         let region = ConformalRegion::new(-2.0, 2.0, -FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05).unwrap();

//         assert!(map.validate_region(&region).is_ok());
//     }

//     #[test]
//     fn region_containing_branch_point_is_invalid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region = region_around(branch, 0.1);

//         assert_eq!(
//             map.validate_region(&region),
//             Err(InPlaneRegionError::CoordinateBranchPoint),
//         );
//     }

//     #[test]
//     fn branch_point_on_left_edge_is_invalid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region =
//             ConformalRegion::new(branch.re, branch.re + 0.2, branch.im - 0.1, branch.im + 0.1)
//                 .unwrap();

//         assert_eq!(
//             map.validate_region(&region),
//             Err(InPlaneRegionError::CoordinateBranchPoint),
//         );
//     }

//     #[test]
//     fn branch_point_on_right_edge_is_invalid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region =
//             ConformalRegion::new(branch.re - 0.2, branch.re, branch.im - 0.1, branch.im + 0.1)
//                 .unwrap();

//         assert!(map.validate_region(&region).is_err());
//     }

//     #[test]
//     fn branch_point_on_lower_edge_is_invalid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region =
//             ConformalRegion::new(branch.re - 0.1, branch.re + 0.1, branch.im, branch.im + 0.2)
//                 .unwrap();

//         assert!(map.validate_region(&region).is_err());
//     }

//     #[test]
//     fn branch_point_on_upper_edge_is_invalid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region =
//             ConformalRegion::new(branch.re - 0.1, branch.re + 0.1, branch.im - 0.2, branch.im)
//                 .unwrap();

//         assert!(map.validate_region(&region).is_err());
//     }

//     #[test]
//     fn branch_imaginary_overlap_without_real_overlap_is_valid() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 0).unwrap();

//         let region = ConformalRegion::new(
//             branch.re + 1.0,
//             branch.re + 2.0,
//             branch.im - 0.1,
//             branch.im + 0.1,
//         )
//         .unwrap();

//         assert!(map.validate_region(&region).is_ok());
//     }

//     #[test]
//     fn validator_detects_periodic_copy_of_branch_point() {
//         let map = lossy_map();

//         let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

//         let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

//         let [root, _] = coordinate_branch_roots(s, r);
//         let branch = finite_branch_point(root, 3).unwrap();

//         let region = region_around(branch, 0.1);

//         assert_eq!(
//             map.validate_region(&region),
//             Err(InPlaneRegionError::CoordinateBranchPoint),
//         );
//     }

//     #[test]
//     fn zero_t_root_is_not_a_finite_branch_point() {
//         let map = symmetric_map();

//         let region = ConformalRegion::new(-100.0, -90.0, -0.4, 0.4).unwrap();

//         assert!(map.validate_region(&region).is_ok());
//     }
// }
