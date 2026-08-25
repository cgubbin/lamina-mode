//! Conformal uniformisation of the complex in-plane wavenumber.
//!
//! This module parameterises a fixed-frequency planar optical problem by a
//! complex conformal coordinate `z`. The transformation removes the
//! square-root branch structure of the two exterior transverse wavevectors and
//! provides analytic exterior radiation conditions suitable for argument-
//! principle mode searches.
//!
//! # Definitions
//!
//! Let
//!
//! ```text
//! chi_L = epsilon_L * mu_L
//! chi_R = epsilon_R * mu_R
//!
//! r = (chi_L - chi_R) / 4
//! s = (chi_L + chi_R) / 2.
//! ```
//!
//! For fixed vacuum wavenumber `k0`, introduce
//!
//! ```text
//! t = exp(2z).
//! ```
//!
//! The conformal map is
//!
//! ```text
//! kx²(z)
//!     = k0² [s + t + r²/t],
//! ```
//!
//! while the exterior transverse wavevectors in the `lamina-core` convention
//! are
//!
//! ```text
//! kappa_L(z)
//!     = i k0 [exp(z) - r exp(-z)]
//!
//! kappa_R(z)
//!     = i k0 [exp(z) + r exp(-z)].
//! ```
//!
//! These satisfy identically
//!
//! ```text
//! kappa_L² = chi_L k0² - kx²
//! kappa_R² = chi_R k0² - kx².
//! ```
//!
//! Unlike pointwise square roots of the exterior dispersion relations, these
//! expressions define the desired exterior sheets analytically in `z`.
//!
//! # Fundamental period
//!
//! The squared coordinate is periodic under
//!
//! ```text
//! z -> z + i*pi.
//! ```
//!
//! The exterior wavevectors change sign under the same translation. One
//! convenient representative of the conformal surface is therefore
//!
//! ```text
//! -pi/2 <= Im(z) <= pi/2,
//! ```
//!
//! with the upper and lower boundaries periodically identified.
//!
//! A numerical search contour normally retains a finite margin from these
//! boundaries.
//!
//! # Physical real-kx geometry
//!
//! When both exterior constitutive products are real and positive, the image
//! of real physical `kx` can be described explicitly.
//!
//! Write
//!
//! ```text
//! u = (kx/k0)².
//! ```
//!
//! The inverse satisfies
//!
//! ```text
//! t² - (u - s)t + r² = 0.
//! ```
//!
//! Its discriminant factorises as
//!
//! ```text
//! (u - chi_L)(u - chi_R).
//! ```
//!
//! Suppose, without loss of generality,
//!
//! ```text
//! chi_outer > chi_inner.
//! ```
//!
//! ## Above both light lines
//!
//! For
//!
//! ```text
//! u > chi_outer,
//! ```
//!
//! both exterior channels are evanescent. The two `t` roots are positive and
//! the two physical preimages lie on the real `z` axis.
//!
//! These form the two high-|kx| branches on either side of the conformal
//! surface.
//!
//! ## Between the light lines
//!
//! For
//!
//! ```text
//! chi_inner < u < chi_outer,
//! ```
//!
//! one exterior channel is propagating and the other evanescent. The two roots
//! are complex conjugates and obey
//!
//! ```text
//! |t| = |r|.
//! ```
//!
//! Therefore both physical preimages lie on the vertical line
//!
//! ```text
//! Re(z) = 1/2 ln|r|.
//! ```
//!
//! This is the singly-leaky physical sector.
//!
//! ## Below both light lines
//!
//! For
//!
//! ```text
//! u < chi_inner,
//! ```
//!
//! both exterior channels are propagating. The two roots are negative real and
//! their conformal representatives lie on the periodically identified
//! boundaries
//!
//! ```text
//! Im(z) = +/- pi/2.
//! ```
//!
//! ## Light lines
//!
//! At either exterior light line the two inverse roots coalesce.
//!
//! The outer light line appears as a coincident point on the real axis. The
//! inner light line may appear as two representatives on the upper and lower
//! boundaries; these differ by exactly `i*pi` and therefore represent the same
//! point on the periodic conformal surface.
//!
//! # Linear-coordinate branch points
//!
//! `lamina-core` consumes the linear coordinate `kx`, not merely `kx²`.
//! Zeros of `kx²` therefore remain branch points of the linear coordinate.
//!
//! They satisfy
//!
//! ```text
//! t² + s t + r² = 0.
//! ```
//!
//! For real positive exterior constitutive products these roots are negative
//! real, so their conformal preimages lie on
//!
//! ```text
//! Im(z) = +/- pi/2.
//! ```
//!
//! Consequently a trimmed fundamental strip
//!
//! ```text
//! -pi/2 + delta < Im(z) < pi/2 - delta
//! ```
//!
//! contains no linear-coordinate branch point and supports one analytic
//! [`ConformalChart`].
//!
//! This single chart can contain analytic continuations of guided and leaky
//! modes and therefore permits mode trajectories to cross exterior light
//! lines.
//!
//! # Complex or lossy exterior media
//!
//! [`ComplexInPlaneMap`] itself does not require lossless exterior media.
//! At fixed `k0`, dispersive exterior materials may be sampled once and the
//! resulting finite complex `epsilon*mu` values can be used directly.
//!
//! For complex exterior constitutive products:
//!
//! - the conformal map and analytic exterior `kappa` expressions remain valid;
//! - the zeros of `kx²` move away from the lossless strip boundaries;
//! - the real physical axis no longer separates cleanly into propagating and
//!   evanescent sectors;
//! - "inner" and "outer" real light lines cease to provide a canonical search
//!   geometry.
//!
//! Arbitrary complex-exterior searches should therefore construct an explicit
//! [`ConformalRegion`], validate it using [`ValidateConformalRegion`], and
//! build a [`ConformalChart`] over that region.
//!
//! The convenience operations [`ComplexInPlaneMap::landmarks`],
//! [`ComplexInPlaneMap::physical_preimages`], and
//! [`ComplexInPlaneMap::search_region`] are intentionally restricted to real,
//! positive exterior constitutive products.

pub(crate) mod landmarks;
pub(crate) mod region;

pub use region::InPlaneSearchError;

use lamina_core::{CanonicalCoordinates, ComplexScalar, ExteriorWavevectors, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::{Ix0, arr0};
use num_traits::{One, Zero};

use super::{
    ConformalPoint, RawConformalMap,
    branch::stable_quadratic_roots,
    chart::{ConformalChart, LinearCoordinateMap},
};

pub type InPlaneConformalChart<A> = ConformalChart<ComplexInPlaneMap<A>, A>;

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

    fn k0_scalar(&self) -> A::Scalar {
        self.k0().value()[()]
    }

    fn r_scalar(&self) -> A::Scalar {
        self.r().value()[()]
    }

    fn s_scalar(&self) -> A::Scalar {
        self.s().value()[()]
    }

    /// Recover the two exterior constitutive products
    /// ```text
    ///     chi = epsilon * mu
    /// ```
    /// from the compiled conformal coefficients.
    ///
    /// Since
    /// ```text
    ///     r = (chi_L - chi_R) / 4
    ///     s = (chi_L + chi_R) / 2,
    /// ```
    /// we have
    /// ```text
    ///     chi_L = s + 2r
    ///     chi_R = s - 2r.
    /// ```
    fn exterior_chi(&self) -> (A::Scalar, A::Scalar) {
        let two = A::Scalar::one() + A::Scalar::one();

        let r = self.r_scalar();
        let s = self.s_scalar();

        (s + two * r, s - two * r)
    }

    /// Return the exterior constitutive products when both are real and
    /// strictly positive.
    fn real_positive_exterior_chi(
        &self,
    ) -> Result<
        (
            <A::Scalar as ComplexField>::RealField,
            <A::Scalar as ComplexField>::RealField,
        ),
        InPlaneSearchError,
    > {
        let (left, right) = self.exterior_chi();

        let left_im = left.imaginary();
        let right_im = right.imaginary();

        if left_im != A::Scalar::zero().real() || right_im != A::Scalar::zero().real() {
            return Err(InPlaneSearchError::UnsupportedExterior);
        }

        let left = left.real();
        let right = right.real();

        if !left.is_finite()
            || !right.is_finite()
            || left <= A::Scalar::zero().real()
            || right <= A::Scalar::zero().real()
        {
            return Err(InPlaneSearchError::UnsupportedExterior);
        }

        Ok((left, right))
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
    /// ```text
    /// q'/q =
    ///
    ///     2 [exp(2z) - r² exp(-2z)]
    ///     ---------------------------------
    ///       s + exp(2z) + r² exp(-2z).
    /// ```
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

impl<A> RawConformalMap<A> for ComplexInPlaneMap<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
{
    fn map_raw(&self, z: A) -> ConformalPoint<A> {
        self.map_with_kx_sign(z, A::Scalar::one())
    }
}

impl<A> LinearCoordinateMap<A> for ComplexInPlaneMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0> + Clone,
    A::Scalar: ComplexScalar,
{
    fn coordinate_squared_scalar(&self, z: A::Scalar) -> A::Scalar {
        self.kx_squared_scalar(z)
    }

    fn coordinate_squared_log_derivative_scalar(&self, z: A::Scalar) -> A::Scalar {
        self.kx_squared_log_derivative_scalar(z)
    }

    fn map_with_coordinate_sign(&self, z: A, sign: A::Scalar) -> ConformalPoint<A> {
        self.map_with_kx_sign(z, sign)
    }
}

#[cfg(test)]
mod tests {
    use crate::conformal::{
        ConformalRegion, ValidateConformalRegion,
        in_plane::region::{InPlaneRegionError, InPlaneSearchExtent},
    };

    use super::*;

    use approx::assert_relative_eq;
    use lamina_core::{ModeJet1, SeedJet};
    use nalgebra::ComplexField;
    use ndarray::arr0;
    use num_complex::Complex64;
    use std::f64::consts::{FRAC_PI_2, PI};

    type J = ModeJet1<Complex64, Ix0>;

    const ABS_TOL: f64 = 1.0e-11;
    const REL_TOL: f64 = 1.0e-11;

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

    // -------------------------------------------------------------------------
    // Jet helpers
    // -------------------------------------------------------------------------

    fn constant(value: Complex64) -> J {
        J::lift_constant(arr0(value))
    }

    fn variable(value: Complex64) -> J {
        J::variable(arr0(value), 0).unwrap()
    }

    // Adapt only these two helpers if your first-order jet exposes its
    // coefficients under different names.
    fn jet_value(value: &J) -> Complex64 {
        value.value()[()]
    }

    fn jet_derivative(value: &J) -> Complex64 {
        value.first()[()]
    }

    // -------------------------------------------------------------------------
    // Fixtures
    // -------------------------------------------------------------------------

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

    /// Lossy and magnetic exterior.
    ///
    /// This is intentionally not reducible to epsilon alone, so it catches
    /// accidental reintroduction of a mu = 1 assumption.
    fn magnetic_lossy_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(1.7, c(3.2, 0.4), c(1.15, 0.08), c(1.4, 0.25), c(0.92, 0.06))
            .unwrap()
    }

    fn symmetric_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(2.25, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn nearly_symmetric_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(
            2.0,
            c(2.25, 0.0),
            c(1.0, 0.0),
            c(2.25 + 1.0e-10, 0.0),
            c(1.0, 0.0),
        )
        .unwrap()
    }

    fn test_points() -> [Complex64; 5] {
        [
            c(0.0, 0.0),
            c(0.30, 0.20),
            c(-0.70, 0.40),
            c(0.60, -0.50),
            c(-0.20, -0.70),
        ]
    }

    // -------------------------------------------------------------------------
    // Physical/scalar fixture quantities
    // -------------------------------------------------------------------------

    fn lossless_chi_left() -> Complex64 {
        c(4.0, 0.0)
    }

    fn lossless_chi_right() -> Complex64 {
        c(1.0, 0.0)
    }

    fn lossy_chi_left() -> Complex64 {
        c(4.0, 0.8)
    }

    fn lossy_chi_right() -> Complex64 {
        c(1.5, 0.35)
    }

    fn magnetic_lossy_chi_left() -> Complex64 {
        c(3.2, 0.4) * c(1.15, 0.08)
    }

    fn magnetic_lossy_chi_right() -> Complex64 {
        c(1.4, 0.25) * c(0.92, 0.06)
    }

    fn r_from_chi(left: Complex64, right: Complex64) -> Complex64 {
        (left - right) / 4.0
    }

    fn s_from_chi(left: Complex64, right: Complex64) -> Complex64 {
        (left + right) / 2.0
    }

    fn expected_kx_squared(k0: f64, r: Complex64, s: Complex64, z: Complex64) -> Complex64 {
        k0 * k0 * (s + (2.0 * z).exp() + r * r * (-2.0 * z).exp())
    }

    fn expected_kappa_left(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * k0 * (z.exp() - r * (-z).exp())
    }

    fn expected_kappa_right(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * k0 * (z.exp() + r * (-z).exp())
    }

    fn expected_kx_derivative(k0: f64, r: Complex64, z: Complex64, kx: Complex64) -> Complex64 {
        k0 * k0 * ((2.0 * z).exp() - r * r * (-2.0 * z).exp()) / kx
    }

    fn expected_kappa_left_derivative(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * k0 * (z.exp() + r * (-z).exp())
    }

    fn expected_kappa_right_derivative(k0: f64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * k0 * (z.exp() - r * (-z).exp())
    }

    // -------------------------------------------------------------------------
    // Conformal-region helpers
    // -------------------------------------------------------------------------

    fn finite_branch_point(t: Complex64, period: i32) -> Option<Complex64> {
        if t.modulus_squared() == 0.0 {
            return None;
        }

        Some(c(
            0.5 * t.modulus().ln(),
            0.5 * t.argument() + f64::from(period) * PI,
        ))
    }

    fn region_around(z: Complex64, half_width: f64) -> ConformalRegion<f64> {
        ConformalRegion::new(
            z.re - half_width,
            z.re + half_width,
            z.im - half_width,
            z.im + half_width,
        )
        .unwrap()
    }

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn rejects_zero_wavenumber() {
        let result =
            ComplexInPlaneMap::<J>::new(0.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0));

        assert_eq!(result.unwrap_err(), InPlaneMapError::InvalidWavenumber);
    }

    #[test]
    fn rejects_non_finite_wavenumber() {
        let result = ComplexInPlaneMap::<J>::new(
            f64::NAN,
            c(4.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
        );

        assert_eq!(result.unwrap_err(), InPlaneMapError::InvalidWavenumber);
    }

    #[test]
    fn rejects_non_finite_left_permittivity() {
        let result = ComplexInPlaneMap::<J>::new(
            1.0,
            c(f64::NAN, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneMapError::InvalidExteriorMaterial
        );
    }

    #[test]
    fn rejects_non_finite_left_permeability() {
        let result = ComplexInPlaneMap::<J>::new(
            1.0,
            c(4.0, 0.0),
            c(f64::INFINITY, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneMapError::InvalidExteriorMaterial
        );
    }

    #[test]
    fn rejects_non_finite_right_permittivity() {
        let result = ComplexInPlaneMap::<J>::new(
            1.0,
            c(4.0, 0.0),
            c(1.0, 0.0),
            c(1.0, f64::NAN),
            c(1.0, 0.0),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneMapError::InvalidExteriorMaterial
        );
    }

    #[test]
    fn rejects_non_finite_right_permeability() {
        let result = ComplexInPlaneMap::<J>::new(
            1.0,
            c(4.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, f64::INFINITY),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneMapError::InvalidExteriorMaterial
        );
    }

    #[test]
    fn constructor_uses_epsilon_times_mu() {
        let map = magnetic_lossy_map();

        let expected_r = r_from_chi(magnetic_lossy_chi_left(), magnetic_lossy_chi_right());

        let expected_s = s_from_chi(magnetic_lossy_chi_left(), magnetic_lossy_chi_right());

        assert_complex_close(jet_value(map.r()), expected_r);
        assert_complex_close(jet_value(map.s()), expected_s);

        assert_complex_close(jet_derivative(map.r()), Complex64::new(0.0, 0.0));

        assert_complex_close(jet_derivative(map.s()), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn compiled_parameters_are_derivative_free() {
        let map = lossy_map();

        assert_complex_close(jet_derivative(map.k0()), Complex64::new(0.0, 0.0));

        assert_complex_close(jet_derivative(map.r()), Complex64::new(0.0, 0.0));

        assert_complex_close(jet_derivative(map.s()), Complex64::new(0.0, 0.0));
    }

    // =========================================================================
    // Mapping identities
    // =========================================================================

    #[test]
    fn mapped_kx_satisfies_uniformisation_for_lossless_exterior() {
        let map = lossless_map();

        let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            assert_complex_close(kx * kx, expected_kx_squared(2.0, r, s, z0));
        }
    }

    #[test]
    fn mapped_kx_satisfies_uniformisation_for_lossy_exterior() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            assert_complex_close(kx * kx, expected_kx_squared(2.0, r, s, z0));
        }
    }

    #[test]
    fn exterior_wavevectors_satisfy_core_dispersion_relation_lossless() {
        let map = lossless_map();

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());
            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            let left = jet_value(point.exterior_wavevectors().left());

            let right = jet_value(point.exterior_wavevectors().right());

            assert_complex_close(left * left, lossless_chi_left() * k0 * k0 - kx * kx);

            assert_complex_close(right * right, lossless_chi_right() * k0 * k0 - kx * kx);
        }
    }

    #[test]
    fn exterior_wavevectors_satisfy_core_dispersion_relation_lossy() {
        let map = lossy_map();

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());
            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            let left = jet_value(point.exterior_wavevectors().left());

            let right = jet_value(point.exterior_wavevectors().right());

            assert_complex_close(left * left, lossy_chi_left() * k0 * k0 - kx * kx);

            assert_complex_close(right * right, lossy_chi_right() * k0 * k0 - kx * kx);
        }
    }

    #[test]
    fn exterior_wavevectors_satisfy_core_dispersion_relation_magnetic_lossy() {
        let map = magnetic_lossy_map();

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());
            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            let left = jet_value(point.exterior_wavevectors().left());

            let right = jet_value(point.exterior_wavevectors().right());

            assert_complex_close(left * left, magnetic_lossy_chi_left() * k0 * k0 - kx * kx);

            assert_complex_close(
                right * right,
                magnetic_lossy_chi_right() * k0 * k0 - kx * kx,
            );
        }
    }

    #[test]
    fn exterior_wavevectors_match_selected_conformal_sheet() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        for z0 in test_points() {
            let point = map.map_raw(constant(z0));

            assert_complex_close(
                jet_value(point.exterior_wavevectors().left()),
                expected_kappa_left(2.0, r, z0),
            );

            assert_complex_close(
                jet_value(point.exterior_wavevectors().right()),
                expected_kappa_right(2.0, r, z0),
            );
        }
    }

    // =========================================================================
    // Constant versus variable evaluation
    // =========================================================================

    #[test]
    fn constant_and_variable_z_have_identical_mapped_values() {
        let map = lossy_map();
        let z0 = c(0.31, -0.27);

        let constant_point = map.map_raw(constant(z0));
        let variable_point = map.map_raw(variable(z0));

        assert_complex_close(
            jet_value(constant_point.coordinates().vacuum_angular_wavenumber()),
            jet_value(variable_point.coordinates().vacuum_angular_wavenumber()),
        );

        assert_complex_close(
            jet_value(constant_point.coordinates().parallel_angular_wavenumber()),
            jet_value(variable_point.coordinates().parallel_angular_wavenumber()),
        );

        assert_complex_close(
            jet_value(constant_point.exterior_wavevectors().left()),
            jet_value(variable_point.exterior_wavevectors().left()),
        );

        assert_complex_close(
            jet_value(constant_point.exterior_wavevectors().right()),
            jet_value(variable_point.exterior_wavevectors().right()),
        );
    }

    #[test]
    fn fixed_k0_has_zero_conformal_derivative() {
        let map = lossless_map();
        let point = map.map_raw(variable(c(0.31, 0.27)));

        assert_complex_close(
            jet_derivative(point.coordinates().vacuum_angular_wavenumber()),
            c(0.0, 0.0),
        );
    }

    // =========================================================================
    // Analytic z derivatives
    // =========================================================================

    #[test]
    fn mapped_kx_derivative_matches_closed_form() {
        let map = lossless_map();

        let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

        for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
            let point = map.map_raw(variable(z0));

            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            assert_complex_close(
                jet_derivative(point.coordinates().parallel_angular_wavenumber()),
                expected_kx_derivative(2.0, r, z0, kx),
            );
        }
    }

    #[test]
    fn left_kappa_derivative_matches_closed_form() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
            let point = map.map_raw(variable(z0));

            assert_complex_close(
                jet_derivative(point.exterior_wavevectors().left()),
                expected_kappa_left_derivative(2.0, r, z0),
            );
        }
    }

    #[test]
    fn right_kappa_derivative_matches_closed_form() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        for z0 in [c(0.31, 0.27), c(-0.42, 0.18), c(0.15, -0.36)] {
            let point = map.map_raw(variable(z0));

            assert_complex_close(
                jet_derivative(point.exterior_wavevectors().right()),
                expected_kappa_right_derivative(2.0, r, z0),
            );
        }
    }

    // =========================================================================
    // Periodicity / sheet topology
    // =========================================================================

    #[test]
    fn squared_coordinate_is_periodic_under_i_pi_shift() {
        let map = lossy_map();
        let z0 = c(0.37, 0.21);

        let first = map.map_raw(constant(z0));
        let second = map.map_raw(constant(z0 + c(0.0, PI)));

        let first_kx = jet_value(first.coordinates().parallel_angular_wavenumber());
        let second_kx = jet_value(second.coordinates().parallel_angular_wavenumber());

        assert_complex_close(second_kx * second_kx, first_kx * first_kx);
    }

    #[test]
    fn exterior_wavevectors_change_sign_under_i_pi_shift() {
        let map = lossy_map();
        let z0 = c(0.37, 0.21);

        let first = map.map_raw(constant(z0));
        let second = map.map_raw(constant(z0 + c(0.0, PI)));

        assert_complex_close(
            jet_value(second.exterior_wavevectors().left()),
            -jet_value(first.exterior_wavevectors().left()),
        );

        assert_complex_close(
            jet_value(second.exterior_wavevectors().right()),
            -jet_value(first.exterior_wavevectors().right()),
        );
    }

    // =========================================================================
    // Branch-root algebra
    // =========================================================================

    #[test]
    fn branch_roots_satisfy_quadratic_lossless() {
        let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

        let roots = coordinate_branch_roots(s, r);

        for root in roots {
            let residual = root * root + s * root + r * r;
            assert!(residual.modulus() < 1.0e-12);
        }
    }

    #[test]
    fn branch_roots_satisfy_quadratic_lossy() {
        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let roots = coordinate_branch_roots(s, r);

        for root in roots {
            let residual = root * root + s * root + r * r;
            assert!(residual.modulus() < 1.0e-11);
        }
    }

    #[test]
    fn branch_root_product_is_r_squared() {
        for (left, right) in [
            (lossless_chi_left(), lossless_chi_right()),
            (lossy_chi_left(), lossy_chi_right()),
            (magnetic_lossy_chi_left(), magnetic_lossy_chi_right()),
        ] {
            let r = r_from_chi(left, right);
            let s = s_from_chi(left, right);

            let [first, second] = coordinate_branch_roots(s, r);

            assert_complex_close(first * second, r * r);
        }
    }

    #[test]
    fn lossless_branch_roots_are_negative_real() {
        let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

        let roots = coordinate_branch_roots(s, r);

        for root in roots {
            assert!(root.re < 0.0);
            assert_relative_eq!(root.im, 0.0, epsilon = ABS_TOL);
        }
    }

    #[test]
    fn lossless_branch_points_lie_at_half_pi_mod_pi() {
        let r = r_from_chi(lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(lossless_chi_left(), lossless_chi_right());

        for root in coordinate_branch_roots(s, r) {
            let z = finite_branch_point(root, 0).unwrap();

            assert_relative_eq!(z.im.abs(), FRAC_PI_2, epsilon = ABS_TOL,);
        }
    }

    #[test]
    fn lossy_media_move_branch_points_off_lossless_boundaries() {
        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let roots = coordinate_branch_roots(s, r);

        assert!(roots.into_iter().any(|root| {
            let z = finite_branch_point(root, 0).unwrap();
            (z.im.abs() - FRAC_PI_2).abs() > 1.0e-4
        }));
    }

    #[test]
    fn symmetric_exterior_has_zero_t_root() {
        let chi = c(2.25, 0.0);
        let r = r_from_chi(chi, chi);
        let s = s_from_chi(chi, chi);

        let [first, second] = coordinate_branch_roots(s, r);

        assert!(first.modulus_squared() == 0.0 || second.modulus_squared() == 0.0);
    }

    #[test]
    fn nearly_symmetric_branch_roots_remain_accurate() {
        let left = c(2.25, 0.0);
        let right = c(2.25 + 1.0e-10, 0.0);

        let r = r_from_chi(left, right);
        let s = s_from_chi(left, right);

        let [first, second] = coordinate_branch_roots(s, r);

        let residual_first = first * first + s * first + r * r;

        let residual_second = second * second + s * second + r * r;

        assert!(residual_first.modulus() < 1.0e-12);
        assert!(residual_second.modulus() < 1.0e-12);

        assert_complex_close(first * second, r * r);
    }

    // =========================================================================
    // Region validation
    // =========================================================================

    #[test]
    fn lossless_region_inside_standard_strip_is_valid() {
        let map = lossless_map();

        let region = ConformalRegion::new(-2.0, 2.0, -FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05).unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn region_containing_branch_point_is_invalid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region = region_around(branch, 0.1);

        assert_eq!(
            map.validate_region(&region),
            Err(InPlaneRegionError::CoordinateBranchPoint),
        );
    }

    #[test]
    fn branch_point_on_left_edge_is_invalid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region =
            ConformalRegion::new(branch.re, branch.re + 0.2, branch.im - 0.1, branch.im + 0.1)
                .unwrap();

        assert_eq!(
            map.validate_region(&region),
            Err(InPlaneRegionError::CoordinateBranchPoint),
        );
    }

    #[test]
    fn branch_point_on_right_edge_is_invalid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region =
            ConformalRegion::new(branch.re - 0.2, branch.re, branch.im - 0.1, branch.im + 0.1)
                .unwrap();

        assert!(map.validate_region(&region).is_err());
    }

    #[test]
    fn branch_point_on_lower_edge_is_invalid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region =
            ConformalRegion::new(branch.re - 0.1, branch.re + 0.1, branch.im, branch.im + 0.2)
                .unwrap();

        assert!(map.validate_region(&region).is_err());
    }

    #[test]
    fn branch_point_on_upper_edge_is_invalid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region =
            ConformalRegion::new(branch.re - 0.1, branch.re + 0.1, branch.im - 0.2, branch.im)
                .unwrap();

        assert!(map.validate_region(&region).is_err());
    }

    #[test]
    fn branch_imaginary_overlap_without_real_overlap_is_valid() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 0).unwrap();

        let region = ConformalRegion::new(
            branch.re + 1.0,
            branch.re + 2.0,
            branch.im - 0.1,
            branch.im + 0.1,
        )
        .unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn validator_detects_periodic_copy_of_branch_point() {
        let map = lossy_map();

        let r = r_from_chi(lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(lossy_chi_left(), lossy_chi_right());

        let [root, _] = coordinate_branch_roots(s, r);
        let branch = finite_branch_point(root, 3).unwrap();

        let region = region_around(branch, 0.1);

        assert_eq!(
            map.validate_region(&region),
            Err(InPlaneRegionError::CoordinateBranchPoint),
        );
    }

    #[test]
    fn zero_t_root_is_not_a_finite_branch_point() {
        let map = symmetric_map();

        let region = ConformalRegion::new(-100.0, -90.0, -0.4, 0.4).unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    // =============================================================================
    // Physical inverse: real kx -> conformal preimages
    // =============================================================================

    #[test]
    fn physical_preimages_round_trip_above_both_light_lines() {
        let map = lossless_map();

        // k0 = 2, chi_L = 4, chi_R = 1.
        //
        // Exterior light lines are therefore
        //
        //     kx = 4
        //     kx = 2.
        //
        // kx = 5 lies above both.
        let kx = 5.0;

        let preimages = map.physical_preimages(kx).unwrap();

        for z in preimages.into_iter().flatten() {
            let q = map.kx_squared_scalar(z);

            assert_complex_close(q, c(kx * kx, 0.0));
        }
    }

    #[test]
    fn physical_preimages_round_trip_between_light_lines() {
        let map = lossless_map();

        // 2 < kx < 4: one exterior channel propagating,
        // one evanescent.
        let kx = 3.0;

        let preimages = map.physical_preimages(kx).unwrap();

        for z in preimages.into_iter().flatten() {
            let q = map.kx_squared_scalar(z);

            assert_complex_close(q, c(kx * kx, 0.0));
        }
    }

    #[test]
    fn physical_preimages_round_trip_below_both_light_lines() {
        let map = lossless_map();

        // kx < 2: both exterior channels propagating.
        let kx = 1.0;

        let preimages = map.physical_preimages(kx).unwrap();

        for z in preimages.into_iter().flatten() {
            let q = map.kx_squared_scalar(z);

            assert_complex_close(q, c(kx * kx, 0.0));
        }
    }

    #[test]
    fn guided_preimages_are_real() {
        let map = lossless_map();

        let [first, second] = map.physical_preimages(5.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        assert_relative_eq!(first.im, 0.0, epsilon = ABS_TOL,);

        assert_relative_eq!(second.im, 0.0, epsilon = ABS_TOL,);

        assert!(first.re < second.re);
    }

    #[test]
    fn singly_leaky_preimages_have_equal_real_parts() {
        let map = lossless_map();

        let [first, second] = map.physical_preimages(3.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        assert_relative_eq!(
            first.re,
            second.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    #[test]
    fn singly_leaky_preimages_are_complex_conjugates() {
        let map = lossless_map();

        let [first, second] = map.physical_preimages(3.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        assert_relative_eq!(
            first.re,
            second.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            first.im,
            -second.im,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    #[test]
    fn singly_leaky_preimages_lie_on_half_log_abs_r() {
        let map = lossless_map();

        let [first, second] = map.physical_preimages(3.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        let r = jet_value(map.r());
        let expected_re = 0.5 * r.modulus().ln();

        assert_relative_eq!(
            first.re,
            expected_re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            second.re,
            expected_re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    #[test]
    fn doubly_propagating_preimages_lie_on_strip_boundaries() {
        let map = lossless_map();

        let [first, second] = map.physical_preimages(1.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        assert_relative_eq!(
            first.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            second.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    #[test]
    fn outer_light_line_maps_to_branch_point() {
        let map = lossless_map();

        // chi_L = 4 and k0 = 2 -> outer light line kx = 4.
        let [first, second] = map.physical_preimages(4.0).unwrap();

        let points = [first, second];

        // At the light line, the two t-plane roots coalesce.
        let finite: Vec<_> = points.into_iter().flatten().collect();

        assert_eq!(finite.len(), 2);

        assert_complex_close(finite[0], finite[1]);

        // The mapped coordinate is exactly the exterior threshold.
        let q = map.kx_squared_scalar(finite[0]);

        assert_complex_close(q, c(16.0, 0.0));
    }

    #[test]
    fn inner_light_line_maps_to_periodically_equivalent_branch_points() {
        let map = lossless_map();

        // chi_R = 1 and k0 = 2 -> inner light line kx = 2.
        let [first, second] = map.physical_preimages(2.0).unwrap();

        let first = first.unwrap();
        let second = second.unwrap();

        // Both representatives have the same real coordinate.
        assert_relative_eq!(
            first.re,
            second.re,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        // The repeated negative t-root appears on the two boundaries of the
        // chosen fundamental strip.
        assert_relative_eq!(
            first.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        assert_relative_eq!(
            second.im.abs(),
            std::f64::consts::FRAC_PI_2,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        // They differ by one period of the conformal map.
        assert_relative_eq!(
            (first.im - second.im).abs(),
            std::f64::consts::PI,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );

        // Both nevertheless map to the same physical threshold.
        assert_complex_close(map.kx_squared_scalar(first), c(4.0, 0.0));

        assert_complex_close(map.kx_squared_scalar(second), c(4.0, 0.0));
    }

    #[test]
    fn physical_preimages_are_continuous_through_outer_light_line() {
        let map = lossless_map();

        let below = map.physical_preimages(4.0 - 1.0e-5).unwrap();

        let at = map.physical_preimages(4.0).unwrap();

        let above = map.physical_preimages(4.0 + 1.0e-5).unwrap();

        let below = below.map(Option::unwrap);
        let at = at.map(Option::unwrap);
        let above = above.map(Option::unwrap);

        // Both branches should approach the same branch point.
        for z in below {
            assert!(
                (z - at[0]).modulus() < 1.0e-2,
                "below-light-line preimage did not approach branch point: \
             z={z:?}, branch={:?}",
                at[0],
            );
        }

        for z in above {
            assert!(
                (z - at[0]).modulus() < 1.0e-2,
                "above-light-line preimage did not approach branch point: \
             z={z:?}, branch={:?}",
                at[0],
            );
        }
    }

    #[test]
    fn physical_preimages_reject_lossy_exteriors() {
        let map = lossy_map();

        assert_eq!(
            map.physical_preimages(3.0),
            Err(InPlaneSearchError::UnsupportedExterior),
        );
    }

    #[test]
    fn physical_preimages_reject_negative_kx() {
        let map = lossless_map();

        assert_eq!(
            map.physical_preimages(-1.0),
            Err(InPlaneSearchError::InvalidParallelWavenumber,),
        );
    }

    #[test]
    fn symmetric_exterior_has_one_infinite_preimage() {
        let map = symmetric_map();

        // k0 = 2, chi = 2.25 -> light line 3.
        let [first, second] = map.physical_preimages(4.0).unwrap();

        assert!(
            first.is_none() ^ second.is_none(),
            "expected exactly one finite preimage: \
         first={first:?}, second={second:?}",
        );

        let finite = first.or(second).unwrap();

        let q = map.kx_squared_scalar(finite);

        assert_complex_close(q, c(16.0, 0.0));
    }

    #[test]
    fn guided_preimages_are_symmetric_about_half_log_abs_r() {
        let map = lossless_map();

        let [left, right] = map.physical_preimages(5.0).unwrap();

        let left = left.unwrap();
        let right = right.unwrap();

        let r = jet_value(map.r());

        let centre = 0.5 * r.modulus().ln();

        assert_relative_eq!(
            0.5 * (left.re + right.re),
            centre,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    // =============================================================================
    // Canonical search-region construction
    // =============================================================================

    #[test]
    fn search_region_real_bounds_round_trip_to_max_kx() {
        let map = lossless_map();

        let max_kx = 6.0;

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: max_kx,
                imaginary_margin: 0.05,
            })
            .unwrap()
            .into_inner();

        let (re_min, re_max) = region.real_bounds();

        let q_left = map.kx_squared_scalar(c(re_min, 0.0));

        let q_right = map.kx_squared_scalar(c(re_max, 0.0));

        assert_complex_close(q_left, c(max_kx * max_kx, 0.0));

        assert_complex_close(q_right, c(max_kx * max_kx, 0.0));
    }

    #[test]
    fn search_region_is_centred_on_half_log_abs_r() {
        let map = lossless_map();

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: 0.05,
            })
            .unwrap()
            .into_inner();

        let (re_min, re_max) = region.real_bounds();

        let r = jet_value(map.r());

        let expected_centre = 0.5 * r.modulus().ln();

        assert_relative_eq!(
            0.5 * (re_min + re_max),
            expected_centre,
            epsilon = ABS_TOL,
            max_relative = REL_TOL,
        );
    }

    #[test]
    fn search_region_uses_requested_imaginary_margin() {
        let map = lossless_map();

        let margin = 0.07;

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: margin,
            })
            .unwrap()
            .into_inner();

        let (im_min, im_max) = region.imaginary_bounds();

        assert_relative_eq!(
            im_min,
            -std::f64::consts::FRAC_PI_2 + margin,
            epsilon = ABS_TOL,
        );

        assert_relative_eq!(
            im_max,
            std::f64::consts::FRAC_PI_2 - margin,
            epsilon = ABS_TOL,
        );
    }

    #[test]
    fn search_region_rejects_lossy_exteriors() {
        let map = lossy_map();

        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: 0.05,
            },),
            Err(InPlaneSearchError::UnsupportedExterior),
        );
    }

    #[test]
    fn search_region_rejects_cutoff_below_outer_light_line() {
        let map = lossless_map();

        // Outer light line is kx = 4.
        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 3.9,
                imaginary_margin: 0.05,
            },),
            Err(InPlaneSearchError::CutoffBelowExteriorLightLine,),
        );
    }

    #[test]
    fn search_region_rejects_cutoff_at_outer_light_line() {
        let map = lossless_map();

        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 4.0,
                imaginary_margin: 0.05,
            },),
            Err(InPlaneSearchError::CutoffBelowExteriorLightLine,),
        );
    }

    #[test]
    fn search_region_rejects_zero_margin() {
        let map = lossless_map();

        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: 0.0,
            },),
            Err(InPlaneSearchError::InvalidImaginaryMargin,),
        );
    }

    #[test]
    fn search_region_rejects_margin_at_half_pi() {
        let map = lossless_map();

        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: std::f64::consts::FRAC_PI_2,
            },),
            Err(InPlaneSearchError::InvalidImaginaryMargin,),
        );
    }

    #[test]
    fn search_region_rejects_symmetric_exterior() {
        let map = symmetric_map();

        assert_eq!(
            map.search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 5.0,
                imaginary_margin: 0.05,
            },),
            Err(InPlaneSearchError::SymmetricExterior),
        );
    }

    #[test]
    fn full_trimmed_fundamental_strip_is_branch_point_free() {
        let map = lossless_map();

        let [left, right] = map.physical_preimages(6.0).unwrap();

        let left = left.unwrap();
        let right = right.unwrap();

        let region = ConformalRegion::new(
            left.re,
            right.re,
            -std::f64::consts::FRAC_PI_2 + 0.05,
            std::f64::consts::FRAC_PI_2 - 0.05,
        )
        .unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn constructed_search_region_passes_generic_validation() {
        let map = lossless_map();

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: 6.0,
                imaginary_margin: 0.05,
            })
            .unwrap()
            .into_inner();

        assert!(map.validate_region(&region).is_ok());
    }
}
