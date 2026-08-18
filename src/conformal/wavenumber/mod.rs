//! Conformal uniformisation of the complex vacuum wavenumber.
//!
//! This module parameterises a fixed-in-plane-wavenumber planar optical
//! problem by a complex conformal coordinate `z`. The transformation removes
//! the square-root branch structure of the exterior transverse wavevectors and
//! provides analytic exterior radiation conditions suitable for
//! argument-principle mode searches.
//!
//! # Exterior-material restriction
//!
//! Unlike the fixed-frequency in-plane map, this map varies `k0` itself.
//! Consequently the exterior constitutive products
//!
//! ```text
//! chi_L = epsilon_L * mu_L
//! chi_R = epsilon_R * mu_R
//! ```
//!
//! must be constant over the search.
//!
//! Complex, lossy and magnetic constant exterior media are supported. A
//! dispersive exterior material cannot in general be represented by this
//! uniformisation, because `chi(k0)` would make the map coefficients themselves
//! depend on the conformal coordinate.
//!
//! Interior layers remain free to be dispersive: this restriction applies only
//! to the two semi-infinite exterior media.
//!
//! # Definitions
//!
//! For fixed in-plane wavenumber `kx`, define
//!
//! ```text
//! r = kx² (chi_L - chi_R) / (4 chi_L chi_R)
//!
//! s = kx² (chi_L + chi_R) / (2 chi_L chi_R).
//! ```
//!
//! With
//!
//! ```text
//! t = exp(2z),
//! ```
//!
//! the conformal map is
//!
//! ```text
//! k0²(z) = s - t - r²/t.
//! ```
//!
//! The exterior transverse wavevectors in the `lamina-core` convention are
//!
//! ```text
//! kappa_L(z)
//!     = i sqrt(chi_L) [exp(z) - r exp(-z)]
//!
//! kappa_R(z)
//!     = i sqrt(chi_R) [exp(z) + r exp(-z)].
//! ```
//!
//! They satisfy identically
//!
//! ```text
//! kappa_L² = chi_L k0² - kx²
//! kappa_R² = chi_R k0² - kx².
//! ```
//!
//! The square roots of the constant exterior constitutive products are chosen
//! once when the map is constructed. Their signs therefore form part of the
//! selected exterior radiation sheet.
//!
//! # Fundamental period
//!
//! `k0²(z)` is periodic under
//!
//! ```text
//! z -> z + i*pi.
//! ```
//!
//! The exterior wavevectors reverse sign under the same translation.
//! A convenient representative is therefore
//!
//! ```text
//! -pi/2 <= Im(z) <= pi/2,
//! ```
//!
//! with the upper and lower boundaries periodically identified.
//!
//! # Physical real-k0 geometry
//!
//! When both exterior constitutive products are real and positive, define
//!
//! ```text
//! u   = k0²
//! u_L = kx² / chi_L
//! u_R = kx² / chi_R.
//! ```
//!
//! The inverse map obeys
//!
//! ```text
//! t² - (s - u)t + r² = 0
//! ```
//!
//! with discriminant
//!
//! ```text
//! (u - u_L)(u - u_R).
//! ```
//!
//! ## Below both light lines
//!
//! For
//!
//! ```text
//! u < min(u_L, u_R),
//! ```
//!
//! both exterior channels are evanescent. Both inverse roots are positive and
//! the physical preimages lie on the real `z` axis.
//!
//! ## Between the light lines
//!
//! For
//!
//! ```text
//! min(u_L, u_R) < u < max(u_L, u_R),
//! ```
//!
//! one exterior channel is propagating and the other evanescent. The roots are
//! complex conjugates with
//!
//! ```text
//! |t| = |r|,
//! ```
//!
//! so the physical preimages lie on
//!
//! ```text
//! Re(z) = 1/2 ln|r|.
//! ```
//!
//! ## Above both light lines
//!
//! For
//!
//! ```text
//! u > max(u_L, u_R),
//! ```
//!
//! both exterior channels are propagating. The inverse roots are negative real
//! and the physical real-k0 locus lies on the periodically identified
//! boundaries
//!
//! ```text
//! Im(z) = +/- pi/2.
//! ```
//!
//! # Linear-k0 branch points
//!
//! `lamina-core` consumes the linear coordinate `k0`, not only `k0²`.
//! Zeros of `k0²` therefore remain branch points of the linear coordinate.
//!
//! They satisfy
//!
//! ```text
//! t² - s t + r² = 0.
//! ```
//!
//! For real positive exterior constitutive products these roots are positive,
//! so their conformal preimages lie on
//!
//! ```text
//! Im(z) = 0.
//! ```
//!
//! A single chart spanning the whole fundamental strip would therefore contain
//! linear-coordinate branch points. The canonical lossless search is instead
//! divided into two branch-point-free charts:
//!
//! ```text
//! lower:
//!     -pi/2 + delta < Im(z) < -delta
//!
//! upper:
//!      delta < Im(z) < pi/2 - delta.
//! ```
//!
//! Together these cover the two analytic continuations adjacent to the real
//! physical-frequency axis.
//!
//! # Complex exterior media
//!
//! The raw map remains valid for finite complex constant exterior constitutive
//! products. In that case the linear-k0 branch points move away from the real
//! axis and the simple upper/lower lossless geometry is no longer canonical.
//!
//! Such searches should use an explicit [`ConformalRegion`], validate it using
//! [`ValidateConformalRegion`], and construct a [`ConformalChart`] directly.
//!
//! Automatic continuation of a known lossless chart into complex exterior
//! media is deferred to the mode-continuation layer.

mod landmarks;
mod region;

pub use region::{
    WavenumberRegionError, WavenumberSearchError, WavenumberSearchExtent, WavenumberSearchRegions,
};

use lamina_core::{CanonicalCoordinates, ComplexScalar, ExteriorWavevectors, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::{Ix0, arr0};
use num_traits::{Float, FloatConst, One, Zero};

use super::{
    ConformalPoint, ConformalRegion, RawConformalMap, ValidateConformalRegion,
    branch::{
        branch_root_intersects_region, should_swap_preimages, stable_quadratic_roots, t_to_z,
    },
    chart::{ConformalChart, LinearCoordinateMap},
};

pub type WavenumberConformalChart<A> = ConformalChart<ComplexWavenumberMap<A>, A>;

#[derive(Clone, Debug, PartialEq)]
pub struct ComplexWavenumberMap<A>
where
    A: ScalarAlgebra,
{
    kx: A,

    r: A,
    s: A,

    // These fix the constant exterior-index branches used by
    // kappa_L(z) and kappa_R(z).
    sqrt_chi_left: A,
    sqrt_chi_right: A,

    branch_roots: [A::Scalar; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WavenumberMapError {
    #[error("in-plane wavenumber must be finite")]
    InvalidInPlaneWavenumber,

    #[error("exterior constitutive parameters must be finite and non-zero")]
    InvalidExteriorMaterial,
}

impl<A, R> ComplexWavenumberMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField<RealField = R> + Copy,
{
    pub fn new(
        kx: R,
        epsilon_left: A::Scalar,
        mu_left: A::Scalar,
        epsilon_right: A::Scalar,
        mu_right: A::Scalar,
    ) -> Result<Self, WavenumberMapError>
    where
        R: Float + Copy,
    {
        if !kx.is_finite() {
            return Err(WavenumberMapError::InvalidInPlaneWavenumber);
        }

        if !epsilon_left.is_finite()
            || !mu_left.is_finite()
            || !epsilon_right.is_finite()
            || !mu_right.is_finite()
        {
            return Err(WavenumberMapError::InvalidExteriorMaterial);
        }

        let chi_left = epsilon_left * mu_left;
        let chi_right = epsilon_right * mu_right;

        // The transformation contains 1 / (chi_L chi_R).
        if chi_left == <A::Scalar as Zero>::zero() || chi_right == <A::Scalar as Zero>::zero() {
            return Err(WavenumberMapError::InvalidExteriorMaterial);
        }

        let kx_complex = A::Scalar::from_real(kx);
        let kx_squared = kx_complex * kx_complex;

        let two = <A::Scalar as One>::one() + <A::Scalar as One>::one();
        let four = two + two;

        let product = chi_left * chi_right;

        let r = kx_squared * (chi_left - chi_right) / (four * product);

        let s = kx_squared * (chi_left + chi_right) / (two * product);

        let branch_roots = coordinate_branch_roots(s, r);

        Ok(Self {
            kx: A::lift_constant(arr0(kx_complex)),

            r: A::lift_constant(arr0(r)),
            s: A::lift_constant(arr0(s)),

            sqrt_chi_left: A::lift_constant(arr0(chi_left.sqrt())),
            sqrt_chi_right: A::lift_constant(arr0(chi_right.sqrt())),

            branch_roots,
        })
    }

    fn kx_scalar(&self) -> A::Scalar {
        self.kx().value()[()]
    }

    fn r_scalar(&self) -> A::Scalar {
        self.r().value()[()]
    }

    fn s_scalar(&self) -> A::Scalar {
        self.s().value()[()]
    }

    fn sqrt_chi_left_scalar(&self) -> A::Scalar {
        self.sqrt_chi_left.value()[()]
    }

    fn sqrt_chi_right_scalar(&self) -> A::Scalar {
        self.sqrt_chi_right.value()[()]
    }

    fn exterior_chi(&self) -> (A::Scalar, A::Scalar) {
        let left = self.sqrt_chi_left_scalar();
        let right = self.sqrt_chi_right_scalar();

        (left * left, right * right)
    }

    pub(crate) fn k0_squared_scalar(&self, z: A::Scalar) -> A::Scalar {
        let two = A::Scalar::one() + A::Scalar::one();

        let exp_2z = (z * two).exp();
        let exp_minus_2z = (-z * two).exp();

        let r_squared = self.r_scalar() * self.r_scalar();

        self.s_scalar() - exp_2z - r_squared * exp_minus_2z
    }

    pub(crate) fn k0_squared_log_derivative_scalar(&self, z: A::Scalar) -> A::Scalar {
        let two = A::Scalar::one() + A::Scalar::one();

        let exp_2z = (z * two).exp();
        let exp_minus_2z = (-z * two).exp();

        let r_squared = self.r_scalar() * self.r_scalar();

        let numerator = two * (r_squared * exp_minus_2z - exp_2z);

        let denominator = self.s_scalar() - exp_2z - r_squared * exp_minus_2z;

        numerator / denominator
    }

    fn real_positive_exterior_chi(&self) -> Result<(R, R), WavenumberSearchError>
    where
        R: Float + Copy,
    {
        let (left, right) = self.exterior_chi();

        if left.imaginary() != R::zero() || right.imaginary() != R::zero() {
            return Err(WavenumberSearchError::UnsupportedExterior);
        }

        let left = left.real();
        let right = right.real();

        if !left.is_finite() || !right.is_finite() || left <= R::zero() || right <= R::zero() {
            return Err(WavenumberSearchError::UnsupportedExterior);
        }

        Ok((left, right))
    }
}

impl<A> ComplexWavenumberMap<A>
where
    A: ScalarAlgebra,
{
    pub fn kx(&self) -> &A {
        &self.kx
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
}

impl<A> ComplexWavenumberMap<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
{
    pub(crate) fn map_with_k0_sign(&self, z: A, k0_sign: A::Scalar) -> ConformalPoint<A> {
        let one = A::Scalar::one();
        let i = <A::Scalar as ComplexScalar>::i();

        let exp_z = z.exp();
        let exp_minus_z = z.scale(-one).exp();

        let exp_2z = exp_z.multiply(&exp_z);

        let exp_minus_2z = exp_minus_z.multiply(&exp_minus_z);

        let r_squared = self.r.square();

        let k0_squared = self
            .s
            .subtract(&exp_2z)
            .subtract(&r_squared.multiply(&exp_minus_2z));

        let k0 = k0_squared.sqrt().scale(k0_sign);

        let r_exp_minus_z = self.r.multiply(&exp_minus_z);

        let kappa_left = exp_z
            .subtract(&r_exp_minus_z)
            .multiply(&self.sqrt_chi_left)
            .scale(i);

        let kappa_right = exp_z
            .add(&r_exp_minus_z)
            .multiply(&self.sqrt_chi_right)
            .scale(i);

        ConformalPoint::new(
            CanonicalCoordinates::new(k0, self.kx.clone()),
            ExteriorWavevectors::new(kappa_left, kappa_right),
        )
    }
}

pub(crate) fn coordinate_branch_roots<C>(s: C, r: C) -> [C; 2]
where
    C: ComplexField + Copy,
{
    // t² - s t + r² = 0
    stable_quadratic_roots(-s, r * r)
}

impl<A> RawConformalMap<A> for ComplexWavenumberMap<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
{
    fn map_raw(&self, z: A) -> ConformalPoint<A> {
        self.map_with_k0_sign(z, A::Scalar::one())
    }
}

impl<A> LinearCoordinateMap<A> for ComplexWavenumberMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0> + Clone,
    A::Scalar: ComplexScalar,
{
    type RegionError = WavenumberRegionError;

    fn coordinate_squared_scalar(&self, z: A::Scalar) -> A::Scalar {
        self.k0_squared_scalar(z)
    }

    fn coordinate_squared_log_derivative_scalar(&self, z: A::Scalar) -> A::Scalar {
        self.k0_squared_log_derivative_scalar(z)
    }

    fn map_with_coordinate_sign(&self, z: A, sign: A::Scalar) -> ConformalPoint<A> {
        self.map_with_k0_sign(z, sign)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::conformal::{
        ConformalRegion, RawConformalMap, ValidateConformalRegion, chart::LinearCoordinateMap,
    };
    use approx::assert_relative_eq;
    use lamina_core::{ModeJet1, ScalarAlgebra, SeedJet};
    use nalgebra::ComplexField;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;
    use std::f64::consts::PI;

    type J = ModeJet1<Complex64, Ix0>;

    const ABS_TOL: f64 = 1.0e-10;
    const REL_TOL: f64 = 1.0e-10;

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

    fn assert_complex_close_tol(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn constant(value: Complex64) -> J {
        J::lift_constant(arr0(value))
    }

    fn variable(value: Complex64) -> J {
        J::variable(arr0(value), 0).unwrap()
    }

    fn jet_value(value: &J) -> Complex64 {
        value.value()[()]
    }

    fn jet_derivative(value: &J) -> Complex64 {
        value.first()[()]
    }

    // -------------------------------------------------------------------------
    // Fixtures
    // -------------------------------------------------------------------------

    fn lossless_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn lossy_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
    }

    fn magnetic_lossy_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.2, c(3.2, 0.4), c(1.15, 0.08), c(1.4, 0.25), c(0.92, 0.06))
            .unwrap()
    }

    fn symmetric_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(1.5, c(2.25, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
            .unwrap()
    }

    fn zero_kx_map() -> ComplexWavenumberMap<J> {
        ComplexWavenumberMap::new(0.0, c(2.25, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn test_points() -> [Complex64; 5] {
        [
            c(0.20, 0.30),
            c(-0.35, 0.45),
            c(0.60, -0.40),
            c(-0.50, -0.25),
            c(0.15, 0.70),
        ]
    }

    // -------------------------------------------------------------------------
    // Physical fixture quantities
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

    fn r_from_chi(kx: f64, left: Complex64, right: Complex64) -> Complex64 {
        let kx2 = kx * kx;

        kx2 * (left - right) / (4.0 * left * right)
    }

    fn s_from_chi(kx: f64, left: Complex64, right: Complex64) -> Complex64 {
        let kx2 = kx * kx;

        kx2 * (left + right) / (2.0 * left * right)
    }

    fn expected_k0_squared(r: Complex64, s: Complex64, z: Complex64) -> Complex64 {
        s - (2.0 * z).exp() - r * r * (-2.0 * z).exp()
    }

    fn expected_kappa_left(chi_left: Complex64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * chi_left.sqrt() * (z.exp() - r * (-z).exp())
    }

    fn expected_kappa_right(chi_right: Complex64, r: Complex64, z: Complex64) -> Complex64 {
        c(0.0, 1.0) * chi_right.sqrt() * (z.exp() + r * (-z).exp())
    }

    fn expected_k0_derivative(r: Complex64, z: Complex64, k0: Complex64) -> Complex64 {
        (r * r * (-2.0 * z).exp() - (2.0 * z).exp()) / k0
    }

    fn expected_kappa_left_derivative(
        chi_left: Complex64,
        r: Complex64,
        z: Complex64,
    ) -> Complex64 {
        c(0.0, 1.0) * chi_left.sqrt() * (z.exp() + r * (-z).exp())
    }

    fn expected_kappa_right_derivative(
        chi_right: Complex64,
        r: Complex64,
        z: Complex64,
    ) -> Complex64 {
        c(0.0, 1.0) * chi_right.sqrt() * (z.exp() - r * (-z).exp())
    }

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
    fn rejects_non_finite_in_plane_wavenumber() {
        let result = ComplexWavenumberMap::<J>::new(
            f64::NAN,
            c(4.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
        );

        assert_eq!(
            result.unwrap_err(),
            WavenumberMapError::InvalidInPlaneWavenumber,
        );
    }

    #[test]
    fn accepts_zero_in_plane_wavenumber() {
        assert!(
            ComplexWavenumberMap::<J>::new(
                0.0,
                c(4.0, 0.0),
                c(1.0, 0.0),
                c(1.0, 0.0),
                c(1.0, 0.0),
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_zero_left_constitutive_product() {
        let result =
            ComplexWavenumberMap::<J>::new(1.0, c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0));

        assert_eq!(
            result.unwrap_err(),
            WavenumberMapError::InvalidExteriorMaterial,
        );
    }

    #[test]
    fn rejects_zero_right_constitutive_product() {
        let result =
            ComplexWavenumberMap::<J>::new(1.0, c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0));

        assert_eq!(
            result.unwrap_err(),
            WavenumberMapError::InvalidExteriorMaterial,
        );
    }

    #[test]
    fn rejects_non_finite_exterior_material() {
        let result = ComplexWavenumberMap::<J>::new(
            1.0,
            c(f64::NAN, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
        );

        assert_eq!(
            result.unwrap_err(),
            WavenumberMapError::InvalidExteriorMaterial,
        );
    }

    #[test]
    fn fixed_parameters_are_derivative_free() {
        let map = magnetic_lossy_map();

        assert_complex_close(jet_derivative(map.kx()), c(0.0, 0.0));

        assert_complex_close(jet_derivative(map.r()), c(0.0, 0.0));

        assert_complex_close(jet_derivative(map.s()), c(0.0, 0.0));
    }

    // =========================================================================
    // Raw mapping
    // =========================================================================

    #[test]
    fn mapped_k0_satisfies_uniformisation() {
        let map = lossless_map();

        let r = r_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        for z in test_points() {
            let point = map.map_raw(constant(z));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());

            assert_complex_close(k0 * k0, expected_k0_squared(r, s, z));
        }
    }

    #[test]
    fn mapped_k0_satisfies_uniformisation_for_lossy_media() {
        let map = lossy_map();

        let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        for z in test_points() {
            let point = map.map_raw(constant(z));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());

            assert_complex_close(k0 * k0, expected_k0_squared(r, s, z));
        }
    }

    // =========================================================================
    // Exterior-wavevector identities
    // =========================================================================

    #[test]
    fn exterior_wavevectors_satisfy_core_dispersion_relation() {
        let map = lossless_map();

        for z in test_points() {
            let point = map.map_raw(constant(z));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());

            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            let left = jet_value(point.exterior_wavevectors().left());

            let right = jet_value(point.exterior_wavevectors().right());

            assert_complex_close(left * left, lossless_chi_left() * k0 * k0 - kx * kx);

            assert_complex_close(right * right, lossless_chi_right() * k0 * k0 - kx * kx);
        }
    }

    #[test]
    fn lossy_exterior_wavevectors_satisfy_core_dispersion_relation() {
        let map = lossy_map();

        for z in test_points() {
            let point = map.map_raw(constant(z));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());

            let kx = jet_value(point.coordinates().parallel_angular_wavenumber());

            let left = jet_value(point.exterior_wavevectors().left());

            let right = jet_value(point.exterior_wavevectors().right());

            assert_complex_close(left * left, lossy_chi_left() * k0 * k0 - kx * kx);

            assert_complex_close(right * right, lossy_chi_right() * k0 * k0 - kx * kx);
        }
    }

    #[test]
    fn magnetic_lossy_exterior_wavevectors_satisfy_core_dispersion_relation() {
        let map = magnetic_lossy_map();

        for z in test_points() {
            let point = map.map_raw(constant(z));

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

        let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        for z in test_points() {
            let point = map.map_raw(constant(z));

            assert_complex_close(
                jet_value(point.exterior_wavevectors().left()),
                expected_kappa_left(lossy_chi_left(), r, z),
            );

            assert_complex_close(
                jet_value(point.exterior_wavevectors().right()),
                expected_kappa_right(lossy_chi_right(), r, z),
            );
        }
    }

    // =========================================================================
    // Jet derivatives
    // =========================================================================

    #[test]
    fn fixed_kx_has_zero_conformal_derivative() {
        let map = lossless_map();

        let point = map.map_raw(variable(c(0.3, 0.4)));

        assert_complex_close(
            jet_derivative(point.coordinates().parallel_angular_wavenumber()),
            c(0.0, 0.0),
        );
    }

    #[test]
    fn k0_derivative_matches_closed_form() {
        let map = lossless_map();

        let r = r_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        for z in [c(0.25, 0.35), c(-0.40, 0.25), c(0.20, -0.45)] {
            let point = map.map_raw(variable(z));

            let k0 = jet_value(point.coordinates().vacuum_angular_wavenumber());

            let derivative = jet_derivative(point.coordinates().vacuum_angular_wavenumber());

            assert_complex_close_tol(derivative, expected_k0_derivative(r, z, k0), 1.0e-9);
        }
    }

    #[test]
    fn left_kappa_derivative_matches_closed_form() {
        let map = lossy_map();

        let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        for z in [c(0.25, 0.35), c(-0.40, 0.25), c(0.20, -0.45)] {
            let point = map.map_raw(variable(z));

            assert_complex_close_tol(
                jet_derivative(point.exterior_wavevectors().left()),
                expected_kappa_left_derivative(lossy_chi_left(), r, z),
                1.0e-9,
            );
        }
    }

    #[test]
    fn right_kappa_derivative_matches_closed_form() {
        let map = lossy_map();

        let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        for z in [c(0.25, 0.35), c(-0.40, 0.25), c(0.20, -0.45)] {
            let point = map.map_raw(variable(z));

            assert_complex_close_tol(
                jet_derivative(point.exterior_wavevectors().right()),
                expected_kappa_right_derivative(lossy_chi_right(), r, z),
                1.0e-9,
            );
        }
    }

    // =========================================================================
    // LinearCoordinateMap hooks
    // =========================================================================

    #[test]
    fn generic_squared_coordinate_hook_matches_raw_map() {
        let map = lossy_map();

        for z in test_points() {
            let raw = map.map_raw(constant(z));

            let k0 = jet_value(raw.coordinates().vacuum_angular_wavenumber());

            assert_complex_close(map.coordinate_squared_scalar(z), k0 * k0);
        }
    }

    #[test]
    fn squared_coordinate_log_derivative_matches_direct_derivative() {
        let map = lossy_map();

        for z in test_points() {
            let q = map.coordinate_squared_scalar(z);

            let log_derivative = map.coordinate_squared_log_derivative_scalar(z);

            let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

            let expected_q_prime = 2.0 * (r * r * (-2.0 * z).exp() - (2.0 * z).exp());

            assert_complex_close_tol(q * log_derivative, expected_q_prime, 1.0e-9);
        }
    }

    #[test]
    fn coordinate_sign_changes_only_linear_k0() {
        let map = lossy_map();
        let z = variable(c(0.30, -0.20));

        let plus = map.map_with_coordinate_sign(z.clone(), c(1.0, 0.0));

        let minus = map.map_with_coordinate_sign(z, c(-1.0, 0.0));

        let plus_k0 = plus.coordinates().vacuum_angular_wavenumber();

        let minus_k0 = minus.coordinates().vacuum_angular_wavenumber();

        assert_complex_close(jet_value(minus_k0), -jet_value(plus_k0));

        assert_complex_close(jet_derivative(minus_k0), -jet_derivative(plus_k0));

        // Fixed kx is unchanged.
        assert_complex_close(
            jet_value(minus.coordinates().parallel_angular_wavenumber()),
            jet_value(plus.coordinates().parallel_angular_wavenumber()),
        );

        // Exterior analytic branches are unchanged.
        assert_complex_close(
            jet_value(minus.exterior_wavevectors().left()),
            jet_value(plus.exterior_wavevectors().left()),
        );

        assert_complex_close(
            jet_value(minus.exterior_wavevectors().right()),
            jet_value(plus.exterior_wavevectors().right()),
        );
    }

    // =========================================================================
    // Periodicity and sheet topology
    // =========================================================================

    #[test]
    fn squared_coordinate_is_periodic_under_i_pi_shift() {
        let map = lossy_map();

        let z = c(0.30, 0.20);

        assert_complex_close(
            map.coordinate_squared_scalar(z + c(0.0, PI)),
            map.coordinate_squared_scalar(z),
        );
    }

    #[test]
    fn exterior_wavevectors_reverse_sign_under_i_pi_shift() {
        let map = lossy_map();

        let z = c(0.30, 0.20);

        let first = map.map_raw(constant(z));

        let second = map.map_raw(constant(z + c(0.0, PI)));

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
    fn branch_roots_satisfy_wavenumber_quadratic() {
        let left = lossless_chi_left();
        let right = lossless_chi_right();

        let r = r_from_chi(1.5, left, right);
        let s = s_from_chi(1.5, left, right);

        let roots = coordinate_branch_roots(s, r);

        for root in roots {
            let residual = root * root - s * root + r * r;

            assert!(
                residual.modulus() < 1.0e-12,
                "root={root:?}, residual={residual:?}",
            );
        }
    }

    #[test]
    fn branch_root_product_is_r_squared() {
        for (kx, left, right) in [
            (1.5, lossless_chi_left(), lossless_chi_right()),
            (1.5, lossy_chi_left(), lossy_chi_right()),
            (1.2, magnetic_lossy_chi_left(), magnetic_lossy_chi_right()),
        ] {
            let r = r_from_chi(kx, left, right);
            let s = s_from_chi(kx, left, right);

            let [first, second] = coordinate_branch_roots(s, r);

            assert_complex_close(first * second, r * r);
        }
    }

    #[test]
    fn lossless_branch_roots_are_positive_real() {
        let r = r_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        for root in coordinate_branch_roots(s, r) {
            assert!(root.re > 0.0);

            assert_relative_eq!(root.im, 0.0, epsilon = ABS_TOL,);
        }
    }

    #[test]
    fn lossless_branch_points_lie_on_real_z_axis_mod_pi() {
        let r = r_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        let s = s_from_chi(1.5, lossless_chi_left(), lossless_chi_right());

        for root in coordinate_branch_roots(s, r) {
            let point = finite_branch_point(root, 0).unwrap();

            assert_relative_eq!(point.im, 0.0, epsilon = ABS_TOL,);
        }
    }

    #[test]
    fn lossy_media_move_branch_points_off_real_axis() {
        let r = r_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        let s = s_from_chi(1.5, lossy_chi_left(), lossy_chi_right());

        let roots = coordinate_branch_roots(s, r);

        assert!(
            roots
                .into_iter()
                .any(|root| { finite_branch_point(root, 0).unwrap().im.abs() > 1.0e-4 })
        );
    }

    #[test]
    fn symmetric_exterior_has_zero_t_root() {
        let left = c(2.25, 0.0);
        let right = c(2.25, 0.0);

        let r = r_from_chi(1.5, left, right);
        let s = s_from_chi(1.5, left, right);

        let [first, second] = coordinate_branch_roots(s, r);

        assert!(first.modulus_squared() == 0.0 || second.modulus_squared() == 0.0);
    }

    #[test]
    fn zero_kx_has_only_zero_t_roots() {
        let map = zero_kx_map();

        let roots = map.branch_roots();

        assert_eq!(roots, [c(0.0, 0.0), c(0.0, 0.0)],);
    }

    // =========================================================================
    // Region validation
    // =========================================================================

    #[test]
    fn lossless_upper_half_region_is_valid() {
        let map = lossless_map();

        let region = ConformalRegion::new(-2.0, 2.0, 0.05, 1.0).unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn lossless_lower_half_region_is_valid() {
        let map = lossless_map();

        let region = ConformalRegion::new(-2.0, 2.0, -1.0, -0.05).unwrap();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn region_containing_branch_point_is_invalid() {
        let map = lossy_map();

        let [root, _] = map.branch_roots();

        let branch = finite_branch_point(root, 0).unwrap();

        let region = region_around(branch, 0.1);

        assert_eq!(
            map.validate_region(&region),
            Err(WavenumberRegionError::CoordinateBranchPoint,),
        );
    }

    #[test]
    fn branch_point_on_region_boundary_is_invalid() {
        let map = lossy_map();

        let [root, _] = map.branch_roots();

        let branch = finite_branch_point(root, 0).unwrap();

        let region =
            ConformalRegion::new(branch.re, branch.re + 0.2, branch.im - 0.1, branch.im + 0.1)
                .unwrap();

        assert!(map.validate_region(&region).is_err());
    }

    #[test]
    fn validator_detects_periodic_branch_point_copy() {
        let map = lossy_map();

        let [root, _] = map.branch_roots();

        let branch = finite_branch_point(root, 3).unwrap();

        let region = region_around(branch, 0.1);

        assert_eq!(
            map.validate_region(&region),
            Err(WavenumberRegionError::CoordinateBranchPoint,),
        );
    }

    #[test]
    fn zero_t_root_is_not_a_finite_branch_point() {
        let map = symmetric_map();

        let region = ConformalRegion::new(-100.0, -90.0, 0.2, 0.8).unwrap();

        assert!(map.validate_region(&region).is_ok());
    }
}
