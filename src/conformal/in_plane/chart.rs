//! Analytic charts for the linear in-plane wavenumber.
//!
//! # Why this type exists
//!
//! [`ComplexInPlaneMap`] uniformises the branch structure associated with the
//! two exterior transverse wavevectors. For fixed vacuum wavenumber `k0`, it
//! defines
//!
//! ```text
//!              [                         r²       ]
//! kx²(z) = k0² [ s + exp(2z) + ---------------- ]
//!              [                     exp(2z)     ]
//! ```
//!
//! together with analytic exterior wavevectors
//!
//! ```text
//! κ_L(z) = i k0 [exp(z) - r exp(-z)]
//! κ_R(z) = i k0 [exp(z) + r exp(-z)].
//! ```
//!
//! Here
//!
//! ```text
//! χ_L = ε_L μ_L
//! χ_R = ε_R μ_R
//!
//! r = (χ_L - χ_R) / 4
//! s = (χ_L + χ_R) / 2.
//! ```
//!
//! The exterior `κ` functions are already single-valued analytic functions of
//! `z`. The remaining complication is the *linear* in-plane coordinate `kx`.
//! `lamina-core` consumes linear `kx`, while the natural conformal
//! uniformisation produces `kx²`.
//!
//! Simply evaluating
//!
//! ```text
//! kx = sqrt(kx²)
//! ```
//!
//! with the pointwise principal complex square root is not sufficient in
//! general. For lossy exterior media, `kx²(z)` may cross the principal
//! square-root cut even though it never vanishes. The mathematical square root
//! remains analytic on the search region, but the library principal square root
//! can change sheet.
//!
//! # Branch points
//!
//! Introduce
//!
//! ```text
//! t = exp(2z).
//! ```
//!
//! The zeros of `kx²` satisfy
//!
//! ```text
//! t² + s t + r² = 0.
//! ```
//!
//! If the roots are `t_1` and `t_2`, their finite preimages are
//!
//! ```text
//! z_{j,n}
//!     = 1/2 ln|t_j|
//!       + i [1/2 arg(t_j) + n π],
//!
//! n ∈ ℤ.
//! ```
//!
//! A rectangular conformal region is admissible when it contains or touches
//! none of these points. Because the rectangle is simply connected, this is
//! sufficient for a single-valued analytic square root of `kx²` to exist over
//! the whole region.
//!
//! # Selecting the analytic square root
//!
//! Let
//!
//! ```text
//! q(z) = kx²(z).
//! ```
//!
//! On an admissible region, `q(z)` is nonzero, so
//!
//! ```text
//! q'(z) / q(z)
//! ```
//!
//! is analytic. Choose a reference point `z_ref` and one of the two roots
//!
//! ```text
//! kx_ref² = q(z_ref).
//! ```
//!
//! Analytic continuation from that point is then
//!
//! ```text
//!                       [ 1   z
//! kx(z) = kx_ref exp    [ -   ∫ q'(ζ)/q(ζ) dζ ].
//!                       [ 2  z_ref
//! ```
//!
//! The integral is independent of path inside the admissible rectangle. This
//! implementation uses the straight segment from `z_ref` to `z`; rectangles
//! are convex, so that segment remains inside the validated region.
//!
//! This construction has several useful properties:
//!
//! - it is independent of the order in which an APM integrator evaluates
//!   points;
//! - it does not track mutable branch state;
//! - it remains valid when `q(z)` crosses the principal square-root cut;
//! - the choice between the two sheets is made once, at the reference point;
//! - lossy and magnetic exterior media require no special branch heuristic.
//!
//! # Interaction with jets
//!
//! [`ComplexInPlaneMap`] still evaluates its algebraic expressions using the
//! caller's [`ScalarAlgebra`]. In particular, the raw map obtains a local
//! square root of `kx²` through the algebra's normal `sqrt` implementation.
//!
//! The chart separately computes the analytically continued scalar `kx(z)`.
//! Since both this value and the raw algebraic square root square to the same
//! `kx²`, they differ only by a sign. The chart determines that sign and
//! multiplies the complete algebra value by `+1` or `-1`.
//!
//! This preserves every derivative carried by the algebra. Locally,
//!
//! ```text
//! d kx / dz = (d kx² / dz) / (2 kx),
//! ```
//!
//! so once the correct root value is selected, the derivative propagated
//! through the algebra is also on the correct analytic sheet.
//!
//! # Numerical integration
//!
//! The logarithmic derivative is integrated with adaptive Simpson quadrature.
//! Failure to reach the requested tolerance is reported rather than silently
//! accepting the final subdivision. This is why chart evaluation is fallible
//! even though the underlying conformal formulas themselves are algebraic.

use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FloatConst, FromPrimitive, One};

use super::{ComplexInPlaneMap, InPlaneRegionError};
use crate::conformal::{ConformalPoint, ConformalRegion, ValidateConformalRegion};

/// Choice of linear `kx` sheet at the chart reference point.
///
/// The two choices differ globally by a sign throughout a connected chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InPlaneSheet {
    /// Use the principal square root of `kx²` at the reference point.
    Principal,

    /// Use the negative of the principal square root at the reference point.
    Opposite,
}

/// Numerical settings used for analytic continuation of `kx`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InPlaneChartConfig<R> {
    /// Absolute tolerance for the logarithmic-derivative path integral.
    pub integration_tolerance: R,

    /// Maximum adaptive-Simpson subdivision depth.
    pub max_depth: u32,
}

impl<R> Default for InPlaneChartConfig<R>
where
    R: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            integration_tolerance: R::from_f64(1.0e-11).unwrap(),
            max_depth: 20,
        }
    }
}

/// Errors raised while constructing or evaluating an in-plane conformal chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InPlaneChartError {
    #[error("invalid conformal search region: {0}")]
    InvalidRegion(#[from] InPlaneRegionError),

    #[error("chart reference point lies outside the conformal search region")]
    ReferenceOutsideRegion,

    #[error("reference point lies at a zero of the linear in-plane coordinate")]
    ZeroReferenceCoordinate,

    #[error("requested conformal coordinate lies outside the chart region")]
    CoordinateOutsideRegion,

    #[error("analytic continuation integral failed to converge")]
    IntegrationDidNotConverge,

    #[error("analytic continuation produced a non-finite value")]
    NonFiniteContinuation,

    #[error("continued kx is inconsistent with the conformal kx² mapping")]
    InconsistentContinuation,
}

/// A single analytic chart for the linear in-plane wavenumber.
///
/// The underlying [`ComplexInPlaneMap`] supplies the conformal uniformisation
/// and analytic exterior wavevectors. This type additionally chooses one of
/// the two analytic square roots of `kx²` over a validated rectangular region.
///
/// Charts are immutable and contain no continuation history. Evaluating one
/// point does not affect any later evaluation, so they are suitable for
/// adaptive or parallel contour integration.
#[derive(Clone, Debug)]
pub struct InPlaneConformalChart<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField,
{
    map: ComplexInPlaneMap<A>,
    region: ConformalRegion<<A::Scalar as ComplexField>::RealField>,

    reference: A::Scalar,
    reference_kx: A::Scalar,

    config: InPlaneChartConfig<<A::Scalar as ComplexField>::RealField>,
}

impl<A, R> InPlaneConformalChart<A>
where
    A: ScalarAlgebra<Dimension = Ix0> + Clone,
    A::Scalar: ComplexScalar<RealField = R> + Copy,
    R: Float + FromPrimitive + FloatConst + Copy,
{
    /// Construct a chart using the centre of `region` as its reference point.
    ///
    /// `sheet` determines which of the two square roots of `kx²` is selected
    /// at that point. Analytic continuation fixes the choice everywhere else
    /// in the connected chart.
    pub fn new(
        map: ComplexInPlaneMap<A>,
        region: ConformalRegion<R>,
        sheet: InPlaneSheet,
        config: InPlaneChartConfig<R>,
    ) -> Result<Self, InPlaneChartError> {
        map.validate_region(&region)?;

        let (re_min, re_max) = region.real_bounds();
        let (im_min, im_max) = region.imaginary_bounds();

        let two = R::one() + R::one();

        let reference = A::Scalar::from_real((re_min + re_max) / two)
            + A::Scalar::i() * A::Scalar::from_real((im_min + im_max) / two);

        Self::with_reference(map, region, reference, sheet, config)
    }

    /// Construct a chart with an explicitly selected reference point.
    ///
    /// This is useful when the desired physical sheet is most naturally
    /// identified at a particular point in the conformal plane.
    pub fn with_reference(
        map: ComplexInPlaneMap<A>,
        region: ConformalRegion<R>,
        reference: A::Scalar,
        sheet: InPlaneSheet,
        config: InPlaneChartConfig<R>,
    ) -> Result<Self, InPlaneChartError> {
        map.validate_region(&region)?;

        if !contains(&region, reference) {
            return Err(InPlaneChartError::ReferenceOutsideRegion);
        }

        let q = map.kx_squared_scalar(reference);

        if q.modulus_squared() == R::zero() {
            return Err(InPlaneChartError::ZeroReferenceCoordinate);
        }

        let principal = q.sqrt();

        let reference_kx = match sheet {
            InPlaneSheet::Principal => principal,
            InPlaneSheet::Opposite => -principal,
        };

        Ok(Self {
            map,
            region,
            reference,
            reference_kx,
            config,
        })
    }

    pub fn map(&self) -> &ComplexInPlaneMap<A> {
        &self.map
    }

    pub fn region(&self) -> &ConformalRegion<R> {
        &self.region
    }

    pub fn reference(&self) -> A::Scalar {
        self.reference
    }

    pub fn reference_kx(&self) -> A::Scalar {
        self.reference_kx
    }

    pub fn config(&self) -> InPlaneChartConfig<R> {
        self.config
    }

    /// Evaluate the analytically continued scalar linear in-plane coordinate.
    ///
    /// This performs no jet operations. It is the scalar sheet-selection
    /// calculation used by [`Self::map_point`].
    pub fn kx(&self, z: A::Scalar) -> Result<A::Scalar, InPlaneChartError> {
        if !contains(&self.region, z) {
            return Err(InPlaneChartError::CoordinateOutsideRegion);
        }

        if z == self.reference {
            return Ok(self.reference_kx);
        }

        let delta = z - self.reference;

        // Parameterise the straight path by
        //
        //     ζ(u) = z_ref + u (z - z_ref),   u ∈ [0, 1].
        //
        // Therefore
        //
        //     ∫ q'(ζ)/q(ζ) dζ
        //
        // becomes
        //
        //     ∫_0^1 [q'(ζ(u))/q(ζ(u))] (z-z_ref) du.
        let integrand = |u: R| {
            let z_u = self.reference + delta * A::Scalar::from_real(u);

            self.map.kx_squared_log_derivative_scalar(z_u) * delta
        };

        let integral = adaptive_simpson(
            &integrand,
            R::zero(),
            R::one(),
            self.config.integration_tolerance,
            self.config.max_depth,
        )?;

        let two = R::one() + R::one();

        let kx = self.reference_kx * (integral / A::Scalar::from_real(two)).exp();

        if !kx.is_finite() {
            return Err(InPlaneChartError::NonFiniteContinuation);
        }

        // The logarithmic continuation should square to q(z). This check is
        // intentionally relative to the natural scale of q rather than an
        // exact equality because the path integral is numerical.
        let q = self.map.kx_squared_scalar(z);
        let residual = kx * kx - q;

        let scale = R::one().max(q.modulus());

        let verification_tolerance =
            self.config.integration_tolerance * R::from_f64(100.0).unwrap();

        if residual.modulus() > verification_tolerance * scale {
            return Err(InPlaneChartError::InconsistentContinuation);
        }

        Ok(kx)
    }

    /// Evaluate the full conformal point on this chart.
    ///
    /// The underlying map performs the complete `ScalarAlgebra` evaluation.
    /// The raw linear `kx` produced by its local square root is then multiplied
    /// by the sign required to agree with the chart's analytically continued
    /// scalar `kx`.
    ///
    /// Since the two candidate square roots differ only by sign, this operation
    /// preserves all jet derivatives while putting them onto the selected
    /// analytic sheet.
    pub fn map_point(&self, z: A) -> Result<ConformalPoint<A>, InPlaneChartError> {
        let z_scalar = z.value()[()];

        if !contains(&self.region, z_scalar) {
            return Err(InPlaneChartError::CoordinateOutsideRegion);
        }

        let analytic_kx = self.kx(z_scalar)?;

        let principal_kx = self.map.kx_squared_scalar(z_scalar).sqrt();

        let plus_distance = (analytic_kx - principal_kx).modulus_squared();
        let minus_distance = (analytic_kx + principal_kx).modulus_squared();

        let sign = if plus_distance <= minus_distance {
            A::Scalar::one()
        } else {
            -A::Scalar::one()
        };

        Ok(self.map.map_with_kx_sign(z, sign))
    }
}

// =============================================================================
// Scalar geometry
// =============================================================================

fn contains<C>(region: &ConformalRegion<C::RealField>, z: C) -> bool
where
    C: ComplexField + Copy,
    C::RealField: Float + Copy,
{
    let (re_min, re_max) = region.real_bounds();
    let (im_min, im_max) = region.imaginary_bounds();

    let re = z.real();
    let im = z.imaginary();

    re >= re_min && re <= re_max && im >= im_min && im <= im_max
}

// =============================================================================
// Adaptive complex Simpson integration
// =============================================================================

/// Integrate a complex-valued function of a real parameter using adaptive
/// Simpson quadrature.
///
/// The error estimator is based on the complex modulus of the difference
/// between one Simpson panel and its two-panel refinement.
fn adaptive_simpson<C, R, F>(
    f: &F,
    a: R,
    b: R,
    tolerance: R,
    max_depth: u32,
) -> Result<C, InPlaneChartError>
where
    C: ComplexField<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
    F: Fn(R) -> C,
{
    let two = R::one() + R::one();

    let fa = f(a);
    let fb = f(b);

    let midpoint = (a + b) / two;
    let fm = f(midpoint);

    if !fa.is_finite() || !fm.is_finite() || !fb.is_finite() {
        return Err(InPlaneChartError::NonFiniteContinuation);
    }

    let whole = simpson_panel(a, b, fa, fm, fb);

    adaptive_simpson_recursive(f, a, b, fa, fm, fb, whole, tolerance, max_depth)
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_recursive<C, R, F>(
    f: &F,
    a: R,
    b: R,
    fa: C,
    fm: C,
    fb: C,
    whole: C,
    tolerance: R,
    depth: u32,
) -> Result<C, InPlaneChartError>
where
    C: ComplexField<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
    F: Fn(R) -> C,
{
    let two = R::one() + R::one();
    let fifteen = R::from_f64(15.0).unwrap();

    let midpoint = (a + b) / two;
    let left_midpoint = (a + midpoint) / two;
    let right_midpoint = (midpoint + b) / two;

    let flm = f(left_midpoint);
    let frm = f(right_midpoint);

    if !flm.is_finite() || !frm.is_finite() {
        return Err(InPlaneChartError::NonFiniteContinuation);
    }

    let left = simpson_panel(a, midpoint, fa, flm, fm);

    let right = simpson_panel(midpoint, b, fm, frm, fb);

    let refined = left + right;
    let correction = refined - whole;

    if correction.modulus() <= fifteen * tolerance {
        // Richardson correction for Simpson's rule.
        return Ok(refined + correction / C::from_real(fifteen));
    }

    if depth == 0 {
        return Err(InPlaneChartError::IntegrationDidNotConverge);
    }

    let half_tolerance = tolerance / two;

    let left_result =
        adaptive_simpson_recursive(f, a, midpoint, fa, flm, fm, left, half_tolerance, depth - 1)?;

    let right_result = adaptive_simpson_recursive(
        f,
        midpoint,
        b,
        fm,
        frm,
        fb,
        right,
        half_tolerance,
        depth - 1,
    )?;

    Ok(left_result + right_result)
}

fn simpson_panel<C, R>(a: R, b: R, fa: C, fm: C, fb: C) -> C
where
    C: ComplexField<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
{
    let four = R::from_f64(4.0).unwrap();
    let six = R::from_f64(6.0).unwrap();

    let weighted = fa + fm * C::from_real(four) + fb;

    weighted * C::from_real((b - a) / six)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use lamina_core::{ScalarAlgebra, SeedJet};
    use nalgebra::ComplexField;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;
    use std::f64::consts::PI;

    // Replace with the actual exported first-order univariate jet type.
    type J = FirstJet<Complex64, Ix0>;

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

    // -------------------------------------------------------------------------
    // Jet helpers
    // -------------------------------------------------------------------------

    fn constant(value: Complex64) -> J {
        J::lift_constant(arr0(value))
    }

    fn variable(value: Complex64) -> J {
        J::variable(arr0(value))
    }

    // Adapt these two helpers to the exact coefficient accessors of J.
    fn jet_value(value: &J) -> Complex64 {
        value.value()[()]
    }

    fn jet_derivative(value: &J) -> Complex64 {
        value.derivative()[()]
    }

    // -------------------------------------------------------------------------
    // Ordinary dielectric fixture
    // -------------------------------------------------------------------------

    fn lossless_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn lossless_region() -> ConformalRegion<f64> {
        ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap()
    }

    fn lossless_chart(sheet: InPlaneSheet) -> InPlaneConformalChart<J> {
        InPlaneConformalChart::with_reference(
            lossless_map(),
            lossless_region(),
            c(0.0, 0.0),
            sheet,
            InPlaneChartConfig::default(),
        )
        .unwrap()
    }

    // -------------------------------------------------------------------------
    // Lossy dielectric fixture
    // -------------------------------------------------------------------------

    fn lossy_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
    }

    fn lossy_region() -> ConformalRegion<f64> {
        ConformalRegion::new(-0.5, 0.4, -0.8, 0.7).unwrap()
    }

    fn lossy_chart() -> InPlaneConformalChart<J> {
        InPlaneConformalChart::with_reference(
            lossy_map(),
            lossy_region(),
            c(0.0, 0.0),
            InPlaneSheet::Principal,
            InPlaneChartConfig::default(),
        )
        .unwrap()
    }

    // -------------------------------------------------------------------------
    // Principal-cut-crossing fixture
    // -------------------------------------------------------------------------
    //
    // This fixture is intentionally chosen so that q(z) = kx²(z) crosses the
    // negative real axis inside a branch-point-free conformal rectangle.
    //
    // Consequently:
    //
    //   * a holomorphic kx exists throughout the rectangle;
    //   * pointwise principal sqrt(q) changes sheet across the crossing;
    //   * InPlaneConformalChart must remain continuous.
    //
    // The constitutive values are deliberately strongly complex. They are a
    // numerical branch-stress fixture, not intended as a representative
    // passive dielectric.

    fn cut_crossing_map() -> ComplexInPlaneMap<J> {
        ComplexInPlaneMap::new(
            2.0,
            c(-7.037_440_270_253_994, -5.086_084_917_885_638),
            c(1.0, 0.0),
            c(-4.003_045_984_369_058, 5.328_848_852_382_237),
            c(1.0, 0.0),
        )
        .unwrap()
    }

    fn cut_crossing_region() -> ConformalRegion<f64> {
        ConformalRegion::new(0.03, 0.63, -0.40, 0.00).unwrap()
    }

    fn cut_crossing_chart() -> InPlaneConformalChart<J> {
        InPlaneConformalChart::with_reference(
            cut_crossing_map(),
            cut_crossing_region(),
            c(0.10, -0.20),
            InPlaneSheet::Principal,
            InPlaneChartConfig {
                integration_tolerance: 1.0e-12,
                max_depth: 24,
            },
        )
        .unwrap()
    }

    // q crosses the principal sqrt cut close to this point.
    const CUT_CROSSING_X: f64 = 0.329_942_900_121_112_44;
    const CUT_CROSSING_Y: f64 = -0.20;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn chart_accepts_valid_region() {
        let result = InPlaneConformalChart::with_reference(
            lossless_map(),
            lossless_region(),
            c(0.0, 0.0),
            InPlaneSheet::Principal,
            InPlaneChartConfig::default(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn chart_rejects_region_containing_coordinate_branch_point() {
        let map = lossless_map();

        // For chi_L = 4 and chi_R = 1 the finite branch points lie at
        // Im(z) = +/- pi/2 mod pi. This region deliberately reaches one.
        let region = ConformalRegion::new(-1.0, 1.0, 1.4, 1.7).unwrap();

        let result = InPlaneConformalChart::with_reference(
            map,
            region,
            c(0.0, 1.5),
            InPlaneSheet::Principal,
            InPlaneChartConfig::default(),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneChartError::InvalidRegion(InPlaneRegionError::CoordinateBranchPoint,),
        );
    }

    #[test]
    fn chart_rejects_reference_outside_region() {
        let result = InPlaneConformalChart::with_reference(
            lossless_map(),
            lossless_region(),
            c(2.0, 0.0),
            InPlaneSheet::Principal,
            InPlaneChartConfig::default(),
        );

        assert_eq!(
            result.unwrap_err(),
            InPlaneChartError::ReferenceOutsideRegion,
        );
    }

    #[test]
    fn automatically_selected_reference_is_region_centre() {
        let region = ConformalRegion::new(-1.2, 0.8, -0.6, 0.4).unwrap();

        let chart = InPlaneConformalChart::new(
            lossless_map(),
            region,
            InPlaneSheet::Principal,
            InPlaneChartConfig::default(),
        )
        .unwrap();

        assert_complex_close(chart.reference(), c(-0.2, -0.1));
    }

    // =========================================================================
    // Reference-point semantics
    // =========================================================================

    #[test]
    fn principal_sheet_uses_principal_root_at_reference() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        let z = chart.reference();
        let expected = chart.map().kx_squared_scalar(z).sqrt();

        assert_complex_close(chart.reference_kx(), expected);
    }

    #[test]
    fn opposite_sheet_uses_negative_principal_root_at_reference() {
        let chart = lossless_chart(InPlaneSheet::Opposite);

        let z = chart.reference();
        let expected = -chart.map().kx_squared_scalar(z).sqrt();

        assert_complex_close(chart.reference_kx(), expected);
    }

    #[test]
    fn evaluation_at_reference_returns_reference_kx_exactly() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        assert_eq!(chart.kx(chart.reference()).unwrap(), chart.reference_kx(),);
    }

    // =========================================================================
    // Defining analytic-square-root invariant
    // =========================================================================

    #[test]
    fn continued_kx_squares_to_mapped_kx_squared() {
        let chart = lossy_chart();

        for z in [
            c(-0.35, -0.55),
            c(-0.20, 0.20),
            c(0.00, 0.00),
            c(0.25, -0.30),
            c(0.30, 0.45),
        ] {
            let kx = chart.kx(z).unwrap();
            let q = chart.map().kx_squared_scalar(z);

            assert_complex_close_tol(kx * kx, q, 1.0e-9);
        }
    }

    #[test]
    fn opposite_and_principal_charts_differ_by_global_sign() {
        let principal = lossless_chart(InPlaneSheet::Principal);

        let opposite = lossless_chart(InPlaneSheet::Opposite);

        for z in [
            c(-0.7, -0.5),
            c(-0.3, 0.4),
            c(0.0, 0.0),
            c(0.4, -0.2),
            c(0.8, 0.6),
        ] {
            let a = principal.kx(z).unwrap();
            let b = opposite.kx(z).unwrap();

            assert_complex_close(a, -b);
        }
    }

    // =========================================================================
    // Agreement with principal sqrt where principal sqrt is analytic
    // =========================================================================

    #[test]
    fn lossless_chart_agrees_with_principal_sqrt_inside_standard_chart() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        for z in [
            c(-0.8, -0.7),
            c(-0.4, 0.3),
            c(0.0, 0.0),
            c(0.3, -0.4),
            c(0.7, 0.6),
        ] {
            let expected = chart.map().kx_squared_scalar(z).sqrt();

            assert_complex_close_tol(chart.kx(z).unwrap(), expected, 1.0e-9);
        }
    }

    // =========================================================================
    // Principal-square-root cut crossing
    // =========================================================================

    #[test]
    fn stress_fixture_is_a_valid_branch_point_free_chart() {
        let map = cut_crossing_map();
        let region = cut_crossing_region();

        assert!(map.validate_region(&region).is_ok());
    }

    #[test]
    fn stress_fixture_kx_squared_crosses_negative_real_axis() {
        let map = cut_crossing_map();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let q_left = map.kx_squared_scalar(left);
        let q_right = map.kx_squared_scalar(right);

        // Both points lie very close to the negative real axis.
        assert!(q_left.re < 0.0);
        assert!(q_right.re < 0.0);

        // They lie on opposite sides of the principal complex cut.
        assert!(
            q_left.im * q_right.im < 0.0,
            "expected opposite imaginary signs: \
             q_left={q_left:?}, q_right={q_right:?}",
        );
    }

    #[test]
    fn principal_sqrt_changes_sheet_across_stress_fixture_cut() {
        let map = cut_crossing_map();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let principal_left = map.kx_squared_scalar(left).sqrt();

        let principal_right = map.kx_squared_scalar(right).sqrt();

        // Near a negative-real-axis crossing the two principal roots approach
        // opposite values rather than one another.
        let same_sheet_distance = (principal_left - principal_right).modulus();

        let opposite_sheet_distance = (principal_left + principal_right).modulus();

        assert!(
            opposite_sheet_distance < same_sheet_distance,
            "principal roots did not exhibit expected sheet flip: \
             left={principal_left:?}, right={principal_right:?}",
        );
    }

    #[test]
    fn chart_remains_continuous_across_principal_sqrt_cut() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let kx_left = chart.kx(left).unwrap();
        let kx_right = chart.kx(right).unwrap();

        let same_sheet_distance = (kx_left - kx_right).modulus();

        let opposite_sheet_distance = (kx_left + kx_right).modulus();

        assert!(
            same_sheet_distance < opposite_sheet_distance,
            "chart changed sheet across principal sqrt cut: \
             left={kx_left:?}, right={kx_right:?}",
        );

        // The points are only 2e-5 apart, so the analytically continued
        // coordinate should itself be close.
        assert!(
            same_sheet_distance < 1.0e-2,
            "continued kx was unexpectedly discontinuous: \
             distance={same_sheet_distance}",
        );
    }

    #[test]
    fn chart_selects_opposite_raw_sqrt_sign_across_cut_when_required() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left_z = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right_z = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let analytic_left = chart.kx(left_z).unwrap();
        let analytic_right = chart.kx(right_z).unwrap();

        let principal_left = chart.map().kx_squared_scalar(left_z).sqrt();

        let principal_right = chart.map().kx_squared_scalar(right_z).sqrt();

        let left_matches_plus =
            (analytic_left - principal_left).modulus() < (analytic_left + principal_left).modulus();

        let right_matches_plus = (analytic_right - principal_right).modulus()
            < (analytic_right + principal_right).modulus();

        // A principal-root cut crossing must cause the relationship between
        // the analytic branch and principal root to change sign.
        assert_ne!(left_matches_plus, right_matches_plus);
    }

    // =========================================================================
    // Full algebra map
    // =========================================================================

    #[test]
    fn chart_map_point_uses_analytic_kx_value() {
        let chart = cut_crossing_chart();

        for z0 in [
            c(0.10, -0.20),
            c(0.25, -0.20),
            c(0.40, -0.20),
            c(0.55, -0.20),
        ] {
            let point = chart.map_point(constant(z0)).unwrap();

            let mapped_kx = jet_value(point.coordinates().kx());

            let expected = chart.kx(z0).unwrap();

            assert_complex_close_tol(mapped_kx, expected, 1.0e-9);
        }
    }

    #[test]
    fn constant_and_variable_map_points_select_same_kx_sheet() {
        let chart = cut_crossing_chart();

        for z0 in [
            c(0.20, -0.20),
            c(CUT_CROSSING_X - 1.0e-4, CUT_CROSSING_Y),
            c(CUT_CROSSING_X + 1.0e-4, CUT_CROSSING_Y),
            c(0.50, -0.20),
        ] {
            let constant_point = chart.map_point(constant(z0)).unwrap();

            let variable_point = chart.map_point(variable(z0)).unwrap();

            assert_complex_close(
                jet_value(constant_point.coordinates().kx()),
                jet_value(variable_point.coordinates().kx()),
            );
        }
    }

    // =========================================================================
    // Jet derivative
    // =========================================================================

    #[test]
    fn chart_kx_jet_derivative_matches_analytic_derivative() {
        let chart = cut_crossing_chart();

        for z0 in [
            c(0.15, -0.20),
            c(0.30, -0.20),
            c(0.36, -0.20),
            c(0.52, -0.20),
        ] {
            let point = chart.map_point(variable(z0)).unwrap();

            let kx = jet_value(point.coordinates().kx());

            let derivative = jet_derivative(point.coordinates().kx());

            let q = chart.map().kx_squared_scalar(z0);

            let log_derivative = chart.map().kx_squared_log_derivative_scalar(z0);

            let dq_dz = q * log_derivative;

            let expected = dq_dz / (kx + kx);

            assert_complex_close_tol(derivative, expected, 1.0e-9);
        }
    }

    #[test]
    fn chart_kx_derivative_is_continuous_across_principal_cut() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left = chart
            .map_point(variable(c(CUT_CROSSING_X - delta, CUT_CROSSING_Y)))
            .unwrap();

        let right = chart
            .map_point(variable(c(CUT_CROSSING_X + delta, CUT_CROSSING_Y)))
            .unwrap();

        let derivative_left = jet_derivative(left.coordinates().kx());

        let derivative_right = jet_derivative(right.coordinates().kx());

        assert!(
            (derivative_left - derivative_right).modulus() < 1.0e-2,
            "analytic derivative changed discontinuously across principal cut: \
             left={derivative_left:?}, right={derivative_right:?}",
        );
    }

    // =========================================================================
    // Exterior wavevectors are unaffected by kx charting
    // =========================================================================

    #[test]
    fn chart_preserves_analytic_exterior_wavevectors() {
        let chart = cut_crossing_chart();

        for z0 in [
            c(0.20, -0.20),
            c(0.32, -0.20),
            c(0.35, -0.20),
            c(0.50, -0.20),
        ] {
            let raw = chart.map().map(variable(z0));

            let mapped = chart.map_point(variable(z0)).unwrap();

            assert_complex_close(
                jet_value(raw.exterior_wavevectors().left()),
                jet_value(mapped.exterior_wavevectors().left()),
            );

            assert_complex_close(
                jet_value(raw.exterior_wavevectors().right()),
                jet_value(mapped.exterior_wavevectors().right()),
            );

            assert_complex_close(
                jet_derivative(raw.exterior_wavevectors().left()),
                jet_derivative(mapped.exterior_wavevectors().left()),
            );

            assert_complex_close(
                jet_derivative(raw.exterior_wavevectors().right()),
                jet_derivative(mapped.exterior_wavevectors().right()),
            );
        }
    }

    // =========================================================================
    // Domain checking
    // =========================================================================

    #[test]
    fn scalar_kx_rejects_point_outside_chart() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        assert_eq!(
            chart.kx(c(1.1, 0.0)).unwrap_err(),
            InPlaneChartError::CoordinateOutsideRegion,
        );
    }

    #[test]
    fn map_point_rejects_point_outside_chart() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        assert_eq!(
            chart.map_point(constant(c(0.0, 1.1))).unwrap_err(),
            InPlaneChartError::CoordinateOutsideRegion,
        );
    }

    #[test]
    fn chart_accepts_points_on_regular_region_boundary() {
        let chart = lossless_chart(InPlaneSheet::Principal);

        for z in [c(-1.0, 0.0), c(1.0, 0.0), c(0.0, -1.0), c(0.0, 1.0)] {
            assert!(
                chart.kx(z).is_ok(),
                "regular chart boundary point rejected: {z:?}",
            );
        }
    }

    // =========================================================================
    // Logarithmic continuation
    // =========================================================================

    #[test]
    fn continuation_is_reversible() {
        let chart = lossy_chart();

        let target = c(0.30, -0.45);
        let target_kx = chart.kx(target).unwrap();

        // Construct a second chart using the analytically continued target
        // point as its reference. We choose the local sheet according to which
        // principal-root sign matches target_kx.
        let q_target = chart.map().kx_squared_scalar(target);

        let principal = q_target.sqrt();

        let sheet = if (target_kx - principal).modulus_squared()
            <= (target_kx + principal).modulus_squared()
        {
            InPlaneSheet::Principal
        } else {
            InPlaneSheet::Opposite
        };

        let reverse = InPlaneConformalChart::with_reference(
            lossy_map(),
            lossy_region(),
            target,
            sheet,
            InPlaneChartConfig::default(),
        )
        .unwrap();

        assert_complex_close_tol(
            reverse.kx(chart.reference()).unwrap(),
            chart.reference_kx(),
            1.0e-9,
        );
    }

    #[test]
    fn charts_with_consistent_reference_sheets_agree_everywhere() {
        let first = lossy_chart();

        let second_reference = c(0.25, -0.30);
        let continued = first.kx(second_reference).unwrap();

        let principal = first.map().kx_squared_scalar(second_reference).sqrt();

        let sheet = if (continued - principal).modulus_squared()
            <= (continued + principal).modulus_squared()
        {
            InPlaneSheet::Principal
        } else {
            InPlaneSheet::Opposite
        };

        let second = InPlaneConformalChart::with_reference(
            lossy_map(),
            lossy_region(),
            second_reference,
            sheet,
            InPlaneChartConfig::default(),
        )
        .unwrap();

        for z in [
            c(-0.30, -0.50),
            c(-0.10, 0.30),
            c(0.10, -0.20),
            c(0.30, 0.40),
        ] {
            assert_complex_close_tol(first.kx(z).unwrap(), second.kx(z).unwrap(), 1.0e-9);
        }
    }

    // =========================================================================
    // Adaptive Simpson helper
    // =========================================================================

    #[test]
    fn adaptive_simpson_integrates_constant_complex_function() {
        let result = adaptive_simpson(&|_| c(2.0, -3.0), 0.0, 1.0, 1.0e-13, 10).unwrap();

        assert_complex_close(result, c(2.0, -3.0));
    }

    #[test]
    fn adaptive_simpson_integrates_linear_complex_function() {
        let result = adaptive_simpson(&|x| c(x, 2.0 * x), 0.0, 1.0, 1.0e-13, 10).unwrap();

        assert_complex_close(result, c(0.5, 1.0));
    }

    #[test]
    fn adaptive_simpson_integrates_complex_exponential() {
        // Integral_0^1 exp(i pi x) dx = 2 i / pi.
        let result = adaptive_simpson(&|x| c(0.0, PI * x).exp(), 0.0, 1.0, 1.0e-12, 20).unwrap();

        assert_complex_close_tol(result, c(0.0, 2.0 / PI), 1.0e-10);
    }

    #[test]
    fn adaptive_simpson_reports_non_convergence() {
        // Deliberately oscillatory function and zero refinement budget.
        let result = adaptive_simpson(&|x| c(0.0, 100.0 * x).exp(), 0.0, 1.0, 1.0e-15, 0);

        assert_eq!(
            result.unwrap_err(),
            InPlaneChartError::IntegrationDidNotConverge,
        );
    }
}
