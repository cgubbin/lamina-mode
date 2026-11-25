//! Analytic charts for linear conformal coordinates.
//!
//! Raw conformal maps naturally parameterise a *squared* physical coordinate,
//! such as `kx²` or `k0²`. The associated exterior transverse wavevectors can
//! be written directly as analytic functions of the conformal coordinate `z`,
//! but `lamina-core` consumes the corresponding *linear* physical coordinate.
//!
//! A pointwise complex square root is insufficient in general. In particular,
//! for lossy exterior media the squared coordinate may cross the branch cut of
//! the principal square root without vanishing. The physical coordinate still
//! possesses an analytic square root over a simply connected, zero-free search
//! region, but the pointwise principal square root can change sheet.
//!
//! [`ConformalChart`] selects one analytic square-root sheet over a validated
//! rectangular region.
//!
//! # Analytic continuation
//!
//! Let
//!
//! ```text
//! q(z) = coordinate(z)².
//! ```
//!
//! If `q` has no zeros in a simply connected region, then `q'/q` is analytic
//! there. Given a reference point `z_ref` and one chosen square root
//! `w_ref² = q(z_ref)`, the unique analytic continuation is
//!
//! ```text
//!                             z
//! w(z) = w_ref exp[ 1/2 ∫          q'(ζ) / q(ζ) dζ ].
//!                           z_ref
//! ```
//!
//! The integral is path-independent inside the chart. Since
//! [`ConformalRegion`] is rectangular and therefore convex, this
//! implementation integrates along the straight segment from `z_ref` to `z`.
//!
//! # Interaction with `ScalarAlgebra`
//!
//! The raw map still evaluates its squared coordinate with the normal algebra
//! operations, including its local `sqrt`. The chart independently evaluates
//! the analytically continued scalar coordinate. The raw and continued roots
//! must differ only by a sign, so the chart multiplies the *entire algebra
//! value* by `+1` or `-1`.
//!
//! This is important for jets: changing the sign of the complete jet places
//! both the value and all carried derivatives on the selected analytic sheet.
//!
//! The chart is immutable and contains no continuation history. Evaluation is
//! therefore independent of call order and suitable for adaptive or parallel
//! contour integration.

use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FromPrimitive, One};

use super::{ConformalPoint, ConformalRegion, ValidateConformalRegion};

/// Internal interface required by [`ConformalChart`].
///
/// Implementations provide the scalar squared-coordinate map and its
/// logarithmic derivative, together with a way to evaluate the complete raw
/// conformal problem while selecting either sign of the linear coordinate.
///
/// This trait is intentionally narrower than the public physical map types.
/// It represents exactly the operations common to complex-`kx` and
/// complex-`k0` analytic continuation.
pub(crate) trait LinearCoordinateMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
{
    /// Evaluate the squared varying physical coordinate `q(z)`.
    fn coordinate_squared_scalar(&self, z: A::Scalar) -> A::Scalar;

    /// Evaluate `q'(z) / q(z)`.
    fn coordinate_squared_log_derivative_scalar(&self, z: A::Scalar) -> A::Scalar;

    /// Evaluate the complete conformal map, applying `sign` to the linear
    /// varying coordinate.
    ///
    /// `sign` is expected to be exactly `+1` or `-1`.
    fn map_with_coordinate_sign(&self, z: A, sign: A::Scalar) -> ConformalPoint<A>;
}

/// Choice of analytic sheet at a chart's reference point.
///
/// The two sheets differ by a global sign throughout a connected chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateSheet {
    /// Select the principal square root at the reference point.
    Principal,

    /// Select its negative.
    Opposite,
}

/// Numerical settings for analytic continuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalChartConfig<R> {
    /// Absolute tolerance for integration of `q'/q`.
    pub integration_tolerance: R,

    /// Maximum recursive subdivision depth of adaptive Simpson quadrature.
    pub max_depth: u32,
}

impl<R> Default for ConformalChartConfig<R>
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

/// Errors raised while constructing or evaluating a conformal chart.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ConformalChartError<E> {
    #[error("invalid conformal search region: {0}")]
    InvalidRegion(#[source] E),

    #[error("chart reference point lies outside the conformal search region")]
    ReferenceOutsideRegion,

    #[error("reference point lies at a zero of the linear coordinate")]
    ZeroReferenceCoordinate,

    #[error("requested conformal coordinate lies outside the chart region")]
    CoordinateOutsideRegion,

    #[error("cannot construct a finite surrounding evaluation margin")]
    InvalidEvaluationMargin,

    #[error("analytic continuation integral failed to converge")]
    IntegrationDidNotConverge,

    #[error("analytic continuation produced a non-finite value")]
    NonFiniteContinuation,

    #[error("continued coordinate is inconsistent with the squared-coordinate map")]
    InconsistentContinuation,

    #[error(
        "a stack closed by walls on both sides has no exterior branch: search a direct β domain"
    )]
    ClosedStack,
}

/// One analytic linear-coordinate chart over a validated conformal region.
///
/// `M` may be, for example, a complex-in-plane-wavenumber map or a
/// complex-vacuum-wavenumber map.
///
/// The chart owns its map because the sheet and validity region are part of
/// the compiled modal problem.
#[derive(Clone, Debug)]
pub struct ConformalChart<M, A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField,
{
    map: M,
    region: ConformalRegion<<A::Scalar as ComplexField>::RealField>,
    reference: A::Scalar,
    reference_coordinate: A::Scalar,
    config: ConformalChartConfig<<A::Scalar as ComplexField>::RealField>,
}

impl<M, A, R> ConformalChart<M, A>
where
    A: ScalarAlgebra<Dimension = Ix0> + Clone,
    A::Scalar: ComplexScalar + ComplexField<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
    M: LinearCoordinateMap<A> + ValidateConformalRegion<R>,
{
    /// Construct a chart using the centre of `region` as its reference point.
    pub fn new(
        map: M,
        region: ConformalRegion<R>,
        sheet: CoordinateSheet,
        config: ConformalChartConfig<R>,
    ) -> Result<Self, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>> {
        let (re_min, re_max) = region.real_bounds();
        let (im_min, im_max) = region.imaginary_bounds();

        let two = R::one() + R::one();

        let reference = A::Scalar::from_real((re_min + re_max) / two)
            + A::Scalar::i() * A::Scalar::from_real((im_min + im_max) / two);

        Self::with_reference(map, region, reference, sheet, config)
    }

    /// Construct a chart with an explicitly selected reference point.
    pub fn with_reference(
        map: M,
        region: ConformalRegion<R>,
        reference: A::Scalar,
        sheet: CoordinateSheet,
        config: ConformalChartConfig<R>,
    ) -> Result<Self, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>> {
        map.validate_region(&region)
            .map_err(ConformalChartError::InvalidRegion)?;

        if !contains(&region, reference) {
            return Err(ConformalChartError::ReferenceOutsideRegion);
        }

        let q = map.coordinate_squared_scalar(reference);

        if q.modulus_squared() == R::zero() {
            return Err(ConformalChartError::ZeroReferenceCoordinate);
        }

        let principal = q.sqrt();

        let reference_coordinate = match sheet {
            CoordinateSheet::Principal => principal,
            CoordinateSheet::Opposite => -principal,
        };

        Ok(Self {
            map,
            region,
            reference,
            reference_coordinate,
            config,
        })
    }

    /// Construct a chart from an explicitly supplied linear coordinate at the
    /// reference point.
    ///
    /// This is particularly useful when subdividing an existing chart: the
    /// parent chart can supply the already-continued coordinate and therefore
    /// preserve the physical sheet without referring back to a local
    /// principal-root convention.
    pub fn with_reference_coordinate(
        map: M,
        region: ConformalRegion<R>,
        reference: A::Scalar,
        reference_coordinate: A::Scalar,
        config: ConformalChartConfig<R>,
    ) -> Result<Self, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>> {
        map.validate_region(&region)
            .map_err(ConformalChartError::InvalidRegion)?;

        if !contains(&region, reference) {
            return Err(ConformalChartError::ReferenceOutsideRegion);
        }

        let q = map.coordinate_squared_scalar(reference);

        if q.modulus_squared() == R::zero() {
            return Err(ConformalChartError::ZeroReferenceCoordinate);
        }

        let residual = reference_coordinate * reference_coordinate - q;

        let scale = R::one().max(q.modulus());

        let tolerance = config.integration_tolerance * R::from_f64(100.0).unwrap();

        if residual.modulus() > tolerance * scale {
            return Err(ConformalChartError::InconsistentContinuation);
        }

        Ok(Self {
            map,
            region,
            reference,
            reference_coordinate,
            config,
        })
    }

    /// Extend only the evaluation chart for boundary roundoff. Search callers
    /// must retain their original rectangle; the reference coordinate fixes the
    /// analytic sheet and the enlarged region is validated before use.
    pub(crate) fn with_evaluation_margin(
        &self,
    ) -> Result<Self, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>>
    where
        M: Clone,
        R: num_traits::float::FloatCore,
    {
        let expand = |lo: R, hi: R| {
            let width = hi - lo;
            let scale = Float::max(
                R::one(),
                Float::max(Float::abs(lo), Float::max(Float::abs(hi), width)),
            );
            let margin = R::from_u32(64).unwrap() * <R as Float>::epsilon() * scale;
            let lower = lo - margin;
            let upper = hi + margin;
            if !(Float::is_finite(lower) && Float::is_finite(upper) && lower < lo && upper > hi) {
                return Err(ConformalChartError::InvalidEvaluationMargin);
            }
            Ok((lower, upper))
        };
        let (re_min, re_max) = expand(self.region.real_bounds().0, self.region.real_bounds().1)?;
        let (im_min, im_max) = expand(
            self.region.imaginary_bounds().0,
            self.region.imaginary_bounds().1,
        )?;
        let region = ConformalRegion::new(re_min, re_max, im_min, im_max)
            .map_err(|_| ConformalChartError::InvalidEvaluationMargin)?;
        Self::with_reference_coordinate(
            self.map.clone(),
            region,
            self.reference,
            self.reference_coordinate,
            self.config,
        )
    }

    pub fn raw_map(&self) -> &M {
        &self.map
    }

    pub fn region(&self) -> &ConformalRegion<R> {
        &self.region
    }

    pub fn reference(&self) -> A::Scalar {
        self.reference
    }

    pub fn reference_coordinate(&self) -> A::Scalar {
        self.reference_coordinate
    }

    pub fn config(&self) -> ConformalChartConfig<R> {
        self.config
    }

    /// Evaluate the analytically continued scalar linear coordinate.
    pub fn coordinate(
        &self,
        z: A::Scalar,
    ) -> Result<A::Scalar, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>> {
        if !contains(&self.region, z) {
            return Err(ConformalChartError::CoordinateOutsideRegion);
        }

        if z == self.reference {
            return Ok(self.reference_coordinate);
        }

        let delta = z - self.reference;

        let integrand = |u: R| {
            let z_u = self.reference + delta * A::Scalar::from_real(u);

            self.map.coordinate_squared_log_derivative_scalar(z_u) * delta
        };

        let integral = adaptive_simpson(
            &integrand,
            R::zero(),
            R::one(),
            self.config.integration_tolerance,
            self.config.max_depth,
        )
        .map_err(|err| match err {
            ContinuationError::NonFinite => ConformalChartError::NonFiniteContinuation,
            ContinuationError::DidNotConverge => ConformalChartError::IntegrationDidNotConverge,
        })?;

        let two = R::one() + R::one();

        let coordinate = self.reference_coordinate * (integral / A::Scalar::from_real(two)).exp();

        if !coordinate.is_finite() {
            return Err(ConformalChartError::NonFiniteContinuation);
        }

        let q = self.map.coordinate_squared_scalar(z);
        let residual = coordinate * coordinate - q;

        let scale = R::one().max(q.modulus());

        let verification_tolerance =
            self.config.integration_tolerance * R::from_f64(100.0).unwrap();

        if residual.modulus() > verification_tolerance * scale {
            return Err(ConformalChartError::InconsistentContinuation);
        }

        Ok(coordinate)
    }

    /// Evaluate the complete conformal problem on this analytic chart.
    pub fn map(
        &self,
        z: A,
    ) -> Result<ConformalPoint<A>, ConformalChartError<<M as ValidateConformalRegion<R>>::Error>>
    {
        let z_scalar = z.value()[()];

        if !contains(&self.region, z_scalar) {
            return Err(ConformalChartError::CoordinateOutsideRegion);
        }

        let analytic = self.coordinate(z_scalar)?;
        let local = self.map.coordinate_squared_scalar(z_scalar).sqrt();

        let plus_distance = (analytic - local).modulus_squared();

        let minus_distance = (analytic + local).modulus_squared();

        let sign = if plus_distance <= minus_distance {
            A::Scalar::one()
        } else {
            -A::Scalar::one()
        };

        Ok(self.map.map_with_coordinate_sign(z, sign))
    }
}

// =============================================================================
// Geometry
// =============================================================================

#[cfg(test)]
thread_local! {
    static REGION_MISSES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn contains<C>(region: &ConformalRegion<C::RealField>, z: C) -> bool
where
    C: ComplexField + Copy,
    C::RealField: Float + Copy,
{
    let (re_min, re_max) = region.real_bounds();
    let (im_min, im_max) = region.imaginary_bounds();

    let re = z.real();
    let im = z.imaginary();

    let inside = re >= re_min && re <= re_max && im >= im_min && im <= im_max;
    #[cfg(test)]
    if !inside {
        REGION_MISSES.with(|points| points.borrow_mut().push(format!("{z:?}")));
    }
    inside
}

// =============================================================================
// Complex adaptive Simpson quadrature
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContinuationError {
    DidNotConverge,
    NonFinite,
}

fn adaptive_simpson<C, R, F>(
    f: &F,
    a: R,
    b: R,
    tolerance: R,
    max_depth: u32,
) -> Result<C, ContinuationError>
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
        return Err(ContinuationError::NonFinite);
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
) -> Result<C, ContinuationError>
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
        return Err(ContinuationError::NonFinite);
    }

    let left = simpson_panel(a, midpoint, fa, flm, fm);

    let right = simpson_panel(midpoint, b, fm, frm, fb);

    let refined = left + right;
    let correction = refined - whole;

    if correction.modulus() <= fifteen * tolerance {
        return Ok(refined + correction / C::from_real(fifteen));
    }

    if depth == 0 {
        return Err(ContinuationError::DidNotConverge);
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

    use crate::conformal::{
        ConformalRegion, ValidateConformalRegion,
        in_plane::{ComplexInPlaneMap, region::InPlaneRegionError},
    };

    use approx::assert_relative_eq;
    use lamina_core::{ModeJet1, ScalarAlgebra, SeedJet};
    use nalgebra::ComplexField;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;
    use std::f64::consts::PI;

    type J = ModeJet1<Complex64, Ix0>;
    type Map = ComplexInPlaneMap<J>;
    type Chart = ConformalChart<Map, J>;
    type ChartError = ConformalChartError<InPlaneRegionError>;

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
        J::variable(arr0(value), 0).unwrap()
    }

    fn jet_value(value: &J) -> Complex64 {
        value.value()[()]
    }

    fn jet_derivative(value: &J) -> Complex64 {
        value.first()[()]
    }

    // -------------------------------------------------------------------------
    // Ordinary lossless fixture
    // -------------------------------------------------------------------------

    fn lossless_map() -> Map {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)).unwrap()
    }

    fn lossless_region() -> ConformalRegion<f64> {
        ConformalRegion::new(-1.0, 1.0, -1.0, 1.0).unwrap()
    }

    fn lossless_chart(sheet: CoordinateSheet) -> Chart {
        Chart::with_reference(
            lossless_map(),
            lossless_region(),
            c(0.0, 0.0),
            sheet,
            ConformalChartConfig::default(),
        )
        .unwrap()
    }

    // -------------------------------------------------------------------------
    // Lossy fixture
    // -------------------------------------------------------------------------

    fn lossy_map() -> Map {
        ComplexInPlaneMap::new(2.0, c(4.0, 0.8), c(1.0, 0.0), c(1.5, 0.35), c(1.0, 0.0)).unwrap()
    }

    fn lossy_region() -> ConformalRegion<f64> {
        ConformalRegion::new(-0.5, 0.4, -0.8, 0.7).unwrap()
    }

    fn lossy_chart() -> Chart {
        Chart::with_reference(
            lossy_map(),
            lossy_region(),
            c(0.0, 0.0),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap()
    }

    // -------------------------------------------------------------------------
    // Principal-sqrt-cut-crossing fixture
    // -------------------------------------------------------------------------
    //
    // q(z) = kx²(z) crosses the negative real axis inside this region, but
    // has no zero there. Thus:
    //
    //   * an analytic linear coordinate exists throughout the chart;
    //   * the pointwise principal sqrt changes local sheet;
    //   * ConformalChart must remain continuous.
    //
    // The material values are intentionally strongly complex. This fixture is
    // designed to stress branch continuation, not model a representative
    // passive optical material.

    fn cut_crossing_map() -> Map {
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

    fn cut_crossing_chart() -> Chart {
        Chart::with_reference(
            cut_crossing_map(),
            cut_crossing_region(),
            c(0.10, -0.20),
            CoordinateSheet::Principal,
            ConformalChartConfig {
                integration_tolerance: 1.0e-12,
                max_depth: 24,
            },
        )
        .unwrap()
    }

    const CUT_CROSSING_X: f64 = 0.329_942_900_121_112_44;

    const CUT_CROSSING_Y: f64 = -0.20;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn chart_accepts_valid_region() {
        let result = Chart::with_reference(
            lossless_map(),
            lossless_region(),
            c(0.0, 0.0),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn chart_rejects_invalid_map_region() {
        let region = ConformalRegion::new(-1.0, 1.0, 1.4, 1.7).unwrap();

        let result = Chart::with_reference(
            lossless_map(),
            region,
            c(0.0, 1.5),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        );

        assert_eq!(
            result.unwrap_err(),
            ChartError::InvalidRegion(InPlaneRegionError::CoordinateBranchPoint,),
        );
    }

    #[test]
    fn chart_rejects_reference_outside_region() {
        let result = Chart::with_reference(
            lossless_map(),
            lossless_region(),
            c(2.0, 0.0),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        );

        assert_eq!(result.unwrap_err(), ChartError::ReferenceOutsideRegion,);
    }

    #[test]
    fn automatically_selected_reference_is_region_centre() {
        let region = ConformalRegion::new(-1.2, 0.8, -0.6, 0.4).unwrap();

        let chart = Chart::new(
            lossless_map(),
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        assert_complex_close(chart.reference(), c(-0.2, -0.1));
    }

    // =========================================================================
    // Reference-sheet semantics
    // =========================================================================

    #[test]
    fn principal_sheet_uses_principal_root_at_reference() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        let z = chart.reference();

        let expected = chart.raw_map().coordinate_squared_scalar(z).sqrt();

        assert_complex_close(chart.reference_coordinate(), expected);
    }

    #[test]
    fn opposite_sheet_uses_negative_principal_root_at_reference() {
        let chart = lossless_chart(CoordinateSheet::Opposite);

        let z = chart.reference();

        let expected = -chart.raw_map().coordinate_squared_scalar(z).sqrt();

        assert_complex_close(chart.reference_coordinate(), expected);
    }

    #[test]
    fn coordinate_at_reference_is_reference_coordinate_exactly() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        assert_eq!(
            chart.coordinate(chart.reference()).unwrap(),
            chart.reference_coordinate(),
        );
    }

    #[test]
    fn explicit_reference_coordinate_is_accepted() {
        let map = lossless_map();
        let region = lossless_region();
        let reference = c(0.2, -0.1);

        let reference_coordinate = map.coordinate_squared_scalar(reference).sqrt();

        let chart = Chart::with_reference_coordinate(
            map,
            region,
            reference,
            reference_coordinate,
            ConformalChartConfig::default(),
        )
        .unwrap();

        assert_eq!(chart.reference_coordinate(), reference_coordinate,);
    }

    #[test]
    fn inconsistent_reference_coordinate_is_rejected() {
        let map = lossless_map();
        let region = lossless_region();
        let reference = c(0.2, -0.1);

        let result = Chart::with_reference_coordinate(
            map,
            region,
            reference,
            c(123.0, -45.0),
            ConformalChartConfig::default(),
        );

        assert_eq!(result.unwrap_err(), ChartError::InconsistentContinuation,);
    }

    // =========================================================================
    // Analytic-square-root invariant
    // =========================================================================

    #[test]
    fn continued_coordinate_squares_to_squared_map() {
        let chart = lossy_chart();

        for z in [
            c(-0.35, -0.55),
            c(-0.20, 0.20),
            c(0.00, 0.00),
            c(0.25, -0.30),
            c(0.30, 0.45),
        ] {
            let coordinate = chart.coordinate(z).unwrap();

            let q = chart.raw_map().coordinate_squared_scalar(z);

            assert_complex_close_tol(coordinate * coordinate, q, 1.0e-9);
        }
    }

    #[test]
    fn opposite_and_principal_charts_differ_by_global_sign() {
        let principal = lossless_chart(CoordinateSheet::Principal);

        let opposite = lossless_chart(CoordinateSheet::Opposite);

        for z in [
            c(-0.7, -0.5),
            c(-0.3, 0.4),
            c(0.0, 0.0),
            c(0.4, -0.2),
            c(0.8, 0.6),
        ] {
            let a = principal.coordinate(z).unwrap();
            let b = opposite.coordinate(z).unwrap();

            assert_complex_close(a, -b);
        }
    }

    // =========================================================================
    // Principal sqrt where no correction is required
    // =========================================================================

    #[test]
    fn chart_agrees_with_principal_sqrt_on_principal_chart() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        for z in [
            c(-0.8, -0.7),
            c(-0.4, 0.3),
            c(0.0, 0.0),
            c(0.3, -0.4),
            c(0.7, 0.6),
        ] {
            let expected = chart.raw_map().coordinate_squared_scalar(z).sqrt();

            assert_complex_close_tol(chart.coordinate(z).unwrap(), expected, 1.0e-9);
        }
    }

    // =========================================================================
    // Principal-square-root cut crossing
    // =========================================================================

    #[test]
    fn cut_crossing_fixture_is_branch_point_free() {
        let map = cut_crossing_map();

        assert!(map.validate_region(&cut_crossing_region()).is_ok());
    }

    #[test]
    fn squared_coordinate_crosses_negative_real_axis() {
        let map = cut_crossing_map();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let q_left = map.coordinate_squared_scalar(left);

        let q_right = map.coordinate_squared_scalar(right);

        assert!(q_left.re < 0.0);
        assert!(q_right.re < 0.0);

        assert!(
            q_left.im * q_right.im < 0.0,
            "expected squared coordinate on opposite sides \
             of principal cut: left={q_left:?}, \
             right={q_right:?}",
        );
    }

    #[test]
    fn principal_sqrt_changes_sheet_across_cut() {
        let map = cut_crossing_map();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let principal_left = map.coordinate_squared_scalar(left).sqrt();

        let principal_right = map.coordinate_squared_scalar(right).sqrt();

        let same_sheet_distance = (principal_left - principal_right).modulus();

        let opposite_sheet_distance = (principal_left + principal_right).modulus();

        assert!(
            opposite_sheet_distance < same_sheet_distance,
            "principal roots did not exhibit expected \
             sheet flip: left={principal_left:?}, \
             right={principal_right:?}",
        );
    }

    #[test]
    fn chart_remains_continuous_across_principal_sqrt_cut() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let coordinate_left = chart.coordinate(left).unwrap();

        let coordinate_right = chart.coordinate(right).unwrap();

        let same_sheet_distance = (coordinate_left - coordinate_right).modulus();

        let opposite_sheet_distance = (coordinate_left + coordinate_right).modulus();

        assert!(
            same_sheet_distance < opposite_sheet_distance,
            "chart changed sheet across principal sqrt cut: \
             left={coordinate_left:?}, \
             right={coordinate_right:?}",
        );

        assert!(
            same_sheet_distance < 1.0e-2,
            "continued coordinate was unexpectedly \
             discontinuous: distance={same_sheet_distance}",
        );
    }

    #[test]
    fn chart_changes_relationship_to_principal_root_across_cut() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left = c(CUT_CROSSING_X - delta, CUT_CROSSING_Y);

        let right = c(CUT_CROSSING_X + delta, CUT_CROSSING_Y);

        let analytic_left = chart.coordinate(left).unwrap();

        let analytic_right = chart.coordinate(right).unwrap();

        let principal_left = chart.raw_map().coordinate_squared_scalar(left).sqrt();

        let principal_right = chart.raw_map().coordinate_squared_scalar(right).sqrt();

        let left_matches_plus = (analytic_left - principal_left).modulus_squared()
            <= (analytic_left + principal_left).modulus_squared();

        let right_matches_plus = (analytic_right - principal_right).modulus_squared()
            <= (analytic_right + principal_right).modulus_squared();

        assert_ne!(left_matches_plus, right_matches_plus,);
    }

    // =========================================================================
    // Full algebra map
    // =========================================================================

    #[test]
    fn chart_map_uses_analytic_coordinate_value() {
        let chart = cut_crossing_chart();

        for z in [
            c(0.10, -0.20),
            c(0.25, -0.20),
            c(0.40, -0.20),
            c(0.55, -0.20),
        ] {
            let point = chart.map(constant(z)).unwrap();

            let mapped_coordinate = jet_value(point.coordinates().parallel_angular_wavenumber());

            let expected = chart.coordinate(z).unwrap();

            assert_complex_close_tol(mapped_coordinate, expected, 1.0e-9);
        }
    }

    #[test]
    fn constant_and_variable_evaluations_select_same_sheet() {
        let chart = cut_crossing_chart();

        for z in [
            c(0.20, -0.20),
            c(CUT_CROSSING_X - 1.0e-4, CUT_CROSSING_Y),
            c(CUT_CROSSING_X + 1.0e-4, CUT_CROSSING_Y),
            c(0.50, -0.20),
        ] {
            let constant_point = chart.map(constant(z)).unwrap();

            let variable_point = chart.map(variable(z)).unwrap();

            assert_complex_close(
                jet_value(constant_point.coordinates().parallel_angular_wavenumber()),
                jet_value(variable_point.coordinates().parallel_angular_wavenumber()),
            );
        }
    }

    // =========================================================================
    // Jet derivatives
    // =========================================================================

    #[test]
    fn charted_jet_derivative_matches_analytic_coordinate_derivative() {
        let chart = cut_crossing_chart();

        for z in [
            c(0.15, -0.20),
            c(0.30, -0.20),
            c(0.36, -0.20),
            c(0.52, -0.20),
        ] {
            let point = chart.map(variable(z)).unwrap();

            let coordinate = jet_value(point.coordinates().parallel_angular_wavenumber());

            let derivative = jet_derivative(point.coordinates().parallel_angular_wavenumber());

            let q = chart.raw_map().coordinate_squared_scalar(z);

            let log_derivative = chart.raw_map().coordinate_squared_log_derivative_scalar(z);

            let dq_dz = q * log_derivative;

            let expected = dq_dz / (coordinate + coordinate);

            assert_complex_close_tol(derivative, expected, 1.0e-9);
        }
    }

    #[test]
    fn charted_jet_derivative_is_continuous_across_principal_cut() {
        let chart = cut_crossing_chart();

        let delta = 1.0e-5;

        let left = chart
            .map(variable(c(CUT_CROSSING_X - delta, CUT_CROSSING_Y)))
            .unwrap();

        let right = chart
            .map(variable(c(CUT_CROSSING_X + delta, CUT_CROSSING_Y)))
            .unwrap();

        let derivative_left = jet_derivative(left.coordinates().parallel_angular_wavenumber());

        let derivative_right = jet_derivative(right.coordinates().parallel_angular_wavenumber());

        assert!(
            (derivative_left - derivative_right).modulus() < 1.0e-2,
            "analytic derivative changed discontinuously \
             across principal cut: \
             left={derivative_left:?}, \
             right={derivative_right:?}",
        );
    }

    // =========================================================================
    // Domain checking
    // =========================================================================

    #[test]
    fn scalar_coordinate_rejects_point_outside_chart() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        assert_eq!(
            chart.coordinate(c(1.1, 0.0)).unwrap_err(),
            ChartError::CoordinateOutsideRegion,
        );
    }

    #[test]
    fn algebra_map_rejects_point_outside_chart() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        assert_eq!(
            chart.map(constant(c(0.0, 1.1))).unwrap_err(),
            ChartError::CoordinateOutsideRegion,
        );
    }

    #[test]
    fn chart_accepts_regular_region_boundary_points() {
        let chart = lossless_chart(CoordinateSheet::Principal);

        for z in [c(-1.0, 0.0), c(1.0, 0.0), c(0.0, -1.0), c(0.0, 1.0)] {
            assert!(
                chart.coordinate(z).is_ok(),
                "regular chart boundary point rejected: \
                 {z:?}",
            );
        }
    }

    // =========================================================================
    // Reference independence / analytic continuation
    // =========================================================================

    #[test]
    fn continuation_is_reversible() {
        let first = lossy_chart();

        let target = c(0.30, -0.45);

        let target_coordinate = first.coordinate(target).unwrap();

        let reverse = Chart::with_reference_coordinate(
            lossy_map(),
            lossy_region(),
            target,
            target_coordinate,
            ConformalChartConfig::default(),
        )
        .unwrap();

        assert_complex_close_tol(
            reverse.coordinate(first.reference()).unwrap(),
            first.reference_coordinate(),
            1.0e-9,
        );
    }

    #[test]
    fn charts_with_consistent_reference_coordinates_agree_everywhere() {
        let first = lossy_chart();

        let second_reference = c(0.25, -0.30);

        let second_reference_coordinate = first.coordinate(second_reference).unwrap();

        let second = Chart::with_reference_coordinate(
            lossy_map(),
            lossy_region(),
            second_reference,
            second_reference_coordinate,
            ConformalChartConfig::default(),
        )
        .unwrap();

        for z in [
            c(-0.30, -0.50),
            c(-0.10, 0.30),
            c(0.10, -0.20),
            c(0.30, 0.40),
        ] {
            assert_complex_close_tol(
                first.coordinate(z).unwrap(),
                second.coordinate(z).unwrap(),
                1.0e-9,
            );
        }
    }

    // =========================================================================
    // Adaptive Simpson quadrature
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
        // Integral_0^1 exp(i pi x) dx = 2i/pi.
        let result = adaptive_simpson(&|x| c(0.0, PI * x).exp(), 0.0, 1.0, 1.0e-12, 20).unwrap();

        assert_complex_close_tol(result, c(0.0, 2.0 / PI), 1.0e-10);
    }

    #[test]
    fn adaptive_simpson_reports_non_convergence() {
        let result = adaptive_simpson(&|x| c(0.0, 100.0 * x).exp(), 0.0, 1.0, 1.0e-15, 0);

        assert_eq!(result.unwrap_err(), ContinuationError::DidNotConverge,);
    }

    #[test]
    fn adaptive_simpson_rejects_non_finite_integrand() {
        let result = adaptive_simpson(
            &|x| {
                if x == 0.5 {
                    c(f64::NAN, 0.0)
                } else {
                    c(1.0, 0.0)
                }
            },
            0.0,
            1.0,
            1.0e-12,
            10,
        );

        assert_eq!(result.unwrap_err(), ContinuationError::NonFinite,);
    }
    #[test]
    fn evaluation_margin_preserves_sheet_and_accepts_boundary_roundtrips() {
        for sheet in [CoordinateSheet::Principal, CoordinateSheet::Opposite] {
            let chart = lossless_chart(sheet);
            let evaluation = chart.with_evaluation_margin().unwrap();
            assert_eq!(chart.reference(), evaluation.reference());
            assert_eq!(
                chart.reference_coordinate(),
                evaluation.reference_coordinate()
            );
            let region = *chart.region();
            let (xmin, xmax) = region.real_bounds();
            let (ymin, ymax) = region.imaginary_bounds();
            // Simulate affine normalization and reverse mapping, including corners.
            for x in [xmin, 0.0, xmax] {
                for y in [ymin, 0.0, ymax] {
                    let rx = xmin + ((x - xmin) / (xmax - xmin)) * (xmax - xmin);
                    let ry = ymin + ((y - ymin) / (ymax - ymin)) * (ymax - ymin);
                    let point = c(rx, ry);
                    assert_complex_close(
                        evaluation.coordinate(point).unwrap(),
                        chart.coordinate(c(x, y)).unwrap(),
                    );
                }
            }
            let rounded = c(xmax + f64::EPSILON, 0.0);
            REGION_MISSES.with(|points| points.borrow_mut().clear());
            assert!(matches!(
                chart.coordinate(rounded),
                Err(ChartError::CoordinateOutsideRegion)
            ));
            REGION_MISSES.with(|points| assert_eq!(points.borrow().len(), 1));
            assert!(evaluation.coordinate(rounded).is_ok());
            assert!(matches!(
                evaluation.coordinate(c(xmax + 1e-6, 0.0)),
                Err(ChartError::CoordinateOutsideRegion)
            ));
            assert_eq!(*chart.region(), region);
        }
    }

    #[test]
    fn evaluation_margin_rejects_branchpoint_extension() {
        let map = lossless_map();
        let branch = map.landmarks().unwrap().coordinate_branch_points()[0];
        let region = ConformalRegion::new(
            branch.re - 0.1,
            branch.re + 0.1,
            branch.im - 0.2,
            branch.im - 1e-15,
        )
        .unwrap();
        let chart = Chart::new(
            map,
            region,
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();
        assert!(matches!(
            chart.with_evaluation_margin(),
            Err(ChartError::InvalidRegion(_))
        ));
    }

    #[test]
    fn evaluation_margin_rejects_nonfinite_expansion() {
        let map = lossless_map();
        // Build an internal chart fixture so the margin check is isolated from
        // constitutive exponent evaluation at extreme coordinates.
        let mut chart = lossless_chart(CoordinateSheet::Principal);
        chart.map = map;
        chart.region = ConformalRegion::new(-f64::MAX, f64::MAX, -1.0, 1.0).unwrap();
        assert!(matches!(
            chart.with_evaluation_margin(),
            Err(ChartError::InvalidEvaluationMargin)
        ));
    }
}
