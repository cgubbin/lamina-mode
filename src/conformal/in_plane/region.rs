use argloc::Rect;
use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FloatConst, Zero, float::FloatCore};

use crate::conformal::{
    ConformalRegion, ValidateConformalRegion,
    branch::{
        branch_root_intersects_region, should_swap_preimages, stable_quadratic_roots, t_to_z,
    },
};

use super::ComplexInPlaneMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InPlaneSearchError {
    #[error("canonical physical search regions require real positive exterior epsilon*mu")]
    UnsupportedExterior,

    #[error("physical in-plane wavenumber must be finite and non-negative")]
    InvalidParallelWavenumber,

    #[error(
        "maximum in-plane wavenumber used to truncate the search region must \
         lie above both exterior light lines"
    )]
    CutoffBelowExteriorLightLine,

    #[error("imaginary margin must satisfy 0 < margin < pi/2")]
    InvalidImaginaryMargin,

    #[error(
        "automatic finite search-region construction is degenerate for \
         identical exterior constitutive products; supply an explicit \
         ConformalRegion instead"
    )]
    SymmetricExterior,

    #[error("constructed conformal region failed map validation")]
    InvalidRegion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InPlaneRegionError {
    #[error("search region contains or touches a branch point of the linear in-plane coordinate")]
    CoordinateBranchPoint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InPlaneSearchExtent<R> {
    /// Maximum magnitude of the real in-plane wavenumber represented by the
    /// canonical search region.
    ///
    /// This must lie above both exterior light lines. This requirement applies
    /// only to the *outer truncation* of the conformal region: the resulting
    /// region itself contains guided and leaky sectors on both sides of the
    /// light lines.
    pub max_parallel_wavenumber: R,

    /// Distance retained from the periodic boundaries
    /// ```text
    ///     Im(z) = +/- pi/2.
    ///
    /// The contour therefore lies at
    ///
    ///     -pi/2 + margin
    ///
    /// and
    ///
    ///      pi/2 - margin.
    /// ```
    pub imaginary_margin: R,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InPlaneSearchRegion<R>(ConformalRegion<R>);

impl<R> TryInto<Rect<R>> for InPlaneSearchRegion<R>
where
    ConformalRegion<R>: TryInto<Rect<R>>,
{
    type Error = <ConformalRegion<R> as TryInto<Rect<R>>>::Error;

    fn try_into(self) -> Result<Rect<R>, Self::Error> {
        self.0.try_into()
    }
}

impl<R> InPlaneSearchRegion<R> {
    fn new(region: ConformalRegion<R>) -> Self {
        Self(region)
    }

    pub(crate) fn into_inner(self) -> ConformalRegion<R> {
        self.0
    }
}

impl<A> ComplexInPlaneMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField + Copy,
{
    /// Construct the canonical one-period in-plane modal search region.
    ///
    /// The region spans essentially one full fundamental conformal strip,
    ///
    /// ```text
    /// -pi/2 < Im(z) < pi/2,
    /// ```
    ///
    /// and therefore contains the conformal continuation of guided, singly
    /// leaky, and doubly leaky sectors. The two real bounds are obtained from
    /// the two high-|kx| preimages of `max_parallel_wavenumber`.
    ///
    /// `max_parallel_wavenumber` must lie above both exterior light lines so
    /// that those two outer preimages are real and finite. This is a truncation
    /// condition only; it is not a restriction on modes inside the resulting
    /// region.
    pub fn search_region(
        &self,
        extent: InPlaneSearchExtent<<A::Scalar as ComplexField>::RealField>,
    ) -> Result<InPlaneSearchRegion<<A::Scalar as ComplexField>::RealField>, InPlaneSearchError>
    where
        A::Scalar: ComplexScalar,
        <A::Scalar as ComplexField>::RealField: Float + FloatConst + FloatCore,
    {
        let (chi_left, chi_right) = self.real_positive_exterior_chi()?;

        let max_kx = extent.max_parallel_wavenumber;

        if !Float::is_finite(max_kx) || max_kx <= A::Scalar::zero().real() {
            return Err(InPlaneSearchError::InvalidParallelWavenumber);
        }

        let half_pi = <<A::Scalar as ComplexField>::RealField as FloatConst>::FRAC_PI_2();

        let margin = extent.imaginary_margin;

        if !Float::is_finite(margin) || margin <= A::Scalar::zero().real() || margin >= half_pi {
            return Err(InPlaneSearchError::InvalidImaginaryMargin);
        }

        // r = 0 is the equal-exterior degeneracy. One high-kx preimage tends
        // to Re(z) -> -infinity, so no unique finite canonical rectangle can
        // be inferred from max_kx alone.
        if self.r_scalar().modulus_squared() == A::Scalar::zero().real() {
            return Err(InPlaneSearchError::SymmetricExterior);
        }

        let k0 = Float::abs(self.k0_scalar().real());

        let outer_light_line = k0 * Float::sqrt(Float::max(chi_left, chi_right));

        if max_kx <= outer_light_line {
            return Err(InPlaneSearchError::CutoffBelowExteriorLightLine);
        }

        let preimages = self.physical_preimages(max_kx)?;

        let [Some(left), Some(right)] = preimages else {
            // Already guarded by r != 0, so this indicates a numerical or
            // algebraic inconsistency.
            return Err(InPlaneSearchError::InvalidRegion);
        };

        // Above both light lines both physical preimages are real.
        //
        // Do not silently accept a complex pair here: the high-kx inverse is
        // what defines our real truncation boundaries.
        if left.imaginary() != A::Scalar::zero().real()
            || right.imaginary() != A::Scalar::zero().real()
        {
            return Err(InPlaneSearchError::InvalidRegion);
        }

        let re_min = left.real();
        let re_max = right.real();

        if !(re_min < re_max) {
            return Err(InPlaneSearchError::InvalidRegion);
        }

        let region = ConformalRegion::new(re_min, re_max, -half_pi + margin, half_pi - margin)
            .map_err(|_| InPlaneSearchError::InvalidRegion)?;

        self.validate_region(&region)
            .map_err(|_| InPlaneSearchError::InvalidRegion)?;

        Ok(InPlaneSearchRegion::new(region))
    }

    /// Return the two conformal preimages of a real physical in-plane
    /// wavenumber within one fundamental `i*pi` period.
    ///
    /// This inverse is defined for lossless exterior media with real positive
    /// constitutive products
    /// ```text
    ///     chi_j = epsilon_j * mu_j.
    /// ```
    /// No restriction is placed on the position of `kx` relative to the two
    /// exterior light lines. Consequently the returned points cover:
    ///
    /// - the guided / doubly-evanescent sector;
    /// - the singly-leaky sector between unequal light lines;
    /// - the doubly-propagating sector below both light lines;
    /// - the light-line branch points themselves.
    ///
    /// The inverse follows from
    ///
    /// ```text
    /// u = (kx / k0)^2
    ///
    /// t^2 - (u - s)t + r^2 = 0,
    ///
    /// t = exp(2z).
    /// ```
    ///
    /// Each non-zero `t` root is represented using the principal argument,
    /// giving a representative with
    /// ```text
    ///     -pi/2 < Im(z) <= pi/2.
    /// ```
    /// A zero `t` root corresponds to `Re(z) -> -infinity`. Such a root can
    /// occur for identical exterior media and cannot be represented by a
    /// finite conformal coordinate.
    pub fn physical_preimages(
        &self,
        kx: <A::Scalar as ComplexField>::RealField,
    ) -> Result<[Option<A::Scalar>; 2], InPlaneSearchError>
    where
        A::Scalar: ComplexScalar,
        <A::Scalar as ComplexField>::RealField: Float + Copy,
    {
        self.real_positive_exterior_chi()?;

        if !kx.is_finite() || kx < A::Scalar::zero().real() {
            return Err(InPlaneSearchError::InvalidParallelWavenumber);
        }

        let k0 = self.k0_scalar().real();

        let ratio = kx / k0;
        let u = ratio * ratio;

        let s = self.s_scalar().real();
        let r = self.r_scalar().real();

        // t² - (u - s)t + r² = 0
        //
        // The shared solver expects
        //
        // t² + linear*t + product = 0.
        let linear = A::Scalar::from_real(s - u);

        let product = A::Scalar::from_real(r * r);

        let roots = stable_quadratic_roots(linear, product);

        let mut z = roots.map(t_to_z);

        // Keep the result deterministic. In the guided regime this orders the
        // two real preimages from left to right. Between the light lines they
        // have equal real parts and are ordered by imaginary part.
        if should_swap_preimages(z[0], z[1]) {
            z.swap(0, 1);
        }

        Ok(z)
    }
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
