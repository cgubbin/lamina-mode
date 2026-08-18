use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FloatConst, float::FloatCore};

use crate::conformal::{
    ConformalRegion, ValidateConformalRegion,
    branch::{
        branch_root_intersects_region, should_swap_preimages, stable_quadratic_roots, t_to_z,
    },
};

use super::ComplexWavenumberMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WavenumberSearchError {
    #[error("canonical physical search geometry requires real positive exterior epsilon*mu")]
    UnsupportedExterior,

    #[error("physical vacuum wavenumber must be finite and non-negative")]
    InvalidVacuumWavenumber,

    #[error(
        "maximum vacuum wavenumber used to truncate the search must lie above \
         both exterior light lines"
    )]
    MaxVacuumWavenumberBelowUpperLightLine,

    #[error("imaginary margin must satisfy 0 < margin < pi/4")]
    InvalidImaginaryMargin,

    #[error(
        "automatic finite search-region construction is degenerate for \
         symmetric exterior constitutive products or zero in-plane wavenumber; \
         supply explicit ConformalRegions instead"
    )]
    DegenerateExterior,

    #[error("constructed conformal region failed map validation")]
    InvalidRegion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WavenumberRegionError {
    #[error(
        "search region contains or touches a branch point \
         of the linear vacuum wavenumber"
    )]
    CoordinateBranchPoint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavenumberSearchExtent<R> {
    /// Maximum positive real vacuum wavenumber represented by the
    /// canonical search regions.
    ///
    /// This must exceed both exterior light-line wavenumbers. It is used only
    /// to choose finite left/right conformal truncation bounds.
    pub max_vacuum_wavenumber: R,

    /// Clearance from both the real-z coordinate-branch-point line and the
    /// periodically identified +/-pi/2 boundaries.
    pub imaginary_margin: R,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WavenumberSearchRegions<R> {
    lower: ConformalRegion<R>,
    upper: ConformalRegion<R>,
}

impl<R> WavenumberSearchRegions<R> {
    pub fn lower(&self) -> &ConformalRegion<R> {
        &self.lower
    }

    pub fn upper(&self) -> &ConformalRegion<R> {
        &self.upper
    }

    pub fn into_regions(self) -> [ConformalRegion<R>; 2] {
        [self.lower, self.upper]
    }
}

impl<A, R> ComplexWavenumberMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexScalar<RealField = R> + Copy,
    R: Float + FloatConst + FloatCore + Copy,
{
    pub fn search_regions(
        &self,
        extent: WavenumberSearchExtent<R>,
    ) -> Result<WavenumberSearchRegions<R>, WavenumberSearchError> {
        let (chi_left, chi_right) = self.real_positive_exterior_chi()?;

        let max_k0 = extent.max_vacuum_wavenumber;

        if !Float::is_finite(max_k0) || max_k0 <= R::zero() {
            return Err(WavenumberSearchError::InvalidVacuumWavenumber);
        }

        let half_pi = <R as FloatConst>::FRAC_PI_2();

        let quarter_pi = half_pi / (R::one() + R::one());

        let margin = extent.imaginary_margin;

        if !Float::is_finite(margin) || margin <= R::zero() || margin >= quarter_pi {
            return Err(WavenumberSearchError::InvalidImaginaryMargin);
        }

        // r = 0 includes equal exterior chi. kx = 0 makes both r and s zero.
        //
        // In either case one inverse branch escapes to Re(z) -> -infinity,
        // so max_k0 alone cannot determine a finite canonical rectangle.
        if self.r_scalar().modulus_squared() == R::zero() {
            return Err(WavenumberSearchError::DegenerateExterior);
        }

        let kx = Float::abs(self.kx_scalar().real());

        let left_light_line = kx / Float::sqrt(chi_left);

        let right_light_line = kx / Float::sqrt(chi_right);

        let upper_light_line = Float::max(left_light_line, right_light_line);

        if max_k0 <= upper_light_line {
            return Err(WavenumberSearchError::MaxVacuumWavenumberBelowUpperLightLine);
        }

        let [left, right] = self.physical_preimages(max_k0)?;

        let [Some(left), Some(right)] = [left, right] else {
            return Err(WavenumberSearchError::InvalidRegion);
        };

        let re_min = left.real();
        let re_max = right.real();

        if !(re_min < re_max) {
            return Err(WavenumberSearchError::InvalidRegion);
        }

        let lower = ConformalRegion::new(re_min, re_max, -half_pi + margin, -margin)
            .map_err(|_| WavenumberSearchError::InvalidRegion)?;

        let upper = ConformalRegion::new(re_min, re_max, margin, half_pi - margin)
            .map_err(|_| WavenumberSearchError::InvalidRegion)?;

        self.validate_region(&lower)
            .map_err(|_| WavenumberSearchError::InvalidRegion)?;

        self.validate_region(&upper)
            .map_err(|_| WavenumberSearchError::InvalidRegion)?;

        Ok(WavenumberSearchRegions { lower, upper })
    }

    /// Return the two conformal preimages of a non-negative real vacuum
    /// wavenumber within one closed representative of the fundamental period.
    ///
    /// This convenience inverse is available for real positive constant
    /// exterior constitutive products.
    pub fn physical_preimages(&self, k0: R) -> Result<[Option<A::Scalar>; 2], WavenumberSearchError>
    where
        A::Scalar: ComplexScalar<RealField = R>,
        R: Float + Copy,
    {
        self.real_positive_exterior_chi()?;

        if !Float::is_finite(k0) || k0 < R::zero() {
            return Err(WavenumberSearchError::InvalidVacuumWavenumber);
        }

        let u = k0 * k0;

        let s = self.s_scalar().real();
        let r = self.r_scalar().real();

        // t² - (s - u)t + r² = 0
        //
        // stable_quadratic_roots solves
        //
        // t² + linear*t + product = 0.
        let linear = A::Scalar::from_real(u - s);

        let product = A::Scalar::from_real(r * r);

        let roots = stable_quadratic_roots(linear, product);

        let mut z = roots.map(t_to_z);

        if should_swap_preimages(z[0], z[1]) {
            z.swap(0, 1);
        }

        Ok(z)
    }
}

impl<A, R> ValidateConformalRegion<R> for ComplexWavenumberMap<A>
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField<RealField = R> + Copy,
    R: Float + FloatConst + Copy,
{
    type Error = WavenumberRegionError;

    fn validate_region(&self, region: &ConformalRegion<R>) -> Result<(), Self::Error> {
        if self
            .branch_roots
            .iter()
            .copied()
            .any(|root| branch_root_intersects_region(root, region))
        {
            Err(WavenumberRegionError::CoordinateBranchPoint)
        } else {
            Ok(())
        }
    }
}
