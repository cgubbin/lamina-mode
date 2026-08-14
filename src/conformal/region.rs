use num_traits::float::FloatCore;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConformalRegionError {
    #[error("conformal-region bounds must be finite")]
    NonFiniteBound,

    #[error("real bounds must satisfy min < max")]
    InvalidRealBounds,

    #[error("imaginary bounds must satisfy min < max")]
    InvalidImaginaryBounds,
}

/// Rectangular region of the conformal plane.
///
/// Bounds are expressed directly in `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalRegion<R> {
    re_min: R,
    re_max: R,
    im_min: R,
    im_max: R,
}

impl<R> ConformalRegion<R>
where
    R: Copy,
{
    pub fn new(re_min: R, re_max: R, im_min: R, im_max: R) -> Result<Self, ConformalRegionError>
    where
        R: FloatCore + PartialOrd,
    {
        if !re_min.is_finite() || !re_max.is_finite() || !im_min.is_finite() || !im_max.is_finite()
        {
            return Err(ConformalRegionError::NonFiniteBound);
        }

        if re_min >= re_max {
            return Err(ConformalRegionError::InvalidRealBounds);
        }

        if im_min >= im_max {
            return Err(ConformalRegionError::InvalidImaginaryBounds);
        }

        Ok(Self {
            re_min,
            re_max,
            im_min,
            im_max,
        })
    }

    pub fn real_bounds(&self) -> (R, R) {
        (self.re_min, self.re_max)
    }

    pub fn imaginary_bounds(&self) -> (R, R) {
        (self.im_min, self.im_max)
    }
}
