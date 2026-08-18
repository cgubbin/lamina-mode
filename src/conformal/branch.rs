use lamina_core::ComplexScalar;
use nalgebra::ComplexField;
use num_traits::{Float, FloatConst, One, Zero};

use crate::conformal::ConformalRegion;

/// Roots of x² + s x + product = 0.
///
/// The second argument is the product of the roots.  Computing the
/// smaller root from Vieta's relation avoids catastrophic cancellation
/// when the two exterior media are nearly identical.
pub(super) fn stable_quadratic_roots<C>(s: C, product: C) -> [C; 2]
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

pub(super) fn branch_root_intersects_region<C>(t: C, region: &ConformalRegion<C::RealField>) -> bool
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

pub(super) fn t_to_z<C>(t: C) -> Option<C>
where
    C: ComplexScalar + Copy,
    C::RealField: Float + Copy,
{
    if t.modulus_squared() == C::RealField::zero() {
        // exp(2z) cannot vanish for finite z.
        return None;
    }

    let two = C::RealField::one() + C::RealField::one();

    let re = Float::ln(t.modulus()) / two;

    let im = t.argument() / two;

    Some(C::from_real(re) + C::i() * C::from_real(im))
}

pub(super) fn should_swap_preimages<C>(first: Option<C>, second: Option<C>) -> bool
where
    C: ComplexField + Copy,
    C::RealField: Float + Copy,
{
    match (first, second) {
        // Put the finite root before an infinite/preimage-at-minus-infinity
        // marker only if both aren't finite. For the normal asymmetric case
        // both are Some.
        (None, Some(_)) => true,
        (Some(_), None) | (None, None) => false,

        (Some(first), Some(second)) => {
            if first.real() > second.real() {
                true
            } else if first.real() < second.real() {
                false
            } else {
                first.imaginary() > second.imaginary()
            }
        }
    }
}
