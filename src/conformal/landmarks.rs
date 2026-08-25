use lamina_core::ComplexScalar;
use num_traits::{Float, One, Zero};

/// Which semi-infinite exterior defines a light-line landmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExteriorSide {
    Left,
    Right,
}

pub(super) fn require_finite_pair<C>(values: [Option<C>; 2]) -> Option<[C; 2]>
where
    C: Copy,
{
    match values {
        [Some(first), Some(second)] => Some([first, second]),

        _ => None,
    }
}

/// Return the principal finite conformal preimage of a root in the
/// `t = exp(2z)` plane.
///
/// For non-zero `t`,
///
/// ```text
/// z = 1/2 log(t)
///   = 1/2 ln|t| + i/2 arg(t),
/// ```
///
/// where `arg(t)` is the principal argument. The returned representative
/// therefore lies in
///
/// ```text
/// -pi/2 < Im(z) <= pi/2.
/// ```
///
/// Periodically equivalent preimages are obtained by adding integer multiples
/// of `i*pi`.
///
/// A zero `t` root has no finite preimage because `exp(2z)` never vanishes at
/// finite `z`; it corresponds to `Re(z) -> -infinity`.
pub(super) fn finite_t_preimage<C>(t: C) -> Option<C>
where
    C: ComplexScalar + Copy,
    C::RealField: Float + Copy,
{
    if t.modulus_squared() == C::RealField::zero() {
        return None;
    }

    let two = C::RealField::one() + C::RealField::one();

    let re = Float::ln(t.modulus()) / two;

    let im = t.argument() / two;

    Some(C::from_real(re) + C::i() * C::from_real(im))
}
