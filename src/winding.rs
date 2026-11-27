//! Zeros of an analytic function inside a rectangle from the change of its
//! argument around the boundary.
use argloc::Rect;
use nalgebra::Complex;

/// Why a boundary winding could not count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindingFailure {
    /// A zero within the clearance of the boundary.
    BoundaryZero,
    /// The function could not be evaluated, was not finite, or the winding
    /// was not integer-close or was negative (a pole inside).
    NonInteger,
}

/// Zeros of `f` inside `window` from the change of arg f around its
/// boundary, tracked in steps over which the phase turns by less than π/4
/// and agrees with the integrated logarithmic derivative, and the first
/// moment ∮ z d(log f) / 2πi, the sum of the zeros. `evaluate` returns f and
/// f′ at a point. On failure, the point where tracking stopped and why: a
/// zero within `clearance` of the boundary, or a function that could not be
/// evaluated.
pub(crate) fn boundary_winding(
    window: Rect<f64>,
    clearance: f64,
    mut evaluate: impl FnMut(Complex<f64>) -> Option<(Complex<f64>, Complex<f64>)>,
) -> Result<(usize, Complex<f64>), (Complex<f64>, WindingFailure)> {
    let corner = |x: f64, y: f64| Complex::new(x, y);
    let (x, y) = ([window.x_min, window.x_max], [window.y_min, window.y_max]);
    let path = [
        corner(x[0], y[0]),
        corner(x[1], y[0]),
        corner(x[1], y[1]),
        corner(x[0], y[1]),
        corner(x[0], y[0]),
    ];
    let failed = |z| (z, WindingFailure::NonInteger);
    let mut evaluate = |z: Complex<f64>| {
        let (value, derivative) = evaluate(z).ok_or_else(|| failed(z))?;
        if value == Complex::ZERO {
            return Err((z, WindingFailure::BoundaryZero));
        }
        let log = derivative / value;
        if !value.re.is_finite()
            || !value.im.is_finite()
            || !log.re.is_finite()
            || !log.im.is_finite()
        {
            return Err(failed(z));
        }
        Ok((value, log))
    };
    let mut turns = 0.0;
    let mut moment = Complex::<f64>::ZERO;
    for edge in path.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        let length = (b - a).norm();
        let (mut t, mut step) = (0.0_f64, 1.0_f64 / 16.0);
        let mut here = evaluate(a)?;
        while t < 1.0 {
            step = step.min(1.0 - t);
            let (z0, z1) = (a + (b - a) * t, a + (b - a) * (t + step));
            let there = evaluate(z1)?;
            let turn = (there.0 / here.0).arg();
            let predicted = ((here.1 + there.1) * 0.5 * (z1 - z0)).im;
            if turn.abs() > std::f64::consts::FRAC_PI_4 || (turn - predicted).abs() > 0.05 {
                if step * length <= clearance {
                    return Err((z1, WindingFailure::BoundaryZero));
                }
                step *= 0.5;
                continue;
            }
            turns += turn;
            moment += (z0 + z1) * 0.5 * Complex::new((there.0 / here.0).norm().ln(), turn);
            t += step;
            here = there;
            step *= 2.0;
        }
    }
    let winding = turns / std::f64::consts::TAU;
    let count = winding.round();
    if (winding - count).abs() > 1e-6 || count < 0.0 {
        return Err(failed(path[0]));
    }
    Ok((
        count as usize,
        moment / Complex::new(0.0, std::f64::consts::TAU),
    ))
}
