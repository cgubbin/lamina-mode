//! Branches followed through folds by switching the control coordinate.
//!
//! A dispersion branch is a curve F(k0, β) = 0. Continuation in k0 stops at a
//! fold, where ∂F/∂β = 0, dβ/dk0 diverges and the branch turns back in k0 (a
//! zero-group-velocity point of a lossless mode). Along the curve the tangent
//! (dk0, dβ) ∝ (∂F/∂β, −∂F/∂k0) stays finite, so each step holds fixed the
//! coordinate along which the tangent moves most: k0 away from a fold, β
//! through it. Steps are measured by arclength in the real (k0, Re β) plane,
//! in cm⁻¹, and the tangent keeps its orientation from point to point, so
//! after a fold the branch continues back in k0. A sign change of dk0/dβ
//! between accepted points is a fold; it is located on β-controlled stations
//! and reported with the points on both sides.
use super::*;
use crate::problem as p;
use lamina_core::{
    Backend, ComplexPlane, ModalDerivativeBackend, PlaneWaveDeterminant,
    ProjectPlaneWaveModeDeterminant,
};

/// Settings of [`ModePoint::continue_through_folds_with`].
#[derive(Debug, Clone)]
pub struct ArclengthSettings {
    /// Assumptions, slope threshold, refinement and retries. `radius` is the
    /// half-width of conformal charts (in-plane under k0 control, spectral
    /// under β control for a point found on one); `minimum_step` is the
    /// smallest arclength step in cm⁻¹.
    pub branch: BranchSettings,
    /// The largest arclength step in the (k0, Re β) plane, cm⁻¹.
    pub step: f64,
    /// Half-width in cm⁻¹ of the k0 boxes of stations that solve k0
    /// directly under β control.
    pub frequency_radius: f64,
    /// The frequency-domain evidence of those stations for a point that has
    /// none (one found at fixed k0).
    pub frequency: FrequencyStation,
    /// The trace ends when k0 leaves this interval (cm⁻¹).
    pub k0_bounds: [f64; 2],
    /// The trace ends when Re β leaves this interval (cm⁻¹).
    pub beta_bounds: [f64; 2],
    /// The trace ends after this many accepted points past the seed.
    pub max_points: usize,
    /// Folds are located until their β bracket is narrower than this
    /// fraction of |β|.
    pub fold_tolerance: f64,
}

/// A fold of the branch: an extremum of k0 along it, where dk0/dβ = 0 (for
/// a lossless mode, zero group velocity).
#[derive(Debug, Clone)]
pub struct Fold {
    /// k0 at the fold, cm⁻¹.
    pub k0: Complex,
    /// β at the fold, cm⁻¹.
    pub beta: Complex,
    /// The station nearest the fold, at fixed β.
    pub point: ModePoint,
    /// dk0/dβ at `point`.
    pub slope: Complex,
    /// The accepted points before and after the fold along the trace, on its
    /// two sides.
    pub sides: [ModePoint; 2],
    /// dk0/dβ at the two sides, of opposite sign.
    pub side_slopes: [Complex; 2],
    /// Width of the final β bracket about the fold, cm⁻¹.
    pub bracket: f64,
}

/// Why an arclength trace ended.
#[derive(Debug)]
pub enum ArclengthEnd {
    /// k0 or Re β left its bounds; the last point is outside them.
    Bounds,
    /// `max_points` points were accepted.
    MaxPoints,
    /// A step failed at the minimum step.
    Failed(BranchFailure),
}

/// The points of a branch followed by arclength.
#[derive(Debug)]
pub struct ArclengthTrace {
    /// The seed followed by every accepted point.
    pub points: Vec<ModePoint>,
    /// The coordinate held fixed to reach each point after the seed.
    pub controls: Vec<Control>,
    /// Folds passed, in order.
    pub folds: Vec<Fold>,
    /// Refused steps: the control, its attempted value and the cause.
    pub retries: Vec<(Control, f64, BranchFailure)>,
    pub end: ArclengthEnd,
}

/// The tangent at a point and the slopes under either control.
#[derive(Clone, Copy, Debug)]
struct Tangent {
    /// Unit (dk0, d Re β).
    direction: [f64; 2],
    /// dβ/dk0, `None` at a fold.
    beta_per_k0: Option<Complex>,
    /// dk0/dβ, `None` where β is stationary in k0.
    k0_per_beta: Option<Complex>,
}

impl ModePoint {
    fn tangent<B: ModalDerivativeBackend + Clone>(
        &self,
        backend: B,
        s: &BranchSettings,
    ) -> Result<Tangent, BranchFailure> {
        let report = self
            .local_derivatives_with(
                backend,
                &[PhysicalParameter::K0, PhysicalParameter::Beta],
                s.local_assumption,
                Default::default(),
            )
            .map_err(|e| BranchFailure::Derivative(e.to_string()))?;
        let f_k0 = report.partial(PhysicalParameter::K0)?.value;
        let f_beta = report.partial(PhysicalParameter::Beta)?.value;
        // (dk0, dβ) ∝ (∂F/∂β, −∂F/∂k0); for a lossless mode the two share a
        // phase, removed with that of the larger.
        let larger = if f_k0.norm() > f_beta.norm() {
            f_k0
        } else {
            f_beta
        };
        if larger.norm() == 0. || !larger.norm().is_finite() {
            return Err(BranchFailure::Derivative(
                "the characteristic is stationary in k0 and β".into(),
            ));
        }
        let phase = larger.conj() / larger.norm();
        let (a, b) = ((f_beta * phase).re, (-f_k0 * phase).re);
        let norm = a.hypot(b);
        let ratio = |num: Complex, den: Complex| {
            let threshold = s.slope.denominator_atol
                + s.slope.denominator_rtol * s.slope.denominator_reference_scale;
            (den.norm() > threshold).then(|| -num / den)
        };
        Ok(Tangent {
            direction: [a / norm, b / norm],
            beta_per_k0: ratio(f_k0, f_beta),
            k0_per_beta: ratio(f_beta, f_k0),
        })
    }

    fn coordinates(&self) -> Result<(Complex, Complex), BranchFailure> {
        let sig = self.signature().map_err(|_| BranchFailure::Exterior)?;
        Ok((sig.k0, sig.k_parallel))
    }

    /// Follow this point's branch by arclength through any folds, starting
    /// with k0 increasing (`direction` > 0) or decreasing, until it leaves the
    /// bounds, reaches `max_points` or a step fails at the minimum step.
    ///
    /// Each step holds fixed the coordinate, k0 or β, along which the
    /// branch's tangent moves most, so a fold (a zero of dk0/dβ, for a
    /// lossless mode a zero-group-velocity point) is crossed at fixed β and
    /// the branch continues back in k0. Every station counts the roots in its
    /// corrector chart and accepts only one, as in
    /// [`ModePoint::continue_with`]. Folds are reported with their location,
    /// the station there and the points on both sides.
    pub fn continue_through_folds_with<B, E>(
        &self,
        backend: B,
        direction: f64,
        settings: &ArclengthSettings,
    ) -> Result<ArclengthTrace, BranchFailure>
    where
        B: Backend<p::J, ComplexPlane, Entries = E> + ModalDerivativeBackend + Clone,
        E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<p::J>>,
    {
        let s = &settings.branch;
        let positive = |v: f64| v.is_finite() && v > 0.;
        let interval = |[lo, hi]: [f64; 2]| lo.is_finite() && hi.is_finite() && lo < hi;
        if !(direction == 1. || direction == -1.)
            || !positive(settings.step)
            || !positive(settings.frequency_radius)
            || !positive(s.radius)
            || !positive(s.minimum_step)
            || s.minimum_step > settings.step
            || !positive(settings.fold_tolerance)
            || !interval(settings.k0_bounds)
            || !interval(settings.beta_bounds)
            || s.slope.validate().is_err()
            || self.backend() != std::any::type_name::<B>()
        {
            return Err(BranchFailure::InvalidRequest);
        }
        let config = s
            .refinement
            .resolve(self.root.provenance().wavenumber())
            .validate()
            .map_err(|_| BranchFailure::InvalidRequest)?;

        let mut trace = ArclengthTrace {
            points: vec![self.clone()],
            controls: vec![],
            folds: vec![],
            retries: vec![],
            end: ArclengthEnd::MaxPoints,
        };
        let mut tangent = self.tangent(backend.clone(), s)?;
        let a = tangent.direction[0];
        let mut orientation = if a != 0. { direction * a.signum() } else { 1. };
        let mut step = settings.step;
        let mut retries = s.retries;
        while trace.points.len() <= settings.max_points {
            let here = trace.points.last().unwrap().clone();
            let [a, b] = tangent.direction.map(|t| t * orientation);
            let (k0, beta) = here.coordinates()?;
            let (control, target, prediction) = if a.abs() >= b.abs() {
                let slope = tangent.beta_per_k0.ok_or_else(|| {
                    BranchFailure::Derivative("no slope dβ/dk0 off a fold".into())
                })?;
                let delta = step * a;
                (Control::K0, k0.re + delta, beta + slope * delta)
            } else {
                let slope = tangent.k0_per_beta.ok_or_else(|| {
                    BranchFailure::Derivative("no slope dk0/dβ near a fold".into())
                })?;
                let delta = step * b;
                (Control::Beta, beta.re + delta, k0 + slope * delta)
            };
            let station = here.station(
                backend.clone(),
                here.station_for(control, target, prediction, settings),
                config,
            );
            let next = match station.and_then(|q| Ok((q.tangent(backend.clone(), s)?, q))) {
                Ok(next) => next,
                Err(cause) => {
                    if retries == 0 || step * 0.5 < s.minimum_step {
                        trace.end = ArclengthEnd::Failed(cause);
                        return Ok(trace);
                    }
                    retries -= 1;
                    trace.retries.push((control, target, cause));
                    step *= 0.5;
                    continue;
                }
            };
            let (next_tangent, point) = next;
            // Keep the orientation: the tangent turns continuously.
            let [na, nb] = next_tangent.direction;
            let next_orientation = if na * a + nb * b >= 0. { 1. } else { -1. };
            if (na * next_orientation).signum() != a.signum() && a != 0. && na != 0. {
                match self.locate_fold(
                    backend.clone(),
                    [&here, &point],
                    [tangent, next_tangent],
                    settings,
                    config,
                ) {
                    Ok(fold) => trace.folds.push(fold),
                    Err(cause) => {
                        // The fold lies between the last two points; keep them.
                        trace.controls.push(control);
                        trace.points.push(point);
                        trace.end = ArclengthEnd::Failed(cause);
                        return Ok(trace);
                    }
                }
            }
            trace.controls.push(control);
            trace.points.push(point);
            tangent = next_tangent;
            orientation = next_orientation;
            step = (step * 2.).min(settings.step);
            retries = s.retries;
            let (k0, beta) = trace.points.last().unwrap().coordinates()?;
            let outside = |v: f64, [lo, hi]: [f64; 2]| v < lo || v > hi;
            if outside(k0.re, settings.k0_bounds) || outside(beta.re, settings.beta_bounds) {
                trace.end = ArclengthEnd::Bounds;
                return Ok(trace);
            }
        }
        Ok(trace)
    }

    fn station_for<'a>(
        &self,
        control: Control,
        target: f64,
        prediction: Complex,
        settings: &'a ArclengthSettings,
    ) -> Station<'a> {
        let direct =
            control == Control::Beta && self.provenance().strategy() != Strategy::ConformalSpectral;
        Station {
            control,
            target,
            prediction,
            radius: if direct {
                settings.frequency_radius
            } else {
                settings.branch.radius
            },
            frequency: Some(&settings.frequency),
            domain_assumption: settings.branch.domain_assumption,
        }
    }

    /// Locate the fold between `sides` by regula falsi (Illinois) on
    /// g(β) = Re dk0/dβ over stations at fixed β.
    fn locate_fold<B, E>(
        &self,
        backend: B,
        sides: [&ModePoint; 2],
        tangents: [Tangent; 2],
        settings: &ArclengthSettings,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<Fold, BranchFailure>
    where
        B: Backend<p::J, ComplexPlane, Entries = E> + ModalDerivativeBackend + Clone,
        E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<p::J>>,
    {
        let s = &settings.branch;
        let slope_of = |t: &Tangent| {
            t.k0_per_beta
                .ok_or_else(|| BranchFailure::Derivative("no slope dk0/dβ at a fold side".into()))
        };
        let side_slopes = [slope_of(&tangents[0])?, slope_of(&tangents[1])?];
        // (β, g, point, slope) at each end of the bracket.
        let mut ends = [0, 1].map(|i| {
            let (_, beta) = sides[i].coordinates().expect("accepted point");
            (beta.re, side_slopes[i].re, sides[i].clone(), side_slopes[i])
        });
        let mut best = ends[0].clone();
        let mut stale = [0usize; 2];
        for _ in 0..80 {
            let width = (ends[1].0 - ends[0].0).abs();
            if width <= settings.fold_tolerance * ends[0].0.abs().max(ends[1].0.abs()) {
                break;
            }
            let weight = |i: usize| if stale[i] >= 2 { 0.5 } else { 1. };
            let (g0, g1) = (ends[0].1 * weight(0), ends[1].1 * weight(1));
            let mut beta = (ends[0].0 * g1 - ends[1].0 * g0) / (g1 - g0);
            if !beta.is_finite() || (beta - ends[0].0) * (beta - ends[1].0) >= 0. {
                beta = 0.5 * (ends[0].0 + ends[1].0);
            }
            // Predict k0 from the nearer end; dk0/dβ is finite through a fold.
            let near = if (beta - ends[0].0).abs() <= (beta - ends[1].0).abs() {
                0
            } else {
                1
            };
            let from = &ends[near].2;
            let (k0, at) = from.coordinates()?;
            let prediction = k0 + ends[near].3 * (beta - at.re);
            let point = from.station(
                backend.clone(),
                from.station_for(Control::Beta, beta, prediction, settings),
                config,
            )?;
            let slope = slope_of(&point.tangent(backend.clone(), s)?)?;
            let g = slope.re;
            let entry = (beta, g, point, slope);
            let replace = if g.signum() == ends[0].1.signum() {
                0
            } else {
                1
            };
            stale[replace] = 0;
            stale[1 - replace] += 1;
            best = entry.clone();
            ends[replace] = entry;
            if g == 0. {
                break;
            }
        }
        let (k0, beta) = best.2.coordinates()?;
        Ok(Fold {
            k0,
            beta,
            point: best.2,
            slope: best.3,
            sides: [sides[0].clone(), sides[1].clone()],
            side_slopes,
            bracket: (ends[1].0 - ends[0].0).abs(),
        })
    }
}
