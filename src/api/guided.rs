//! Counted enumeration on the real guided sheet. Counts refer to open |β| windows.
//! On a stack walled on both sides every real-β mode in the window is counted,
//! including those below the exterior light lines.
use super::*;
use crate::problem as p;
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, EvaluateMeromorphicMaterial, Polarisation,
    Scalar, Scatter2, WallKind,
};
use lamina_units::Length;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidedDirection {
    Forward,
    Backward,
}
#[derive(Clone, Copy, Debug)]
pub struct GuidedWindow {
    pub wavelength: Length<f64>,
    pub polarisation: Polarisation,
    pub direction: GuidedDirection,
    /// Open interval in inverse centimetres.
    pub abs_beta: [f64; 2],
    /// Lower end of eligible interval is at least this far above the light
    /// line of the open exteriors. A stack walled on both sides has no light
    /// line: its eligible interval is the window itself, below the upper bound.
    pub cutoff_margin: f64,
    /// Distance in phase cycles within which an endpoint count is ambiguous.
    pub phase_tolerance: f64,
    pub max_subdivisions: usize,
    pub refinement: RefinementConfig,
    pub assumption: AnalyticAssumption,
}
#[derive(Debug, thiserror::Error)]
pub enum EnumerationError {
    #[error("invalid guided interval or numerical controls")]
    InvalidRequest,
    #[error(
        "ineligible material in region {0} (left exterior = 0, then layers, then right exterior)"
    )]
    Material(usize),
    #[error("invalid finite-layer thickness at index {0}")]
    Thickness(usize),
    #[error("physical propagation problem: {0}")]
    Problem(String),
    #[error("guided enumeration needs open or walled sides; the {side:?} side is closed by a PML")]
    Termination { side: lamina_core::StackSide },
}
#[derive(Debug)]
pub enum GuidedLimit {
    EndpointPhase,
    PhaseResolution,
    Budget,
    Correction(BranchFailure),
    OutsideBracket,
}
#[derive(Debug)]
pub struct GuidedGap {
    pub abs_beta: [f64; 2],
    pub expected: Option<usize>,
    pub cause: GuidedLimit,
}
#[derive(Debug)]
pub struct GuidedBasis {
    pub request: GuidedWindow,
    pub light_line: f64,
    pub upper_bound: f64,
    pub eligible: Option<[f64; 2]>,
    pub expected: Option<usize>,
    pub modes: Vec<ModePoint>,
    pub unresolved: Vec<GuidedGap>,
}
impl GuidedBasis {
    /// Numerical coverage of the eligible open interval at the stated tolerances.
    pub fn is_complete(&self) -> bool {
        self.expected == Some(self.modes.len()) && self.unresolved.is_empty()
    }
}
#[derive(Clone, Copy)]
struct Medium {
    line: f64,
    weight: f64,
}
/// The condition at one end of the counted interval.
#[derive(Clone, Copy)]
enum End {
    /// The exterior decays away from the stack.
    Open(Medium),
    /// A wall imposing ψ = 0 (Dirichlet) or ∂ψ/∂z = 0.
    Wall { dirichlet: bool },
}
struct Counter {
    k: f64,
    ends: [End; 2],
    /// The exteriors of the conformal chart: the open sides.
    chart: [Medium; 2],
    /// Layers left to right, wall gaps as layers of their exterior material.
    layers: Vec<(Medium, f64)>,
    tol: f64,
}
impl Counter {
    // Phase is atan2(primary, weighted normal derivative / k0). At an
    // eigenvalue the right end's phase is attained modulo π. As |β| increases
    // the crossing index decreases, counting one state per crossing: the
    // separated end conditions keep the phase monotone (Sturm–Liouville).
    fn index(&self, b: f64) -> Option<i64> {
        let alpha = |m: Medium| ((b - m.line) * (b + m.line)).sqrt();
        let (mut y, mut v) = match self.ends[0] {
            End::Open(m) => (1.0_f64, alpha(m) / (m.weight * self.k)),
            End::Wall { dirichlet: true } => (0., 1.),
            End::Wall { dirichlet: false } => (1., 0.),
        };
        let mut phase = y.atan2(v);
        let mut work = phase.abs();
        for &(m, d) in &self.layers {
            let q2 = (m.line - b) * (m.line + b);
            let q = q2.abs().sqrt();
            let t = q * d;
            let w = m.weight * self.k;
            let (c, s, turns) = if t < 1e-6 {
                (1. - q2 * d * d / 2., d * (1. - q2 * d * d / 6.), 0.)
            } else if q2 > 0. {
                let rest = t % PI;
                let turns = ((t - rest) / PI).round();
                (rest.cos(), rest.sin() / q, turns)
            } else {
                // Divide the evanescent propagator by cosh(qd).
                (1., t.tanh() / q, 0.)
            };
            let next_y = c * y + w * s * v;
            let next_v = c * v - q2 * s * y / w;
            let norm = next_y.hypot(next_v);
            let scale = (c * y).abs() + (w * s * v).abs() + (c * v).abs() + (q2 * s * y / w).abs();
            if !norm.is_finite() || norm <= 64. * f64::EPSILON * scale {
                return None;
            }
            let angle = (v * next_y - y * next_v).atan2(y * next_y + v * next_v);
            phase += turns * PI + angle;
            work += t.abs() + angle.abs();
            y = next_y / norm;
            v = next_v / norm;
        }
        let target = match self.ends[1] {
            End::Open(m) => 1_f64.atan2(-alpha(m) / (m.weight * self.k)),
            End::Wall { dirichlet: true } => 0.,
            End::Wall { dirichlet: false } => FRAC_PI_2,
        };
        let cycles = (phase - target) / PI;
        let err = self.tol + 64. * f64::EPSILON * (1. + work);
        if !cycles.is_finite() || (cycles - cycles.round()).abs() <= err {
            None
        } else {
            Some(cycles.floor() as i64)
        }
    }
    fn native(&self, b: f64) -> f64 {
        let decay = |m: Medium| ((b - m.line) * (b + m.line)).sqrt();
        ((decay(self.chart[0]) + decay(self.chart[1])) / (2. * self.k)).ln()
    }
}
/// Enumerate one polarization and direction using the checked Scatter2 characteristic.
pub fn enumerate_guided_modes(
    stack: AnalyticalMaterialStack<Complex>,
    req: GuidedWindow,
) -> Result<GuidedBasis, EnumerationError> {
    search_guided_modes(stack, req, None).map(|(basis, _)| basis)
}

pub(super) fn search_guided_modes(
    stack: AnalyticalMaterialStack<Complex>,
    req: GuidedWindow,
    predictions: Option<&[f64]>,
) -> Result<(GuidedBasis, bool), EnumerationError> {
    let k = TAU / req.wavelength.as_centimetres();
    let [lo, hi] = req.abs_beta;
    if !(k.is_finite()
        && k > 0.
        && lo.is_finite()
        && hi.is_finite()
        && lo >= 0.
        && lo < hi
        && req.cutoff_margin.is_finite()
        && req.cutoff_margin > 0.
        && req.phase_tolerance > 0.
        && req.phase_tolerance < 0.25)
    {
        return Err(EnumerationError::InvalidRequest);
    }
    let cfg = crate::refinement::RefinementConfig::from(req.refinement)
        .validate()
        .map_err(|_| EnumerationError::InvalidRequest)?;
    let medium = |mat: &AnalyticalMaterialHandle<Complex>, region| {
        let at = Scalar(k.into());
        let e = mat.evaluate_relative_permittivity_complex(at);
        let mu = mat.evaluate_relative_permeability_complex(at);
        let line = k * (e.re * mu.re).sqrt();
        if !line.is_finite() || line <= 0. || e.im != 0. || mu.im != 0. || e.re <= 0. || mu.re <= 0.
        {
            return Err(EnumerationError::Material(region));
        }
        Ok(Medium {
            line,
            weight: match req.polarisation {
                Polarisation::TransverseElectric => mu.re,
                Polarisation::TransverseMagnetic => e.re,
            },
        })
    };
    let exteriors = [
        medium(stack.left_exterior(), 0)?,
        medium(stack.right_exterior(), stack.len() + 1)?,
    ];
    let walls = p::walls(&stack).map_err(|side| EnumerationError::Termination { side })?;
    let ends = [0, 1].map(|side| match walls[side] {
        None => End::Open(exteriors[side]),
        Some((kind, _)) => End::Wall {
            dirichlet: (kind == WallKind::PerfectElectric)
                == (req.polarisation == Polarisation::TransverseElectric),
        },
    });
    // The light line and its margin come from the open sides only.
    let open: Vec<_> = (0..2).filter(|&side| walls[side].is_none()).collect();
    let light_line = open
        .iter()
        .map(|&side| exteriors[side].line)
        .fold(0., f64::max);
    let mut upper_bound = exteriors[0].line.max(exteriors[1].line);
    let gap = |side: usize| walls[side].map(|(_, gap)| (exteriors[side], gap));
    let mut layers: Vec<_> = gap(0).into_iter().collect();
    for (i, layer) in stack.layers_left_to_right().iter().enumerate() {
        let m = medium(layer.material(), i + 1)?;
        let d = layer.thickness().as_centimetres();
        if !d.is_finite() || d < 0. {
            return Err(EnumerationError::Thickness(i));
        }
        layers.push((m, d));
        upper_bound = upper_bound.max(m.line);
    }
    layers.extend(gap(1));
    let margin = if open.is_empty() {
        0.
    } else {
        req.cutoff_margin
    };
    let (lo, hi) = (lo.max(light_line + margin), hi.min(upper_bound));
    let mut report = GuidedBasis {
        request: req,
        eligible: (lo < hi).then_some([lo, hi]),
        expected: Some(0),
        light_line,
        upper_bound,
        modes: vec![],
        unresolved: vec![],
    };
    if lo >= hi {
        return Ok((report, false));
    }
    let counter = Counter {
        k,
        ends,
        chart: match *open {
            [side] => [exteriors[side]; 2],
            _ => exteriors,
        },
        layers,
        tol: req.phase_tolerance,
    };
    let mut budget = req.max_subdivisions;
    let (Some(low), Some(high)) = (counter.index(lo), counter.index(hi)) else {
        report.expected = None;
        report.unresolved.push(GuidedGap {
            abs_beta: [lo, hi],
            expected: None,
            cause: GuidedLimit::EndpointPhase,
        });
        return Ok((report, false));
    };
    if low < high {
        report.expected = None;
        report.unresolved.push(GuidedGap {
            abs_beta: [lo, hi],
            expected: None,
            cause: GuidedLimit::PhaseResolution,
        });
        return Ok((report, false));
    }
    report.expected = Some((low - high) as usize);
    if low == high {
        return Ok((report, false));
    }
    let problem = p::InPlaneProblem::in_plane(stack, k, req.polarisation, Scatter2::new())
        .map_err(|e| EnumerationError::Problem(e.to_string()))?;
    let correct = |l: f64, h: f64| {
        let mid = l + 0.5 * (h - l);
        let sign = if req.direction == GuidedDirection::Forward {
            1.
        } else {
            -1.
        };
        let accepted = (|| {
            let fail = |e: String| BranchFailure::Domain(e);
            if open.is_empty() {
                // Walls on both sides: search β directly.
                let rad = (h - l) * 0.5;
                let [a, b] = if sign > 0. { [l, h] } else { [-h, -l] };
                let bounds =
                    ConformalRegion::new(a, b, -rad, rad).map_err(|e| fail(e.to_string()))?;
                let d = problem
                    .direct_domain(bounds, req.assumption)
                    .map_err(|e| fail(e.to_string()))?;
                return problem.corrected_point(&d, (mid * sign).into(), cfg);
            }
            let z = counter.native(mid);
            let zl = counter.native(l);
            let zh = counter.native(h);
            let rad = (zh - zl) * 0.5;
            let bounds =
                ConformalRegion::new(zl, zh, -rad, rad).map_err(|e| fail(e.to_string()))?;
            let d = problem
                .domain(
                    bounds,
                    ReferenceSelection::Coordinate {
                        reference: z.into(),
                        coordinate: (mid * sign).into(),
                    },
                    Default::default(),
                    req.assumption,
                )
                .map_err(|e| fail(e.to_string()))?;
            problem.corrected_point(&d, z.into(), cfg)
        })();
        match accepted {
            Ok((root, sig, enclosure))
                if sig.k_parallel.re * sign > l
                    && sig.k_parallel.re * sign < h
                    && sig.k_parallel.im.abs() <= 1e-9 * mid
                    && sig
                        .exteriors()
                        .iter()
                        .flatten()
                        .all(|r| r.im > 0. && r.re.abs() <= 1e-8 * r.im) =>
            {
                Ok(ModePoint {
                    root,
                    signature: Ok(sig),
                    enclosure,
                    chart: 0,
                    candidate: 0,
                    backend: std::any::type_name::<Scatter2>(),
                })
            }
            Ok(_) => Err(GuidedLimit::OutsideBracket),
            Err(e) => Err(GuidedLimit::Correction(e)),
        }
    };
    if let Some(predictions) = predictions
        && predictions.len() == (low - high) as usize
    {
        let local = (|| {
            let mut roots = Vec::with_capacity(predictions.len());
            let mut ordinals = std::collections::BTreeSet::new();
            for &prediction in predictions {
                let b = prediction.abs();
                if !(lo < b && b < hi) {
                    return None;
                }
                let mut radius = 1e-7 * b;
                let mut bracket = None;
                for _ in 0..18 {
                    let l = (b - radius).max(lo);
                    let h = (b + radius).min(hi);
                    if let (Some(a), Some(c)) = (counter.index(l), counter.index(h)) {
                        if a - c > 1 {
                            break;
                        }
                        if a - c == 1 {
                            bracket = Some((l, h, a));
                            break;
                        }
                    }
                    radius *= 2.0;
                }
                let (l, h, ordinal) = bracket?;
                if !ordinals.insert(ordinal) {
                    return None;
                }
                roots.push(correct(l, h).ok()?);
            }
            roots.sort_by(|a, b| {
                b.physical_coordinate()
                    .re
                    .abs()
                    .total_cmp(&a.physical_coordinate().re.abs())
            });
            Some(roots)
        })();
        if let Some(roots) = local {
            report.modes = roots;
            return Ok((report, true));
        }
    }
    let mut pending = vec![(lo, hi, low, high)];
    while let Some((l, h, a, b)) = pending.pop() {
        let count = (a - b) as usize;
        if count == 0 {
            continue;
        }
        let gap = |cause| GuidedGap {
            abs_beta: [l, h],
            expected: Some(count),
            cause,
        };
        let mid = l + 0.5 * (h - l);
        if count == 1 && h - l <= 1e-6 * mid {
            match correct(l, h) {
                Ok(point) => report.modes.push(point),
                Err(cause) => report.unresolved.push(gap(cause)),
            }
            continue;
        }
        if budget == 0 {
            report.unresolved.push(gap(GuidedLimit::Budget));
            continue;
        }
        budget -= 1;
        let split = [0.5, 0.47, 0.53].into_iter().find_map(|f| {
            let x = l + (h - l) * f;
            if x <= l || x >= h {
                return None;
            }
            counter
                .index(x)
                .filter(|i| *i >= b && *i <= a)
                .map(|i| (x, i))
        });
        if let Some((x, i)) = split {
            pending.extend([(l, x, a, i), (x, h, i, b)]);
        } else {
            report.unresolved.push(gap(GuidedLimit::PhaseResolution));
        }
    }
    Ok((report, false))
}
