//! Open sides: sheets, chart windows and the covering of a β rectangle.
//!
//! A stack with an open side is searched on the conformal chart of its open
//! exteriors. With two open sides β² = k0² [s + e^{2z} + r² e^{−2z}],
//! κ_L = i k0 (e^z − r e^{−z}) and κ_R = i k0 (e^z + r e^{−z}), with
//! r = (χ_L − χ_R)/4 and s = (χ_L + χ_R)/2; with one open side r = 0 and
//! s = χ of that side, so κ = i k0 e^z. z ↦ (β², κ) is periodic in z with
//! period 2πi and uniformises the Riemann surface of the exterior
//! wavevectors: a chart rectangle less than one period tall holds each point
//! of the surface at most once. An open side's exterior field is exp(iκζ) at
//! outward distance ζ: guided modes lie on the proper sheet, Im κ > 0, and
//! leaky modes on the improper one, Im κ < 0.
use super::*;
use crate::problem as p;
use lamina_core::{StackRegion, StackSide};
use lamina_units::Length;
use std::f64::consts::{PI, TAU};

/// A chart window's imaginary extent must stay below 2π less this.
pub const PERIOD_CLEARANCE: f64 = 1e-3;
/// A root whose exterior wavevector on an open side is within this fraction
/// of k0 of zero lies at a branch point: its exterior field is flat and the
/// tail of its self-pairing, i/(2κ), diverges.
pub const BRANCH_POINT_CLEARANCE: f64 = 1e-4;
/// An open tail whose |κ_a + κ_b| (or |κ_a − κ_b*|) is below this fraction of
/// k0 is singular.
pub const TAIL_SINGULAR_FLOOR: f64 = 1e-8;

/// An open side's sheet of the exterior wavevector κ, exterior field exp(iκζ).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExteriorSheet {
    /// Im κ > 0: the field decays away from the stack.
    Proper,
    /// Im κ ≤ 0: the field grows, or stays bounded, away from the stack.
    Improper,
}

impl ExteriorSheet {
    pub fn of(kappa: Complex) -> Self {
        if kappa.im > 0.0 {
            Self::Proper
        } else {
            Self::Improper
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WindowError {
    #[error("invalid rectangle or wavelength")]
    Invalid,
    #[error("a chart window needs a stack with an open side")]
    ClosedStack,
    #[error("a chart window's imaginary extent must stay below one period, 2π")]
    Period,
    #[error("the window holds a zero of β², a branch point of β")]
    CoordinateBranchPoint,
    #[error("the β rectangle meets the cut of the stated sheet on the {side:?} side")]
    CrossesCut { side: StackSide },
    #[error("sheets must be stated exactly for the open sides, and be equal for equal exteriors")]
    Sheets,
    #[error("the open exterior {region:?} is not isotropic")]
    AnisotropicExterior { region: StackRegion },
}

/// The chart window covering a β rectangle on stated sheets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Covering {
    pub window: ModalWindow,
    /// The area of the window's β image less the requested rectangle's, cm^-2.
    pub extra_area_cm2: f64,
}

/// The conformal chart of a stack's open sides at fixed real k0.
#[derive(Clone, Copy, Debug)]
pub(super) struct OpenChart {
    k0: f64,
    s: Complex,
    r: Complex,
    /// Each side's χ = εμ when open.
    chi: [Option<Complex>; 2],
}

impl OpenChart {
    /// `None` without an open side. An open exterior must be isotropic, so
    /// that its channel is unscaled and χ = εμ in both polarisations.
    pub fn new(stack: &ModeStack, k0: f64) -> Result<Option<Self>, WindowError> {
        let open = p::open_sides(stack);
        for (side, region, material) in [
            (0, StackRegion::LeftExterior, stack.left_exterior()),
            (1, StackRegion::RightExterior, stack.right_exterior()),
        ] {
            if open[side] && material.scalar().is_none() {
                return Err(WindowError::AnisotropicExterior { region });
            }
        }
        let polarisation = lamina_core::Polarisation::TransverseElectric;
        let channels = p::material::exterior_channels(stack, Complex::new(k0, 0.0), polarisation)
            .map_err(|_| WindowError::Invalid)?;
        let chi = channels.map(|c| c.chi());
        let [left, right] = match open {
            [false, false] => return Ok(None),
            [true, false] => [chi[0]; 2],
            [false, true] => [chi[1]; 2],
            _ => chi,
        };
        Ok(Some(Self {
            k0,
            s: (left + right) * 0.5,
            r: (left - right) * 0.25,
            chi: [0, 1].map(|side| open[side].then_some(chi[side])),
        }))
    }

    pub fn beta_squared(&self, z: Complex) -> Complex {
        let t = (z * 2.0).exp();
        self.k0 * self.k0 * (self.s + t + self.r * self.r / t)
    }

    pub fn kappas(&self, z: Complex) -> [Option<Complex>; 2] {
        let (e, f) = (z.exp(), self.r * (-z).exp());
        let ik0 = Complex::new(0.0, self.k0);
        [
            self.chi[0].map(|_| ik0 * (e - f)),
            self.chi[1].map(|_| ik0 * (e + f)),
        ]
    }

    /// The point z, Im z unwrapped towards `near`, whose exterior
    /// wavevectors are `kappa`, if one is.
    fn point(&self, kappa: [Option<Complex>; 2], near: f64) -> Option<Complex> {
        let ik0 = Complex::new(0.0, self.k0);
        let w = match kappa {
            [Some(l), Some(r)] => (l + r) / (ik0 * 2.0),
            [Some(k), None] | [None, Some(k)] => k / ik0,
            [None, None] => return None,
        };
        let z = w.ln();
        let z = Complex::new(z.re, z.im + TAU * ((near - z.im) / TAU).round());
        let consistent = self.kappas(z).iter().zip(kappa).all(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => (a - b).norm() <= 1e-9 * self.k0.max(b.norm()),
            _ => true,
        });
        (z.re.is_finite() && z.im.is_finite() && consistent).then_some(z)
    }

    /// Whether a zero of β² lies in `window`, closed.
    pub fn holds_branch_point(&self, window: Rect<f64>) -> bool {
        let disc = (self.s * self.s - self.r * self.r * 4.0).sqrt();
        [(-self.s + disc) * 0.5, (-self.s - disc) * 0.5]
            .into_iter()
            .filter(|t| t.norm() > 0.0)
            .any(|t| {
                // z = ln(t)/2 + iπk.
                let z = t.ln() * 0.5;
                let first = ((window.y_min - z.im) / PI).ceil();
                window.x_min <= z.re && z.re <= window.x_max && z.im + PI * first <= window.y_max
            })
    }

    /// The bounding box of β over the boundary of `window`, continued
    /// from `start`'s sign at the first corner.
    pub fn beta_image(&self, window: Rect<f64>, start: Option<Complex>) -> Rect<f64> {
        let betas = continued_betas(self, &boundary(window, 64), start);
        bounding_box(&betas)
    }
}

/// The closed boundary of `r`, `n` samples per edge, counter-clockwise.
fn boundary(r: Rect<f64>, n: usize) -> Vec<Complex> {
    let corners = [
        Complex::new(r.x_min, r.y_min),
        Complex::new(r.x_max, r.y_min),
        Complex::new(r.x_max, r.y_max),
        Complex::new(r.x_min, r.y_max),
    ];
    (0..4)
        .flat_map(|edge| {
            let (a, b) = (corners[edge], corners[(edge + 1) % 4]);
            (0..n).map(move |i| a + (b - a) * (i as f64 / n as f64))
        })
        .chain([corners[0]])
        .collect()
}

/// β along a path of z, continuous, starting nearest `start`.
fn continued_betas(chart: &OpenChart, path: &[Complex], start: Option<Complex>) -> Vec<Complex> {
    let mut betas: Vec<Complex> = vec![];
    for &z in path {
        let beta = chart.beta_squared(z).sqrt();
        let previous = betas.last().copied().or(start);
        betas.push(match previous {
            Some(p) if (beta + p).norm() < (beta - p).norm() => -beta,
            _ => beta,
        });
    }
    betas
}

fn bounding_box(points: &[Complex]) -> Rect<f64> {
    points.iter().fold(
        Rect {
            x_min: f64::INFINITY,
            x_max: f64::NEG_INFINITY,
            y_min: f64::INFINITY,
            y_max: f64::NEG_INFINITY,
        },
        |r, z| Rect {
            x_min: r.x_min.min(z.re),
            x_max: r.x_max.max(z.re),
            y_min: r.y_min.min(z.im),
            y_max: r.y_max.max(z.im),
        },
    )
}

/// Whether κ² = `c` − β² reaches [0, ∞) on the rectangle: the cut of both
/// sheets. The cut is an unbounded connected curve, so it meets the
/// rectangle only across its boundary; along each edge Im κ² is linear.
fn meets_cut(c: Complex, corners: [Complex; 4]) -> bool {
    (0..4).any(|edge| {
        let (a, d) = (corners[edge], corners[(edge + 1) % 4] - corners[edge]);
        let kappa2 = |t: f64| c - (a + d * t) * (a + d * t);
        let (constant, slope) = ((c - a * a).im, -2.0 * (a * d).im);
        if slope != 0.0 {
            let t = -constant / slope;
            (0.0..=1.0).contains(&t) && kappa2(t).re >= 0.0
        } else {
            // Im κ² is constant along the edge; Re κ² is quadratic in t.
            let vertex = -(a * d).re / (d * d).re;
            constant == 0.0
                && [0.0, 1.0, vertex]
                    .into_iter()
                    .filter(|t| (0.0..=1.0).contains(t))
                    .any(|t| kappa2(t).re >= 0.0)
        }
    })
}

/// The chart window's period check.
pub(super) fn check_period(window: Rect<f64>) -> Result<(), WindowError> {
    if window.y_max - window.y_min < TAU - PERIOD_CLEARANCE {
        Ok(())
    } else {
        Err(WindowError::Period)
    }
}

impl ModalWindow {
    /// The smallest chart rectangle whose image covers the β rectangle
    /// (cm^-1) on `sheets`, one per open side, `None` on closed sides: the
    /// bounding box of its boundary mapped to z on those sheets. A rectangle
    /// that meets a stated sheet's cut is refused naming the side. The basis
    /// of the covering holds every mode in it, each with its sheet.
    pub fn covering(
        stack: impl Into<ModeStack>,
        wavelength: Length<f64>,
        re_cm: [f64; 2],
        im_cm: [f64; 2],
        sheets: [Option<ExteriorSheet>; 2],
    ) -> Result<Covering, WindowError> {
        let k0 = TAU / wavelength.as_centimetres();
        let finite = re_cm.iter().chain(&im_cm).all(|v| v.is_finite());
        if !(k0.is_finite() && k0 > 0.0 && finite && re_cm[0] < re_cm[1] && im_cm[0] < im_cm[1]) {
            return Err(WindowError::Invalid);
        }
        let chart = OpenChart::new(&stack.into(), k0)?.ok_or(WindowError::ClosedStack)?;
        if (0..2).any(|side| chart.chi[side].is_some() != sheets[side].is_some())
            || (chart.r == Complex::ZERO && sheets[0].is_some() && sheets[1].is_some())
                && sheets[0] != sheets[1]
        {
            return Err(WindowError::Sheets);
        }
        let corners = [
            Complex::new(re_cm[0], im_cm[0]),
            Complex::new(re_cm[1], im_cm[0]),
            Complex::new(re_cm[1], im_cm[1]),
            Complex::new(re_cm[0], im_cm[1]),
        ];
        for (side, which) in [StackSide::Left, StackSide::Right].into_iter().enumerate() {
            if let Some(chi) = chart.chi[side]
                && meets_cut(chi * k0 * k0, corners)
            {
                return Err(WindowError::CrossesCut { side: which });
            }
        }
        let requested = Rect {
            x_min: re_cm[0],
            x_max: re_cm[1],
            y_min: im_cm[0],
            y_max: im_cm[1],
        };
        let mut path: Vec<Complex> = vec![];
        for beta in boundary(requested, 256) {
            let kappa = [0, 1].map(|side| {
                let chi = chart.chi[side]?;
                // The proper root has Im κ ≥ 0: i √(β² − χ k0²), principal.
                let proper = Complex::I * (beta * beta - chi * k0 * k0).sqrt();
                Some(match sheets[side]? {
                    ExteriorSheet::Proper => proper,
                    ExteriorSheet::Improper => -proper,
                })
            });
            let near = path.last().map_or(0.0, |z| z.im);
            path.push(chart.point(kappa, near).ok_or(WindowError::Sheets)?);
        }
        let step = path
            .windows(2)
            .map(|w| (w[1] - w[0]).norm())
            .fold(0.0, f64::max);
        let mut window = bounding_box(&path);
        let pad = 0.5 * step;
        let shift = TAU * (0.5 * (window.y_min + window.y_max) / TAU).round();
        window = Rect {
            x_min: window.x_min - pad,
            x_max: window.x_max + pad,
            y_min: window.y_min - pad - shift,
            y_max: window.y_max + pad - shift,
        };
        check_period(window)?;
        if chart.holds_branch_point(window) {
            return Err(WindowError::CoordinateBranchPoint);
        }
        // The image's area by the shoelace formula of its boundary.
        let betas = continued_betas(&chart, &boundary(window, 1024), None);
        let area = 0.5
            * betas
                .windows(2)
                .map(|w| (w[0].conj() * w[1]).im)
                .sum::<f64>();
        Ok(Covering {
            window: ModalWindow::Chart {
                re: [window.x_min, window.x_max],
                im: [window.y_min, window.y_max],
            },
            extra_area_cm2: area - (re_cm[1] - re_cm[0]) * (im_cm[1] - im_cm[0]),
        })
    }
}
