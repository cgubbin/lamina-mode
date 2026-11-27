//! The guided β chart of a stack with full-tensor regions.
//!
//! On the real guided sheet every open exterior's normal wavenumbers are
//! non-real: two decay away from the stack (outgoing) and two grow (incoming).
//! In units of k0, with ξ = β/k0 and n the normal wavenumber, they are the
//! roots of the exterior's quartic `p(n) = det [[K, −μ], [ε, K]]`, `K` the
//! cross product with `(ξ, 0, n)`.
//!
//! A separable exterior (no `xy`, `yx`, `yz` or `zy` entries) factorises into
//! one quadratic per polarisation. With `A` the inverse of the `xz` block of ε
//! (TM) or μ (TE) and `v` = μ_yy (TM) or ε_yy (TE),
//!
//! ```text
//! a n² − b ξ n + c ξ² = v,   a = A_xx, b = A_xz + A_zx, c = A_zz,
//! n = (± b ξ + i √w) / 2a,   w = (4ac − b²) ξ² − 4av,
//! ```
//!
//! outward (`+` on the right, `−` on the left, where the outgoing root is
//! `−n`), with its branch point (the channel's light line) at
//! `ξ² = 4av / (4ac − b²)`: `ε_zz μ_yy` (TM) or `ε_yy μ_zz` (TE) for a
//! symmetric block. Any other exterior's roots are found numerically, the
//! quartic's coefficients by a discrete Fourier transform of `p` on a circle.
//! Such an exterior is refused when it is open ([`GuidedExteriors::new`]):
//! the 4×4 backends' reference basis of its incoming waves can lose rank at
//! a real β above the light line, which no sampled check excludes. The
//! numerical model remains for root rates along directions that couple TE
//! and TM.
//!
//! The chart is linear: β itself, in cm⁻¹, at fixed real k0, with each open
//! side's outward wavenumbers continued as functions of β. A rectangle is a
//! counting region only if, on samples of its boundary and its real
//! midline, every exterior keeps two roots strictly in each half plane, the
//! outgoing roots stay apart from the incoming ones, and the exterior
//! reference basis of the 4×4 backends stays conditioned.
use super::frequency::{contains, halo};
use super::material::{self, Tensor};
use super::*;
use crate::conformal::ConformalPoint;
use lamina_core::{
    CanonicalCoordinates, ExteriorBranch, ExteriorWavevectors, ScalarAlgebra, SeedJet, StackRegion,
};
use nalgebra::{Matrix4, Matrix6};
use ndarray::arr0;
use std::f64::consts::TAU;

/// A cause for which a β rectangle cannot carry a hybrid count, naming the side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GuidedRegionError {
    #[error("β lies outside the chart region")]
    OutsideRegion,
    #[error("cannot construct a finite β evaluation halo")]
    InvalidMargin,
    /// An outgoing root of the exterior meets an incoming one: an exterior
    /// branch point on or near the region.
    #[error("the {side:?} exterior has a branch point on or near the region")]
    BranchClearance { side: StackSide },
    /// An exterior root crosses the real axis: the outgoing pair is not
    /// continuous over the region.
    #[error("the {side:?} exterior's outgoing pair is not continuous over the region")]
    PairContinuity { side: StackSide },
    /// The exterior reference basis of the 4×4 backends collapses: a spurious
    /// zero of the characteristic.
    #[error("the {side:?} exterior's reference basis loses conditioning on the region")]
    GaugeDegenerate { side: StackSide },
}

/// Recorded conditions of a counting region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuidedRegionEvidence {
    /// Half-height of the rectangle about the real axis, cm⁻¹.
    pub half_height: f64,
    /// Least distance between an outgoing and an incoming exterior root,
    /// relative to the roots' scale, over the samples.
    pub exterior_separation: f64,
    /// Least |Im n| of an exterior root relative to the roots' scale.
    pub pair_margin: f64,
    /// Least σ_min/σ_max of the exterior reference rows of the incoming
    /// waves; one for a separable exterior, whose waves are TE and TM.
    pub gauge_conditioning: f64,
    pub samples: usize,
}

/// Separation, pair margin and gauge conditioning below which a region is refused.
const SEPARATION_FLOOR: f64 = 1e-6;
const PAIR_FLOOR: f64 = 1e-9;
const GAUGE_FLOOR: f64 = 1e-8;

/// One scalar channel of a separable region: `a n² − b ξ n + c ξ² = v`.
#[derive(Clone, Copy, Debug)]
struct Quadratic {
    a: C,
    b: C,
    c: C,
    v: C,
}

impl Quadratic {
    fn of(tensors: &[Tensor; 2], polarisation: Polarisation) -> Self {
        let (block, v) = match polarisation {
            Polarisation::TransverseMagnetic => (&tensors[0], tensors[1][1][1]),
            Polarisation::TransverseElectric => (&tensors[1], tensors[0][1][1]),
        };
        let det = block[0][0] * block[2][2] - block[0][2] * block[2][0];
        Self {
            a: block[2][2] / det,
            b: -(block[0][2] + block[2][0]) / det,
            c: block[0][0] / det,
            v,
        }
    }

    /// `ξ²` of the branch point.
    fn line_squared(&self) -> C {
        4.0 * self.a * self.v / (4.0 * self.a * self.c - self.b * self.b)
    }

    /// The outward root and its rate in ξ; `sign` is +1 on the right, −1 on the left.
    fn outward(&self, xi: C, sign: f64) -> (C, C) {
        let d = 4.0 * self.a * self.c - self.b * self.b;
        let root = (d * xi * xi - 4.0 * self.a * self.v).sqrt();
        let two_a = 2.0 * self.a;
        (
            (sign * self.b * xi + C::I * root) / two_a,
            (sign * self.b + C::I * xi * d / root) / two_a,
        )
    }

    /// The incoming root as an eigenvalue: the other root of the quadratic.
    fn incoming(&self, xi: C, sign: f64) -> C {
        let d = 4.0 * self.a * self.c - self.b * self.b;
        let root = (d * xi * xi - 4.0 * self.a * self.v).sqrt();
        (self.b * xi - sign * C::I * root) / (2.0 * self.a)
    }
}

/// `det [[K, −μ], [ε, K]]` at `(ξ, n)`.
fn maxwell(tensors: &[Tensor; 2], xi: C, n: C) -> Matrix6<C> {
    let k = [
        [C::ZERO, -n, C::ZERO],
        [n, C::ZERO, -xi],
        [C::ZERO, xi, C::ZERO],
    ];
    Matrix6::from_fn(|row, column| match (row / 3, column / 3) {
        (0, 0) | (1, 1) => k[row % 3][column % 3],
        (0, 1) => -tensors[1][row % 3][column % 3],
        _ => tensors[0][row % 3][column % 3],
    })
}

/// The Taylor coefficients 0..=4 of `f` about the origin from eight samples
/// on a circle of radius `r`.
fn taylor(f: impl Fn(C) -> C, r: f64) -> [C; 5] {
    let samples: Vec<C> = (0..8)
        .map(|j| f(C::from_polar(r, TAU * j as f64 / 8.0)))
        .collect();
    std::array::from_fn(|k| {
        let sum: C = samples
            .iter()
            .enumerate()
            .map(|(j, value)| value * C::from_polar(1.0, -TAU * (j * k) as f64 / 8.0))
            .sum();
        sum / (8.0 * r.powi(k as i32))
    })
}

/// The scale of the roots of a general exterior's quartic at ξ.
fn root_scale(tensors: &[Tensor; 2], xi: C) -> f64 {
    1.0 + xi.norm() + norm(&tensors[0]).max(norm(&tensors[1]))
}

/// The four roots of a general exterior's quartic at ξ.
fn quartic_roots(tensors: &[Tensor; 2], xi: C) -> Option<[C; 4]> {
    let p = |n: C| maxwell(tensors, xi, n).determinant();
    let coefficients = taylor(p, root_scale(tensors, xi));
    let lead = coefficients[4];
    if lead.norm() == 0.0 || !lead.re.is_finite() || !lead.im.is_finite() {
        return None;
    }
    let companion = Matrix4::from_fn(|row, column| {
        if row == 0 {
            -coefficients[3 - column] / lead
        } else if row == column + 1 {
            C::ONE
        } else {
            C::ZERO
        }
    });
    let roots = companion.schur().eigenvalues()?;
    let value = |n: C| {
        coefficients
            .iter()
            .rev()
            .fold(C::ZERO, |acc, c| acc * n + c)
    };
    let slope = |n: C| {
        (1..5)
            .rev()
            .fold(C::ZERO, |acc, k| acc * n + coefficients[k] * k as f64)
    };
    let mut out = [C::ZERO; 4];
    for (slot, mut n) in out.iter_mut().zip(roots.iter().copied()) {
        for _ in 0..3 {
            let d = slope(n);
            let step = value(n) / d;
            if !step.re.is_finite() || !step.im.is_finite() {
                break;
            }
            n -= step;
        }
        *slot = n;
    }
    out.iter()
        .all(|n| n.re.is_finite() && n.im.is_finite())
        .then_some(out)
}

/// `dn/dξ = −(∂p/∂ξ)/(∂p/∂n)` at a simple root `n`.
fn root_rate(tensors: &[Tensor; 2], xi: C, n: C) -> C {
    let r = root_scale(tensors, xi);
    let along_xi = taylor(|t| maxwell(tensors, xi + t, n).determinant(), r)[1];
    let along_n = taylor(|t| maxwell(tensors, xi, n + t).determinant(), r)[1];
    -along_xi / along_n
}

/// Two roots closer than this, relative to the roots' scale, are treated
/// as one double root: only their symmetric functions enter the 4×4
/// backends' exterior projector, and those are analytic.
const DOUBLE_ROOT: f64 = 1e-6;

fn norm(t: &Tensor) -> f64 {
    t.iter().flatten().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}

/// The xy, yx, yz and zy entries of both tensors vanish to round-off of
/// their norm. At an isotropic exterior's light line four roots meet, so
/// round-off there would move the numerical roots by its fourth root.
fn separable(tensors: &[Tensor; 2]) -> bool {
    tensors.iter().all(|t| {
        [(0, 1), (1, 0), (1, 2), (2, 1)]
            .into_iter()
            .all(|(r, c)| t[r][c].norm() <= 64.0 * f64::EPSILON * norm(t))
    })
}

/// One open exterior.
#[derive(Clone, Debug)]
enum Exterior {
    Separable {
        te: Quadratic,
        tm: Quadratic,
        tensors: Box<[Tensor; 2]>,
    },
    General(Box<[Tensor; 2]>),
}

/// The waves of one side at one ξ: outward roots with their rates, the
/// incoming roots, and the reference-basis conditioning.
struct Waves {
    outgoing: Vec<(C, C)>,
    incoming: Vec<C>,
    gauge: f64,
}

impl Exterior {
    fn new(tensors: [Tensor; 2]) -> Self {
        if separable(&tensors) {
            Self::Separable {
                te: Quadratic::of(&tensors, Polarisation::TransverseElectric),
                tm: Quadratic::of(&tensors, Polarisation::TransverseMagnetic),
                tensors: Box::new(tensors),
            }
        } else {
            Self::General(Box::new(tensors))
        }
    }

    /// The waves of `channel` at ξ on the side with outward `sign`.
    fn waves(&self, xi: C, sign: f64, channel: Channel) -> Option<Waves> {
        match self {
            Self::Separable { te, tm, .. } => {
                let quadratics: Vec<&Quadratic> = match channel {
                    Channel::TransverseElectric => vec![te],
                    Channel::TransverseMagnetic => vec![tm],
                    Channel::Coupled => vec![te, tm],
                };
                Some(Waves {
                    outgoing: quadratics.iter().map(|q| q.outward(xi, sign)).collect(),
                    // Incoming roots as eigenvalues; outgoing outward values
                    // as eigenvalues are `sign · n`.
                    incoming: quadratics.iter().map(|q| q.incoming(xi, sign)).collect(),
                    gauge: 1.0,
                })
            }
            Self::General(tensors) => {
                let split = |xi: C| {
                    let roots = quartic_roots(tensors, xi)?;
                    // Eigenvalues outgoing on this side: Im n of sign `sign`.
                    let (out, inc): (Vec<C>, Vec<C>) =
                        roots.iter().partition(|n| n.im * sign > 0.0);
                    (out.len() == 2).then_some((out, inc))
                };
                let (out, inc) = split(xi)?;
                let scale = root_scale(tensors, xi);
                let outgoing = if (out[0] - out[1]).norm() <= DOUBLE_ROOT * scale {
                    // A double root: its mean and the mean's rate.
                    let mean = |xi: C| {
                        split(xi).map_or(C::new(f64::NAN, 0.0), |(o, _)| 0.5 * (o[0] + o[1]))
                    };
                    let rate = taylor(|t| mean(xi + t), 1e-3 * scale)[1];
                    let n = 0.5 * (out[0] + out[1]);
                    vec![(sign * n, sign * rate); 2]
                } else {
                    out.iter()
                        .map(|&n| (sign * n, sign * root_rate(tensors, xi, n)))
                        .collect()
                };
                if outgoing
                    .iter()
                    .any(|(_, r)| !r.re.is_finite() || !r.im.is_finite())
                {
                    return None;
                }
                Some(Waves {
                    outgoing,
                    gauge: gauge(tensors, xi, [inc[0], inc[1]], scale),
                    incoming: inc,
                })
            }
        }
    }

    fn tensors(&self) -> &[Tensor; 2] {
        match self {
            Self::Separable { tensors, .. } | Self::General(tensors) => tensors,
        }
    }

    /// ξ² of each channel's branch point, for a separable exterior.
    fn lines(&self, channel: Channel) -> Option<Vec<C>> {
        match self {
            Self::Separable { te, tm, .. } => Some(match channel {
                Channel::TransverseElectric => vec![te.line_squared()],
                Channel::TransverseMagnetic => vec![tm.line_squared()],
                Channel::Coupled => vec![te.line_squared(), tm.line_squared()],
            }),
            Self::General(_) => None,
        }
    }
}

/// σ_min/σ_max of the `(E_x, −Z₀H_x)` rows of the two incoming waves, each
/// of unit tangential norm: zero exactly when the outgoing projection of the
/// reference pair `(E_y, Z₀H_y)` is singular.
fn gauge(tensors: &[Tensor; 2], xi: C, incoming: [C; 2], scale: f64) -> f64 {
    // A null vector of the Maxwell matrix at `n`, the `rank`-th smallest.
    let row = |n: C, rank: usize| {
        let svd = maxwell(tensors, xi, n).svd(false, true);
        let Some(v_t) = svd.v_t else {
            return [C::ZERO; 2];
        };
        let mut order: Vec<usize> = (0..6).collect();
        order.sort_by(|&a, &b| svd.singular_values[a].total_cmp(&svd.singular_values[b]));
        let f: Vec<C> = (0..6).map(|k| v_t[(order[rank], k)].conj()).collect();
        // ψ = (E_x, η_y, E_y, −η_x).
        let psi = [f[0], f[4], f[1], -f[3]];
        let size = psi.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if size == 0.0 {
            return [C::ZERO; 2];
        }
        [psi[0] / size, psi[3] / size]
    };
    // A double root has a two-dimensional null space.
    let rows = if (incoming[0] - incoming[1]).norm() <= DOUBLE_ROOT * scale {
        let n = 0.5 * (incoming[0] + incoming[1]);
        [row(n, 0), row(n, 1)]
    } else {
        incoming.map(|n| row(n, 0))
    };
    let m = nalgebra::Matrix2::new(rows[0][0], rows[1][0], rows[0][1], rows[1][1]);
    let s = m.singular_values();
    if s.max() == 0.0 {
        0.0
    } else {
        s.min() / s.max()
    }
}

/// Outward wavenumbers of one side with their β-rates.
type Waves2 = Vec<(C, C)>;

/// The exteriors of a guided chart at fixed real k0: each open side's model,
/// `None` on a side closed by a wall.
#[derive(Clone, Debug)]
pub struct GuidedExteriors {
    k0: f64,
    channel: Channel,
    sides: [Option<Exterior>; 2],
}

const SIGNS: [f64; 2] = [-1.0, 1.0];
const SIDES: [StackSide; 2] = [StackSide::Left, StackSide::Right];

impl GuidedExteriors {
    /// Refuses, before any search, an open exterior whose tensors at `k0` couple
    /// TE and TM (`ConstitutiveError::Unsupported`). Its incoming waves can make
    /// the `(E_y, Z₀H_y)` reference of the 4×4 backends lose rank at a real β
    /// above the light line, a zero of the characteristic for every stack.
    /// A separable exterior's reference loses rank only on its light lines.
    pub(crate) fn new(stack: &ModeStack, k0: f64, channel: Channel) -> Result<Self, ProblemError> {
        let open = open_sides(stack);
        let exterior = |region, material: &Region, open: bool| {
            open.then(|| {
                let [values, _] = material::full_tensors(region, material, C::new(k0, 0.0))?;
                if !separable(&values) {
                    return Err(lamina_core::ConstitutiveError::Unsupported {
                        region,
                        class: material.class(),
                    });
                }
                Ok(Exterior::new(values))
            })
            .transpose()
        };
        Ok(Self {
            k0,
            channel,
            sides: [
                exterior(StackRegion::LeftExterior, stack.left_exterior(), open[0])?,
                exterior(StackRegion::RightExterior, stack.right_exterior(), open[1])?,
            ],
        })
    }

    /// The outward wavenumbers of each open side at β, in cm⁻¹, with their
    /// β-rates; `None` at a point where a side's roots do not split.
    fn outward(&self, beta: C) -> Result<[Option<Waves2>; 2], StackSide> {
        let xi = beta / self.k0;
        let mut out = [None, None];
        for side in 0..2 {
            if let Some(exterior) = &self.sides[side] {
                let waves = exterior
                    .waves(xi, SIGNS[side], self.channel)
                    .ok_or(SIDES[side])?;
                out[side] = Some(
                    waves
                        .outgoing
                        .into_iter()
                        .map(|(n, rate)| (self.k0 * n, rate))
                        .collect(),
                );
            }
        }
        Ok(out)
    }

    /// The exterior branches at the β jet.
    fn branches(&self, beta: &J) -> Result<ExteriorWavevectors<J>, StackSide> {
        let b = beta.value()[()];
        let jet = |(kappa, rate): (C, C)| {
            beta.scale(rate)
                .add(&<J as SeedJet>::constant(arr0(kappa - rate * b)))
        };
        let [left, right] = self.outward(b)?.map(|side| {
            side.map(|mut waves| match waves.len() {
                1 => ExteriorBranch::Scalar(jet(waves.remove(0))),
                _ => ExteriorBranch::Pair([jet(waves[0]), jet(waves[1])]),
            })
        });
        Ok(ExteriorWavevectors::from_branches(left, right))
    }

    /// The β-rate of each open side's outward wavenumbers along a direction
    /// in which k0 changes at `k0_rate`, β at `beta_rate` and the exterior
    /// tensors at `rates` ([ε', μ'] per side), from a Cauchy integral of the
    /// continued roots; a double root enters through its mean.
    pub(crate) fn rates(
        &self,
        beta: C,
        k0_rate: f64,
        beta_rate: C,
        rates: [[Tensor; 2]; 2],
    ) -> Option<[Option<[C; 2]>; 2]> {
        let mut out = [None, None];
        for side in 0..2 {
            let Some(exterior) = &self.sides[side] else {
                continue;
            };
            let base = exterior.tensors();
            let values = |t: C| -> Option<Vec<C>> {
                let k0 = self.k0 + t * k0_rate;
                let tensors = std::array::from_fn(|m| {
                    std::array::from_fn(|r| {
                        std::array::from_fn(|c| base[m][r][c] + t * rates[side][m][r][c])
                    })
                });
                let waves = Exterior::new(tensors).waves(
                    (beta + t * beta_rate) / k0,
                    SIGNS[side],
                    self.channel,
                )?;
                Some(waves.outgoing.iter().map(|(n, _)| k0 * n).collect())
            };
            let centre = values(C::ZERO)?;
            let count = centre.len();
            let scale = centre.iter().map(|v| v.norm()).fold(self.k0, f64::max);
            let double = count == 2 && (centre[0] - centre[1]).norm() <= DOUBLE_ROOT * scale;
            // A radius over which k0, β and the tensors change by at most
            // 1e-5 of themselves: the roots' scale is k0, which an absolute
            // radius would leave unresolved at large k0 or overshoot along a
            // fast direction.
            let speed = (k0_rate.abs() / self.k0)
                .max(beta_rate.norm() / beta.norm().max(f64::MIN_POSITIVE))
                .max(
                    norm(&rates[side][0]).max(norm(&rates[side][1]))
                        / norm(&base[0]).max(norm(&base[1])).max(f64::MIN_POSITIVE),
                );
            if speed == 0.0 {
                out[side] = Some([C::ZERO; 2]);
                continue;
            }
            let r = 1e-5 / speed;
            // The roots at t, ordered as at the centre.
            let at = |t: C, which: usize| -> C {
                let Some(v) = values(t) else {
                    return C::new(f64::NAN, 0.0);
                };
                if v.len() != count {
                    return C::new(f64::NAN, 0.0);
                }
                if double {
                    return 0.5 * (v[0] + v[1]);
                }
                let near = |x: &C| (x - centre[which]).norm();
                *v.iter()
                    .min_by(|a, b| near(a).total_cmp(&near(b)))
                    .unwrap_or(&v[0])
            };
            let rate = [0, 1].map(|which| taylor(|t| at(t, which.min(count - 1)), r)[1]);
            if rate.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                return None;
            }
            out[side] = Some(rate);
        }
        Some(out)
    }

    /// The outward exterior wavenumbers at real or complex β, left then
    /// right; `None` on a closed side.
    pub(crate) fn wavenumbers(&self, beta: C) -> Option<[Option<Vec<C>>; 2]> {
        let out = self.outward(beta).ok()?;
        Some(out.map(|side| side.map(|w| w.into_iter().map(|(k, _)| k).collect())))
    }

    /// The light line of the open sides in cm⁻¹: above it every open
    /// exterior's roots are non-real. Zero with no open side. `upper` bounds
    /// the search for a general exterior; `None` if there is no guided sheet
    /// below it.
    pub(crate) fn light_line(&self, upper: f64) -> Option<f64> {
        let mut line: f64 = 0.0;
        for (side, exterior) in self.sides.iter().enumerate() {
            let Some(exterior) = exterior else { continue };
            let xi = match exterior.lines(self.channel) {
                Some(lines) => lines
                    .iter()
                    .map(|l| l.re.max(0.0).sqrt())
                    .fold(0.0, f64::max),
                None => {
                    // Bisect for the lowest ξ above which the roots split.
                    let splits = |xi: f64| {
                        exterior
                            .waves(C::new(xi, 0.0), SIGNS[side], self.channel)
                            .is_some_and(|w| {
                                let scale = w
                                    .outgoing
                                    .iter()
                                    .map(|(n, _)| n.norm())
                                    .chain(w.incoming.iter().map(|n| n.norm()))
                                    .fold(1.0, f64::max);
                                w.outgoing.iter().all(|(n, _)| n.im > DOUBLE_ROOT * scale)
                                    && w.incoming
                                        .iter()
                                        .all(|n| n.im * SIGNS[side] < -DOUBLE_ROOT * scale)
                            })
                    };
                    let top = upper / self.k0;
                    if !splits(top) {
                        return None;
                    }
                    let steps = 512;
                    let mut low = 0.0;
                    for i in (0..steps).rev() {
                        let xi = top * i as f64 / steps as f64;
                        if !splits(xi) {
                            low = xi;
                            break;
                        }
                    }
                    let mut high = top;
                    for i in 0..steps {
                        let xi = top * (i + 1) as f64 / steps as f64;
                        if xi > low && splits(xi) {
                            high = xi;
                            break;
                        }
                    }
                    for _ in 0..80 {
                        let mid = 0.5 * (low + high);
                        if splits(mid) {
                            high = mid;
                        } else {
                            low = mid;
                        }
                    }
                    high
                }
            };
            line = line.max(self.k0 * xi);
        }
        Some(line)
    }

    /// Check a β rectangle `[lo, hi] × [−h, h]` (or its mirror for backward
    /// modes) as a counting region on samples of its boundary and midline.
    pub(crate) fn region(
        &self,
        rect: argloc::Rect<f64>,
    ) -> Result<GuidedRegionEvidence, GuidedRegionError> {
        let per_edge = 64;
        let corners = [
            C::new(rect.x_min, rect.y_min),
            C::new(rect.x_max, rect.y_min),
            C::new(rect.x_max, rect.y_max),
            C::new(rect.x_min, rect.y_max),
        ];
        let mid = 0.5 * (rect.y_min + rect.y_max);
        let points: Vec<C> = (0..4)
            .flat_map(|e| {
                let (a, b) = (corners[e], corners[(e + 1) % 4]);
                (0..per_edge).map(move |i| a + (b - a) * (i as f64 / per_edge as f64))
            })
            .chain((0..=per_edge).map(|i| {
                C::new(
                    rect.x_min + (rect.x_max - rect.x_min) * i as f64 / per_edge as f64,
                    mid,
                )
            }))
            .collect();
        let mut evidence = GuidedRegionEvidence {
            half_height: 0.5 * (rect.y_max - rect.y_min),
            exterior_separation: f64::INFINITY,
            pair_margin: f64::INFINITY,
            gauge_conditioning: 1.0,
            samples: points.len(),
        };
        for (side, exterior) in self.sides.iter().enumerate() {
            let Some(exterior) = exterior else { continue };
            let fault = SIDES[side];
            for &beta in &points {
                let xi = beta / self.k0;
                let waves = exterior
                    .waves(xi, SIGNS[side], self.channel)
                    .ok_or(GuidedRegionError::BranchClearance { side: fault })?;
                // As eigenvalues: outgoing `sign · n`.
                let outgoing: Vec<C> = waves
                    .outgoing
                    .iter()
                    .map(|(n, _)| SIGNS[side] * n)
                    .collect();
                let scale = outgoing
                    .iter()
                    .chain(&waves.incoming)
                    .map(|n| n.norm())
                    .fold(f64::MIN_POSITIVE, f64::max);
                let separation = outgoing
                    .iter()
                    .flat_map(|o| waves.incoming.iter().map(move |i| (o - i).norm()))
                    .fold(f64::INFINITY, f64::min)
                    / scale;
                let margin = outgoing
                    .iter()
                    .map(|n| n.im * SIGNS[side])
                    .chain(waves.incoming.iter().map(|n| -n.im * SIGNS[side]))
                    .fold(f64::INFINITY, f64::min)
                    / scale;
                if separation < SEPARATION_FLOOR {
                    return Err(GuidedRegionError::BranchClearance { side: fault });
                }
                if margin < PAIR_FLOOR {
                    return Err(GuidedRegionError::PairContinuity { side: fault });
                }
                if waves.gauge < GAUGE_FLOOR {
                    return Err(GuidedRegionError::GaugeDegenerate { side: fault });
                }
                evidence.exterior_separation = evidence.exterior_separation.min(separation);
                evidence.pair_margin = evidence.pair_margin.min(margin);
                evidence.gauge_conditioning = evidence.gauge_conditioning.min(waves.gauge);
            }
        }
        Ok(evidence)
    }
}

/// The map of a guided problem: its fixed k0 and exteriors.
#[derive(Clone, Debug)]
pub struct GuidedMap {
    exteriors: Arc<GuidedExteriors>,
}

pub type GuidedProblem<B> = ModalProblem<GuidedMap, Compiled<B>>;
pub type GuidedDomain<S> = ProblemDomain<GuidedMap, S, GuidedChart>;

/// A β rectangle in cm⁻¹ at fixed real k0, with continued exterior roots.
#[derive(Clone, Debug)]
pub struct GuidedChart {
    region: ConformalRegion<f64>,
    exteriors: Arc<GuidedExteriors>,
}

impl AnalyticChart<J> for GuidedChart {
    type Error = GuidedRegionError;
    fn region(&self) -> &ConformalRegion<f64> {
        &self.region
    }
    fn reference(&self) -> C {
        let (x0, x1) = self.region.real_bounds();
        let (y0, y1) = self.region.imaginary_bounds();
        C::new(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    }
    fn reference_coordinate(&self) -> C {
        self.reference()
    }
    fn coordinate_kind(&self) -> CoordinateKind {
        CoordinateKind::Wavenumber
    }
    fn coordinate(&self, beta: C) -> Result<C, GuidedRegionError> {
        if contains(self.region, beta) {
            Ok(beta)
        } else {
            Err(GuidedRegionError::OutsideRegion)
        }
    }
    fn map(&self, beta: J) -> Result<ConformalPoint<J>, GuidedRegionError> {
        self.coordinate(beta.value()[()])?;
        let exterior = self
            .exteriors
            .branches(&beta)
            .map_err(|side| GuidedRegionError::BranchClearance { side })?;
        Ok(ConformalPoint::new(
            CanonicalCoordinates::new(beta.constant(C::new(self.exteriors.k0, 0.0)), beta),
            exterior,
        ))
    }
    fn with_evaluation_margin(&self) -> Result<Self, GuidedRegionError> {
        Ok(Self {
            region: halo(self.region).ok_or(GuidedRegionError::InvalidMargin)?,
            exteriors: self.exteriors.clone(),
        })
    }
}

impl<B> GuidedProblem<B>
where
    B: Backend<J, ComplexPlane>,
{
    /// A problem on the guided sheet at real `k0` for `channel`, for a stack
    /// whose sides are open or walled.
    pub(crate) fn guided(
        stack: ModeStack,
        k0: f64,
        channel: Channel,
        backend: B,
    ) -> Result<Self, ProblemError> {
        if !k0.is_finite() || k0 <= 0.0 {
            return Err(InPlaneMapError::InvalidWavenumber.into());
        }
        check_terminations(&stack)?;
        let exteriors = Arc::new(GuidedExteriors::new(&stack, k0, channel)?);
        let solver = Compiled::compile(&stack, backend, channel)?;
        Ok(Self {
            context: Arc::new(ProblemContext {
                constants: None,
                stack,
                channel,
                coordinate: SearchCoordinate::InPlane { fixed_k0: k0 },
            }),
            map: GuidedMap { exteriors },
            solver,
        })
    }

    pub(crate) fn exteriors(&self) -> &GuidedExteriors {
        &self.map.exteriors
    }
}

impl<S: ModeDeterminantSolver<J> + Clone> ModalProblem<GuidedMap, S> {
    /// Search a β rectangle (cm⁻¹). The region's conditions are checked by
    /// [`GuidedExteriors::region`] before it is counted.
    pub(crate) fn guided_domain(
        &self,
        bounds: ConformalRegion<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<GuidedDomain<S>, GuidedRegionError> {
        let chart = GuidedChart {
            region: bounds,
            exteriors: self.map.exteriors.clone(),
        };
        let reference = chart.reference();
        let search = SingleChartModalSearch::new(ConformalDispersion::<_, _, C, _>::with_chart(
            self.solver.clone(),
            chart,
        ))
        .map_err(|_| GuidedRegionError::InvalidMargin)?;
        let provenance = DomainProvenance {
            domain_identity: Arc::new(()),
            context: self.context.clone(),
            requested: bounds,
            evaluation: *search.dispersion().chart().region(),
            selection: ReferenceSelection::Coordinate {
                reference,
                coordinate: reference,
            },
            reference,
            reference_coordinate: reference,
            assumption,
            radiation: radiation::RadiationProvenance::Unspecified,
            coordinate_kind: CoordinateKind::Wavenumber,
            frequency: None,
        };
        Ok(ProblemDomain { search, provenance })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(e: [f64; 3]) -> Tensor {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| if r == c { C::from(e[r]) } else { C::ZERO })
        })
    }

    /// A uniaxial tensor with ordinary `o`, extraordinary `e` and its optic
    /// axis at angle θ from z in the xz plane (`phi = 0`) or towards y.
    fn rotated(o: f64, e: f64, theta: f64, towards_y: bool) -> Tensor {
        let axis = if towards_y {
            [0.0, theta.sin(), theta.cos()]
        } else {
            [theta.sin(), 0.0, theta.cos()]
        };
        std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                let delta = if r == c { o } else { 0.0 };
                C::from(delta + (e - o) * axis[r] * axis[c])
            })
        })
    }

    #[test]
    fn quartic_roots_match_the_uniaxial_closed_forms() {
        // Optic axis along z: TE n² = ε_o − ξ², TM n² = ε_o (1 − ξ²/ε_e).
        let (o, e) = (2.25, 3.1);
        let tensors = [rotated(o, e, 0.0, false), diagonal([1.0; 3])];
        let general = Exterior::General(Box::new(tensors));
        for xi in [0.3, 1.2, 2.5] {
            let xi = C::from(xi);
            let w = general.waves(xi, 1.0, Channel::Coupled).unwrap();
            let mut expected = [(o - xi * xi).sqrt(), (o * (1.0 - xi * xi / e)).sqrt()].map(|n| {
                if n.im < 0.0 || (n.im == 0.0 && n.re < 0.0) {
                    -n
                } else {
                    n
                }
            });
            expected.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            let mut found: Vec<C> = w.outgoing.iter().map(|(n, _)| *n).collect();
            found.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            if xi.re > o.sqrt() {
                for (f, x) in found.iter().zip(expected) {
                    assert!((f - x).norm() < 1e-12, "{f} {x}");
                }
            }
        }
        // Optic axis in xz at 35°: the extraordinary pair against the
        // separable closed form.
        let tensors = [rotated(o, e, 0.6, false), diagonal([1.0; 3])];
        let general = Exterior::General(Box::new(tensors));
        let closed = Exterior::new(tensors);
        assert!(matches!(closed, Exterior::Separable { .. }));
        for sign in SIGNS {
            for xi in [2.0, 2.6] {
                let xi = C::from(xi);
                let a = general.waves(xi, sign, Channel::Coupled).unwrap();
                let b = closed.waves(xi, sign, Channel::Coupled).unwrap();
                for (n, rate) in &b.outgoing {
                    let (m, r) = a
                        .outgoing
                        .iter()
                        .min_by(|x, y| (x.0 - n).norm().total_cmp(&(y.0 - n).norm()))
                        .unwrap();
                    assert!((m - n).norm() < 1e-12, "{m} {n}");
                    assert!((r - rate).norm() < 1e-9, "{r} {rate}");
                }
            }
        }
    }

    #[test]
    fn separable_branch_points_are_the_closed_form_lines() {
        // Optic axis in xz: the extraordinary line is ξ² = ε_zz (μ = 1).
        let (o, e, theta) = (2.25, 3.1, 0.6);
        let tensors = [rotated(o, e, theta, false), diagonal([1.0; 3])];
        let exterior = Exterior::new(tensors);
        let lines = exterior.lines(Channel::Coupled).unwrap();
        assert!((lines[0] - o).norm() < 1e-14);
        assert!((lines[1] - tensors[0][2][2]).norm() < 1e-14);
        // A rotation out of xz: the numerical light line is the largest
        // real branch point, bracketed by the ordinary and extraordinary.
        let tensors = [rotated(o, e, theta, true), diagonal([1.0; 3])];
        let exteriors = GuidedExteriors {
            k0: 1.0,
            channel: Channel::Coupled,
            sides: [None, Some(Exterior::new(tensors))],
        };
        let line = exteriors.light_line(3.0).unwrap();
        // Optic axis in yz: the extraordinary wave of k = (ξ, 0, n) sees
        // ε_xx = o along x, so its branch point is ξ² = o·... ; the light
        // line is the larger root of the quartic's real discriminant.
        assert!(line >= o.sqrt() - 1e-9 && line <= e.sqrt() + 1e-9, "{line}");
        let probe = |xi: f64| {
            exteriors.sides[1]
                .as_ref()
                .unwrap()
                .waves(C::from(xi), 1.0, Channel::Coupled)
                .is_some()
        };
        assert!(probe(line * (1.0 + 1e-6)));
    }

    #[test]
    fn regions_over_an_exterior_branch_point_are_refused() {
        let rect = |lo: f64, hi: f64, h: f64| argloc::Rect {
            x_min: lo,
            x_max: hi,
            y_min: -h,
            y_max: h,
        };
        let exteriors = |tensors: [Tensor; 2]| GuidedExteriors {
            k0: 1.0,
            channel: Channel::Coupled,
            sides: [None, Some(Exterior::new(tensors))],
        };
        // A separable exterior: across its light line ξ = 1.5 the outgoing
        // pair meets the real axis.
        let separable = exteriors([diagonal([2.25; 3]), diagonal([1.0; 3])]);
        assert!(separable.region(rect(1.6, 2.0, 0.05)).is_ok());
        assert_eq!(
            separable.region(rect(1.4, 2.0, 0.05)),
            Err(GuidedRegionError::PairContinuity {
                side: StackSide::Right
            })
        );
        // A general exterior: about its light line its roots do not split.
        let general = exteriors([rotated(2.25, 3.1, 0.6, true), diagonal([1.0; 3])]);
        let line = general.light_line(3.0).unwrap();
        assert!(general.region(rect(line * 1.05, 2.5, 0.01)).is_ok());
        let refused = general
            .region(rect(line * 0.95, line * 1.05, 0.01))
            .unwrap_err();
        assert!(matches!(
            refused,
            GuidedRegionError::PairContinuity { .. } | GuidedRegionError::BranchClearance { .. }
        ));
    }
}
