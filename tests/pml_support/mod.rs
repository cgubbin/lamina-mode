//! Fixtures and independent references for stacks closed by wall-backed PMLs:
//! a symmetric slab, the complex-thickness box characteristic and the open
//! slab's conditions, all independent of lamina.
#![allow(dead_code)]
use lamina_core::{
    AnalyticalMaterialStack, Constant, Pml, Polarisation, Stack, StackSide, StretchProfile,
    Termination, Wall, WallKind,
};
use lamina_mode::{AnalyticAssumption, Complex};
use lamina_units::Length;
use std::f64::consts::TAU;

pub const TE: Polarisation = Polarisation::TransverseElectric;
pub const TM: Polarisation = Polarisation::TransverseMagnetic;
pub const PEC: WallKind = WallKind::PerfectElectric;
pub const PMC: WallKind = WallKind::PerfectMagnetic;
pub const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
/// Core and cladding relative permittivities; the wavelength is 1 cm.
pub const CORE: f64 = 4.;
pub const CLAD: f64 = 2.25;
pub const K: f64 = TAU;
/// Half-width of the core.
pub const A: f64 = 0.35;

pub fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

pub fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// A polynomial profile of order `m` and strength `strength`.
#[derive(Clone, Copy)]
pub struct Profile {
    pub m: f64,
    pub strength: Complex,
}

impl Profile {
    /// s̄ = 1 + c/(m+1), so that τ = d s̄.
    pub fn mean(self) -> Complex {
        1. + self.strength / (self.m + 1.)
    }
    /// ζ̃(ζ) = ζ + c d/(m+1) (ζ/d)^(m+1).
    pub fn stretched(self, zeta: f64, d: f64) -> Complex {
        zeta + self.strength * d / (self.m + 1.) * (zeta / d).powf(self.m + 1.)
    }
}

pub const PROFILE: Profile = Profile {
    m: 2.,
    strength: Complex::new(1., 3.),
};

/// A PML `gap` beyond the interface of `side`, `d` thick.
pub fn pml(side: StackSide, gap: f64, d: f64, profile: Profile, backing: WallKind) -> Pml<f64> {
    let start = match side {
        StackSide::Left => -gap,
        StackSide::Right => 2. * A + gap,
    };
    Pml::new(
        cm(start),
        cm(d),
        StretchProfile::polynomial(profile.m, profile.strength),
        backing,
    )
}

/// The core |z − A| < A in cladding, closed by `left` and `right`.
pub fn slab(
    left: impl Into<Termination<f64>>,
    right: impl Into<Termination<f64>>,
) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(2. * A))
        .left_termination(left)
        .right_termination(right)
        .finalise()
}

/// The core closed on both sides by equal PEC-backed PMLs.
pub fn symmetric(gap: f64, d: f64, profile: Profile) -> AnalyticalMaterialStack<Complex> {
    slab(
        pml(StackSide::Left, gap, d, profile, PEC),
        pml(StackSide::Right, gap, d, profile, PEC),
    )
}

/// The weight w (ψ' carries w = 1/μ in TE, 1/ε in TM) of a material.
pub fn weight(pol: Polarisation, epsilon: f64) -> f64 {
    if pol == TE { 1. } else { 1. / epsilon }
}

pub fn dirichlet(kind: WallKind, pol: Polarisation) -> bool {
    (kind == PEC) == (pol == TE)
}

/// cos(pℓ) and sin(pℓ)/p, entire in p².
pub fn trig(p2: Complex, length: Complex) -> (Complex, Complex) {
    let p = p2.sqrt();
    let sinc = if p == Complex::ZERO {
        length
    } else {
        (p * length).sin() / p
    };
    ((p * length).cos(), sinc)
}

/// The characteristic of a box closed by walls, independent of lamina: the
/// state (ψ, w ψ') from the left wall's condition carried through `regions`
/// (permittivity and complex length) and the right wall's condition applied.
/// Each region is scaled by the β-independent exp(i√ε k0 ℓ), so that long
/// complex lengths stay in range.
pub fn box_value(
    pol: Polarisation,
    walls: [WallKind; 2],
    regions: &[(f64, Complex)],
    beta: Complex,
) -> Complex {
    let mut state = if dirichlet(walls[0], pol) {
        (Complex::ZERO, Complex::ONE)
    } else {
        (Complex::ONE, Complex::ZERO)
    };
    for &(epsilon, length) in regions {
        let w = weight(pol, epsilon);
        let p2 = epsilon * K * K - beta * beta;
        let (cos, sinc) = trig(p2, length);
        let scale = (Complex::I * epsilon.sqrt() * K * length).exp();
        state = (
            scale * (cos * state.0 + sinc * state.1 / w),
            scale * (-w * p2 * sinc * state.0 + cos * state.1),
        );
    }
    if dirichlet(walls[1], pol) {
        state.0
    } else {
        state.1
    }
}

/// The open symmetric slab's even or odd condition with exterior wavevector
/// `kappa` (fields ∝ exp(iκ|z − A|) outside), independent of lamina.
pub fn open_value(pol: Polarisation, odd: bool, beta: Complex, kappa: Complex) -> Complex {
    let (w1, w2) = (weight(pol, CORE), weight(pol, CLAD));
    let q2 = CORE * K * K - beta * beta;
    let (cos, sinc) = trig(q2, c(A, 0.));
    if odd {
        w1 * cos - Complex::I * w2 * kappa * sinc
    } else {
        w1 * q2 * sinc + Complex::I * w2 * kappa * cos
    }
}

/// The guided exterior root κ = i√(β² − εk0²) and the leaky (outgoing,
/// growing) root, the principal √(εk0² − β²) for Re β > 0, Im β > 0.
pub fn guided_kappa(beta: Complex) -> Complex {
    Complex::I * (beta * beta - CLAD * K * K).sqrt()
}
pub fn leaky_kappa(beta: Complex) -> Complex {
    (CLAD * K * K - beta * beta).sqrt()
}

pub fn newton(f: &dyn Fn(Complex) -> Complex, mut z: Complex) -> Option<Complex> {
    for _ in 0..100 {
        let h = 1e-6 * z.norm().max(1.);
        let derivative = (f(z + h) - f(z - h)) / (2. * h);
        let step = f(z) / derivative;
        if !step.re.is_finite() || !step.im.is_finite() {
            return None;
        }
        z -= step;
        if step.norm() <= 1e-15 * z.norm() {
            return Some(z);
        }
    }
    None
}

/// The number of zeros of `f` in a rectangle by the argument principle,
/// independent of argloc.
pub fn winding(f: &dyn Fn(Complex) -> Complex, r: [f64; 4]) -> usize {
    let corners = [c(r[0], r[2]), c(r[1], r[2]), c(r[1], r[3]), c(r[0], r[3])];
    let n = 40_000;
    let mut turn = 0.;
    let mut last = f(corners[0]).arg();
    for edge in 0..4 {
        let (a, b) = (corners[edge], corners[(edge + 1) % 4]);
        for i in 1..=n {
            let arg = f(a + (b - a) * (i as f64 / n as f64)).arg();
            let mut step = arg - last;
            step -= TAU * (step / TAU).round();
            assert!(step.abs() < 1., "the boundary passes too close to a zero");
            turn += step;
            last = arg;
        }
    }
    let count = turn / TAU;
    assert!((count - count.round()).abs() < 1e-6, "{count}");
    count.round() as usize
}

/// The open slab's guided ladder above `lo` and the cladding light line, by
/// bracketing the real guided conditions.
pub fn open_guided(pol: Polarisation, lo: f64) -> Vec<(bool, f64)> {
    let lo = lo.max(CLAD.sqrt() * K * (1. + 1e-9));
    let mut roots = vec![];
    for odd in [false, true] {
        // With κ = iα both conditions are real.
        let f = |b: f64| {
            let beta = c(b, 0.);
            open_value(pol, odd, beta, guided_kappa(beta)).re
        };
        let n = 20_000;
        let hi = CORE.sqrt() * K;
        let at = |i: usize| lo + (hi - lo) * i as f64 / n as f64;
        for i in 0..n {
            let (mut a, mut b) = (at(i), at(i + 1));
            if f(a).signum() == f(b).signum() {
                continue;
            }
            for _ in 0..100 {
                let mid = 0.5 * (a + b);
                if f(mid).signum() == f(a).signum() {
                    a = mid
                } else {
                    b = mid
                }
            }
            roots.push((odd, 0.5 * (a + b)));
        }
    }
    roots
}

/// A thick PML, 20 cm, whose normalisation −Im(κ_ref τ) ≈ −151 nearly
/// cancels |Im κτ| at the fundamental guided mode while |Im κτ| reaches
/// several hundred at 3 k0.
pub const LONG: Profile = Profile {
    m: 2.,
    strength: Complex::new(0., 2.4),
};
pub const LONG_D: f64 = 20.;

/// The core between a thick PML 0.2 cm to its left and a PMC wall 0.3 cm to
/// its right.
pub fn long_pml_and_wall() -> AnalyticalMaterialStack<Complex> {
    slab(
        pml(StackSide::Left, 0.2, LONG_D, LONG, PEC),
        Wall::perfect_magnetic(cm(2. * A + 0.3)),
    )
}

/// The open slab's modes in `window`, labelled, with their exterior κ, and
/// the number of leaky ones: guided by bracketing, leaky by Newton from a
/// grid, counted by the argument principle above the axis.
pub fn open_ladder(
    pol: Polarisation,
    window: [f64; 4],
) -> (Vec<(String, Complex, Complex)>, usize) {
    let mut open: Vec<(String, Complex, Complex)> = open_guided(pol, window[0])
        .into_iter()
        .map(|(odd, b)| {
            let beta = c(b, 0.);
            (
                format!("guided {}", if odd { "odd" } else { "even" }),
                beta,
                guided_kappa(beta),
            )
        })
        .collect();
    let above = [window[0], window[1], 0.05 * K, window[3]];
    let mut leaky = 0;
    for odd in [false, true] {
        let f = |beta| open_value(pol, odd, beta, leaky_kappa(beta));
        let expected = winding(&f, above);
        let mut found: Vec<Complex> = vec![];
        for i in 0..=26 {
            for j in 0..=20 {
                let seed = c(
                    above[0] + (above[1] - above[0]) * i as f64 / 26.,
                    above[2] + (above[3] - above[2]) * j as f64 / 20.,
                );
                if let Some(root) = newton(&f, seed)
                    && root.re > above[0]
                    && root.re < above[1]
                    && root.im > above[2]
                    && root.im < above[3]
                    && found.iter().all(|r| (r - root).norm() > 1e-8)
                {
                    found.push(root);
                }
            }
        }
        assert_eq!(found.len(), expected);
        leaky += expected;
        let parity = if odd { "odd" } else { "even" };
        open.extend(
            found
                .into_iter()
                .map(|beta| (format!("leaky {parity}"), beta, leaky_kappa(beta))),
        );
    }
    (open, leaky)
}
