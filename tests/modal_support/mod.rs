//! Fixtures and independent references for modal bases of walled stacks.
#![allow(dead_code)]
use lamina_core::{
    AnalyticalMaterialStack, Constant, Polarisation, Stack, Transfer2, Wall, WallKind,
};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::{PI, TAU};

pub const TE: Polarisation = Polarisation::TransverseElectric;
pub const TM: Polarisation = Polarisation::TransverseMagnetic;
pub const PEC: WallKind = WallKind::PerfectElectric;
pub const PMC: WallKind = WallKind::PerfectMagnetic;
pub const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
/// Core and cladding relative permittivities; the wavelength is 1 cm.
pub const CORE: f64 = 4.;
pub const CLAD: f64 = 2.25;
pub const K: f64 = TAU;

pub fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

/// A core of thickness `2a` in cladding between walls `gap` beyond each interface.
pub fn slab(a: f64, gap: f64, kind: WallKind) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(2. * a))
        .left_termination(Wall::new(kind, cm(-gap)))
        .right_termination(Wall::new(kind, cm(2. * a + gap)))
        .finalise()
}

/// A core of thickness `2a` centred between walls at ±`half` about the
/// core centre, so cores of different widths share their walls.
pub fn centred(a: f64, half: f64, kind: WallKind) -> AnalyticalMaterialStack<Complex> {
    let gap = half - a;
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(2. * a))
        .left_termination(Wall::new(kind, cm(-gap)))
        .right_termination(Wall::new(kind, cm(2. * a + gap)))
        .finalise()
}

pub fn request(pol: Polarisation, window: ModalWindow) -> ModalBasisRequest {
    ModalBasisRequest::new(cm(1.), ModeFamily::Polarised(pol), window)
}

pub fn dirichlet(kind: WallKind, pol: Polarisation) -> bool {
    (kind == PEC) == (pol == TE)
}

/// sin(x)/x and its continuation, entire in x².
fn sinc(x2: f64, x: f64) -> f64 {
    if x2.abs() < 1e-12 {
        1. - x2 / 6.
    } else if x2 > 0. {
        x.sin() / x
    } else {
        x.sinh() / x
    }
}

/// The even and odd wall conditions of [`slab`] as real functions of β²,
/// entire in the transverse wavenumbers squared, so real (β² > 0) and
/// imaginary (β² < 0) roots need no branch choice.
pub fn condition(kind: WallKind, pol: Polarisation, a: f64, gap: f64, odd: bool, b2: f64) -> f64 {
    let (w1, w2) = if pol == TE {
        (1., 1.)
    } else {
        (1. / CORE, 1. / CLAD)
    };
    let q2 = CORE * K * K - b2;
    let p2 = CLAD * K * K - b2;
    let (q, p) = (q2.abs().sqrt(), p2.abs().sqrt());
    let cos = |x2: f64, x: f64| if x2 >= 0. { x.cos() } else { x.cosh() };
    let (sq, cq) = (a * sinc(q2, q * a), cos(q2, q * a));
    let (sp, cp) = (gap * sinc(p2, p * gap), cos(p2, p * gap));
    // q sin(qa) = q² · a sinc(qa).
    match (odd, dirichlet(kind, pol)) {
        (false, true) => w1 * q2 * sq * sp - w2 * cq * cp,
        (false, false) => w1 * q2 * sq * cp + w2 * p2 * sp * cq,
        (true, true) => w1 * cq * sp + w2 * sq * cp,
        (true, false) => w1 * cq * cp - w2 * p2 * sp * sq,
    }
}

/// Roots β (real, or i|β| when β² < 0) of [`slab`] with β² in `range`, by
/// bisection of sign changes on a fine scan.
pub fn ladder(
    kind: WallKind,
    pol: Polarisation,
    a: f64,
    gap: f64,
    range: [f64; 2],
) -> Vec<Complex> {
    let mut roots = vec![];
    for odd in [false, true] {
        let f = |b2| condition(kind, pol, a, gap, odd, b2);
        let n = 400_000;
        let at = |i: usize| range[0] + (range[1] - range[0]) * i as f64 / n as f64;
        for i in 0..n {
            let (mut lo, mut hi) = (at(i), at(i + 1));
            let (fl, fh) = (f(lo), f(hi));
            assert!(fl != 0. && fh != 0., "a root on a scan point");
            if fl.signum() == fh.signum() {
                continue;
            }
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if f(mid).signum() == fl.signum() {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            let b2 = 0.5 * (lo + hi);
            roots.push(if b2 >= 0. {
                Complex::new(b2.sqrt(), 0.)
            } else {
                Complex::new(0., (-b2).sqrt())
            });
        }
    }
    roots
}

/// Gauss–Legendre nodes and weights on [−1, 1].
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut derivative = 0.;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1., x);
                for k in 2..=n {
                    let k = k as f64;
                    (p0, p1) = (p1, ((2. * k - 1.) * x * p1 - (k - 1.) * p0) / k);
                }
                derivative = n as f64 * (x * p1 - p0) / (x * x - 1.);
                let dx = p1 / derivative;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            (x, 2. / ((1. - x * x) * derivative * derivative))
        })
        .collect()
}

/// The zeros in `window` from ∮ D′/D dβ / 2πi with the Transfer2
/// characteristic, by composite Gauss–Legendre quadrature refined until the
/// count is integer to 1e-8.
pub fn winding_count(
    stack: AnalyticalMaterialStack<Complex>,
    pol: Polarisation,
    window: [[f64; 2]; 2],
) -> usize {
    let problem = PropagationProblem::with_backend(stack, K, pol, Transfer2::new()).unwrap();
    let [re, im] = window;
    let bounds = ConformalRegion::new(re[0], re[1], im[0], im[1]).unwrap();
    let domain = problem.direct_domain(bounds, ASSUME).unwrap();
    let corners = [
        Complex::new(re[0], im[0]),
        Complex::new(re[1], im[0]),
        Complex::new(re[1], im[1]),
        Complex::new(re[0], im[1]),
        Complex::new(re[0], im[0]),
    ];
    let rule = gauss_legendre(16);
    let mut previous = f64::NAN;
    for level in 4..14 {
        let panels = 1 << level;
        let mut integral = Complex::ZERO;
        for edge in corners.windows(2) {
            let (a, b) = (edge[0], edge[1]);
            let h = (b - a) / panels as f64;
            for p in 0..panels {
                let centre = a + h * (p as f64 + 0.5);
                for &(x, w) in &rule {
                    let e = problem.evaluate(&domain, centre + h * (0.5 * x)).unwrap();
                    integral += e.derivative / e.value * (h * (0.5 * w));
                }
            }
        }
        let count = (integral / Complex::new(0., TAU)).re;
        if (count - previous).abs() < 1e-8 && (count - count.round()).abs() < 1e-8 {
            return count.round() as usize;
        }
        previous = count;
    }
    panic!("the test-side winding did not converge");
}

/// Uniform samples of a stack between its walls: the left gap, each layer
/// and the right gap, `n + 1` points each, with their depths in cm.
pub fn wall_to_wall(
    stack: &AnalyticalMaterialStack<Complex>,
    gaps: [f64; 2],
    n: usize,
) -> (lamina_core::FieldSampling<f64>, Vec<f64>) {
    use lamina_core::{ExteriorSampling, FieldSampling, LayerSampling};
    let mut sampling =
        FieldSampling::new().left_exterior(ExteriorSampling::uniform(n + 1, cm(gaps[0])));
    let mut depths: Vec<f64> = (0..=n)
        .map(|i| -gaps[0] + gaps[0] * i as f64 / n as f64)
        .collect();
    let mut left = 0.;
    for (i, layer) in stack.layers_left_to_right().iter().enumerate() {
        let width = layer.thickness().as_centimetres();
        sampling = sampling.layer(i, LayerSampling::uniform_with_boundaries(n + 1, true, true));
        depths.extend((0..=n).map(|j| left + width * j as f64 / n as f64));
        left += width;
    }
    sampling = sampling.right_exterior(ExteriorSampling::uniform(n + 1, cm(gaps[1])));
    depths.extend((0..=n).map(|j| left + gaps[1] * j as f64 / n as f64));
    (sampling, depths)
}

/// Samples of `stack` at depths `z` (cm, stack frame), each from inside the
/// region containing it.
pub fn at(stack: &AnalyticalMaterialStack<Complex>, z: &[f64]) -> lamina_core::FieldSampling<f64> {
    let widths: Vec<_> = stack
        .layers_left_to_right()
        .iter()
        .map(|l| l.thickness().as_centimetres())
        .collect();
    let mut sampling = lamina_core::FieldSampling::new();
    for &z in z {
        let mut left = 0.;
        if z < 0. {
            sampling = sampling.left_exterior(lamina_core::ExteriorSampling::point(cm(-z)));
            continue;
        }
        let mut placed = false;
        for (i, w) in widths.iter().enumerate() {
            if z <= left + w {
                sampling = sampling.layer(i, lamina_core::LayerSampling::offsets([cm(z - left)]));
                placed = true;
                break;
            }
            left += w;
        }
        if !placed {
            sampling = sampling.right_exterior(lamina_core::ExteriorSampling::point(cm(z - left)));
        }
    }
    sampling
}
