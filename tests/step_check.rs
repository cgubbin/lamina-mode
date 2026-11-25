//! A completeness and pairing check of two modal bases across a step: two
//! symmetric slabs of different core widths between common PEC walls, joined
//! at x = 0. The tangential fields are expanded in N even modes per side
//! using only the unconjugated pairing matrices of the bases, and the
//! coefficients are compared with the same expansion built from analytic
//! layered-box modes and closed-form overlaps, extrapolated in N. The error
//! must follow the reference's convergence until the floor predicted by the
//! conditioning, κ_N δ, and stay within a factor 10 of max(truncation, κ_N δ).
mod modal_support;
use lamina_core::{Polarisation, Scatter2};
use lamina_mode::*;
use modal_support::*;
use nalgebra::{DMatrix, DVector};

/// W/m per (V/m)² cm.
const C: f64 = 0.01 / 376.730313412;
/// Walls at ±W about the common core centre; core half-widths left and right.
const W: f64 = 0.8;
const CORES: [f64; 2] = [0.3, 0.2];
const N_MAX: usize = 10;
const N_REFERENCE: usize = 120;

/// An even mode of a centred symmetric slab, as Σ c e^{ikz} on [0, a] and [a, W].
struct Even {
    beta: Complex,
    a: f64,
    pieces: [Vec<(Complex, Complex)>; 2],
    /// ψ at the core interface z = a.
    at_interface: Complex,
}

fn even(pol: Polarisation, a: f64, b2: f64) -> Even {
    let i = Complex::I;
    let b2 = Complex::new(b2, 0.);
    let q = (CORE * K * K - b2).sqrt();
    let p = (CLAD * K * K - b2).sqrt();
    let g = W - a;
    let half = Complex::new(0.5, 0.);
    let core = vec![(half, q), (half, -q)];
    let cladding = if pol == TE {
        // A sin(p (W − z)), Dirichlet at the PEC wall.
        let amplitude = (q * a).cos() / (p * g).sin();
        let e = (i * p * W).exp();
        vec![
            (amplitude * e / (2. * i), -p),
            (-amplitude / e / (2. * i), p),
        ]
    } else {
        // A cos(p (W − z)), Neumann at the PEC wall.
        let amplitude = (q * a).cos() / (p * g).cos();
        let e = (i * p * W).exp();
        vec![(amplitude * e * half, -p), (amplitude / e * half, p)]
    };
    Even {
        beta: b2.sqrt(),
        a,
        pieces: [core, cladding],
        at_interface: (q * a).cos(),
    }
}

/// ∫ e^{iκz} over [z0, z1].
fn exponential(kappa: Complex, z0: f64, z1: f64) -> Complex {
    let h = z1 - z0;
    let x = Complex::I * kappa * h;
    let start = (Complex::I * kappa * z0).exp();
    if x.norm() < 1e-4 {
        start * h * (1. + x / 2. + x * x / 6. + x * x * x / 24.)
    } else {
        start * (x.exp() - 1.) / (Complex::I * kappa)
    }
}

/// 2 ∫_0^W ψ_m ψ_n w dz for even modes, with w = 1/ε of `weight`'s slab
/// (TM) or 1 (TE), in closed form piece by piece.
fn overlap(pol: Polarisation, m: &Even, n: &Even, weight: f64) -> Complex {
    let mut edges = [0., m.a, n.a, W];
    edges.sort_by(f64::total_cmp);
    let mut total = Complex::ZERO;
    for e in edges.windows(2).filter(|e| e[1] > e[0]) {
        let mid = 0.5 * (e[0] + e[1]);
        fn piece(mode: &Even, mid: f64) -> &[(Complex, Complex)] {
            &mode.pieces[usize::from(mid > mode.a)]
        }
        let w = if pol == TE {
            1.
        } else if mid < weight {
            1. / CORE
        } else {
            1. / CLAD
        };
        for &(c, k) in piece(m, mid) {
            for &(d, l) in piece(n, mid) {
                total += w * c * d * exponential(k + l, e[0], e[1]);
            }
        }
    }
    2. * total
}

/// The unit unconjugated self-pairing amplitude, with the sign rule at the
/// first interface (ψ is E_y for TE and Z0 H_y for TM).
fn amplitude(pol: Polarisation, m: &Even) -> Complex {
    let n = (Complex::new(2. * K / C, 0.) / (m.beta * overlap(pol, m, m, m.a))).sqrt();
    let l = n * m.at_interface;
    let decisive = if l.re.abs() <= 64. * f64::EPSILON * l.norm() {
        l.im
    } else {
        l.re
    };
    if decisive < 0. { -n } else { n }
}

fn even_ladder(pol: Polarisation, a: f64, count: usize) -> Vec<Even> {
    let gap = W - a;
    let f = |b2| condition(PEC, pol, a, gap, false, b2);
    let top = CORE * K * K;
    let mut roots = vec![];
    let mut hi = top * (1. - 1e-12);
    let step = 0.05;
    while roots.len() < count {
        let lo = hi - step;
        let (fl, fh) = (f(lo), f(hi));
        if fl.signum() != fh.signum() {
            let (mut l, mut h) = (lo, hi);
            for _ in 0..200 {
                let mid = 0.5 * (l + h);
                if f(mid).signum() == fl.signum() {
                    l = mid
                } else {
                    h = mid
                }
            }
            roots.push(even(pol, a, 0.5 * (l + h)));
        }
        hi = lo;
    }
    roots
}

/// Reflection and transmission of the fundamental even mode with N modes.
fn expand(u: &DMatrix<Complex>, v: &DMatrix<Complex>, n: usize) -> Option<[Complex; 2]> {
    let mut a = DVector::zeros(n);
    a[0] = Complex::ONE;
    let t = u.view((0, 0), (n, n)).into_owned().lu().solve(&a)?;
    let r = v.view((0, 0), (n, n)).into_owned() * &t;
    Some([r[0], t[0]])
}

fn condition_number(m: DMatrix<Complex>) -> f64 {
    let s = m.singular_values();
    s.max() / s.min()
}

/// Mirror x → −x of fields [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz].
fn mirror(s: &[Complex; 6]) -> [Complex; 6] {
    [-s[0], s[1], s[2], s[3], -s[4], -s[5]]
}

#[test]
fn step_expansions_converge_to_the_conditioning_floor() {
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        // References: analytic modes and closed-form overlaps.
        let modes = CORES.map(|a| even_ladder(pol, a, N_REFERENCE));
        let amplitudes = modes
            .each_ref()
            .map(|m| m.iter().map(|e| amplitude(pol, e)).collect::<Vec<_>>());
        let entry = |i: usize, j: usize, mirrored: bool| {
            let (f, g) = (&modes[0][i], &modes[1][j]);
            let (jl, jr) = (overlap(pol, f, g, CORES[0]), overlap(pol, f, g, CORES[1]));
            let nn = amplitudes[0][i] * amplitudes[1][j] * (0.25 * C / K);
            match (mirrored, pol == TE) {
                (false, _) => nn * (f.beta * jl + g.beta * jr),
                (true, true) => nn * (f.beta - g.beta) * jl,
                (true, false) => nn * (g.beta * jr - f.beta * jl),
            }
        };
        let reference = |n: usize| {
            let u = DMatrix::from_fn(n, n, |i, j| entry(i, j, false));
            let v = DMatrix::from_fn(n, n, |i, j| entry(i, j, true));
            expand(&u, &v, n).unwrap()
        };
        let limit = reference(N_REFERENCE);

        // lamina: forward and backward bases of each side.
        let decay = 12. * K;
        let margin = 0.05 * K;
        let solve = |a: f64, window: ModalWindow| {
            let basis =
                modal_basis_in_window_with(&backend, centred(a, W, PEC), request(pol, window))
                    .unwrap();
            assert_eq!(
                basis.status,
                BasisStatus::Complete,
                "{:?}",
                basis.unresolved
            );
            basis
        };
        let forward = CORES.map(|a| solve(a, ModalWindow::forward(2.1 * K, decay, margin)));
        let backward = solve(
            CORES[1],
            ModalWindow::Wavenumber {
                re_cm: [-2.1 * K, margin],
                im_cm: [-decay, margin],
            },
        );
        let registration = GuidedOverlapRegistration {
            origins_cm: [-CORES[0], -CORES[1]],
        };
        let primary = if pol == TE { 1 } else { 4 };
        let sample = |basis: &ModalBasis, a: f64, index: usize, z: &[f64]| {
            let stack = centred(a, W, PEC);
            basis.modes[index]
                .fields_with(backend, &at(&stack, z), Default::default())
                .unwrap()
        };
        // Even modes in key order, from the symmetry of the primary field.
        let evens = |basis: &ModalBasis, a: f64| {
            (0..basis.found())
                .filter(|&i| {
                    let s = sample(basis, a, i, &[a - 0.37 * W, a + 0.37 * W]);
                    (s[0][primary] - s[1][primary]).norm() < 1e-6 * s[0][primary].norm()
                })
                .take(N_MAX)
                .collect::<Vec<_>>()
        };
        let even = [evens(&forward[0], CORES[0]), evens(&forward[1], CORES[1])];
        assert!(even.iter().all(|e| e.len() == N_MAX));
        // Backward partners b = c M g, matched by β and sampled for c.
        let probe = [-0.5, -0.1, 0.15, 0.35, 0.8];
        let partners: Vec<_> = even[1]
            .iter()
            .map(|&j| {
                let beta = forward[1].modes[j].point.physical_coordinate();
                let k = (0..backward.found())
                    .min_by(|&x, &y| {
                        let d = |i: usize| {
                            (backward.modes[i].point.physical_coordinate() + beta).norm()
                        };
                        d(x).total_cmp(&d(y))
                    })
                    .unwrap();
                let g = sample(&forward[1], CORES[1], j, &probe);
                let b = sample(&backward, CORES[1], k, &probe);
                let mg: Vec<_> = g.iter().map(mirror).collect();
                let (s, c) = (0..probe.len())
                    .flat_map(|s| (0..6).map(move |c| (s, c)))
                    .max_by(|x, y| mg[x.0][x.1].norm().total_cmp(&mg[y.0][y.1].norm()))
                    .unwrap();
                let factor = b[s][c] / mg[s][c];
                let peak = mg[s][c].norm();
                for (x, y) in b.iter().flatten().zip(mg.iter().flatten()) {
                    assert!((x - factor * y).norm() < 1e-8 * peak.max(1.) * factor.norm());
                }
                (k, factor)
            })
            .collect();
        let u_pairing =
            basis_pairing_with(&backend, [&forward[0], &forward[1]], registration).unwrap();
        let v_pairing =
            basis_pairing_with(&backend, [&forward[0], &backward], registration).unwrap();
        let column = |pairing: &BasisPairing, basis: usize, index: usize| {
            pairing.modes[basis]
                .iter()
                .position(|&m| m == index)
                .unwrap()
        };
        let u = DMatrix::from_fn(N_MAX, N_MAX, |i, j| {
            u_pairing.unconjugated[(
                column(&u_pairing, 0, even[0][i]),
                column(&u_pairing, 1, even[1][j]),
            )]
        });
        let v = DMatrix::from_fn(N_MAX, N_MAX, |i, j| {
            let (k, factor) = partners[j];
            v_pairing.unconjugated[(column(&v_pairing, 0, even[0][i]), column(&v_pairing, 1, k))]
                / factor
        });
        // The lamina matrices against the closed forms, entry by entry.
        let mut entries: f64 = 0.;
        for i in 0..N_MAX {
            for j in 0..N_MAX {
                entries = entries
                    .max((u[(i, j)] - entry(i, j, false)).norm())
                    .max((v[(i, j)] - entry(i, j, true)).norm());
            }
        }
        let delta = [&u_pairing, &v_pairing]
            .iter()
            .flat_map(|p| {
                [
                    p.quadrature_change_w_per_m,
                    p.self_defects[0],
                    p.self_defects[1],
                ]
            })
            .fold(0., f64::max);
        eprintln!(
            "{pol:?}: lamina matrices against closed forms {entries:.1e} W/m; δ = {delta:.1e}; \
             reference limit r = {:.8}, t = {:.8}",
            limit[0], limit[1]
        );
        for n in 2..=N_MAX {
            let lamina = expand(&u, &v, n).unwrap();
            let truncated = reference(n);
            let error = |x: [Complex; 2]| (x[0] - limit[0]).norm().max((x[1] - limit[1]).norm());
            let (err, truncation) = (error(lamina), error(truncated));
            let kappa = condition_number(u.view((0, 0), (n, n)).into_owned());
            let floor = kappa * delta;
            let bound = truncation.max(floor);
            let difference = (lamina[0] - truncated[0])
                .norm()
                .max((lamina[1] - truncated[1]).norm());
            eprintln!(
                "  N = {n:2}: error {err:.3e}, truncation {truncation:.3e}, κ_N δ {floor:.1e} (κ_N {kappa:.2}), \
                 lamina − reference at N {difference:.1e}"
            );
            assert!(err <= 10. * bound && err >= bound / 10., "N = {n}");
        }
    }
}
