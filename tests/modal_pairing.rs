//! Unconjugated and conjugated pairings of modes and bases of stacks closed
//! by walls: closed forms between walls, exchange symmetry, compatible
//! terminations, Gauss–Legendre agreement, bi-orthogonality and continuity.
mod modal_support;
use lamina_core::{AnalyticalMaterialStack, Constant, Scatter2, Stack, StackSide, Wall};
use lamina_mode::*;
use modal_support::*;
use std::f64::consts::PI;

/// W/m per (V/m)² cm: the flux of fields in [E, Z0 H] across 1 cm.
const C: f64 = 0.01 / 376.730313412;
const ORIGIN: GuidedOverlapRegistration = GuidedOverlapRegistration {
    origins_cm: [0., 0.],
};

fn basis(stack: AnalyticalMaterialStack<Complex>, request: ModalBasisRequest) -> ModalBasis {
    let basis = modal_basis_in_window_with(&Scatter2::new(), stack, request).unwrap();
    assert_eq!(
        basis.status,
        BasisStatus::Complete,
        "{:?}",
        basis.unresolved
    );
    basis
}

fn window() -> ModalWindow {
    ModalWindow::forward(2.1 * K, 2. * K, 0.05 * K)
}

/// A box of one material between PEC walls 0.4 cm left and 0.5 cm right of
/// its single interface: no finite layer, only the two gaps.
fn gaps_only(epsilon: f64) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(epsilon), Constant::dielectric(epsilon))
        .left_termination(Wall::perfect_electric(cm(-0.4)))
        .right_termination(Wall::perfect_electric(cm(0.5)))
        .finalise()
}

/// The closed-form TE modes of [`gaps_only`]: E_y = A sin(nπ (z + 0.4)/L),
/// Z0 H_z = β E_y / k0, with ⟨f, f⟩ = 1 and the sign rule at z = 0.
fn sine_mode(epsilon: f64, beta: Complex) -> (usize, Complex) {
    let length = 0.9;
    let n = ((epsilon * K * K - beta * beta).sqrt().re * length / PI).round() as usize;
    let amplitude = (Complex::new(4. * K / (C * length), 0.) / beta).sqrt();
    let l = amplitude * (n as f64 * PI * 0.4 / length).sin();
    let decisive = if l.re.abs() <= 64. * f64::EPSILON * l.norm() {
        l.im
    } else {
        l.re
    };
    (n, if decisive < 0. { -amplitude } else { amplitude })
}

#[test]
fn gaps_pair_as_closed_form_sine_integrals() {
    let (first, second) = (gaps_only(2.25), gaps_only(1.));
    let a = basis(first, request(TE, window()));
    let b = basis(second, request(TE, window()));
    let pairing = basis_pairing_with(&Scatter2::new(), [&a, &b], ORIGIN).unwrap();
    let modes = |basis: &ModalBasis, epsilon| {
        basis
            .modes
            .iter()
            .map(|m| {
                let beta = m.point.physical_coordinate();
                let (n, amplitude) = sine_mode(epsilon, beta);
                (n, beta, amplitude)
            })
            .collect::<Vec<_>>()
    };
    let (x, y) = (modes(&a, 2.25), modes(&b, 1.));
    assert!(x.iter().any(|(_, b, _)| b.im > 0.1 * K) && y.iter().any(|(_, b, _)| b.im > 0.1 * K));
    let mut worst: f64 = 0.;
    let mut matched = 0;
    for (i, &(n, beta, p)) in x.iter().enumerate() {
        for (j, &(m, gamma, q)) in y.iter().enumerate() {
            // ∫ sin sin over the box is L/2 on the diagonal, zero off it.
            let overlap = if n == m { 0.45 } else { 0. };
            matched += usize::from(n == m);
            let unconjugated = 0.25 * C * (beta + gamma) / K * p * q * overlap;
            let conjugated = 0.25 * C * (gamma.conj() + beta) / K * p * q.conj() * overlap;
            worst = worst
                .max((pairing.unconjugated[(i, j)] - unconjugated).norm())
                .max((pairing.conjugated[(i, j)].unwrap() - conjugated).norm());
        }
    }
    assert!(matched > 3);
    assert!(worst < 1e-9, "{worst:e}");
    assert_eq!(pairing.partition_cm, vec![-0.4, 0., 0.5]);
    eprintln!(
        "gaps only, ε 2.25 against ε 1: {}×{} closed-form entries ({matched} matched orders) to {worst:.1e} W/m",
        x.len(),
        y.len()
    );
}

#[test]
fn pairings_are_symmetric_and_need_matching_walls() {
    let backend = Scatter2::new();
    let a = basis(centred(0.3, 0.8, PEC), request(TE, window()));
    let b = basis(centred(0.2, 0.8, PEC), request(TE, window()));
    // Equal walls in the registered frame: the cores' left interfaces differ.
    let registration = GuidedOverlapRegistration {
        origins_cm: [0., 0.1],
    };
    let swapped = GuidedOverlapRegistration {
        origins_cm: [0.1, 0.],
    };
    let shifted = GuidedOverlapRegistration {
        origins_cm: [0.37, 0.47],
    };
    let mut worst: f64 = 0.;
    for f in &a.modes {
        for g in &b.modes {
            let forward =
                modal_pairing_with(&backend, [f, g], registration, Default::default()).unwrap();
            let back = modal_pairing_with(&backend, [g, f], swapped, Default::default()).unwrap();
            assert_eq!(
                forward.unconjugated_w_per_m.re.to_bits(),
                back.unconjugated_w_per_m.re.to_bits()
            );
            assert_eq!(
                forward.unconjugated_w_per_m.im.to_bits(),
                back.unconjugated_w_per_m.im.to_bits()
            );
            assert!(
                (forward.conjugated_w_per_m.unwrap() - back.conjugated_w_per_m.unwrap().conj())
                    .norm()
                    < 1e-12
            );
            let expected = [-0.5, 0., 0.1, 0.5, 0.6, 1.1];
            assert_eq!(forward.partition_cm.len(), expected.len());
            assert!(
                forward
                    .partition_cm
                    .iter()
                    .zip(expected)
                    .all(|(z, e)| (z - e).abs() < 1e-15)
            );
            assert_eq!(forward.regional_w_per_m.len(), 5);
            let moved = modal_pairing_with(&backend, [f, g], shifted, Default::default()).unwrap();
            worst = worst.max((moved.unconjugated_w_per_m - forward.unconjugated_w_per_m).norm());
        }
    }
    assert!(worst < 1e-10, "{worst:e}");
    let refused = |second: AnalyticalMaterialStack<Complex>, registration| {
        let other = basis(second, request(TE, window()));
        basis_pairing_with(&backend, [&a, &other], registration).unwrap_err()
    };
    // The right walls differ in registered position.
    let wider =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
            .analytical_layer(Constant::dielectric(CORE), cm(0.6))
            .left_termination(Wall::perfect_electric(cm(-0.5)))
            .right_termination(Wall::perfect_electric(cm(1.2)))
            .finalise();
    assert!(matches!(
        refused(wider, ORIGIN),
        PairingError::Terminations {
            side: StackSide::Right
        }
    ));
    assert!(matches!(
        refused(centred(0.3, 0.8, PMC), ORIGIN),
        PairingError::Terminations {
            side: StackSide::Left
        }
    ));
    // Different exterior materials between the same walls pair.
    let vacuum_gaps = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(CORE), cm(0.6))
        .left_termination(Wall::perfect_electric(cm(-0.5)))
        .right_termination(Wall::perfect_electric(cm(1.1)))
        .finalise();
    let other = basis(vacuum_gaps, request(TE, window()));
    assert!(basis_pairing_with(&backend, [&a, &other], ORIGIN).is_ok());
    let mut longer = request(TE, window());
    longer.wavelength = cm(1.01);
    let other = basis(centred(0.3, 0.8, PEC), longer);
    assert!(matches!(
        basis_pairing_with(&backend, [&a, &other], ORIGIN),
        Err(PairingError::IncompatibleWavelength)
    ));
    eprintln!(
        "exchange bit for bit; a common shift of both origins changes pairings by {worst:.1e} W/m"
    );
}

/// Both pairings region by region against composite Gauss–Legendre quadrature.
#[test]
fn pairings_agree_with_gauss_legendre_on_every_region() {
    let backend = Scatter2::new();
    let rule = gauss_legendre(24);
    for pol in [TE, TM] {
        let (sa, sb) = (centred(0.3, 0.8, PMC), centred(0.22, 0.8, PMC));
        let a = basis(sa.clone(), request(pol, window()));
        let b = basis(sb.clone(), request(pol, window()));
        let registration = GuidedOverlapRegistration {
            origins_cm: [0., 0.08],
        };
        let mut worst: f64 = 0.;
        let mut regions = 0;
        for f in a.modes.iter().step_by(2) {
            for g in b.modes.iter().step_by(2) {
                let pairing =
                    modal_pairing_with(&backend, [f, g], registration, Default::default()).unwrap();
                for (edges, regional) in pairing
                    .partition_cm
                    .windows(2)
                    .zip(&pairing.regional_w_per_m)
                {
                    let panels = 16;
                    let h = (edges[1] - edges[0]) / panels as f64;
                    let mut z = vec![];
                    let mut w = vec![];
                    for p in 0..panels {
                        for &(x, weight) in &rule {
                            z.push(edges[0] + h * (p as f64 + 0.5 + 0.5 * x));
                            w.push(0.5 * h * weight);
                        }
                    }
                    let local = |origin: f64| z.iter().map(|z| z - origin).collect::<Vec<_>>();
                    let x = f
                        .fields_with(backend, &at(&sa, &local(0.)), Default::default())
                        .unwrap();
                    let y = g
                        .fields_with(backend, &at(&sb, &local(0.08)), Default::default())
                        .unwrap();
                    let (mut u, mut c) = (Complex::ZERO, Complex::ZERO);
                    for ((p, q), w) in x.iter().zip(&y).zip(&w) {
                        u += w * 0.25 * C * (p[1] * q[5] - p[2] * q[4] + q[1] * p[5] - q[2] * p[4]);
                        c += w
                            * 0.25
                            * C
                            * (p[1] * q[5].conj() - p[2] * q[4].conj() + q[1].conj() * p[5]
                                - q[2].conj() * p[4]);
                    }
                    worst = worst
                        .max((regional[0] - u).norm())
                        .max((regional[1] - c).norm());
                    regions += 1;
                }
            }
        }
        assert!(worst < 1e-9, "{worst:e}");
        eprintln!("{pol:?}: {regions} regions agree with Gauss–Legendre to {worst:.1e} W/m");
    }
}

#[test]
fn walled_bases_are_bi_orthonormal_with_evanescent_and_complex_modes() {
    use lamina_units::VacuumAngularWavenumber as W;
    let backend = Scatter2::new();
    let lossy = |kind| {
        let core = lamina_materials::Drude::new(
            4.,
            W::inverse_centimetres(40f64.sqrt()),
            W::inverse_centimetres(K),
        )
        .unwrap();
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(1.))
            .analytical_layer(core, cm(0.6))
            .left_termination(Wall::new(kind, cm(-0.3)))
            .right_termination(Wall::new(kind, cm(1.1)))
            .finalise()
    };
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            for (name, stack) in [("lossless", slab(0.35, 0.45, kind)), ("lossy", lossy(kind))] {
                let basis = basis(stack, request(pol, window()));
                let pairing = basis_pairing_with(&backend, [&basis, &basis], ORIGIN).unwrap();
                let n = basis.found();
                let identity = (pairing.unconjugated.clone() - nalgebra::DMatrix::identity(n, n))
                    .iter()
                    .map(|v| v.norm())
                    .fold(0., f64::max);
                assert!(identity < 1e-9, "{identity:e}");
                assert!(pairing.self_defects.iter().all(|d| *d < 1e-9));
                assert_eq!(pairing.leading_condition_numbers.len(), n);
                assert!(
                    pairing
                        .singular_values
                        .iter()
                        .all(|s| (s - 1.).abs() < 1e-8)
                );
                assert_eq!(pairing.status, [BasisStatus::Complete; 2]);
                assert_eq!(pairing.scope, [BasisScope::Discrete; 2]);
                let off = basis
                    .modes
                    .iter()
                    .filter(|m| m.point.physical_coordinate().im > 1e-6 * K)
                    .count();
                assert!(off > 0);
                eprintln!(
                    "{name} {pol:?} {kind:?}: {n} modes ({off} with Im β > 0), |G − I| {identity:.1e}, \
                     quadrature change {:.1e} W/m",
                    pairing.quadrature_change_w_per_m
                );
            }
        }
    }
}

/// The matrix between a fixed slab and one whose core thickness t changes by
/// h at fixed walls.
#[test]
fn pairing_matrices_change_smoothly_with_core_thickness() {
    let backend = Scatter2::new();
    let half = 0.8;
    let fixed = basis(centred(0.25, half, PEC), request(TE, window()));
    let t = 0.7;
    // The core is centred, so its left interface moves by −h/2.
    let at = |t: f64| {
        let b = basis(centred(t / 2., half, PEC), request(TE, window()));
        let registration = GuidedOverlapRegistration {
            origins_cm: [0., 0.25 - t / 2.],
        };
        let pairing = basis_pairing_with(&backend, [&fixed, &b], registration).unwrap();
        (b, pairing)
    };
    let (base, centre) = at(t);
    let margins = fixed
        .modes
        .iter()
        .chain(&base.modes)
        .filter_map(|m| m.sign_margin);
    assert!(
        margins.clone().all(|m| m > 1e-2),
        "{:?}",
        margins.collect::<Vec<_>>()
    );
    let u0 = &centre.unconjugated;
    let mut derivatives = vec![];
    for step in [1e-4, 1e-3] {
        let h = step * t;
        let (plus, p) = at(t + h);
        let (minus, m) = at(t - h);
        for other in [&plus, &minus] {
            assert_eq!(other.found(), base.found());
            for (x, y) in other.modes.iter().zip(&base.modes) {
                let (bx, by) = (x.point.physical_coordinate(), y.point.physical_coordinate());
                assert!((bx - by).norm() < 1e-2 * K, "reordered: {bx} {by}");
            }
        }
        let mut change: f64 = 0.;
        for (i, j) in (0..u0.nrows()).flat_map(|i| (0..u0.ncols()).map(move |j| (i, j))) {
            for v in [p.unconjugated[(i, j)], m.unconjugated[(i, j)]] {
                change = change.max((v - u0[(i, j)]).norm());
                if u0[(i, j)].norm() > 1e-3 {
                    assert!((v / u0[(i, j)]).re > 0., "sign change at ({i}, {j})");
                }
            }
        }
        derivatives.push((
            (&p.unconjugated - &m.unconjugated) / Complex::new(2. * h, 0.),
            change / h,
        ));
    }
    let (d1, c1) = &derivatives[0];
    let (d2, c2) = &derivatives[1];
    let scale = d2.iter().map(|v| v.norm()).fold(0., f64::max);
    let disagreement = (d1 - d2).iter().map(|v| v.norm()).fold(0., f64::max);
    // O(h): the largest change per unit h is the same at both steps.
    assert!((c1 / c2 - 1.).abs() < 0.05, "{c1} {c2}");
    assert!(disagreement < 1e-4 * scale, "{disagreement:e} of {scale:e}");
    eprintln!(
        "thickness {t} cm: largest |ΔU|/h {c1:.4} and {c2:.4} W/m per cm; central differences agree to {:.1e} of {scale:.3}",
        disagreement / scale
    );
}
