//! Modal bases and pairings of stacks closed by wall-backed PMLs: the PML
//! contribution in closed form and at fixed complex thickness, Gauss–Legendre
//! agreement, independent counts, bi-orthogonality, the open slab's ladders
//! as the PML thickens, compatible terminations and the scale refusal.
use lamina_core::{
    AnalyticalMaterialStack, Constant, ExteriorSampling, FieldSampling, Polarisation, Scatter2,
    Stack, StackSide, Termination, Wall, WallKind,
};
use lamina_mode::*;
mod pml_support;
use pml_support::*;
mod modal_support;
use modal_support::{at, gauss_legendre};

/// W/m per (V/m)² cm: the flux of fields in [E, Z0 H] across 1 cm.
const C: f64 = 0.01 / 376.730313412;
const ORIGIN: GuidedOverlapRegistration = GuidedOverlapRegistration {
    origins_cm: [0., 0.],
};
const GAP: f64 = 0.1;

fn request(pol: Polarisation, r: [f64; 4]) -> ModalBasisRequest {
    ModalBasisRequest::new(
        cm(1.),
        ModeFamily::Polarised(pol),
        ModalWindow::Wavenumber {
            re_cm: [r[0], r[1]],
            im_cm: [r[2], r[3]],
        },
    )
}

fn basis(stack: AnalyticalMaterialStack<Complex>, pol: Polarisation, r: [f64; 4]) -> ModalBasis {
    let basis = modal_basis_in_window_with(&Scatter2::new(), stack, request(pol, r)).unwrap();
    assert_eq!(
        basis.status,
        BasisStatus::Complete,
        "{:?} {:?} {} {:?}",
        basis.unresolved,
        basis.expected,
        basis.found(),
        basis
            .modes
            .iter()
            .map(|m| (m.point.physical_coordinate(), m.normalisation))
            .collect::<Vec<_>>()
    );
    assert_eq!(basis.scope, BasisScope::Discrete);
    basis
}

/// The core closed on both sides by equal PMLs `GAP` beyond its interfaces,
/// `d` thick, with `backing` walls.
fn closed(d: f64, profile: Profile, backing: WallKind) -> AnalyticalMaterialStack<Complex> {
    slab(
        pml(StackSide::Left, GAP, d, profile, backing),
        pml(StackSide::Right, GAP, d, profile, backing),
    )
}

/// Profiles of orders 0, 2 and 4 with the complex thickness of `PROFILE`.
fn equal_tau() -> [Profile; 3] {
    let excess = PROFILE.mean() - 1.;
    [0., 2., 4.].map(|m| Profile {
        m,
        strength: excess * (m + 1.),
    })
}

/// s(ζ) = 1 + c (ζ/d)^m.
fn stretch(profile: Profile, zeta: f64, d: f64) -> Complex {
    1. + profile.strength * (zeta / d).powf(profile.m)
}

/// max |G − I| of a basis's own unconjugated Gram.
fn identity_defect(pairing: &BasisPairing) -> f64 {
    let n = pairing.unconjugated.nrows();
    (pairing.unconjugated.clone() - nalgebra::DMatrix::identity(n, n))
        .iter()
        .map(|v| v.norm())
        .fold(0., f64::max)
}

/// A pairing's contributions summed over each physical region: the left
/// PML, the regions between the PML starts, and the right PML.
fn by_region(pairing: &ModalPairing) -> Vec<[Complex; 2]> {
    let mut regions: Vec<[Complex; 2]> = vec![];
    let mut previous = None;
    for (edges, value) in pairing
        .partition_cm
        .windows(2)
        .zip(&pairing.regional_w_per_m)
    {
        let middle = 0.5 * (edges[0] + edges[1]);
        let region = if middle < -GAP {
            0
        } else if middle > 2. * A + GAP {
            usize::MAX
        } else {
            1 + pairing
                .partition_cm
                .iter()
                .filter(|&&z| z > -GAP && z < middle)
                .count()
        };
        if previous == Some(region) {
            let last = regions.last_mut().unwrap();
            last[0] += value[0];
            last[1] += value[1];
        } else {
            regions.push(*value);
        }
        previous = Some(region);
    }
    regions
}

const SMALL: [f64; 4] = [1.0 * K, 2.03 * K, -0.031 * K, 1.0 * K];

#[test]
fn the_pml_contribution_has_its_closed_form_and_depends_on_tau_only() {
    let d = 0.75;
    let backend = Scatter2::new();
    let tau = d * PROFILE.mean();
    for pol in [TE, TM] {
        let profiles = equal_tau();
        let bases = profiles.map(|p| basis(closed(d, p, WallKind::PerfectElectric), pol, SMALL));
        let n = bases[1].found();
        assert!(n >= 8, "{n}");
        // The modes of equal-τ stacks coincide.
        for b in &bases {
            assert_eq!(b.found(), n);
            for (x, y) in b.modes.iter().zip(&bases[1].modes) {
                let (x, y) = (x.point.physical_coordinate(), y.point.physical_coordinate());
                assert!((x - y).norm() < 1e-9 * K, "{x} {y}");
            }
        }
        let (mut closed_form, mut independence, mut dependence): (f64, f64, f64) = (0., 0., 0.);
        let pairs: Vec<_> = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .filter(|(i, j)| (i + j) % 3 == 0)
            .collect();
        for &(i, j) in &pairs {
            let pairings = bases.each_ref().map(|b| {
                modal_pairing_with(
                    &backend,
                    [&b.modes[i], &b.modes[j]],
                    ORIGIN,
                    Default::default(),
                )
                .unwrap()
            });
            let regions = pairings.each_ref().map(by_region);
            // Left PML, gap, core, gap, right PML.
            for p in &regions {
                assert_eq!(p.len(), 5);
                for (region, (x, y)) in p.iter().zip(&regions[1]).enumerate() {
                    let scale = pairings[1].unconjugated_w_per_m.norm().max(1.);
                    independence = independence.max((x[0] - y[0]).norm() / scale);
                    if region == 0 || region == 4 {
                        dependence = dependence.max((x[1] - y[1]).norm() / x[1].norm());
                    }
                }
            }
            // Closed form over the right PML of the order-2 profile: each
            // component is A e^{iκζ̃} + B e^{−iκζ̃}, fitted at two depths.
            let modes = [&bases[1].modes[i], &bases[1].modes[j]];
            let coefficients = modes.map(|mode| {
                let beta = mode.point.physical_coordinate();
                let kappa = leaky_kappa(beta);
                let stretched = |zeta: f64| PROFILE.stretched(zeta, d);
                // The second depth where e^{±iκζ̃} are best separated.
                let second = [0.25, 0.5, 0.75, 1.]
                    .into_iter()
                    .map(|f| f * d)
                    .max_by(|&x, &y| {
                        let quality = |z: f64| {
                            let p = (Complex::I * kappa * stretched(z)).exp();
                            (p - p.inv()).norm() / p.norm().max(p.inv().norm())
                        };
                        quality(x).total_cmp(&quality(y))
                    })
                    .unwrap();
                let sampling = FieldSampling::new()
                    .right_exterior(ExteriorSampling::distances([GAP, GAP + second].map(cm)));
                let f = mode
                    .fields_with(backend, &sampling, Default::default())
                    .unwrap();
                let p = (Complex::I * kappa * stretched(second)).exp();
                // [1, 1; p, 1/p] [A; B] = [f0; f1].
                let det = p.inv() - p;
                let ab: [(Complex, Complex); 6] = std::array::from_fn(|c| {
                    let (f0, f1) = (f[0][c], f[1][c]);
                    ((f0 * p.inv() - f1) / det, (f1 - p * f0) / det)
                });
                (kappa, ab)
            });
            let integral = |q: Complex| {
                if q.norm() * tau.norm() < 1e-12 {
                    tau
                } else {
                    ((Complex::I * q * tau).exp() - 1.) / (Complex::I * q)
                }
            };
            // ∫ x_c y_c' dζ̃ for components c of the first mode and c' of the second.
            let product = |c: usize, cc: usize, first: usize| {
                let ((ka, x), (kb, y)) = if first == 0 {
                    (coefficients[0], coefficients[1])
                } else {
                    (coefficients[1], coefficients[0])
                };
                let (xa, xb) = x[c];
                let (ya, yb) = y[cc];
                xa * ya * integral(ka + kb)
                    + xa * yb * integral(ka - kb)
                    + xb * ya * integral(-ka + kb)
                    + xb * yb * integral(-ka - kb)
            };
            let expected = 0.25
                * C
                * (product(1, 5, 0) - product(2, 4, 0) + product(1, 5, 1) - product(2, 4, 1));
            let computed = regions[1][4][0];
            closed_form = closed_form.max((computed - expected).norm() / expected.norm().max(1.));
        }
        eprintln!(
            "{pol:?}: {} pairs of {n} modes; unconjugated at equal τ (orders 0, 2, 4) agree to \
             {independence:.1e}; conjugated PML contributions differ by up to {dependence:.1e} \
             relative; right PML against the closed form ∫₀^τ e^(iqζ̃) dζ̃ to {closed_form:.1e}",
            pairs.len()
        );
        assert!(independence < 1e-8, "{independence:e}");
        assert!(dependence > 1e-3, "{dependence:e}");
        assert!(closed_form < 1e-8, "{closed_form:e}");
    }
}

#[test]
fn both_pairings_agree_with_gauss_legendre_through_the_pml() {
    let d = 0.75;
    let backend = Scatter2::new();
    let rule = gauss_legendre(24);
    for pol in [TE, TM] {
        for profile in equal_tau() {
            let stack = closed(d, profile, WallKind::PerfectElectric);
            let b = basis(stack.clone(), pol, SMALL);
            let mut worst: f64 = 0.;
            let mut regions = 0;
            for f in b.modes.iter().step_by(3) {
                for g in b.modes.iter().skip(1).step_by(3) {
                    let pairing =
                        modal_pairing_with(&backend, [f, g], ORIGIN, Default::default()).unwrap();
                    let edges = &pairing.partition_cm;
                    let (start, end) = (-GAP, 2. * A + GAP);
                    for (edge, regional) in edges.windows(2).zip(&pairing.regional_w_per_m) {
                        let panels = 32;
                        let h = (edge[1] - edge[0]) / panels as f64;
                        let mut z = vec![];
                        let mut w = vec![];
                        for p in 0..panels {
                            for &(x, weight) in &rule {
                                z.push(edge[0] + h * (p as f64 + 0.5 + 0.5 * x));
                                w.push(0.5 * h * weight);
                            }
                        }
                        let x = f
                            .fields_with(backend, &at(&stack, &z), Default::default())
                            .unwrap();
                        let y = g
                            .fields_with(backend, &at(&stack, &z), Default::default())
                            .unwrap();
                        let (mut u, mut c) = (Complex::ZERO, Complex::ZERO);
                        for (((p, q), w), z) in x.iter().zip(&y).zip(&w).zip(&z) {
                            let depth = (start - z).max(z - end);
                            let s = if depth > 0. {
                                stretch(profile, depth, d)
                            } else {
                                Complex::ONE
                            };
                            // Equivalent-medium normal components s E_z, s Z0 H_z.
                            let (p2, p5, q2, q5) = (s * p[2], s * p[5], s * q[2], s * q[5]);
                            u += w * 0.25 * C * (p[1] * q5 - p2 * q[4] + q[1] * p5 - q2 * p[4]);
                            c += w
                                * 0.25
                                * C
                                * (p[1] * q5.conj() - p2 * q[4].conj() + q[1].conj() * p5
                                    - q2.conj() * p[4]);
                        }
                        let scale = u.norm().max(c.norm()).max(1.);
                        if std::env::var("DEBUG_GL").is_ok() {
                            eprintln!(
                                "region {edge:?}: u {} vs {u}, c {} vs {c}",
                                regional[0], regional[1]
                            );
                        }
                        worst = worst
                            .max((regional[0] - u).norm() / scale)
                            .max((regional[1] - c).norm() / scale);
                        regions += 1;
                    }
                }
            }
            eprintln!(
                "{pol:?} order {}: {regions} regions agree with Gauss–Legendre to {worst:.1e}",
                profile.m
            );
            assert!(worst < 1e-8, "{worst:e}");
        }
    }
}

#[test]
fn pml_closed_bases_count_independently_and_are_bi_orthonormal() {
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        for backing in [WallKind::PerfectElectric, WallKind::PerfectMagnetic] {
            // The thickest PMLs scale the characteristic by about e^−33.
            for d in [0.75, 1.75] {
                let stack = closed(d, PROFILE, backing);
                let b = basis(stack, pol, SMALL);
                let tau = d * PROFILE.mean();
                let regions = [(CLAD, GAP + tau), (CORE, c(2. * A, 0.)), (CLAD, GAP + tau)];
                let reference = |beta| box_value(pol, [backing; 2], &regions, beta);
                let independent = winding(&reference, SMALL);
                assert_eq!(b.expected, Some(independent));
                assert_eq!(b.found(), independent);
                for mode in &b.modes {
                    let beta = mode.point.physical_coordinate();
                    let polished = newton(&reference, beta).unwrap();
                    assert!((polished - beta).norm() < 1e-9 * K, "{beta}");
                }
                let pairing = basis_pairing_with(&backend, [&b, &b], ORIGIN).unwrap();
                let defect = identity_defect(&pairing);
                assert!(defect < 1e-8, "{defect:e}");
                assert!(pairing.self_defects.iter().all(|d| *d < 1e-8));
                assert!(
                    pairing
                        .singular_values
                        .iter()
                        .all(|s| (s - 1.).abs() < 1e-7)
                );
                eprintln!(
                    "{pol:?} {backing:?} d = {d}: expected = found = independent = {independent}, \
                     |G − I| {defect:.1e}, quadrature change {:.1e} W/m",
                    pairing.quadrature_change_w_per_m
                );
            }
        }
    }
}

#[test]
fn pml_closed_slabs_hold_the_open_slabs_guided_and_exposed_leaky_ladders() {
    // TE up to the even leaky mode near Im β = 2 k0; TM's odd leaky mode is
    // barely exposed and needs thicker PMLs, and the window stays below the
    // rotated-continuum modes whose self-pairing those PMLs make ill-conditioned.
    let cases = [
        (TE, [0.75, 1., 1.25, 1.5, 1.75], 2.07 * K),
        (TM, [1., 1.5, 2., 2.5, 3.], 0.79 * K),
    ];
    // Basis modes off the ladders stay this far from every ladder value; a
    // predicted distance below it resolves its ladder value.
    let floor = 0.05 * K;
    // Distances at the root accuracy.
    let accuracy = 1e-9 * K;
    for (pol, thicknesses, top) in cases {
        let window = [0.77 * K, 2.03 * K, -0.031 * K, top];
        let (open, leaky) = open_ladder(pol, window);
        let length = |d: f64| GAP + d * PROFILE.mean();
        // Exposure, Im(κ (g + τ)) > 0, holds or fails over the whole sequence.
        let exposed: Vec<bool> = open
            .iter()
            .map(|(label, beta, kappa)| {
                let first = (kappa * length(thicknesses[0])).im > 0.;
                assert!(
                    thicknesses
                        .iter()
                        .all(|&d| ((kappa * length(d)).im > 0.) == first),
                    "{label} {beta}"
                );
                first
            })
            .collect();
        let exposed_leaky = (0..open.len())
            .filter(|&i| exposed[i] && open[i].0.starts_with("leaky"))
            .count();
        assert!(exposed_leaky >= 1, "{leaky} {exposed_leaky}");
        let mut ratios = vec![vec![]; open.len()];
        let mut distances = vec![vec![]; open.len()];
        for &d in &thicknesses {
            let stack = closed(d, PROFILE, WallKind::PerfectElectric);
            let b = basis(stack, pol, window);
            let tau = d * PROFILE.mean();
            let regions = [(CLAD, GAP + tau), (CORE, c(2. * A, 0.)), (CLAD, GAP + tau)];
            let independent = winding(&|beta| box_value(pol, [PEC; 2], &regions, beta), window);
            assert_eq!(b.expected, Some(independent));
            assert_eq!(b.found(), independent);
            let betas: Vec<_> = b
                .modes
                .iter()
                .map(|m| m.point.physical_coordinate())
                .collect();
            let mut matched = vec![];
            for (i, (label, beta, kappa)) in open.iter().enumerate() {
                // The PML perturbs an exposed ladder value by O(e^{2iκ(g+τ)}).
                let predicted = 10. * K * (Complex::I * 2. * kappa * length(d)).exp().norm();
                let near: Vec<_> = (0..betas.len())
                    .filter(|&j| (betas[j] - beta).norm() <= predicted + accuracy)
                    .collect();
                let distance = betas
                    .iter()
                    .map(|b| (b - beta).norm())
                    .fold(f64::INFINITY, f64::min);
                if exposed[i] && predicted < floor {
                    assert_eq!(near.len(), 1, "{label} {beta} d = {d}: {distance:e}");
                    matched.push(near[0]);
                    ratios[i].push(10. * distance / predicted);
                    distances[i].push(distance);
                } else if exposed[i] {
                    assert!(d < thicknesses[4], "{label} {beta} unresolved at d = {d}");
                }
            }
            // The rest stay off every ladder value.
            let mut least = f64::INFINITY;
            for (j, beta) in betas.iter().enumerate() {
                if matched.contains(&j) {
                    continue;
                }
                for (i, (_, open, kappa)) in open.iter().enumerate() {
                    let predicted = 10. * K * (Complex::I * 2. * kappa * length(d)).exp().norm();
                    // An exposed value not yet resolved has its partner near.
                    if !exposed[i] || predicted < floor {
                        least = least.min((beta - open).norm());
                    }
                }
            }
            assert!(least > floor, "d = {d}: {least}");
            eprintln!(
                "{pol:?} d = {d}: {} modes, {} on the ladders, the rest at least {:.3} k0 from \
                 every ladder value",
                betas.len(),
                matched.len(),
                least / K
            );
        }
        for (i, (label, beta, _)) in open.iter().enumerate() {
            if exposed[i] {
                eprintln!(
                    "{pol:?} {label} β = {beta:.6} cm⁻¹: |Δβ| {:?}, |Δβ| / (k0 |e^(2iκ(g+τ))|) {:?}",
                    distances[i]
                        .iter()
                        .map(|d| format!("{d:.1e}"))
                        .collect::<Vec<_>>(),
                    ratios[i]
                        .iter()
                        .zip(&distances[i])
                        .map(|(r, d)| if *d > 1e-13 {
                            format!("{r:.2}")
                        } else {
                            "at round-off".into()
                        })
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

/// The core between cladding of permittivity `left` and `right` closed by
/// PMLs of thickness `d` `GAP` beyond each interface, or by `walls` where given.
fn materials(
    left: f64,
    right: f64,
    d: [f64; 2],
    profile: Profile,
    walls: [Option<WallKind>; 2],
) -> AnalyticalMaterialStack<Complex> {
    let close = |side, d: f64, wall: Option<WallKind>| -> Termination<f64> {
        match (wall, side) {
            (Some(kind), StackSide::Left) => Wall::new(kind, cm(-GAP - d)).into(),
            (Some(kind), _) => Wall::new(kind, cm(2. * A + GAP + d)).into(),
            (None, _) => pml(side, GAP, d, profile, WallKind::PerfectElectric).into(),
        }
    };
    Stack::from_analytical_materials(Constant::dielectric(left), Constant::dielectric(right))
        .analytical_layer(Constant::dielectric(CORE), cm(2. * A))
        .left_termination(close(StackSide::Left, d[0], walls[0]))
        .right_termination(close(StackSide::Right, d[1], walls[1]))
        .finalise()
}

#[test]
fn pairing_needs_equal_pmls_and_accepts_different_pml_materials() {
    let backend = Scatter2::new();
    let window = [1.55 * K, 2.03 * K, -0.031 * K, 0.047 * K];
    let d = 0.75;
    let solve = |stack| basis(stack, TE, window);
    let reference = solve(materials(CLAD, CLAD, [d; 2], PROFILE, [None; 2]));
    let refused =
        |other: &ModalBasis| match basis_pairing_with(&backend, [&reference, other], ORIGIN) {
            Err(PairingError::Terminations { side }) => side,
            other => panic!("{other:?}"),
        };
    // A thicker right PML.
    let thicker = solve(materials(CLAD, CLAD, [d, d + 0.1], PROFILE, [None; 2]));
    assert_eq!(refused(&thicker), StackSide::Right);
    // A wall at the left PML's backing wall.
    let walled = solve(materials(
        CLAD,
        CLAD,
        [d; 2],
        PROFILE,
        [Some(WallKind::PerfectElectric), None],
    ));
    assert_eq!(refused(&walled), StackSide::Left);
    // Another profile of equal complex thickness.
    let profile = equal_tau()[2];
    let reshaped = solve(materials(CLAD, CLAD, [d; 2], profile, [None; 2]));
    assert_eq!(refused(&reshaped), StackSide::Left);
    // A shifted registration moves both PMLs.
    let shifted = GuidedOverlapRegistration {
        origins_cm: [0., 0.01],
    };
    assert!(matches!(
        basis_pairing_with(&backend, [&reference, &reference], shifted),
        Err(PairingError::Terminations {
            side: StackSide::Left
        })
    ));
    // Different gap and PML materials on the right, same geometry.
    let other = solve(materials(CLAD, 1.5, [d; 2], PROFILE, [None; 2]));
    let pairing = basis_pairing_with(&backend, [&reference, &other], ORIGIN).unwrap();
    assert!(pairing.self_defects.iter().all(|d| *d < 1e-8));
    let diagonal: Vec<_> = (0..pairing
        .unconjugated
        .nrows()
        .min(pairing.unconjugated.ncols()))
        .map(|i| pairing.unconjugated[(i, i)])
        .collect();
    eprintln!(
        "refused: thicker right PML, a wall against the left PML, an equal-τ profile of order 4, \
         a shifted registration; accepted: right cladding 1.5 against {CLAD}, ⟨f_i, g_i⟩ {diagonal:.6?}"
    );
}

#[test]
fn a_window_beyond_the_pml_scale_is_refused() {
    // Out to 4 k0 the thick PML's scale reaches about exp(315).
    let error = modal_basis_in_window_with(
        &Scatter2::new(),
        long_pml_and_wall(),
        request(TE, [1.85 * K, 4. * K, -0.031 * K, 0.047 * K]),
    )
    .unwrap_err();
    let ModalBasisError::Scale {
        side,
        log_magnitude,
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(side, StackSide::Left);
    assert!((300..330).contains(&log_magnitude), "{log_magnitude}");
    // Within range the same stack's basis is complete.
    let b = basis(
        long_pml_and_wall(),
        TE,
        [1.85 * K, 3. * K, -0.031 * K, 0.047 * K],
    );
    eprintln!(
        "refused to 4 k0: {side:?} side, exp({log_magnitude}); to 3 k0: {} modes, complete",
        b.found()
    );
}
