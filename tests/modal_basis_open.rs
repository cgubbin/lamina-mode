//! Modal bases of stacks with open sides on the conformal chart: the open
//! symmetric slab's guided and leaky ladders and independent counts, a slab
//! on a wall against the mirror-doubled slab, branch points and window edges,
//! continued tails, bi-orthogonality, compatibility, the covering of a β
//! rectangle and an open side facing a PML.
use lamina_core::{
    AnalyticalMaterialStack, Constant, Polarisation, Scatter2, Stack, StackSide, Termination, Wall,
    WallKind,
};
use lamina_mode::*;
mod pml_support;
use pml_support::*;
mod modal_support;
use modal_support::at;
use std::f64::consts::{FRAC_PI_2, PI};

const ORIGIN: GuidedOverlapRegistration = GuidedOverlapRegistration {
    origins_cm: [0., 0.],
};
/// A chart window, [Re z] × [Im z], holding two guided modes on the real axis
/// and one leaky mode near Im z = −2.6 (TE) or −2.1 (TM) of the open slab.
const WINDOW: [f64; 4] = [-3., 0.35, -2.9, 0.4];

fn chart(r: [f64; 4]) -> ModalWindow {
    ModalWindow::Chart {
        re: [r[0], r[1]],
        im: [r[2], r[3]],
    }
}

fn request(pol: Polarisation, window: ModalWindow) -> ModalBasisRequest {
    ModalBasisRequest::new(cm(1.), ModeFamily::Polarised(pol), window)
}

fn solve(stack: AnalyticalMaterialStack<Complex>, pol: Polarisation, r: [f64; 4]) -> ModalBasis {
    modal_basis_in_window_with(&Scatter2::new(), stack, request(pol, chart(r))).unwrap()
}

fn complete(stack: AnalyticalMaterialStack<Complex>, pol: Polarisation, r: [f64; 4]) -> ModalBasis {
    let basis = solve(stack, pol, r);
    assert_eq!(
        basis.status,
        BasisStatus::Complete,
        "{:?} {:?} {}",
        basis.unresolved,
        basis.expected,
        basis.found()
    );
    assert_eq!(basis.scope, BasisScope::DiscreteWithoutContinuum);
    basis
}

/// The open slab, or the core between `left` and `right`.
fn open() -> AnalyticalMaterialStack<Complex> {
    slab(Termination::Open, Termination::Open)
}

/// On the one-exterior chart of the cladding: κ = i k0 e^z and
/// β² = k0² (χ + e^{2z}), independent of lamina.
fn kappa(z: Complex) -> Complex {
    Complex::I * K * z.exp()
}
fn beta2(z: Complex) -> Complex {
    K * K * (CLAD + (z * 2.).exp())
}

/// The open slab's even or odd condition at chart point z.
fn condition(pol: Polarisation, odd: bool, z: Complex) -> Complex {
    open_value(pol, odd, beta2(z).sqrt(), kappa(z))
}

/// Newton with a central difference, to 1e-14 absolute in z.
fn newton_z(f: &dyn Fn(Complex) -> Complex, mut z: Complex) -> Option<Complex> {
    for _ in 0..100 {
        let h = 1e-7;
        let step = f(z) * 2. * h / (f(z + h) - f(z - h));
        if !step.re.is_finite() || !step.im.is_finite() || z.re > 5. {
            return None;
        }
        z -= step;
        if step.norm() <= 1e-14 {
            return Some(z);
        }
    }
    None
}

fn inside(r: [f64; 4], z: Complex) -> bool {
    r[0] < z.re && z.re < r[1] && r[2] < z.im && z.im < r[3]
}

/// The roots of `f` in the chart rectangle by Newton from a grid,
/// checked against the argument-principle count.
fn roots(f: &dyn Fn(Complex) -> Complex, r: [f64; 4]) -> Vec<Complex> {
    let expected = winding(f, r);
    let mut found: Vec<Complex> = vec![];
    for i in 0..=30 {
        for j in 0..=30 {
            let seed = c(
                r[0] + (r[1] - r[0]) * i as f64 / 30.,
                r[2] + (r[3] - r[2]) * j as f64 / 30.,
            );
            if let Some(z) = newton_z(f, seed)
                && inside(r, z)
                && found.iter().all(|x| (x - z).norm() > 1e-8)
            {
                found.push(z);
            }
        }
    }
    assert_eq!(found.len(), expected);
    found
}

/// The open slab's ladder of `parities` in a chart rectangle.
fn ladder(pol: Polarisation, parities: &[bool], r: [f64; 4]) -> Vec<(bool, Complex)> {
    parities
        .iter()
        .flat_map(|&odd| {
            roots(&|z| condition(pol, odd, z), r)
                .into_iter()
                .map(move |z| (odd, z))
        })
        .collect()
}

/// Each basis mode matches one ladder root in z, β² and sheets; returns
/// the worst |Δz| and the numbers of guided and leaky modes.
fn matches(
    basis: &ModalBasis,
    ladder: &[(bool, Complex)],
    sides: [bool; 2],
) -> (f64, usize, usize) {
    assert_eq!(basis.expected, Some(ladder.len()));
    assert_eq!(basis.found(), ladder.len());
    let (mut worst, mut guided, mut leaky) = (0_f64, 0, 0);
    for mode in &basis.modes {
        let z = mode.point.native_coordinate();
        let (_, root) = ladder
            .iter()
            .min_by(|a, b| (a.1 - z).norm().total_cmp(&(b.1 - z).norm()))
            .unwrap();
        worst = worst.max((root - z).norm());
        let beta = mode.point.physical_coordinate();
        assert!((beta * beta - beta2(*root)).norm() < 1e-8 * K * K, "{beta}");
        let sheet = ExteriorSheet::of(kappa(*root));
        let expected = sides.map(|open| open.then_some(sheet));
        assert_eq!(mode.sheets, expected);
        match sheet {
            ExteriorSheet::Proper => guided += 1,
            ExteriorSheet::Improper => leaky += 1,
        }
    }
    assert!(worst < 1e-9, "{worst:e}");
    (worst, guided, leaky)
}

#[test]
fn open_slab_bases_hold_the_guided_and_leaky_ladders_with_independent_counts() {
    for pol in [TE, TM] {
        let basis = complete(open(), pol, WINDOW);
        let ladder = ladder(pol, &[false, true], WINDOW);
        let (worst, guided, leaky) = matches(&basis, &ladder, [true, true]);
        assert!(guided >= 2 && leaky >= 1, "{guided} {leaky}");
        assert_eq!(basis.chart, CoordinateKind::Conformal);
        eprintln!(
            "{pol:?} open slab: expected = found = independent = {}, {guided} guided and {leaky} \
             leaky, |Δz| {worst:.1e}",
            ladder.len()
        );
    }
}

#[test]
fn a_slab_on_a_wall_holds_the_mirror_doubled_slabs_modes() {
    for pol in [TE, TM] {
        for kind in [WallKind::PerfectElectric, WallKind::PerfectMagnetic] {
            // The core of thickness A on a wall at its left face.
            let stack = Stack::from_analytical_materials(
                Constant::dielectric(CLAD),
                Constant::dielectric(CLAD),
            )
            .analytical_layer(Constant::dielectric(CORE), cm(A))
            .left_termination(Wall::new(kind, cm(0.)))
            .finalise();
            let basis = complete(stack, pol, WINDOW);
            let odd = dirichlet(kind, pol);
            let ladder = ladder(pol, &[odd], WINDOW);
            let (worst, guided, leaky) = matches(&basis, &ladder, [false, true]);
            eprintln!(
                "{pol:?} {kind:?} slab on a wall: the doubled slab's {} {} modes ({guided} \
                 guided, {leaky} leaky), |Δz| {worst:.1e}",
                ladder.len(),
                if odd { "odd" } else { "even" }
            );
        }
    }
}

#[test]
fn a_root_at_a_branch_point_or_on_the_window_edge_is_unresolved() {
    // A TE odd mode just above cutoff, κ = i 2e-5 k0: the core half-width
    // for which q cot(q a) = −α.
    let alpha = 2e-5 * K;
    let q = (1.75 * K * K - alpha * alpha).sqrt();
    let a = (FRAC_PI_2 + (alpha / q).atan()) / q;
    let stack =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
            .analytical_layer(Constant::dielectric(CORE), cm(2. * a))
            .finalise();
    let z = (alpha / K).ln();
    let basis = solve(stack, TE, [z - 1., z + 1., -0.5, 0.5]);
    assert_eq!(basis.expected, Some(1));
    assert_eq!(basis.found(), 0);
    assert_eq!(basis.status, BasisStatus::Incomplete);
    assert!(
        matches!(
            basis.unresolved[..],
            [BasisGap {
                cause: BasisLimit::BranchPoint { .. },
                ..
            }]
        ),
        "{:?}",
        basis.unresolved
    );
    // The open slab's TE leaky root on the window's left edge.
    let leaky = ladder(TE, &[false], WINDOW)
        .into_iter()
        .map(|(_, z)| z)
        .find(|z| z.im < -1.)
        .unwrap();
    let basis = solve(open(), TE, [leaky.re, WINDOW[1], WINDOW[2], WINDOW[3]]);
    assert_eq!(basis.status, BasisStatus::CountFailed);
    assert!(
        basis
            .unresolved
            .iter()
            .any(|g| matches!(g.cause, BasisLimit::BoundaryRoot)),
        "{:?}",
        basis.unresolved
    );
}

#[test]
fn windows_are_checked_against_the_stack_and_the_period() {
    let backend = Scatter2::new();
    let beta = ModalWindow::forward(2. * K, K, 0.01 * K);
    assert!(matches!(
        modal_basis_in_window_with(&backend, open(), request(TE, beta)),
        Err(ModalBasisError::OpenSide)
    ));
    assert!(matches!(
        modal_basis_in_window_with(&backend, open(), request(TE, chart([-1., 0., -3.2, 3.1]))),
        Err(ModalBasisError::Window(WindowError::Period))
    ));
    let walled = slab(
        Wall::perfect_electric(cm(-0.1)),
        Wall::perfect_electric(cm(0.8)),
    );
    assert!(matches!(
        modal_basis_in_window_with(&backend, walled, request(TE, chart(WINDOW))),
        Err(ModalBasisError::Window(WindowError::ClosedStack))
    ));
    // An anisotropic open exterior is outside the open-side bases.
    let uniaxial = Region::oriented(
        lamina_core::AnalyticalTensorMaterialHandle::new(
            lamina_materials::PrincipalAxes::uniaxial(
                Constant::dielectric(CLAD),
                Constant::dielectric(3.),
            ),
        ),
        lamina_core::Rotation::identity(),
    );
    let anisotropic: ModeStack = Stack::builder(
        uniaxial,
        Region::from(lamina_core::AnalyticalMaterialHandle::new(
            Constant::dielectric(CLAD),
        )),
    )
    .layer(
        Region::from(lamina_core::AnalyticalMaterialHandle::new(
            Constant::dielectric(CORE),
        )),
        cm(2. * A),
    )
    .finalise();
    assert!(matches!(
        modal_basis_in_window_with(&backend, anisotropic, request(TE, chart(WINDOW))),
        Err(ModalBasisError::Window(WindowError::AnisotropicExterior {
            region: lamina_core::StackRegion::LeftExterior
        }))
    ));
    // β = 0 at z = ln 1.5 − iπ/2.
    assert!(matches!(
        modal_basis_in_window_with(&backend, open(), request(TE, chart([-1., 1., -2., 0.]))),
        Err(ModalBasisError::Window(WindowError::CoordinateBranchPoint))
    ));
}

/// max |G − I| of a basis's own unconjugated Gram.
fn identity_defect(pairing: &BasisPairing) -> f64 {
    let n = pairing.unconjugated.nrows();
    (pairing.unconjugated.clone() - nalgebra::DMatrix::identity(n, n))
        .iter()
        .map(|v| v.norm())
        .fold(0., f64::max)
}

#[test]
fn guided_and_leaky_modes_are_bi_orthonormal_with_continued_tails() {
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        let basis = complete(open(), pol, WINDOW);
        let pairing = basis_pairing_with(&backend, [&basis, &basis], ORIGIN).unwrap();
        let defect = identity_defect(&pairing);
        assert!(defect < 1e-8, "{defect:e}");
        assert!(pairing.self_defects.iter().all(|d| *d < 1e-8));
        // Conjugated entries are absent exactly where the tails diverge.
        let kappas: Vec<_> = basis
            .modes
            .iter()
            .map(|m| m.point.signature().unwrap().right.unwrap())
            .collect();
        let mut absent = 0;
        for i in 0..kappas.len() {
            for j in 0..kappas.len() {
                let diverges = kappas[i].im + kappas[j].im <= 0.;
                match pairing.conjugated[(i, j)] {
                    Err(Absent::Unbounded { side }) => {
                        assert!(diverges && side == StackSide::Left);
                        absent += 1;
                    }
                    Ok(value) => assert!(!diverges && value.re.is_finite()),
                }
            }
        }
        let leaky = kappas.iter().filter(|k| k.im < 0.).count();
        assert!(absent >= leaky * leaky && leaky >= 1);
        // Guided tails are the decaying ones, v0/(α_a + α_b), to rounding:
        // the conjugated self-pairing of a lossless guided mode is its power.
        let guided: Vec<_> = (0..kappas.len()).filter(|&i| kappas[i].im > 0.).collect();
        for &i in &guided {
            let pair = modal_pairing_with(
                &backend,
                [&basis.modes[i], &basis.modes[i]],
                ORIGIN,
                Default::default(),
            )
            .unwrap();
            let power = pair.conjugated_w_per_m.unwrap();
            assert!((power - pair.unconjugated_w_per_m).norm() < 1e-9, "{power}");
        }
        eprintln!(
            "{pol:?} open slab: |G − I| {defect:.1e} over {} modes, {absent} absent conjugated \
             entries, quadrature change {:.1e} W/m",
            kappas.len(),
            pairing.quadrature_change_w_per_m
        );
    }
}

#[test]
fn positive_power_guided_modes_are_the_signed_unit_power_profiles() {
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        let basis = complete(open(), pol, WINDOW);
        let guided = enumerate_guided_modes(
            open(),
            GuidedWindow {
                wavelength: cm(1.),
                family: pol.into(),
                direction: GuidedDirection::Forward,
                abs_beta: [CLAD.sqrt() * K * (1. + 1e-6), CORE.sqrt() * K],
                cutoff_margin: 1e-8,
                phase_tolerance: 1e-11,
                max_subdivisions: 1000,
                refinement: Default::default(),
                assumption: ASSUME,
            },
        )
        .unwrap();
        assert!(guided.is_complete());
        let z: Vec<f64> = (0..=60).map(|i| -0.5 + 1.7 * i as f64 / 60.).collect();
        let sampling = at(&open(), &z);
        let mut worst: f64 = 0.;
        let mut count = 0;
        let mut pairs = vec![];
        for mode in &basis.modes {
            if mode.sheets[0] != Some(ExteriorSheet::Proper) {
                continue;
            }
            let beta = mode.point.physical_coordinate();
            assert!(beta.re > 0. && beta.im.abs() < 1e-9 * K, "{beta}");
            let point = guided
                .modes
                .iter()
                .find(|p| (p.physical_coordinate() - beta).norm() < 1e-9 * K)
                .unwrap();
            let reference = point
                .guided_power_profile_with(backend, &sampling, Default::default())
                .unwrap();
            let reference = reference.fields.quantity();
            let (e, h) = (reference.electric(), reference.magnetic());
            let fields = mode
                .fields_with(backend, &sampling, Default::default())
                .unwrap();
            let (mut peak, mut error) = (0_f64, 0_f64);
            for (i, sample) in fields.iter().enumerate() {
                let expected = [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]];
                for (value, expected) in sample.iter().zip(expected) {
                    peak = peak.max(expected.norm());
                    error = error.max((value - expected).norm());
                }
            }
            worst = worst.max(error / peak);
            count += 1;
            pairs.push((mode, point));
        }
        assert_eq!(count, guided.modes.len());
        assert!(worst < 1e-9, "{worst:e}");
        // The conjugated pairing with continued tails is the guided power
        // overlap with its decaying tails v0/(α_a + α_b).
        let mut tails: f64 = 0.;
        for (f, p) in &pairs {
            for (g, q) in &pairs {
                let pairing = modal_pairing_with(&backend, [*f, *g], ORIGIN, Default::default())
                    .unwrap()
                    .conjugated_w_per_m
                    .unwrap();
                let overlap =
                    guided_power_overlap_with(&backend, [*p, *q], ORIGIN, Default::default())
                        .unwrap()
                        .power_w_per_m;
                tails = tails.max((pairing - overlap).norm());
            }
        }
        assert!(tails < 1e-9, "{tails:e}");
        eprintln!(
            "{pol:?}: {count} guided modes equal their unit-power profiles to {worst:.1e}; \
             conjugated pairings equal the guided power overlaps to {tails:.1e} W/m"
        );
    }
}

#[test]
fn pairing_accepts_both_open_sides_and_refuses_open_against_closed() {
    let backend = Scatter2::new();
    let first = complete(open(), TE, WINDOW);
    // A different cladding on both sides: accepted.
    let other =
        Stack::from_analytical_materials(Constant::dielectric(2.), Constant::dielectric(2.))
            .analytical_layer(Constant::dielectric(CORE), cm(2. * A))
            .finalise();
    let second = complete(other, TE, [-3., 0.3, -0.5, 0.5]);
    let pairing = basis_pairing_with(&backend, [&first, &second], ORIGIN).unwrap();
    assert_eq!(
        pairing.unconjugated.shape(),
        (first.found(), second.found())
    );
    // Walled on the right: refused, naming the right side.
    let walled = slab(Termination::Open, Wall::perfect_electric(cm(2. * A + 0.1)));
    let third = complete(walled, TE, WINDOW);
    assert!(matches!(
        basis_pairing_with(&backend, [&first, &third], ORIGIN),
        Err(PairingError::Terminations {
            side: StackSide::Right
        })
    ));
}

#[test]
fn a_covering_reaches_the_leaky_mode_from_a_beta_rectangle() {
    let improper = [Some(ExteriorSheet::Improper); 2];
    // The TE leaky mode near β = (1.55 + 0.077i) k0.
    let (re, im) = ([1.45 * K, 1.65 * K], [0.03 * K, 0.15 * K]);
    let covering = ModalWindow::covering(open(), cm(1.), re, im, improper).unwrap();
    let ModalWindow::Chart { re: zr, im: zi } = covering.window else {
        panic!()
    };
    let r = [zr[0], zr[1], zi[0], zi[1]];
    // The rectangle's boundary maps into the covering on the improper sheet.
    for i in 0..=100 {
        for (x, y) in [
            (re[0] + (re[1] - re[0]) * i as f64 / 100., im[0]),
            (re[0] + (re[1] - re[0]) * i as f64 / 100., im[1]),
            (re[0], im[0] + (im[1] - im[0]) * i as f64 / 100.),
            (re[1], im[0] + (im[1] - im[0]) * i as f64 / 100.),
        ] {
            let beta = c(x, y);
            let k = -Complex::I * (beta * beta - CLAD * K * K).sqrt();
            assert!(k.im < 0.);
            let w = k / (Complex::I * K);
            let z = w.ln();
            let z = c(
                z.re,
                z.im + 2. * PI * ((r[2] + r[3]) / 2. / (2. * PI) - z.im / (2. * PI)).round(),
            );
            assert!(inside(r, z), "{z} {r:?}");
        }
    }
    assert!(covering.extra_area_cm2 > 0.);
    let basis = complete(open(), TE, r);
    let ladder = ladder(TE, &[false, true], r);
    let leaky: Vec<_> = basis
        .modes
        .iter()
        .filter(|m| {
            let b = m.point.physical_coordinate();
            let b = if b.re < 0. { -b } else { b };
            b.re > re[0] && b.re < re[1] && b.im > im[0] && b.im < im[1]
        })
        .collect();
    assert_eq!(leaky.len(), 1);
    assert_eq!(leaky[0].sheets, improper);
    matches(&basis, &ladder, [true, true]);
    eprintln!(
        "covering {r:?}: {} modes, extra area {:.3} k0²",
        basis.found(),
        covering.extra_area_cm2 / (K * K)
    );
    // Across the improper sheet's cut on the real axis below the light line.
    assert!(matches!(
        ModalWindow::covering(
            open(),
            cm(1.),
            [1.3 * K, 1.6 * K],
            [-0.01 * K, 0.1 * K],
            improper
        ),
        Err(WindowError::CrossesCut {
            side: StackSide::Left
        })
    ));
}

/// The characteristic of the core between the open cladding on its left and
/// a PEC-backed PML `gap` beyond its right face, independent of lamina: the
/// left exterior's state (ψ, wψ') = (1, −i w κ) carried through the core
/// and the cladding of complex length gap + τ to the wall.
fn open_pml_value(pol: Polarisation, gap: f64, tau: Complex, z: Complex) -> Complex {
    let (b2, k) = (beta2(z), kappa(z));
    let mut state = (Complex::ONE, -Complex::I * weight(pol, CLAD) * k);
    for (epsilon, length) in [(CORE, c(2. * A, 0.)), (CLAD, gap + tau)] {
        let w = weight(pol, epsilon);
        let p2 = epsilon * K * K - b2;
        let (cos, sinc) = trig(p2, length);
        let scale = (Complex::I * epsilon.sqrt() * K * length).exp();
        state = (
            scale * (cos * state.0 + sinc * state.1 / w),
            scale * (-w * p2 * sinc * state.0 + cos * state.1),
        );
    }
    if dirichlet(WallKind::PerfectElectric, pol) {
        state.0
    } else {
        state.1
    }
}

#[test]
fn an_open_side_facing_a_pml_counts_and_pairs() {
    let backend = Scatter2::new();
    let (gap, d) = (0.1, 0.75);
    for pol in [TE, TM] {
        let stack = slab(
            Termination::Open,
            pml(StackSide::Right, gap, d, PROFILE, WallKind::PerfectElectric),
        );
        let r = [-3., 0.35, -2.9, 0.4];
        let basis = complete(stack, pol, r);
        let tau = d * PROFILE.mean();
        let reference = |z| open_pml_value(pol, gap, tau, z);
        let independent = roots(&reference, r);
        assert_eq!(basis.expected, Some(independent.len()));
        assert_eq!(basis.found(), independent.len());
        for mode in &basis.modes {
            let z = mode.point.native_coordinate();
            assert!(independent.iter().any(|x| (x - z).norm() < 1e-9), "{z}");
            assert_eq!(mode.sheets[1], None);
            assert!(mode.sheets[0].is_some());
        }
        let pairing = basis_pairing_with(&backend, [&basis, &basis], ORIGIN).unwrap();
        let defect = identity_defect(&pairing);
        assert!(defect < 1e-8, "{defect:e}");
        eprintln!(
            "{pol:?} open | PML: expected = found = independent = {}, |G − I| {defect:.1e}",
            independent.len()
        );
    }
}
