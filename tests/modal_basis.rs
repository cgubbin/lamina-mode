//! Complete modal bases of stacks closed by walls: counts against the
//! layered-box ladder and an independent winding, boundary and cutoff
//! regions, normalisation, sign rule, ordering, groups and determinism.
mod modal_support;
use lamina_core::{
    AnalyticalMaterialStack, Constant, FieldSampling, LayerSampling, Scatter2, Stack, Wall,
    WallKind,
};
use lamina_mode::*;
use modal_support::*;

const A: f64 = 0.35;
const GAP: f64 = 0.45;
const MARGIN: f64 = 0.05 * K;
/// Largest β and decay of the forward windows.
const BETA_MAX: f64 = 2.1 * K;
const DECAY_MAX: f64 = 3. * K;

fn forward() -> ModalWindow {
    ModalWindow::forward(BETA_MAX, DECAY_MAX, MARGIN)
}

fn basis(stack: AnalyticalMaterialStack<Complex>, request: ModalBasisRequest) -> ModalBasis {
    modal_basis_in_window_with(&Scatter2::new(), stack, request).unwrap()
}

/// The ladder roots inside the forward window.
fn ladder_in_window(kind: WallKind, pol: lamina_core::Polarisation) -> Vec<Complex> {
    let mut roots = ladder(
        kind,
        pol,
        A,
        GAP,
        [-(DECAY_MAX * DECAY_MAX), BETA_MAX * BETA_MAX],
    );
    roots.retain(|b| b.re < BETA_MAX && b.im < DECAY_MAX && b.norm() > MARGIN);
    roots
}

/// A Drude core with Re ε ≈ 3.5 and Im ε ≈ 0.5 at the 1 cm wavelength.
fn lossy(kind: WallKind) -> AnalyticalMaterialStack<Complex> {
    use lamina_units::VacuumAngularWavenumber as W;
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
}

/// A homogeneous box whose first TE mode is just above cutoff, and its β.
fn near_cutoff() -> (AnalyticalMaterialStack<Complex>, f64) {
    let width = 0.5 * (1. + 2e-7);
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::vacuum(), cm(width))
        .left_termination(Wall::perfect_electric(cm(0.)))
        .right_termination(Wall::perfect_electric(cm(width)))
        .finalise();
    (
        stack,
        (K * K - (std::f64::consts::PI / width).powi(2)).sqrt(),
    )
}

/// Two identical cores far apart: an even and an odd supermode closer than
/// the default gap but resolved by the search.
fn twin(separation: f64, kind: WallKind) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(0.3))
        .analytical_layer(Constant::dielectric(CLAD), cm(separation))
        .analytical_layer(Constant::dielectric(CORE), cm(0.3))
        .left_termination(Wall::new(kind, cm(-0.3)))
        .right_termination(Wall::new(kind, cm(0.6 + separation + 0.3)))
        .finalise()
}

fn twin_window() -> ModalWindow {
    ModalWindow::forward(2.05 * K, 0.5 * K, MARGIN)
}

#[test]
fn counts_match_the_layered_box_ladder_and_an_independent_winding() {
    let window = [[-MARGIN, BETA_MAX], [-MARGIN, DECAY_MAX]];
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let expected = ladder_in_window(kind, pol);
            assert!(expected.iter().any(|b| b.im > 0.) && expected.iter().any(|b| b.re > 0.));
            let result = basis(slab(A, GAP, kind), request(pol, forward()));
            assert_eq!(result.chart, CoordinateKind::Wavenumber);
            assert_eq!(result.scope, BasisScope::Discrete);
            assert_eq!(result.window_beta_image, result.window);
            let w = result.window;
            assert_eq!(
                [w.x_min, w.x_max, w.y_min, w.y_max],
                [-MARGIN, BETA_MAX, -MARGIN, DECAY_MAX]
            );
            assert_eq!(
                result.status,
                BasisStatus::Complete,
                "{:?}",
                result.unresolved
            );
            let independent = winding_count(slab(A, GAP, kind), pol, window);
            assert_eq!(result.expected, Some(expected.len()));
            assert_eq!(result.found(), expected.len());
            assert_eq!(independent, expected.len());
            let mut worst: f64 = 0.;
            for root in &expected {
                let nearest = result
                    .modes
                    .iter()
                    .map(|m| (m.point.physical_coordinate() - root).norm())
                    .fold(f64::INFINITY, f64::min);
                worst = worst.max(nearest / K);
            }
            assert!(worst < 1e-10, "{worst:e}");
            for mode in &result.modes {
                let details = mode.point.describe(Default::default()).unwrap();
                assert_eq!(
                    details.interpretation.exterior,
                    [ExteriorBehaviour::Closed; 2]
                );
                if mode.point.physical_coordinate().re > MARGIN {
                    assert_eq!(
                        details.interpretation.classification,
                        Classification::Classified(ModeType::Closed)
                    );
                }
            }
            let real = expected.iter().filter(|b| b.re > 0.).count();
            eprintln!(
                "{pol:?} {kind:?}: expected = found = winding = ladder = {} \
                 ({real} real, {} imaginary), worst |Δβ|/k0 {worst:.1e}",
                expected.len(),
                expected.len() - real
            );
        }
    }
    // Complex roots of a lossy core, against the independent winding.
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let window = [[-MARGIN, 2.2 * K], [-MARGIN, 2. * K]];
            let result = basis(
                lossy(kind),
                request(pol, ModalWindow::forward(2.2 * K, 2. * K, MARGIN)),
            );
            assert_eq!(
                result.status,
                BasisStatus::Complete,
                "{:?}",
                result.unresolved
            );
            let independent = winding_count(lossy(kind), pol, window);
            assert_eq!(result.expected, Some(independent));
            assert_eq!(result.found(), independent);
            let complex = result
                .modes
                .iter()
                .filter(|m| {
                    let b = m.point.physical_coordinate();
                    b.re > 1e-3 * K && b.im > 1e-3 * K
                })
                .count();
            assert!(complex > 0);
            eprintln!(
                "lossy {pol:?} {kind:?}: expected = found = winding = {independent}, {complex} complex"
            );
        }
    }
}

#[test]
fn a_root_on_the_window_edge_or_at_the_origin_is_left_unresolved() {
    let (kind, pol) = (PEC, TE);
    let roots = ladder_in_window(kind, pol);
    let top = roots.iter().map(|b| b.re).fold(0., f64::max);
    let edge = ModalWindow::Wavenumber {
        re_cm: [-MARGIN, top],
        im_cm: [-MARGIN, DECAY_MAX],
    };
    let result = basis(slab(A, GAP, kind), request(pol, edge));
    assert_eq!(result.status, BasisStatus::CountFailed);
    assert_eq!(result.expected, None);
    let region = result
        .unresolved
        .iter()
        .find(|r| matches!(r.cause, BasisLimit::BoundaryRoot))
        .unwrap();
    let b = region.beta_image;
    assert!(b.x_min <= top && top <= b.x_max && b.y_min <= 0. && 0. <= b.y_max);
    // Every other mode is returned once; the one on the edge is not invented.
    assert_eq!(result.found(), roots.len() - 1);
    assert!(
        result
            .modes
            .iter()
            .all(|m| (m.point.physical_coordinate().re - top).abs() > 1e-6 * K)
    );
    // A window whose mirror square around β = 0 holds the slowest root.
    let slowest = roots.iter().map(|b| b.norm()).fold(f64::INFINITY, f64::min);
    let wide = ModalWindow::forward(BETA_MAX, DECAY_MAX, 1.5 * slowest);
    let result = basis(slab(A, GAP, kind), request(pol, wide));
    assert_eq!(result.status, BasisStatus::Incomplete);
    let cutoff: Vec<_> = result
        .unresolved
        .iter()
        .filter(|r| matches!(r.cause, BasisLimit::Cutoff))
        .collect();
    assert!(!cutoff.is_empty());
    assert!(
        result
            .modes
            .iter()
            .all(|m| m.point.physical_coordinate().norm() > 1.5 * slowest)
    );
    // The root and its mirror are both counted.
    let inside = cutoff.iter().map(|r| r.count.unwrap_or(0)).sum::<usize>();
    assert_eq!(result.expected, Some(result.found() + inside));
    eprintln!(
        "edge root at β/k0 = {:.6}: count failed, {} other modes kept; \
         mirror square |β| ≤ {:.4} k0 holds {inside} roots",
        top / K,
        roots.len() - 1,
        1.5 * slowest / K
    );
}

#[test]
fn real_modes_are_the_walled_guided_modes_with_their_unit_power_fields() {
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let stack = slab(A, GAP, kind);
            let result = basis(stack.clone(), request(pol, forward()));
            let guided = enumerate_guided_modes(
                stack.clone(),
                GuidedWindow {
                    wavelength: cm(1.),
                    polarisation: pol,
                    direction: GuidedDirection::Forward,
                    abs_beta: [MARGIN, BETA_MAX],
                    cutoff_margin: 1e-8,
                    phase_tolerance: 1e-11,
                    max_subdivisions: 1000,
                    refinement: Default::default(),
                    assumption: ASSUME,
                },
            )
            .unwrap();
            assert!(guided.is_complete());
            let real: Vec<_> = result
                .modes
                .iter()
                .filter(|m| m.point.physical_coordinate().re > MARGIN)
                .collect();
            assert_eq!(real.len(), guided.modes.len());
            let (sampling, _) = wall_to_wall(&stack, [GAP, GAP], 40);
            let mut worst_beta: f64 = 0.;
            let mut worst_field: f64 = 0.;
            for mode in &real {
                let beta = mode.point.physical_coordinate();
                let point = guided
                    .modes
                    .iter()
                    .min_by(|a, b| {
                        let d = |p: &ModePoint| (p.physical_coordinate() - beta).norm();
                        d(a).total_cmp(&d(b))
                    })
                    .unwrap();
                worst_beta = worst_beta.max((point.physical_coordinate() - beta).norm() / beta.re);
                let reference = point
                    .guided_power_profile_with(backend, &sampling, Default::default())
                    .unwrap();
                let reference = reference.fields.quantity();
                let (e, h) = (reference.electric(), reference.magnetic());
                let fields = mode
                    .fields_with(backend, &sampling, Default::default())
                    .unwrap();
                let mut peak: f64 = 0.;
                let mut error: f64 = 0.;
                for (i, sample) in fields.iter().enumerate() {
                    let expected = [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]];
                    for (value, expected) in sample.iter().zip(expected) {
                        peak = peak.max(expected.norm());
                        error = error.max((value - expected).norm());
                    }
                }
                worst_field = worst_field.max(error / peak);
            }
            assert!(worst_beta < 1e-10, "{worst_beta:e}");
            assert!(worst_field < 1e-9, "{worst_field:e}");
            eprintln!(
                "{pol:?} {kind:?}: {} real modes, |Δβ|/β {worst_beta:.1e}, fields {worst_field:.1e}",
                real.len()
            );
        }
    }
}

/// The wall reference ℓ = E_y + Z0 H_y + i (Z0 H_x − E_x) at the first interface.
#[test]
fn a_wall_on_the_first_interface_signs_by_the_wall_reference() {
    let backend = Scatter2::new();
    for (kind, pol) in [(PEC, TE), (PMC, TM)] {
        let stack = Stack::from_analytical_materials(
            Constant::dielectric(CLAD),
            Constant::dielectric(CLAD),
        )
        .analytical_layer(Constant::dielectric(CORE), cm(0.7))
        .left_termination(Wall::new(kind, cm(0.)))
        .right_termination(Wall::new(kind, cm(1.2)))
        .finalise();
        let result = basis(stack, request(pol, forward()));
        assert_eq!(result.status, BasisStatus::Complete);
        let at = FieldSampling::new().layer(0, LayerSampling::offsets([cm(0.)]));
        for mode in &result.modes {
            assert_eq!(mode.sign_reference, SignReference::WallOnFirstInterface);
            assert!(!mode.sign_ambiguous(), "{:?}", mode.sign_margin);
            let s = mode.fields_with(backend, &at, Default::default()).unwrap()[0];
            // The wall's tangential component vanishes there.
            let wall = if kind == PEC { s[1] } else { s[4] };
            assert!(wall.norm() < 1e-9 * s.iter().map(|v| v.norm()).fold(0., f64::max));
            let l = s[1] + s[4] + Complex::I * (s[3] - s[0]);
            assert!(l.re > 0., "{l}");
        }
        let gapped = basis(slab(A, GAP, kind), request(pol, forward()));
        assert!(
            gapped
                .modes
                .iter()
                .all(|m| m.sign_reference == SignReference::FirstInterface && !m.sign_ambiguous())
        );
        let smallest = result
            .modes
            .iter()
            .filter_map(|m| m.sign_margin)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "{pol:?} {kind:?} wall on the first interface: smallest sign margin {smallest:.3e}"
        );
    }
}

#[test]
fn modes_are_ordered_by_decreasing_re_beta_squared_then_decay() {
    let result = basis(slab(A, GAP, PEC), request(TE, forward()));
    let betas: Vec<_> = result
        .modes
        .iter()
        .map(|m| m.point.physical_coordinate())
        .collect();
    let propagating = betas.iter().take_while(|b| b.re > MARGIN).count();
    assert!(propagating > 0 && propagating < betas.len());
    assert!(betas[..propagating].windows(2).all(|w| w[0].re > w[1].re));
    assert!(
        betas[propagating..]
            .iter()
            .all(|b| b.re.abs() < 1e-9 * K && b.im > 0.)
    );
    assert!(betas[propagating..].windows(2).all(|w| w[0].im < w[1].im));
    assert_eq!(
        ModalBasis::ORDERING,
        "decreasing Re β², then decreasing Im β², then decreasing Re β"
    );
}

#[test]
fn a_close_pair_forms_one_group_with_a_bi_orthonormal_basis() {
    let backend = Scatter2::new();
    let stack = twin(2.15, PEC);
    let grouped = basis(stack.clone(), request(TE, twin_window()));
    assert_eq!(
        grouped.status,
        BasisStatus::Complete,
        "{:?}",
        grouped.unresolved
    );
    assert_eq!(grouped.groups.len(), 1, "{:?}", grouped.groups);
    let [i, j] = [grouped.groups[0][0], grouped.groups[0][1]];
    let beta = |k: usize| grouped.modes[k].point.physical_coordinate();
    let split = (beta(i) - beta(j)).norm();
    assert!(split < 1e-6 * K && split > 1e-9 * K, "{split:e}");
    let registration = GuidedOverlapRegistration {
        origins_cm: [0., 0.],
    };
    let pairing = |a: &BasisMode, b: &BasisMode| {
        modal_pairing_with(&backend, [a, b], registration, Default::default())
            .unwrap()
            .unconjugated_w_per_m
    };
    let (a, b) = (&grouped.modes[i], &grouped.modes[j]);
    assert!((pairing(a, a) - 1.).norm() < 1e-9);
    assert!((pairing(b, b) - 1.).norm() < 1e-9);
    assert!(pairing(a, b).norm() < 1e-9);
    // Even and odd supermodes are already bi-orthogonal: the group's basis is
    // the individually normalised members.
    let mut single = request(TE, twin_window());
    single.degeneracy_gap_cm = Some(0.);
    let single = basis(stack.clone(), single);
    assert!(single.groups.is_empty());
    let (sampling, _) = wall_to_wall(&stack, [0.3, 0.3], 30);
    let mut worst: f64 = 0.;
    for k in [i, j] {
        let x = grouped.modes[k]
            .fields_with(backend, &sampling, Default::default())
            .unwrap();
        let y = single.modes[k]
            .fields_with(backend, &sampling, Default::default())
            .unwrap();
        let peak = y.iter().flatten().map(|v| v.norm()).fold(0., f64::max);
        for (p, q) in x.iter().flatten().zip(y.iter().flatten()) {
            worst = worst.max((p - q).norm() / peak);
        }
    }
    assert!(worst < 1e-6, "{worst:e}");
    eprintln!(
        "close pair split {:.2e} k0, grouped at gap {:.1e} k0; members unchanged to {worst:.1e}",
        split / K,
        grouped.degeneracy_gap_cm / K
    );
}

#[test]
fn a_mode_near_cutoff_is_kept_ill_conditioned() {
    let (stack, beta) = near_cutoff();
    let mut ill = request(TE, ModalWindow::forward(1.1 * K, 0.5 * K, 0.1 * beta));
    ill.self_pairing_rtol = 1e-2;
    let result = basis(stack, ill);
    assert_eq!(
        result.found(),
        1,
        "{:?} {:?}",
        result.unresolved,
        result.expected
    );
    let Normalisation::IllConditioned { ratio } = result.modes[0].normalisation else {
        panic!("{:?}", result.modes[0].normalisation);
    };
    assert!(ratio < 1e-2);
    assert_eq!(result.status, BasisStatus::Incomplete);
    assert!(result.modes[0].sign_margin.is_none());
    assert!(matches!(
        result.modes[0].fields_with(
            Scatter2::new(),
            &FieldSampling::new().layer_centres(),
            Default::default()
        ),
        Err(BasisFieldError::IllConditioned)
    ));
    eprintln!("β = {:.2e} k0: self-pairing ratio {ratio:.2e}", beta / K);
}

#[test]
fn repeated_solves_agree_bit_for_bit_and_rescaled_stacks_to_tolerance() {
    let backend = Scatter2::new();
    let solve = |s: f64| {
        let stack =
            Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(1.))
                .analytical_layer(Constant::dielectric(CORE), cm(0.6 * s))
                .left_termination(Wall::perfect_electric(cm(-0.4 * s)))
                .right_termination(Wall::perfect_electric(cm(1.1 * s)))
                .finalise();
        let window = ModalWindow::forward(2.05 * K / s, 2. * K / s, MARGIN / s);
        let r = ModalBasisRequest::new(cm(s), ModeFamily::Polarised(TE), window);
        basis(stack, r)
    };
    let sampling = |s: f64| {
        FieldSampling::new()
            .left_exterior(lamina_core::ExteriorSampling::uniform(5, cm(0.4 * s)))
            .layer(0, LayerSampling::uniform(9))
    };
    let fields = |b: &ModalBasis, s: f64| {
        b.modes
            .iter()
            .map(|m| {
                m.fields_with(backend, &sampling(s), Default::default())
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (first, second) = (solve(1.), solve(1.));
    assert_eq!(first.status, BasisStatus::Complete);
    assert_eq!(first.found(), second.found());
    for (a, b) in first.modes.iter().zip(&second.modes) {
        let bits = |m: &BasisMode| {
            let b = m.point.physical_coordinate();
            [b.re.to_bits(), b.im.to_bits()]
        };
        assert_eq!(bits(a), bits(b));
    }
    assert_eq!(fields(&first, 1.), fields(&second, 1.));
    let s = 3.7e-4;
    let scaled = solve(s);
    assert_eq!(
        scaled.status,
        BasisStatus::Complete,
        "{:?} {:?} {}",
        scaled.unresolved,
        scaled.expected,
        scaled.found()
    );
    assert_eq!(scaled.found(), first.found());
    let mut index: f64 = 0.;
    for (a, b) in first.modes.iter().zip(&scaled.modes) {
        let n = |m: &BasisMode, s: f64| m.point.physical_coordinate() * s / K;
        index = index.max((n(a, 1.) - n(b, s)).norm());
    }
    // ∫ dz scales by s, so the normalised fields scale by 1/√s.
    let (x, y) = (fields(&first, 1.), fields(&scaled, s));
    let mut field: f64 = 0.;
    for (p, q) in x.iter().zip(&y) {
        let peak = p.iter().flatten().map(|v| v.norm()).fold(0., f64::max);
        for (u, v) in p.iter().flatten().zip(q.iter().flatten()) {
            field = field.max((u - v * s.sqrt()).norm() / peak);
        }
    }
    assert!(index < 1e-10, "{index:e}");
    assert!(field < 1e-8, "{field:e}");
    eprintln!("rescaled by {s:e}: n_eff {index:.1e}, fields {field:.1e}");
}

/// The 1e-6 k0 gap and 1e-8 self-pairing threshold against the walled
/// fixtures: the smallest true gap, the largest numerical gap, the smallest
/// healthy self-pairing ratio and the near-cutoff ratio.
#[test]
fn default_gap_and_threshold_separate_the_fixtures() {
    let mut true_gap = f64::INFINITY;
    let mut numerical_gap: f64 = 0.;
    let mut healthy = f64::INFINITY;
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let result = basis(slab(A, GAP, kind), request(pol, forward()));
            let betas: Vec<_> = result
                .modes
                .iter()
                .map(|m| m.point.physical_coordinate())
                .collect();
            for (i, a) in betas.iter().enumerate() {
                for b in &betas[..i] {
                    true_gap = true_gap.min((a - b).norm() / K);
                }
            }
            for root in ladder_in_window(kind, pol) {
                let nearest = betas
                    .iter()
                    .map(|b| (b - root).norm())
                    .fold(f64::INFINITY, f64::min);
                numerical_gap = numerical_gap.max(nearest / K);
                if nearest / K > 1e-3 {
                    eprintln!(
                        "{pol:?} {kind:?} {root} {:?} {:?} {:?}",
                        result.status, result.expected, result.unresolved
                    );
                }
            }
            for mode in &result.modes {
                if let Normalisation::Unit { ratio } = mode.normalisation {
                    healthy = healthy.min(ratio);
                }
            }
        }
    }
    let pair = basis(twin(2.15, PEC), request(TE, twin_window()));
    let split = (pair.modes[0].point.physical_coordinate()
        - pair.modes[1].point.physical_coordinate())
    .norm()
        / K;
    let (stack, beta) = near_cutoff();
    let near = basis(
        stack,
        request(TE, ModalWindow::forward(1.1 * K, 0.5 * K, 0.1 * beta)),
    );
    let Normalisation::Unit { ratio: near_ratio } = near.modes[0].normalisation else {
        panic!("{:?}", near.modes[0].normalisation)
    };
    assert!(numerical_gap < 1e-3 * 1e-6, "{numerical_gap:e}");
    assert!(true_gap > 1e-6, "{true_gap:e}");
    assert!(split < 1e-6, "{split:e}");
    assert!(healthy > 1e3 * 1e-8, "{healthy:e}");
    assert!(near_ratio > 1e-8, "{near_ratio:e}");
    eprintln!(
        "defaults: smallest true gap {true_gap:.2e} k0 (twin cores {split:.2e} k0, grouped), \
         largest numerical gap {numerical_gap:.2e} k0; smallest healthy ratio {healthy:.2e}, \
         near-cutoff ratio {near_ratio:.2e} at β = {:.2e} k0",
        beta / K
    );
}
