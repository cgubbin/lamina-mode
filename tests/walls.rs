//! Stacks closed by perfect electric or magnetic walls: the layered-box ladder,
//! one-wall images, classification, refusals, and fields and derivatives with
//! the walls fixed in the stack frame.
use lamina_core::{
    AnalyticalMaterialStack, Constant, ExteriorSampling, FieldSampling, LayerSampling, Pml,
    Polarisation, Scatter2, Stack, StackCompileError, StackSide, StretchProfile, Wall, WallKind,
};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::TAU;

const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;
const PEC: WallKind = WallKind::PerfectElectric;
const PMC: WallKind = WallKind::PerfectMagnetic;
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
/// Core and cladding relative permittivities; the wavelength is 1 cm.
const CORE: f64 = 4.;
const CLAD: f64 = 2.25;
const K: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

/// A core of thickness `2a` in cladding, closed on each side by an optional
/// wall `gap` beyond its interface.
fn slab(a: f64, gap: f64, walls: [Option<WallKind>; 2]) -> AnalyticalMaterialStack<Complex> {
    let mut stack =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
            .analytical_layer(Constant::dielectric(CORE), cm(2. * a));
    if let Some(kind) = walls[0] {
        stack = stack.left_termination(Wall::new(kind, cm(-gap)));
    }
    if let Some(kind) = walls[1] {
        stack = stack.right_termination(Wall::new(kind, cm(2. * a + gap)));
    }
    stack.finalise()
}

fn dirichlet(kind: WallKind, pol: Polarisation) -> bool {
    (kind == PEC) == (pol == TE)
}

/// The real-β ladder of a symmetric core |z| < a in cladding out to walls at
/// |z| = a + gap, from the even and odd wall conditions in cross-multiplied
/// form. With S = sin(pL)/p and C = cos(pL), both entire in p², these are
/// finite through every tangent pole and through p = 0, so guided (p
/// imaginary) and radiation-like (p real) modes need no branch choice.
fn ladder(kind: WallKind, pol: Polarisation, a: f64, gap: f64, window: [f64; 2]) -> Vec<f64> {
    let (w1, w2) = if pol == TE {
        (1., 1.)
    } else {
        (1. / CORE, 1. / CLAD)
    };
    let condition = |odd: bool, b: f64| {
        let q = (CORE * K * K - b * b).sqrt();
        let p2 = CLAD * K * K - b * b;
        let (s, c) = if p2 >= 0. {
            let p = p2.sqrt();
            (
                if p == 0. { gap } else { (p * gap).sin() / p },
                (p * gap).cos(),
            )
        } else {
            let g = (-p2).sqrt();
            ((g * gap).sinh() / g, (g * gap).cosh())
        };
        let (sin, cos) = ((q * a).sin(), (q * a).cos());
        let sinc = if q == 0. { a } else { sin / q };
        match (odd, dirichlet(kind, pol)) {
            (false, true) => w1 * q * sin * s - w2 * cos * c,
            (false, false) => w1 * q * sin * c + w2 * p2 * s * cos,
            (true, true) => w1 * cos * s + w2 * sinc * c,
            (true, false) => w1 * cos * c - w2 * p2 * s * sinc,
        }
    };
    let mut roots = vec![];
    for odd in [false, true] {
        let n = 200_000;
        let at = |i: usize| window[0] + (window[1] - window[0]) * i as f64 / n as f64;
        for i in 0..n {
            let (mut lo, mut hi) = (at(i), at(i + 1));
            let (fl, fh) = (condition(odd, lo), condition(odd, hi));
            assert!(fl != 0. && fh != 0., "a root on a scan point");
            if fl.signum() == fh.signum() {
                continue;
            }
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if condition(odd, mid).signum() == fl.signum() {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            roots.push(0.5 * (lo + hi));
        }
    }
    roots.sort_by(f64::total_cmp);
    roots
}

fn window(pol: Polarisation, abs_beta: [f64; 2]) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.),
        polarisation: pol,
        direction: GuidedDirection::Forward,
        abs_beta,
        cutoff_margin: 1e-8,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: ASSUME,
    }
}

fn betas(basis: &GuidedBasis) -> Vec<f64> {
    assert!(basis.is_complete(), "{:?}", basis.unresolved);
    let mut betas: Vec<_> = basis
        .modes
        .iter()
        .map(|p| p.physical_coordinate().re)
        .collect();
    betas.sort_by(f64::total_cmp);
    betas
}

fn assert_ladder(found: &[f64], expected: &[f64], tolerance: f64) -> f64 {
    assert_eq!(found.len(), expected.len(), "{found:?} != {expected:?}");
    let mut worst: f64 = 0.;
    for (f, e) in found.iter().zip(expected) {
        worst = worst.max((f - e).abs() / e);
    }
    assert!(worst < tolerance, "{found:?} != {expected:?}");
    worst
}

/// Box parameters: half-width, gap and a |β| window from well below the
/// cladding light line (1.5 k0) to just below the core's (2 k0).
const A: f64 = 0.35;
const GAP: f64 = 0.45;
const BOX_WINDOW: [f64; 2] = [0.4 * K, 1.999 * K];

/// Accepted roots over `bounds` with argloc's count. argloc may leave
/// well-conditioned roots together as one cluster; each cluster's enclosure
/// is searched again in halves until it resolves.
fn resolved_roots(
    problem: &PropagationProblem,
    bounds: ConformalRegion<f64>,
    domain: &dyn Fn(ConformalRegion<f64>) -> Domain,
) -> (Vec<ModePoint>, usize) {
    let report = problem
        .plan(vec![domain(bounds)], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    let mut points = report.observations().to_vec();
    let mut count = 0;
    for chart in report.discovery().entries() {
        for estimate in chart.unwrap().estimates() {
            let raw = estimate.raw();
            if raw.multiplicity == 1 {
                count += 1;
                continue;
            }
            let r = raw.enclosure;
            let mid = 0.5 * (r.x_min + r.x_max);
            for (lo, hi) in [(r.x_min, mid), (mid, r.x_max)] {
                let half = ConformalRegion::new(lo, hi, r.y_min, r.y_max).unwrap();
                let (more, n) = resolved_roots(problem, half, domain);
                points.extend(more);
                count += n;
            }
        }
    }
    (points, count)
}

#[test]
fn direct_search_finds_the_layered_box_ladder_once_each() {
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let expected = ladder(kind, pol, A, GAP, BOX_WINDOW);
            assert!(expected.iter().any(|b| *b < CLAD.sqrt() * K));
            assert!(expected.iter().any(|b| *b > CLAD.sqrt() * K));
            let problem = PropagationProblem::new(slab(A, GAP, [Some(kind); 2]), K, pol).unwrap();
            let bounds = ConformalRegion::new(BOX_WINDOW[0], BOX_WINDOW[1], -0.5, 0.5).unwrap();
            let direct = |bounds| {
                let domain = problem.direct_domain(bounds, ASSUME).unwrap();
                assert_eq!(
                    domain.provenance().coordinate_kind(),
                    CoordinateKind::Wavenumber
                );
                domain
            };
            let (points, count) = resolved_roots(&problem, bounds, &direct);
            assert_eq!(count, expected.len());
            let mut found = vec![];
            for point in &points {
                let beta = point.physical_coordinate();
                assert!(beta.im.abs() < 1e-9 * beta.re, "{beta}");
                assert_eq!(point.native_coordinate(), beta);
                assert_eq!(point.signature().unwrap().exteriors(), [None, None]);
                let details = point.describe(Default::default()).unwrap();
                assert_eq!(
                    details.interpretation.exterior,
                    [ExteriorBehaviour::Closed; 2]
                );
                assert_eq!(details.interpretation.flux, [Flux::Closed; 2]);
                assert_eq!(
                    details.interpretation.classification,
                    Classification::Classified(ModeType::Closed)
                );
                found.push(beta.re);
            }
            found.sort_by(f64::total_cmp);
            let error = assert_ladder(&found, &expected, 1e-10);
            eprintln!(
                "direct box ladder {pol:?} {kind:?}: {} modes, worst relative error {error:.1e}",
                expected.len()
            );
        }
    }
}

#[test]
fn enumeration_returns_the_whole_real_ladder_between_walls() {
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let expected = ladder(kind, pol, A, GAP, BOX_WINDOW);
            let basis =
                enumerate_guided_modes(slab(A, GAP, [Some(kind); 2]), window(pol, BOX_WINDOW))
                    .unwrap();
            assert_eq!(basis.light_line, 0.);
            assert_eq!(basis.eligible, Some(BOX_WINDOW));
            let error = assert_ladder(&betas(&basis), &expected, 1e-10);
            eprintln!(
                "enumerated box ladder {pol:?} {kind:?}: {} modes, worst relative error {error:.1e}",
                expected.len()
            );
        }
    }
}

#[test]
fn a_wall_selects_the_odd_or_even_modes_of_the_mirrored_stack() {
    // A core of thickness d a gap g from one wall; the mirror image through the
    // wall is an open stack with two cores 2g apart.
    let (d, g) = (0.4, 0.15);
    let mirrored =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
            .analytical_layer(Constant::dielectric(CORE), cm(d))
            .analytical_layer(Constant::dielectric(CLAD), cm(2. * g))
            .analytical_layer(Constant::dielectric(CORE), cm(d))
            .finalise();
    let half = |kind: WallKind, left: bool| {
        let stack = Stack::from_analytical_materials(
            Constant::dielectric(CLAD),
            Constant::dielectric(CLAD),
        )
        .analytical_layer(Constant::dielectric(CORE), cm(d));
        if left {
            stack.left_termination(Wall::new(kind, cm(-g)))
        } else {
            stack.right_termination(Wall::new(kind, cm(d + g)))
        }
        .finalise()
    };
    let range = [0., 2. * K];
    for pol in [TE, TM] {
        let doubled = enumerate_guided_modes(mirrored.clone(), window(pol, range)).unwrap();
        let all = betas(&doubled);
        for left in [true, false] {
            let mut union = vec![];
            for kind in [PEC, PMC] {
                let basis = enumerate_guided_modes(half(kind, left), window(pol, range)).unwrap();
                assert_eq!(basis.light_line, CLAD.sqrt() * K);
                for point in &basis.modes {
                    let details = point.describe(Default::default()).unwrap();
                    let interpretation = details.interpretation;
                    let (closed, open) = if left { (0, 1) } else { (1, 0) };
                    assert_eq!(interpretation.exterior[closed], ExteriorBehaviour::Closed);
                    assert_eq!(interpretation.exterior[open], ExteriorBehaviour::Decay);
                    assert_eq!(interpretation.flux[closed], Flux::Closed);
                    assert_eq!(
                        interpretation.classification,
                        Classification::Classified(ModeType::Guided)
                    );
                }
                // The matching mode of the mirrored stack vanishes (Dirichlet)
                // or not (Neumann) on the mirror plane, halfway across the gap.
                for beta in betas(&basis) {
                    let image = doubled
                        .modes
                        .iter()
                        .min_by(|a, b| {
                            let distance =
                                |p: &ModePoint| (p.physical_coordinate().re - beta).abs();
                            distance(a).total_cmp(&distance(b))
                        })
                        .unwrap();
                    assert!((image.physical_coordinate().re - beta).abs() < 1e-10 * beta);
                    let sampling = FieldSampling::new()
                        .layer(0, LayerSampling::uniform(64))
                        .layer(1, LayerSampling::point(cm(g)));
                    let fields = image
                        .recover_fields_with(Scatter2::new(), &sampling, Default::default())
                        .unwrap();
                    let primary = if pol == TE {
                        fields.fields.quantity().electric().y().to_vec()
                    } else {
                        fields.fields.quantity().magnetic().y().to_vec()
                    };
                    let peak = primary.iter().map(|z| z.norm()).fold(0., f64::max);
                    let mirror = primary[primary.len() - 1].norm() / peak;
                    assert_eq!(mirror < 1e-6, dirichlet(kind, pol), "{mirror:e}");
                    union.push(beta);
                }
            }
            union.sort_by(f64::total_cmp);
            let error = assert_ladder(&union, &all, 1e-10);
            eprintln!(
                "images {pol:?} left={left}: {} modes, worst relative error {error:.1e}",
                all.len()
            );
        }
    }
}

#[test]
fn closed_sides_refuse_branch_domains_and_open_sides_refuse_direct_ones() {
    let closed = PropagationProblem::new(slab(A, GAP, [Some(PEC); 2]), K, TE).unwrap();
    let region = ConformalRegion::new(-0.5, 0.5, -0.5, 0.5).unwrap();
    let refused = closed
        .domain(
            region,
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            Default::default(),
            ASSUME,
        )
        .unwrap_err();
    assert!(
        refused
            .find_source::<ConformalChartError<InPlaneRegionError>>()
            .is_some_and(|e| matches!(e, ConformalChartError::ClosedStack))
    );
    assert!(
        closed
            .canonical_plan(
                InPlaneSearchExtent {
                    max_parallel_wavenumber: 3. * K,
                    imaginary_margin: 0.1,
                },
                ReferenceSelection::Sheet(CoordinateSheet::Principal),
                Default::default(),
                ASSUME,
            )
            .is_err()
    );
    let one_wall = PropagationProblem::new(slab(A, GAP, [Some(PMC), None]), K, TE).unwrap();
    assert!(one_wall.direct_domain(region, ASSUME).is_err());
    let frequency =
        FrequencyProblem::new(slab(A, GAP, [None, Some(PEC)]), 1.7 * K, TM).unwrap_err();
    assert!(
        frequency
            .find_source::<ConstructionError>()
            .is_some_and(|e| matches!(e, ConstructionError::ClosedSide))
    );
}

/// The first guided mode of a slab between walls, with walls `gap` beyond
/// each interface (a thicker right gap makes the profile asymmetric).
fn walled_mode(kind: WallKind, pol: Polarisation, d: f64, gaps: [f64; 2]) -> ModePoint {
    let stack =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(1.))
            .analytical_layer(Constant::dielectric(CORE), cm(d))
            .left_termination(Wall::new(kind, cm(-gaps[0])))
            .right_termination(Wall::new(kind, cm(d + gaps[1])))
            .finalise();
    let basis = enumerate_guided_modes(stack, window(pol, [1.6 * K, 1.999 * K])).unwrap();
    assert!(basis.is_complete());
    basis.modes.into_iter().next().unwrap()
}

#[test]
fn unit_power_profiles_meet_the_wall_conditions() {
    let gaps = [0.2, 0.3];
    let d = 0.5;
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let point = walled_mode(kind, pol, d, gaps);
            let n = 2000;
            let sampling = FieldSampling::new()
                .left_exterior(ExteriorSampling::uniform(n + 1, cm(gaps[0])))
                .layer(0, LayerSampling::uniform_with_boundaries(n + 1, true, true))
                .right_exterior(ExteriorSampling::uniform(n + 1, cm(gaps[1])));
            let profile = point
                .guided_power_profile_with(Scatter2::new(), &sampling, Default::default())
                .unwrap();
            let fields = profile.fields.quantity();
            let (e, h) = (fields.electric(), fields.magnetic());
            // The tangential field the wall forces to vanish: E_y or E_x at a
            // PEC wall, H_x or H_y at a PMC wall.
            let tangential = match (kind, pol) {
                (PEC, TE) => e.y(),
                (PEC, TM) => e.x(),
                (_, TE) => h.x(),
                (_, TM) => h.y(),
            };
            let peak = tangential.iter().map(|z| z.norm()).fold(0., f64::max);
            let last = tangential.len() - 1;
            for wall in [0, last] {
                assert!(
                    tangential[wall].norm() < 1e-9 * peak,
                    "{:e}",
                    tangential[wall].norm() / peak
                );
            }
            // Re-integrate the returned profile: one W per metre.
            let flux: Vec<_> = (0..=last)
                .map(|i| 0.5 * (e.y()[i] * h.z()[i].conj() - e.z()[i] * h.y()[i].conj()).re)
                .collect();
            let simpson = |block: usize, width: f64| {
                let f = &flux[block * (n + 1)..(block + 1) * (n + 1)];
                let inner: f64 = (1..n)
                    .map(|j| f[j] * if j % 2 == 1 { 4. } else { 2. })
                    .sum();
                width * (f[0] + f[n] + inner) / (3. * n as f64)
            };
            let power =
                (simpson(0, gaps[0]) + simpson(1, d) + simpson(2, gaps[1])) * 0.01 / 376.730313412;
            assert!((power - 1.).abs() < 1e-8, "{power}");
        }
    }
}

#[test]
fn sensitivities_and_group_index_hold_the_walls_fixed() {
    let (d, gaps) = (0.5, [0.2, 0.3]);
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let point = walled_mode(kind, pol, d, gaps);
            let beta =
                |d: f64, gaps: [f64; 2]| walled_mode(kind, pol, d, gaps).physical_coordinate().re;
            // Thickening the core moves the right interface towards its wall.
            let h = 1e-4;
            let fd = (beta(d + h, [gaps[0], gaps[1] - h]) - beta(d - h, [gaps[0], gaps[1] + h]))
                / (2. * h);
            let direction = GuidedPerturbation::thickness(1, 0).unwrap();
            let sensitivity = point
                .guided_beta_sensitivity_with(
                    &backend,
                    &direction,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            let error = (sensitivity.beta_rate.re - fd).abs() / fd.abs();
            assert!(error < 1e-6, "{} != {fd}", sensitivity.beta_rate);
            let slopes = point
                .local_derivatives_with(
                    backend,
                    &[
                        PhysicalParameter::K0,
                        PhysicalParameter::Beta,
                        PhysicalParameter::LayerThickness(0.into()),
                    ],
                    LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
                    Default::default(),
                )
                .unwrap();
            let slope = slopes
                .implicit_slope(
                    ImplicitSlopeRequest::BetaPerThickness(0.into()),
                    Default::default(),
                )
                .unwrap();
            assert!((slope.value.re - fd).abs() < 1e-6 * fd.abs());
            // Group index: certified by the stored energy over the gaps, and
            // against re-enumerated roots across the wavelength.
            let group = point
                .guided_group_index_with(&backend, Default::default(), 1e-8)
                .unwrap();
            let at = |k: f64| {
                let stack = point.provenance().stack().clone();
                let mut request = window(pol, [1.6 * k, 1.999 * k]);
                request.wavelength = cm(TAU / k);
                betas(&enumerate_guided_modes(stack, request).unwrap())
                    .last()
                    .copied()
                    .unwrap()
            };
            let hk = 1e-4 * K;
            let fd_group = (at(K + hk) - at(K - hk)) / (2. * hk);
            assert!((group.value - fd_group).abs() < 1e-6 * group.value);
            let slope_k0 = slopes
                .implicit_slope(ImplicitSlopeRequest::BetaPerK0, Default::default())
                .unwrap();
            assert!((slope_k0.value.re - fd_group).abs() < 1e-6 * group.value);
            // Regional power fractions and their rates over gap, core, gap.
            let sample = FieldSampling::new().layer(0, LayerSampling::point(cm(0.)));
            let powers = |d: f64, gaps: [f64; 2]| {
                walled_mode(kind, pol, d, gaps)
                    .guided_beta_sensitivity_with(
                        &backend,
                        &direction,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap()
                    .power_field_tangent_with(
                        &backend,
                        &sample,
                        GuidedObservationFrame::MaterialAttached,
                        Complex::ONE,
                    )
                    .unwrap()
                    .regional_power
            };
            let base = powers(d, gaps);
            let total: f64 = base.iter().map(|r| r.power_w_per_m).sum();
            assert!((total - 1.).abs() < 1e-12);
            let (plus, minus) = (
                powers(d + h, [gaps[0], gaps[1] - h]),
                powers(d - h, [gaps[0], gaps[1] + h]),
            );
            for (i, region) in base.iter().enumerate() {
                let fd = (plus[i].power_w_per_m - minus[i].power_w_per_m) / (2. * h);
                assert!(
                    (region.power_rate_w_per_m - fd).abs() < 1e-6,
                    "region {i}: {} != {fd}",
                    region.power_rate_w_per_m
                );
            }
            eprintln!(
                "walled {pol:?} {kind:?}: dβ/dd {:.3e} rel, n_g {:.3e} rel",
                error,
                (group.value - fd_group).abs() / group.value
            );
        }
    }
}

#[test]
fn translation_and_outward_derivatives_on_walls_are_refused() {
    let backend = Scatter2::new();
    let point = walled_mode(PEC, TE, 0.5, [0.2, 0.3]);
    let translation = GuidedPerturbation::new(
        "translation",
        "cm",
        vec![1.; 2],
        0.,
        vec![ConstitutiveRate::default(); 3],
    )
    .unwrap();
    assert!(matches!(
        point.guided_beta_sensitivity_with(
            &backend,
            &translation,
            Default::default(),
            Default::default()
        ),
        Err(GuidedBetaError::Direction(
            DirectionError::MovesAgainstWalls
        ))
    ));
    // A right wall on the last interface: only thinning the layer is defined.
    let on_interface =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(1.))
            .analytical_layer(Constant::dielectric(CORE), cm(0.5))
            .right_termination(Wall::perfect_electric(cm(0.5)))
            .finalise();
    let basis = enumerate_guided_modes(on_interface, window(TE, [1.6 * K, 1.999 * K])).unwrap();
    assert!(basis.is_complete());
    let point = &basis.modes[0];
    let report = point
        .local_derivatives_with(
            backend,
            &[PhysicalParameter::LayerThickness(0.into())],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            Default::default(),
        )
        .unwrap();
    let Err(LocalDerivativeFailure::Compilation(cause)) = &report.outcomes[0].result else {
        panic!("{:?}", report.outcomes[0].result);
    };
    assert!(matches!(
        cause.downcast_ref::<StackCompileError<f64>>(),
        Some(StackCompileError::OutwardThicknessDerivative { .. })
    ));
    let thickening = point.guided_beta_sensitivity_with(
        &backend,
        &GuidedPerturbation::thickness(1, 0).unwrap(),
        Default::default(),
        Default::default(),
    );
    assert!(matches!(thickening, Err(GuidedBetaError::Evaluation(_))));
}

#[test]
fn leaky_modes_of_a_slab_on_a_wall_are_images_too() {
    // A core in vacuum a gap from one wall, and its open mirror image. With one
    // vacuum exterior the chart is the r = 0 map; the image's two vacuum
    // exteriors give the same map, so its roots share their native coordinates.
    let (d, g) = (0.8, 0.15);
    let vacuum = || Constant::dielectric(1.);
    let half = |kind| {
        Stack::from_analytical_materials(vacuum(), vacuum())
            .analytical_layer(Constant::dielectric(CORE), cm(d))
            .left_termination(Wall::new(kind, cm(-g)))
            .finalise()
    };
    let mirrored = Stack::from_analytical_materials(vacuum(), vacuum())
        .analytical_layer(Constant::dielectric(CORE), cm(d))
        .analytical_layer(vacuum(), cm(2. * g))
        .analytical_layer(Constant::dielectric(CORE), cm(d))
        .finalise();
    // Below Im z = −π/2 the open exteriors grow outward: leaky modes.
    let below = -std::f64::consts::FRAC_PI_2;
    let bounds = ConformalRegion::new(-3., -0.02, below - 1.4, below + 0.6).unwrap();
    let search = |stack, pol, expected: ModeType| {
        let problem = PropagationProblem::new(stack, K, pol).unwrap();
        let domain = |bounds| {
            let sheet = ReferenceSelection::Sheet(CoordinateSheet::Principal);
            problem
                .domain(bounds, sheet, Default::default(), ASSUME)
                .unwrap()
        };
        let (points, count) = resolved_roots(&problem, bounds, &domain);
        assert_eq!(points.len(), count);
        let mut roots: Vec<_> = points
            .iter()
            .map(|p| {
                let details = p.describe(Default::default()).unwrap();
                assert_eq!(
                    details.interpretation.classification,
                    Classification::Classified(expected)
                );
                p.native_coordinate()
            })
            .collect();
        roots.sort_by(|a, b| a.re.total_cmp(&b.re));
        roots
    };
    for pol in [TE, TM] {
        let mut union = search(half(PEC), pol, ModeType::RightLeaky);
        union.extend(search(half(PMC), pol, ModeType::RightLeaky));
        union.sort_by(|a, b| a.re.total_cmp(&b.re));
        let image = search(mirrored.clone(), pol, ModeType::TwoSidedLeaky);
        assert!(image.len() >= 2);
        assert_eq!(union.len(), image.len());
        let worst = union
            .iter()
            .zip(&image)
            .map(|(a, b)| (a - b).norm())
            .fold(0., f64::max);
        assert!(worst < 1e-10, "{union:?} != {image:?}");
        eprintln!(
            "leaky images {pol:?}: {} modes, worst native error {worst:.1e}",
            image.len()
        );
    }
}

#[test]
fn terminations_other_than_walls_are_refused_with_their_side() {
    let pml = Pml::new(
        cm(1.2),
        cm(0.5),
        StretchProfile::polynomial(2., Complex::new(0., 3.)),
        PEC,
    );
    let stack =
        Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
            .analytical_layer(Constant::dielectric(CORE), cm(1.))
            .left_termination(Wall::perfect_electric(cm(-0.2)))
            .right_termination(pml)
            .finalise();
    let refused = |error: &Diagnostic| {
        error.find_source::<ConstructionError>().is_some_and(|e| {
            matches!(
                e,
                ConstructionError::UnsupportedTermination {
                    side: StackSide::Right
                }
            )
        })
    };
    assert!(refused(
        &PropagationProblem::new(stack.clone(), K, TE).unwrap_err()
    ));
    assert!(refused(
        &FrequencyProblem::new(stack.clone(), K, TM).unwrap_err()
    ));
    assert!(matches!(
        enumerate_guided_modes(stack, window(TE, [1.6 * K, 1.999 * K])),
        Err(EnumerationError::Problem(_))
    ));
}
