//! Guided modes of stacks with full-tensor regions: counts by the argument
//! principle of the coupled characteristic, labels, separable consistency,
//! isotropic recovery and uniaxial benchmarks.
use lamina_core::{
    AnalyticalTensorMaterialHandle, Constant, Polarisation, Rotation, Scatter4, Stack, Transfer4,
};
use lamina_materials::PrincipalAxes;
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::{FRAC_PI_2, TAU};

const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;
/// k0 at a 1 cm wavelength, in cm⁻¹.
const K0: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn window(family: ModeFamily, abs_beta: [f64; 2]) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.0),
        family,
        direction: GuidedDirection::Forward,
        abs_beta,
        cutoff_margin: 1e-3 * K0,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

/// A uniaxial region with its optic axis at polar angle `theta` from z and
/// azimuth `phi` from x.
fn uniaxial(ordinary: f64, extraordinary: f64, theta: f64, phi: f64) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        Rotation::from_optic_axis(theta, phi),
    )
}

fn isotropic(epsilon: f64, mu: f64) -> Region {
    lamina_core::AnalyticalMaterialHandle::new(Constant::magnetodielectric(epsilon, mu)).into()
}

fn betas(basis: &GuidedBasis) -> Vec<f64> {
    basis
        .modes
        .iter()
        .map(|m| m.physical_coordinate().re)
        .collect()
}

type Enumerate = Box<dyn Fn(ModeStack, GuidedWindow) -> GuidedBasis>;

fn backends() -> [(&'static str, Enumerate); 2] {
    [
        (
            "Scatter4",
            Box::new(|s, w| enumerate_guided_modes_with(Scatter4::new(), s, w).unwrap()),
        ),
        (
            "Transfer4",
            Box::new(|s, w| enumerate_guided_modes_with(Transfer4::new(), s, w).unwrap()),
        ),
    ]
}

/// Every mode of `basis` matches one of `reference` to `tol` relative, in order.
fn same_ladder(basis: &[f64], reference: &[f64], tol: f64) -> f64 {
    assert_eq!(basis.len(), reference.len(), "{basis:?} {reference:?}");
    basis
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .inspect(|d| assert!(*d < tol, "{basis:?} {reference:?}"))
        .fold(0.0, f64::max)
}

/// The TE and TM modes of a separable stack through the 2×2 scattering
/// backend, merged in decreasing β with their labels.
fn union(stack: &ModeStack, abs_beta: [f64; 2]) -> (Vec<f64>, Vec<ModeLabel>) {
    let mut modes = vec![];
    for pol in [TE, TM] {
        let basis = enumerate_guided_modes_with(
            lamina_core::Scatter2::new(),
            stack.clone(),
            window(pol.into(), abs_beta),
        )
        .unwrap();
        assert!(basis.is_complete());
        modes.extend(betas(&basis).into_iter().map(|b| (b, ModeLabel::from(pol))));
    }
    modes.sort_by(|a, b| b.0.total_cmp(&a.0));
    modes.into_iter().unzip()
}

#[test]
fn every_mode_of_an_isotropic_stack_is_the_union_of_te_and_tm() {
    let stack: ModeStack = Stack::builder(isotropic(1.0, 1.0), isotropic(2.25, 1.0))
        .layer(isotropic(4.0, 1.0), cm(0.9))
        .layer(isotropic(3.0, 1.3), cm(0.4))
        .finalise();
    for (name, enumerate) in backends() {
        let basis = enumerate(stack.clone(), window(ModeFamily::All, [0.0, 3.0 * K0]));
        let [lo, hi] = basis.eligible.unwrap();
        let (reference, labels) = union(&stack, [lo, hi]);
        assert!(basis.is_complete(), "{name}: {:?}", basis.unresolved);
        let worst = same_ladder(&betas(&basis), &reference, 1e-10);
        assert_eq!(basis.labels, labels);
        let evidence = basis.hybrid.as_ref().unwrap();
        let counts = evidence.separable_counts.unwrap();
        assert_eq!(
            counts.iter().flatten().sum::<usize>(),
            basis.expected.unwrap()
        );
        let off = evidence.off_family.iter().fold(0.0, |a: f64, b| a.max(*b));
        println!(
            "{name}: {} modes, counts {counts:?}, largest relative |Δβ| = {worst:e}, \
             largest off-family part = {off:e}, {} of {} cells counted by Transfer4",
            reference.len(),
            evidence
                .count_backends
                .iter()
                .filter(|b| b.contains("Transfer4"))
                .count(),
            evidence.count_backends.len()
        );
    }
}

#[test]
fn a_rotated_uniaxial_film_has_a_complete_hybrid_basis() {
    let stack = Stack::builder(isotropic(1.0, 1.0), isotropic(2.1, 1.0))
        .layer(uniaxial(4.0, 6.25, 0.6, 0.7), cm(0.8))
        .finalise();
    for backend in ["scatter4", "transfer4"] {
        let w = window(ModeFamily::All, [0.0, 3.0 * K0]);
        let basis = match backend {
            "scatter4" => enumerate_guided_modes_with(Scatter4::new(), stack.clone(), w),
            _ => enumerate_guided_modes_with(Transfer4::new(), stack.clone(), w),
        }
        .unwrap();
        println!(
            "{backend}: {:?} {:?} {:?}",
            basis.expected,
            betas(&basis),
            basis.unresolved
        );
        println!("{:?}", basis.hybrid);
        assert!(basis.is_complete());
        assert!(basis.labels.iter().all(|l| *l == ModeLabel::Hybrid));
    }
}

/// A uniaxial region whose optic axis lies along stack axis `axis` (0 = x,
/// 1 = y, 2 = z), oriented by a rotation that is a signed permutation only
/// to round-off: a full-class region.
fn full_uniaxial(ordinary: f64, extraordinary: f64, axis: usize) -> Region {
    // A turn by 0.3 about the optic axis leaves the tensor unchanged, but
    // the rotation is not a signed permutation.
    let (s, c) = 0.3f64.sin_cos();
    let turn = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    // Crystal z onto stack `axis`.
    let permutation: [[f64; 3]; 3] = match axis {
        0 => [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        1 => [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        _ => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let matrix: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| permutation[i][k] * turn[k][j]).sum())
    });
    let region = Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        Rotation::from_matrix(matrix).unwrap(),
    );
    assert_eq!(region.class(), lamina_core::ConstitutiveClass::Full);
    region
}

/// Stack-frame `(xx, yy, zz)` of a uniaxial tensor with its optic axis on `axis`.
fn stack_frame(ordinary: f64, extraordinary: f64, axis: usize) -> [f64; 3] {
    let mut entries = [ordinary; 3];
    entries[axis] = extraordinary;
    entries
}

/// The entries `(t, n, v)` a polarisation sees, μ = 1.
fn channel(epsilon: [f64; 3], polarisation: Polarisation) -> [f64; 3] {
    match polarisation {
        TE => [1.0, 1.0, epsilon[1]],
        TM => [epsilon[0], epsilon[2], 1.0],
    }
}

/// Symmetric-slab propagation constants from the closed-form conditions
/// `q tan(q a)/t_core = α/t_clad` (even) and `−q cot(q a)/t_core = α/t_clad`
/// (odd), `q² = s (v n k0² − β²)`, `α² = s (β² − v n k0²)`, `s = t/n`.
fn ladder(core: [f64; 3], clad: [f64; 3], a: f64) -> Vec<f64> {
    let [t, n, v] = core;
    let [tc, nc, vc] = clad;
    let (line, line_c) = (K0 * (v * n).sqrt(), K0 * (vc * nc).sqrt());
    let tail = |b: f64| (tc / nc * (b * b - line_c * line_c)).sqrt();
    let q = |b: f64| (t / n * (line * line - b * b)).sqrt();
    let even = |b: f64| q(b) * (q(b) * a).sin() / t - tail(b) * (q(b) * a).cos() / tc;
    let odd = |b: f64| q(b) * (q(b) * a).cos() / t + tail(b) * (q(b) * a).sin() / tc;
    let (lo, hi) = (line_c * (1.0 + 1e-12), line * (1.0 - 1e-12));
    let mut roots = Vec::new();
    let samples = 200_000;
    for f in [&even as &dyn Fn(f64) -> f64, &odd] {
        let mut previous = (lo, f(lo));
        for i in 1..=samples {
            let b = lo + (hi - lo) * i as f64 / samples as f64;
            let value = f(b);
            if previous.1 * value < 0.0 {
                let (mut l, mut h) = (previous.0, b);
                for _ in 0..200 {
                    let m = 0.5 * (l + h);
                    if f(m) * f(l) <= 0.0 {
                        h = m;
                    } else {
                        l = m;
                    }
                }
                roots.push(0.5 * (l + h));
            }
            previous = (b, value);
        }
    }
    roots.sort_by(|a, b| b.total_cmp(a));
    roots
}

#[test]
fn uniaxial_slabs_through_the_full_tensor_path_match_the_diagonal_path_and_closed_forms() {
    let (o, e, oc, ec, a) = (4.0, 6.25, 2.0, 2.56, 0.6);
    let (mut worst_diagonal, mut worst_closed): (f64, f64) = (0.0, 0.0);
    for axis in 0..3 {
        let full: ModeStack =
            Stack::builder(full_uniaxial(oc, ec, axis), full_uniaxial(oc, ec, axis))
                .layer(full_uniaxial(o, e, axis), cm(2.0 * a))
                .finalise();
        let diagonal: ModeStack = Stack::builder(
            uniaxial(
                oc,
                ec,
                [FRAC_PI_2, FRAC_PI_2, 0.0][axis],
                [0.0, FRAC_PI_2, 0.0][axis],
            ),
            uniaxial(
                oc,
                ec,
                [FRAC_PI_2, FRAC_PI_2, 0.0][axis],
                [0.0, FRAC_PI_2, 0.0][axis],
            ),
        )
        .layer(
            uniaxial(
                o,
                e,
                [FRAC_PI_2, FRAC_PI_2, 0.0][axis],
                [0.0, FRAC_PI_2, 0.0][axis],
            ),
            cm(2.0 * a),
        )
        .finalise();
        let basis =
            enumerate_guided_modes(full.clone(), window(ModeFamily::All, [0.0, 3.0 * K0])).unwrap();
        assert!(basis.is_complete(), "axis {axis}: {:?}", basis.unresolved);
        let [lo, hi] = basis.eligible.unwrap();
        let (reference, labels) = union(&diagonal, [lo, hi]);
        worst_diagonal = worst_diagonal.max(same_ladder(&betas(&basis), &reference, 1e-9));
        let mut closed: Vec<f64> = [TE, TM]
            .into_iter()
            .flat_map(|pol| {
                ladder(
                    channel(stack_frame(o, e, axis), pol),
                    channel(stack_frame(oc, ec, axis), pol),
                    a,
                )
            })
            .filter(|b| *b > lo && *b < hi)
            .collect();
        closed.sort_by(|a, b| b.total_cmp(a));
        worst_closed = worst_closed.max(same_ladder(&betas(&basis), &closed, 1e-9));
        // TE and TM where the rotation is structurally separable (the
        // crystal's y axis on the stack's), hybrid otherwise.
        let separable = full.iter().all(|layer| layer.material().is_separable());
        for (label, expected) in basis.labels.iter().zip(&labels) {
            assert!(
                *label
                    == if separable {
                        *expected
                    } else {
                        ModeLabel::Hybrid
                    }
            );
        }
    }
    println!(
        "uniaxial slabs, optic axis x, y, z: largest relative |Δβ| against the diagonal path \
         {worst_diagonal:e}, against the closed forms {worst_closed:e}"
    );
}

#[test]
fn an_optic_axis_in_the_xz_plane_keeps_te_and_tm_apart() {
    // Separable full-class regions: TE and TM through the 4×4 backends'
    // scalar channels; every mode through the coupled one.
    let (o, e, oc, ec, a, theta) = (4.0, 6.25, 2.0, 2.56, 0.6, 0.6);
    let stack: ModeStack =
        Stack::builder(uniaxial(oc, ec, theta, 0.0), uniaxial(oc, ec, theta, 0.0))
            .layer(uniaxial(o, e, theta, 0.0), cm(2.0 * a))
            .finalise();
    let all =
        enumerate_guided_modes(stack.clone(), window(ModeFamily::All, [0.0, 3.0 * K0])).unwrap();
    assert!(all.is_complete(), "{:?}", all.unresolved);
    let [lo, hi] = all.eligible.unwrap();
    let mut merged = vec![];
    let mut counts = vec![];
    for pol in [TE, TM] {
        let basis = enumerate_guided_modes(stack.clone(), window(pol.into(), [lo, hi])).unwrap();
        assert!(basis.is_complete(), "{pol:?}: {:?}", basis.unresolved);
        assert!(basis.labels.iter().all(|l| *l == ModeLabel::from(pol)));
        counts.push(basis.expected.unwrap());
        merged.extend(betas(&basis).into_iter().map(|b| (b, ModeLabel::from(pol))));
        if pol == TE {
            // TE sees ε_yy = ε_o only.
            let closed: Vec<f64> = ladder([1.0, 1.0, o], [1.0, 1.0, oc], a)
                .into_iter()
                .filter(|b| *b > lo && *b < hi)
                .collect();
            same_ladder(&betas(&basis), &closed, 1e-9);
        }
    }
    merged.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (reference, labels): (Vec<f64>, Vec<ModeLabel>) = merged.into_iter().unzip();
    assert_eq!(all.expected, Some(counts.iter().sum()));
    let worst = same_ladder(&betas(&all), &reference, 1e-9);
    assert_eq!(all.labels, labels);
    let evidence = all.hybrid.unwrap();
    assert_eq!(
        evidence.separable_counts,
        Some([Some(counts[0]), Some(counts[1])])
    );
    let off = evidence.off_family.iter().fold(0.0, |a: f64, b| a.max(*b));
    println!(
        "xz optic axis: TE {} + TM {} = {} modes, largest relative |Δβ| {worst:e}, \
         largest off-family part {off:e}",
        counts[0],
        counts[1],
        reference.len()
    );
}

#[test]
fn full_class_isotropic_tensors_recover_the_isotropic_basis() {
    // Every region a full-class tensor equal to εI to round-off.
    let tilt = |epsilon: f64| uniaxial(epsilon, epsilon, 0.4, 0.3);
    let full: ModeStack = Stack::builder(tilt(1.0), tilt(2.25))
        .layer(tilt(4.0), cm(0.9))
        .layer(tilt(3.0), cm(0.4))
        .finalise();
    let scalar: ModeStack = Stack::builder(isotropic(1.0, 1.0), isotropic(2.25, 1.0))
        .layer(isotropic(4.0, 1.0), cm(0.9))
        .layer(isotropic(3.0, 1.0), cm(0.4))
        .finalise();
    let w = window(ModeFamily::All, [0.0, 3.0 * K0]);
    let a = enumerate_guided_modes(scalar, w).unwrap();
    let b = enumerate_guided_modes(full, w).unwrap();
    assert!(a.is_complete() && b.is_complete(), "{:?}", b.unresolved);
    assert_eq!(a.expected, b.expected);
    assert!((a.light_line - b.light_line).abs() < 1e-12 * a.light_line);
    let worst = same_ladder(&betas(&b), &betas(&a), 1e-12);
    println!("isotropic recovery through full-class regions: largest relative |Δβ| = {worst:e}");
}

#[test]
fn self_dual_slab_modes_are_reported_as_degenerate_subspaces() {
    // ε = μ in every region: TE and TM coincide mode by mode.
    let stack: ModeStack = Stack::builder(isotropic(1.2, 1.2), isotropic(1.2, 1.2))
        .layer(isotropic(2.0, 2.0), cm(0.5))
        .finalise();
    let te = enumerate_guided_modes(stack.clone(), window(TE.into(), [0.0, 3.0 * K0])).unwrap();
    let all = enumerate_guided_modes(stack, window(ModeFamily::All, [0.0, 3.0 * K0])).unwrap();
    assert_eq!(all.expected, Some(2 * te.modes.len()));
    assert!(all.modes.is_empty());
    let mut pairs = vec![];
    for gap in &all.unresolved {
        let GuidedLimit::NearDegenerate { count, cell } = gap.cause else {
            panic!("{gap:?}");
        };
        assert_eq!(count, 2);
        pairs.push(0.5 * (cell.x_min + cell.x_max));
    }
    pairs.sort_by(|a, b| b.total_cmp(a));
    same_ladder(&pairs, &betas(&te), 1e-8);
    println!(
        "self-dual slab: {} degenerate pairs at the TE ladder",
        pairs.len()
    );
}

#[test]
fn walled_rotated_uniaxial_stacks_keep_their_separability() {
    // PEC walls preserve TE and TM: an optic axis in the xz plane gives
    // TE + TM modes; one out of it, hybrid modes. Neither side has a light
    // line, so every real mode below the bound is counted.
    use lamina_core::{Wall, WallKind};
    let walled = |phi: f64| -> ModeStack {
        Stack::builder(isotropic(2.1, 1.0), isotropic(2.1, 1.0))
            .layer(uniaxial(4.0, 6.25, 0.6, phi), cm(1.2))
            .finalise()
            .with_left_termination(Wall::new(WallKind::PerfectElectric, cm(-0.15)))
            .with_right_termination(Wall::new(WallKind::PerfectElectric, cm(1.35)))
    };
    let window = |family| GuidedWindow {
        abs_beta: [1.6 * K0, 3.0 * K0],
        ..window(family, [0.0, 1.0])
    };
    let separable = walled(0.0);
    let all = enumerate_guided_modes(separable.clone(), window(ModeFamily::All)).unwrap();
    assert!(all.is_complete(), "{:?}", all.unresolved);
    let mut counts = vec![];
    for pol in [TE, TM] {
        let basis = enumerate_guided_modes(separable.clone(), window(pol.into())).unwrap();
        assert!(basis.is_complete(), "{pol:?}: {:?}", basis.unresolved);
        counts.push(basis.expected.unwrap());
    }
    assert_eq!(all.expected, Some(counts[0] + counts[1]));
    assert!(all.labels.iter().all(|l| *l != ModeLabel::Hybrid));
    assert_eq!(
        all.hybrid.as_ref().unwrap().separable_counts,
        Some([Some(counts[0]), Some(counts[1])])
    );
    let hybrid = enumerate_guided_modes(walled(0.7), window(ModeFamily::All)).unwrap();
    assert!(hybrid.is_complete(), "{:?}", hybrid.unresolved);
    assert!(hybrid.labels.iter().all(|l| *l == ModeLabel::Hybrid));
    // Unit power with the walls' gaps integrated and no tails beyond them.
    let sampling = lamina_core::FieldSampling::new()
        .left_exterior(lamina_core::ExteriorSampling::point(cm(0.0)));
    for point in &hybrid.modes {
        let profile = point
            .hybrid_power_profile_with(&Scatter4::new(), &sampling)
            .unwrap();
        assert_eq!(profile.signed_power_w_per_m, 1.0);
    }
    println!(
        "walled: xz-plane axis TE {} + TM {} = {:?}; out of the plane {} hybrid modes",
        counts[0],
        counts[1],
        all.expected,
        hybrid.modes.len()
    );
}
