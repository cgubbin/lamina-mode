//! Guided modes of stacks whose ε and μ are diagonal in the stack frame:
//! closed-form uniaxial ladders, isotropic recovery, eligibility, fields,
//! sensitivities, the group index and overlaps.
use lamina_core::{
    AnalyticalTensorMaterialHandle, Constant, ConstitutiveClass, ConstitutiveError,
    ExteriorSampling, FieldSampling, LayerSampling, Polarisation, Rotation, Scatter2, SignedAxis,
    Stack, StackRegion, Transfer2, Wall, WallKind,
};
use lamina_materials::{PrincipalAxes, Sellmeier, SellmeierTerm};
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber};
use std::f64::consts::TAU;
mod pml_support;

const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;
/// k0 at a 1 cm wavelength, in cm⁻¹.
const K0: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn window(polarisation: Polarisation, abs_beta: [f64; 2]) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.0),
        polarisation,
        direction: GuidedDirection::Forward,
        abs_beta,
        cutoff_margin: 1e-8,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

/// Crystal-to-stack rotation putting the optic (c) axis along stack axis
/// `axis` (0 = x, 1 = y, 2 = z).
fn rotation(axis: usize) -> Rotation<f64> {
    use SignedAxis::*;
    match axis {
        0 => Rotation::signed_permutation(PlusY, PlusZ, PlusX).unwrap(),
        1 => Rotation::signed_permutation(PlusZ, PlusX, PlusY).unwrap(),
        _ => Rotation::identity(),
    }
}

/// Stack-frame `(xx, yy, zz)` of a uniaxial tensor with its optic axis on `axis`.
fn stack_frame(ordinary: f64, extraordinary: f64, axis: usize) -> [f64; 3] {
    let mut entries = [ordinary; 3];
    entries[axis] = extraordinary;
    entries
}

fn uniaxial(ordinary: f64, extraordinary: f64, axis: usize) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        rotation(axis),
    )
}

/// A region with stack-frame diagonals `epsilon` and `mu`.
fn biaxial(epsilon: [f64; 3], mu: [f64; 3]) -> Region {
    let axis = |i: usize| Constant::magnetodielectric(epsilon[i], mu[i]);
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::biaxial(axis(0), axis(1), axis(2))),
        Rotation::identity(),
    )
}

fn slab(clad: Region, core: Region, thickness: f64) -> ModeStack {
    Stack::builder(clad.clone(), clad)
        .layer(core, cm(thickness))
        .finalise()
}

/// The entries `(t, n, v)` a polarisation sees: μ_xx, μ_zz, ε_yy (TE) or
/// ε_xx, ε_zz, μ_yy (TM).
fn channel(epsilon: [f64; 3], mu: [f64; 3], polarisation: Polarisation) -> [f64; 3] {
    match polarisation {
        TE => [mu[0], mu[2], epsilon[1]],
        TM => [epsilon[0], epsilon[2], mu[1]],
    }
}

/// Symmetric-slab propagation constants from the closed-form conditions
/// `q tan(q a)/t_core = α/t_clad` (even) and `−q cot(q a)/t_core = α/t_clad`
/// (odd), with `q² = s (v n k0² − β²)` in the core and `α² = s (β² − v n k0²)`
/// in the cladding, `s = t/n`. Optionally walled at `gap` from each
/// interface: `α → α coth(α g)` for a node at the wall, `α tanh(α g)` for an
/// antinode. Descending.
fn ladder(core: [f64; 3], clad: [f64; 3], a: f64, wall: Option<(f64, bool)>) -> Vec<f64> {
    let [t, n, v] = core;
    let [tc, nc, vc] = clad;
    let (line, line_c) = (K0 * (v * n).sqrt(), K0 * (vc * nc).sqrt());
    let tail = |b: f64| {
        let alpha = (tc / nc * (b * b - line_c * line_c)).sqrt();
        match wall {
            None => alpha,
            Some((gap, true)) => alpha / (alpha * gap).tanh(),
            Some((gap, false)) => alpha * (alpha * gap).tanh(),
        }
    };
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

fn betas(basis: &GuidedBasis) -> Vec<f64> {
    basis
        .modes
        .iter()
        .map(|m| m.physical_coordinate().re)
        .collect()
}

#[test]
fn uniaxial_slab_ladders_match_the_closed_form_conditions() {
    // Core and cladding uniaxial with a common optic axis along x, y and z.
    let (o, e, oc, ec, a) = (4.0, 6.25, 2.0, 2.56, 0.6);
    let mut worst: f64 = 0.0;
    for axis in 0..3 {
        let stack = slab(uniaxial(oc, ec, axis), uniaxial(o, e, axis), 2.0 * a);
        for pol in [TE, TM] {
            let core = channel(stack_frame(o, e, axis), [1.0; 3], pol);
            let clad = channel(stack_frame(oc, ec, axis), [1.0; 3], pol);
            let basis =
                enumerate_guided_modes(stack.clone(), window(pol, [0.0, 3.0 * K0])).unwrap();
            let reference = ladder(core, clad, a, None);
            assert!(basis.is_complete(), "{:?}", basis.unresolved);
            assert_eq!(basis.expected, Some(reference.len()), "axis {axis} {pol:?}");
            assert!(reference.len() >= 3);
            // Per-polarisation light line and upper bound.
            assert!((basis.light_line - K0 * (clad[1] * clad[2]).sqrt()).abs() < 1e-12);
            assert!((basis.upper_bound - K0 * (core[1] * core[2]).sqrt()).abs() < 1e-12);
            for (beta, expected) in betas(&basis).into_iter().zip(reference) {
                worst = worst.max((beta - expected).abs());
            }
        }
    }
    println!("uniaxial ladders: largest |Δβ| = {worst:e} cm^-1");
    assert!(worst < 1e-8, "{worst:e}");
}

#[test]
fn walled_uniaxial_slab_ladders_match_the_closed_form_conditions() {
    // PEC walls: a node of Ey (TE) and an antinode of Z0 Hy (TM) at each wall.
    let (o, e, oc, ec, a, gap) = (4.0, 6.25, 2.0, 2.56, 0.6, 0.15);
    let mut worst: f64 = 0.0;
    for axis in [0, 2] {
        let stack = slab(uniaxial(oc, ec, axis), uniaxial(o, e, axis), 2.0 * a)
            .with_left_termination(Wall::new(WallKind::PerfectElectric, cm(-gap)))
            .with_right_termination(Wall::new(WallKind::PerfectElectric, cm(2.0 * a + gap)));
        for pol in [TE, TM] {
            let core = channel(stack_frame(o, e, axis), [1.0; 3], pol);
            let clad = channel(stack_frame(oc, ec, axis), [1.0; 3], pol);
            // Above both cladding lines, where every cladding decays.
            let line_c = K0 * (clad[1] * clad[2]).sqrt();
            let basis =
                enumerate_guided_modes(stack.clone(), window(pol, [line_c * 1.0001, 3.0 * K0]))
                    .unwrap();
            let reference: Vec<_> = ladder(core, clad, a, Some((gap, pol == TE)))
                .into_iter()
                .filter(|b| *b > line_c * 1.0001)
                .collect();
            assert!(basis.is_complete(), "{:?}", basis.unresolved);
            assert_eq!(basis.expected, Some(reference.len()), "axis {axis} {pol:?}");
            assert!(!reference.is_empty());
            for (beta, expected) in betas(&basis).into_iter().zip(reference) {
                worst = worst.max((beta - expected).abs());
            }
        }
    }
    println!("walled uniaxial ladders: largest |Δβ| = {worst:e} cm^-1");
    assert!(worst < 1e-8, "{worst:e}");
}

/// Every region as a diagonal tensor with equal entries reproduces the
/// isotropic stack.
#[test]
fn equal_diagonal_entries_recover_the_isotropic_stack() {
    let scalar = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.0), cm(0.9))
        .analytical_layer(Constant::magnetodielectric(3.0, 1.3), cm(0.4))
        .finalise();
    let tensor = Stack::builder(biaxial([1.0; 3], [1.0; 3]), biaxial([2.25; 3], [1.0; 3]))
        .layer(biaxial([4.0; 3], [1.0; 3]), cm(0.9))
        .layer(biaxial([3.0; 3], [1.3; 3]), cm(0.4))
        .finalise();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::uniform(4, cm(0.6)))
        .layer(0, LayerSampling::uniform_with_boundaries(9, true, true))
        .layer(1, LayerSampling::uniform_with_boundaries(5, true, true))
        .right_exterior(ExteriorSampling::uniform(4, cm(0.6)));
    let (mut beta_error, mut field_error): (f64, f64) = (0.0, 0.0);
    for pol in [TE, TM] {
        let w = window(pol, [0.0, 3.0 * K0]);
        let a = enumerate_guided_modes(scalar.clone(), w).unwrap();
        let b = enumerate_guided_modes(tensor.clone(), w).unwrap();
        assert!(a.is_complete() && b.is_complete());
        assert_eq!(a.expected, b.expected);
        assert_eq!(a.eligible, b.eligible);
        assert_eq!(a.light_line, b.light_line);
        assert_eq!(a.upper_bound, b.upper_bound);
        assert!(a.modes.len() >= 2);
        for (p, q) in a.modes.iter().zip(&b.modes) {
            let (x, y) = (p.physical_coordinate(), q.physical_coordinate());
            beta_error = beta_error.max((x - y).norm() / x.norm());
            let profile = |m: &ModePoint| {
                m.guided_power_profile_with(Scatter2::new(), &sampling, Default::default())
                    .unwrap()
            };
            let (f, g) = (profile(p), profile(q));
            let components = |m: &ModalProfile| {
                let q = m.fields.quantity();
                let (e, h) = (q.electric(), q.magnetic());
                [e.x(), e.y(), e.z(), h.x(), h.y(), h.z()]
                    .into_iter()
                    .flat_map(|c| c.iter().copied().collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            };
            let (u, v) = (components(&f), components(&g));
            let peak = u.iter().map(|z| z.norm()).fold(0.0, f64::max);
            for (x, y) in u.iter().zip(&v) {
                field_error = field_error.max((x - y).norm() / peak);
            }
        }
    }
    println!(
        "isotropic recovery: relative |Δβ| = {beta_error:e}, relative |Δfield| = {field_error:e}"
    );
    assert!(beta_error < 1e-13, "{beta_error:e}");
    assert!(field_error < 1e-11, "{field_error:e}");
}

#[test]
fn transfer2_and_scatter2_give_the_same_uniaxial_basis() {
    let stack = slab(uniaxial(2.0, 2.56, 0), uniaxial(4.0, 6.25, 0), 1.2);
    for pol in [TE, TM] {
        let w = window(pol, [0.0, 3.0 * K0]);
        let scatter = enumerate_guided_modes_with(Scatter2::new(), stack.clone(), w).unwrap();
        let transfer = enumerate_guided_modes_with(Transfer2::new(), stack.clone(), w).unwrap();
        assert!(scatter.is_complete() && transfer.is_complete());
        assert_eq!(scatter.expected, transfer.expected);
        assert!(transfer.modes[0].backend().contains("Transfer2"));
        for (a, b) in betas(&scatter).into_iter().zip(betas(&transfer)) {
            assert!((a - b).abs() < 1e-10 * a, "{a} {b}");
        }
    }
}

#[test]
fn a_hyperbolic_entry_is_refused_only_for_the_polarisation_that_sees_it() {
    // ε_zz < 0 in the core: TM sees it, TE does not.
    let stack = slab(uniaxial(2.25, 2.25, 2), uniaxial(4.0, -2.0, 2), 1.0);
    assert!(matches!(
        enumerate_guided_modes(stack.clone(), window(TM, [0.0, 3.0 * K0])),
        Err(EnumerationError::Material {
            region: 1,
            entry: TensorEntry::EpsilonZz
        })
    ));
    let te = enumerate_guided_modes(stack, window(TE, [0.0, 3.0 * K0])).unwrap();
    let reference = ladder([1.0, 1.0, 4.0], [1.0, 1.0, 2.25], 0.5, None);
    assert!(te.is_complete());
    assert_eq!(te.expected, Some(reference.len()));
    for (beta, expected) in betas(&te).into_iter().zip(reference) {
        assert!((beta - expected).abs() < 1e-8);
    }
    // A negative μ_xx is seen by TE only.
    let magnetic = slab(
        biaxial([2.25; 3], [1.0; 3]),
        biaxial([4.0; 3], [-1.0, 1.0, 1.0]),
        1.0,
    );
    assert!(matches!(
        enumerate_guided_modes(magnetic.clone(), window(TE, [0.0, 3.0 * K0])),
        Err(EnumerationError::Material {
            region: 1,
            entry: TensorEntry::MuXx
        })
    ));
    assert!(enumerate_guided_modes(magnetic, window(TM, [0.0, 3.0 * K0])).is_ok());
}

#[test]
fn a_region_with_off_diagonal_entries_is_refused() {
    let tilted = Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(4.0),
            Constant::dielectric(6.25),
        )),
        Rotation::from_optic_axis(0.3, 0.0),
    );
    assert_eq!(tilted.class(), ConstitutiveClass::Full);
    let stack = slab(uniaxial(2.25, 2.25, 2), tilted, 1.0);
    let refused = ConstitutiveError::Unsupported {
        region: StackRegion::Layer(lamina_core::FiniteLayerIndex::new(0)),
        class: ConstitutiveClass::Full,
    };
    assert!(matches!(
        enumerate_guided_modes(stack.clone(), window(TE, [0.0, 3.0 * K0])),
        Err(EnumerationError::Constitutive(e)) if e == refused
    ));
    assert!(PropagationProblem::new(stack, K0, TE).is_err());
}

/// Simpson's rule over uniformly spaced samples.
fn simpson(values: &[f64], width: f64) -> f64 {
    let n = values.len() - 1;
    assert!(n % 2 == 0);
    let sum: f64 = values
        .iter()
        .enumerate()
        .map(|(j, v)| {
            v * if j == 0 || j == n {
                1.0
            } else if j % 2 == 1 {
                4.0
            } else {
                2.0
            }
        })
        .sum();
    sum * width / (3.0 * n as f64)
}

#[test]
fn biaxial_three_layer_fields_carry_unit_power_with_continuous_tangential_fields() {
    let eps = [4.0, 5.0, 3.5];
    let mu = [1.2, 1.0, 0.9];
    let right = stack_frame(2.0, 2.56, 0);
    let stack = Stack::builder(biaxial([1.0; 3], [1.0; 3]), uniaxial(2.0, 2.56, 0))
        .layer(biaxial(eps, mu), cm(0.8))
        .finalise();
    let entries = [([1.0; 3], [1.0; 3]), (eps, mu), (right, [1.0; 3])];
    let (deep, points, layer_points) = (4.0, 4001, 2001);
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::uniform(points, cm(deep)))
        .layer(
            0,
            LayerSampling::uniform_with_boundaries(layer_points, true, true),
        )
        .right_exterior(ExteriorSampling::uniform(points, cm(deep)));
    let (mut checked, mut worst_power, mut worst_jump) = (0, 0.0_f64, 0.0_f64);
    for pol in [TE, TM] {
        let basis = enumerate_guided_modes(stack.clone(), window(pol, [0.0, 3.0 * K0])).unwrap();
        assert!(basis.is_complete());
        for point in &basis.modes {
            let profile = point
                .guided_power_profile_with(Scatter2::new(), &sampling, Default::default())
                .unwrap();
            let q = profile.fields.quantity();
            let (e, h) = (q.electric(), q.magnetic());
            let flux =
                |i: usize| 0.5 * (e.y()[i] * h.z()[i].conj() - e.z()[i] * h.y()[i].conj()).re;
            // Samples run left to right: the left exterior from its deepest
            // sample to its interface, the right exterior from its interface.
            let left: Vec<_> = (0..points).rev().map(flux).collect();
            let core: Vec<_> = (points..points + layer_points).map(flux).collect();
            let right: Vec<_> = (points + layer_points..2 * points + layer_points)
                .map(flux)
                .collect();
            // Beyond the deepest samples the flux decays as exp(−2 Im κ d).
            let decay = point
                .signature()
                .unwrap()
                .exteriors()
                .map(|k| 2.0 * k.unwrap().im);
            let tails = left[points - 1] / decay[0] + right[points - 1] / decay[1];
            let power = 0.01 / 376.730313412
                * (simpson(&left, deep) + simpson(&core, 0.8) + simpson(&right, deep) + tails);
            worst_power = worst_power.max((power - 1.0).abs());
            assert!((power - 1.0).abs() < 1e-8, "{pol:?} power {power}");
            // Phase rule: the primary field is real and positive at the left interface.
            let primary = if pol == TE { e.y() } else { h.y() };
            let reference = primary[points - 1];
            assert!(reference.re > 0.0 && reference.im.abs() < 1e-12 * reference.re);
            // Tangential E and H, normal D and B continuous at both interfaces.
            let sides = [
                (points - 1, points),
                (points + layer_points - 1, points + layer_points),
            ];
            for (interface, (a, b)) in sides.into_iter().enumerate() {
                let (ea, eb) = (entries[interface], entries[interface + 1]);
                let pairs = [
                    (e.x()[a], e.x()[b]),
                    (e.y()[a], e.y()[b]),
                    (h.x()[a], h.x()[b]),
                    (h.y()[a], h.y()[b]),
                    (ea.0[2] * e.z()[a], eb.0[2] * e.z()[b]),
                    (ea.1[2] * h.z()[a], eb.1[2] * h.z()[b]),
                ];
                let scale = pairs.iter().map(|p| p.0.norm()).fold(1e-300, f64::max);
                for (x, y) in pairs {
                    worst_jump = worst_jump.max((x - y).norm() / scale);
                    assert!(
                        (x - y).norm() < 1e-10 * scale,
                        "{pol:?} interface {interface}"
                    );
                }
            }
            // Both tails decay monotonically away from the stack.
            for tail in [
                (0..points)
                    .rev()
                    .map(|i| primary[i].norm())
                    .collect::<Vec<_>>(),
                (points + layer_points..2 * points + layer_points)
                    .map(|i| primary[i].norm())
                    .collect(),
            ] {
                assert!(tail.windows(2).all(|w| w[1] < w[0]));
            }
            checked += 1;
        }
    }
    println!(
        "biaxial fields: {checked} modes, largest |P − 1| = {worst_power:e}, \
         largest relative interface jump = {worst_jump:e}"
    );
    assert!(checked >= 3);
}

/// Unit rate of one stack-frame entry: ε (0..3) or μ (3..6).
fn entry_rate(entry: usize) -> ConstitutiveRate {
    let mut epsilon = [0.0; 3];
    let mut mu = [0.0; 3];
    if entry < 3 {
        epsilon[entry] = 1.0;
    } else {
        mu[entry - 3] = 1.0;
    }
    ConstitutiveRate::diagonal(epsilon, mu)
}

#[test]
fn entry_sensitivities_match_central_differences_of_re_enumerated_modes() {
    let eps = [4.0, 5.0, 3.5];
    let mu = [1.2, 1.0, 0.9];
    let stack = |entry: usize, step: f64| {
        let (mut e, mut m) = (eps, mu);
        if entry < 3 {
            e[entry] += step;
        } else {
            m[entry - 3] += step;
        }
        slab(uniaxial(2.0, 2.56, 0), biaxial(e, m), 1.0)
    };
    let h = 1e-5;
    let mut worst: f64 = 0.0;
    for pol in [TE, TM] {
        let w = window(pol, [0.0, 3.0 * K0]);
        let base = enumerate_guided_modes(stack(0, 0.0), w).unwrap();
        let seen = match pol {
            TE => [3, 5, 1],
            TM => [0, 2, 4],
        };
        for entry in 0..6 {
            let mut rates = vec![ConstitutiveRate::default(); 3];
            rates[1] = entry_rate(entry);
            let direction =
                GuidedPerturbation::new("entry", "1", vec![0.0; 2], 0.0, rates).unwrap();
            let plus = betas(&enumerate_guided_modes(stack(entry, h), w).unwrap());
            let minus = betas(&enumerate_guided_modes(stack(entry, -h), w).unwrap());
            for (i, point) in base.modes.iter().enumerate() {
                let rate = point
                    .guided_beta_sensitivity_with(
                        &Scatter2::new(),
                        &direction,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap()
                    .beta_rate;
                if !seen.contains(&entry) {
                    // An entry the polarisation does not see has no effect.
                    assert!(rate.norm() < 1e-9, "{pol:?} entry {entry}: {rate}");
                    continue;
                }
                let fd = (plus[i] - minus[i]) / (2.0 * h);
                worst = worst.max((rate - fd).norm() / fd.abs().max(1.0));
                assert!(rate.im.abs() < 1e-9 * rate.norm().max(1.0));
            }
        }
    }
    println!("entry sensitivities: largest relative |dβ − fd| = {worst:e}");
    assert!(worst < 1e-6, "{worst:e}");
}

#[test]
fn extraordinary_index_derivative_and_field_tangent_match_central_differences() {
    // Optic axis along z: TM sees ε_zz = ε_e, TE does not.
    let stack = |step: f64| slab(uniaxial(2.25, 2.25, 2), uniaxial(4.0, 6.25 + step, 2), 1.0);
    let mut rates = vec![ConstitutiveRate::default(); 3];
    rates[1] = entry_rate(2);
    let direction = GuidedPerturbation::new("ε_zz", "1", vec![0.0; 2], 0.0, rates).unwrap();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::uniform(3, cm(0.4)))
        .layer(0, LayerSampling::uniform_with_boundaries(7, true, true))
        .right_exterior(ExteriorSampling::uniform(3, cm(0.4)));
    let h = 1e-5;
    for pol in [TE, TM] {
        let w = window(pol, [0.0, 3.0 * K0]);
        let base = enumerate_guided_modes(stack(0.0), w).unwrap();
        let plus = enumerate_guided_modes(stack(h), w).unwrap();
        let minus = enumerate_guided_modes(stack(-h), w).unwrap();
        for (i, point) in base.modes.iter().enumerate() {
            let sensitivity = point
                .guided_beta_sensitivity_with(
                    &Scatter2::new(),
                    &direction,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            let fd = (plus.modes[i].physical_coordinate() - minus.modes[i].physical_coordinate())
                / (2.0 * h);
            if pol == TE {
                assert!(sensitivity.beta_rate.norm() < 1e-9);
            } else {
                assert!(
                    (sensitivity.beta_rate - fd).norm() < 1e-6,
                    "{} {fd}",
                    sensitivity.beta_rate
                );
            }
            let tangent = sensitivity
                .power_field_tangent_with(
                    &Scatter2::new(),
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                    Complex::ONE,
                )
                .unwrap();
            let profile = |m: &ModePoint| {
                let p = m
                    .guided_power_profile_with(Scatter2::new(), &sampling, Default::default())
                    .unwrap();
                let q = p.fields.quantity();
                let (e, h) = (q.electric(), q.magnetic());
                (0..e.x().len())
                    .map(|j| [e.x()[j], e.y()[j], e.z()[j], h.x()[j], h.y()[j], h.z()[j]])
                    .collect::<Vec<_>>()
            };
            let (up, down) = (profile(&plus.modes[i]), profile(&minus.modes[i]));
            let peak = tangent
                .samples
                .iter()
                .flat_map(|s| s.field)
                .map(|z| z.norm())
                .fold(0.0, f64::max);
            for (j, sample) in tangent.samples.iter().enumerate() {
                let analytic = sample.tangent.unwrap();
                for c in 0..6 {
                    let fd = (up[j][c] - down[j][c]) / (2.0 * h);
                    assert!(
                        (analytic[c] - fd).norm() < 1e-5 * peak,
                        "{pol:?} sample {j} component {c}: {} {fd}",
                        analytic[c]
                    );
                }
            }
        }
    }
}

#[test]
fn group_index_of_a_dispersive_uniaxial_slab_matches_finite_differences() {
    let pole = |b: f64, q: f64| {
        SellmeierTerm::new(b, VacuumAngularWavenumber::inverse_centimetres(q)).unwrap()
    };
    let core = PrincipalAxes::uniaxial(
        Sellmeier::new(1.5, vec![pole(1.2, 9.0)]).unwrap(),
        Sellmeier::new(1.8, vec![pole(1.0, 11.0)]).unwrap(),
    );
    let substrate = Sellmeier::new(1.6, vec![pole(0.3, 15.0)]).unwrap();
    // Optic axis along x: TM sees the extraordinary ε_xx and the ordinary ε_zz.
    let stack = Stack::builder(
        Region::from(lamina_core::AnalyticalMaterialHandle::new(
            Constant::vacuum(),
        )),
        Region::from(lamina_core::AnalyticalMaterialHandle::new(substrate)),
    )
    .layer(
        Region::oriented(AnalyticalTensorMaterialHandle::new(core), rotation(0)),
        cm(1.2),
    )
    .finalise();
    let modes = |k0: f64, pol| {
        let mut w = window(pol, [0.0, 3.0 * k0]);
        w.wavelength = cm(TAU / k0);
        let basis = enumerate_guided_modes(stack.clone(), w).unwrap();
        assert!(basis.is_complete());
        basis
    };
    let mut worst: f64 = 0.0;
    for pol in [TE, TM] {
        let base = modes(K0, pol);
        assert!(base.modes.len() >= 2);
        let h = 1e-5;
        let plus = betas(&modes(K0 * (1.0 + h), pol));
        let minus = betas(&modes(K0 * (1.0 - h), pol));
        for (i, point) in base.modes.iter().enumerate() {
            let group = point
                .guided_group_index_with(&Scatter2::new(), Default::default(), 1e-8)
                .unwrap();
            let fd = (plus[i] - minus[i]) / (2.0 * h * K0);
            assert!(group.relative_discrepancy < 1e-8);
            assert!(group.value > point.physical_coordinate().re / K0);
            worst = worst.max((group.value - fd).abs() / group.value);
        }
    }
    println!("uniaxial group index: largest relative |n_g − fd| = {worst:e}");
    assert!(worst < 1e-7, "{worst:e}");
}

#[test]
fn a_uniaxial_basis_is_power_orthonormal_and_reciprocal() {
    let stack = slab(uniaxial(2.0, 2.56, 0), uniaxial(4.0, 6.25, 0), 1.0);
    let registration = GuidedOverlapRegistration {
        origins_cm: [0.0; 2],
    };
    for pol in [TE, TM] {
        let mut w = window(pol, [0.0, 3.0 * K0]);
        let forward = enumerate_guided_modes(stack.clone(), w).unwrap().modes;
        w.direction = GuidedDirection::Backward;
        let backward = enumerate_guided_modes(stack.clone(), w).unwrap().modes;
        assert!(forward.len() >= 2);
        for (i, a) in forward.iter().enumerate() {
            for (j, b) in forward.iter().enumerate() {
                let overlap = guided_power_overlap_with(
                    &Scatter2::new(),
                    [a, b],
                    registration,
                    Default::default(),
                )
                .unwrap()
                .power_w_per_m;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (overlap - expected).norm() < 1e-8,
                    "{pol:?} {i} {j} {overlap}"
                );
            }
            let flux = |a, b| {
                guided_reciprocity_with(&Scatter2::new(), [a, b], registration, Default::default())
                    .unwrap()
                    .flux_w_per_m
            };
            let b = &backward[i];
            assert!(flux(a, a).norm() < 1e-12);
            assert!((flux(a, b) + flux(b, a)).norm() < 1e-12);
            assert!((flux(a, b).norm() - 4.0).abs() < 1e-8);
        }
    }
}

#[test]
fn local_coupling_of_a_uniaxial_guide_matches_rates_and_overlaps() {
    // A uniaxial core (optic axis x) beside an isotropic layer: move their
    // interface, or change the core's ε_xx or ε_zz.
    let stack = |interface: f64, rate: [f64; 3]| {
        let core = stack_frame(4.0, 6.25, 0);
        Stack::builder(biaxial([1.0; 3], [1.0; 3]), biaxial([1.44; 3], [1.0; 3]))
            .layer(
                biaxial(std::array::from_fn(|i| core[i] + rate[i]), [1.0; 3]),
                cm(0.42 + interface),
            )
            .layer(biaxial([2.25; 3], [1.0; 3]), cm(0.10 - interface))
            .finalise()
    };
    let backend = Scatter2::new();
    for pol in [TE, TM] {
        let w = window(pol, [0.0, 3.0 * K0]);
        let base = enumerate_guided_modes(stack(0.0, [0.0; 3]), w).unwrap();
        assert!(base.modes.len() >= 2);
        for (motion, material) in [
            (1.0, [0.0; 3]),
            (0.0, [1.0, 0.0, 0.0]),
            (0.0, [0.0, 0.0, 1.0]),
        ] {
            let mut rates = vec![ConstitutiveRate::default(); 4];
            rates[1] = ConstitutiveRate::diagonal(material, [0.0; 3]);
            let direction =
                GuidedPerturbation::new("slab", "1", vec![0.0, motion, 0.0], 0.0, rates).unwrap();
            let coupling =
                guided_local_coupling_with(&backend, &base, &direction, 1e-7, Default::default())
                    .unwrap();
            for (i, mode) in base.modes.iter().enumerate() {
                let beta_rate = mode
                    .guided_beta_sensitivity_with(
                        &backend,
                        &direction,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap()
                    .beta_rate;
                assert!(
                    (coupling.perturbation_matrix[(i, i)] - beta_rate).norm() < 1e-6,
                    "{pol:?} motion {motion} {material:?}: {} {beta_rate}",
                    coupling.perturbation_matrix[(i, i)]
                );
            }
            let evaluate = |step: f64| {
                let displaced =
                    enumerate_guided_modes(stack(step * motion, material.map(|r| step * r)), w)
                        .unwrap();
                guided_power_overlap_with(
                    &backend,
                    [&displaced.modes[1], &base.modes[0]],
                    GuidedOverlapRegistration {
                        origins_cm: [0.0; 2],
                    },
                    Default::default(),
                )
                .unwrap()
                .power_w_per_m
            };
            let step = 1e-5;
            let fd = (evaluate(step) - evaluate(-step)) / (2.0 * step);
            let coefficient = coupling.coefficients[(0, 1)].unwrap();
            assert!(
                (fd - coefficient).norm() < 1e-5,
                "{pol:?} motion {motion} {material:?}: fd {fd}, coefficient {coefficient}"
            );
        }
    }
}

#[test]
fn interpretation_follows_the_channel_each_exterior_presents() {
    // A nonmagnetic uniaxial cladding with its optic axis along x presents
    // TE an isotropic channel (ε_yy, μ = 1) and TM a scaled one (ε_xx ≠ ε_zz).
    let stack = slab(uniaxial(2.0, 2.56, 0), uniaxial(4.0, 6.25, 0), 1.0);
    for (pol, supported) in [(TE, true), (TM, false)] {
        let basis = enumerate_guided_modes(stack.clone(), window(pol, [0.0, 3.0 * K0])).unwrap();
        let details = basis.modes[0].describe(Default::default()).unwrap();
        let classification = details.interpretation.classification;
        if supported {
            assert_eq!(classification, Classification::Classified(ModeType::Guided));
            assert_eq!(details.data.epsilon, [Complex::new(2.0, 0.0); 2]);
        } else {
            assert_eq!(
                classification,
                Classification::Unsupported(Unsupported::ExteriorMaterial)
            );
        }
    }
}

/// The characteristic of a box closed by PEC walls: (ψ, ψ′/t) carried through
/// regions of entries (t, n, v) and complex lengths, with ψ′/t = 0 (TM) or
/// ψ = 0 (TE) at both walls.
fn box_characteristic(
    pol: Polarisation,
    regions: &[([f64; 3], Complex)],
    beta: Complex,
) -> Complex {
    let mut state = if pol == TE {
        [Complex::ZERO, Complex::ONE]
    } else {
        [Complex::ONE, Complex::ZERO]
    };
    for &([t, n, v], length) in regions {
        let q = (t / n * (v * n * K0 * K0 - beta * beta)).sqrt();
        let (c, s) = ((q * length).cos(), (q * length).sin());
        state = [
            c * state[0] + t * s / q * state[1],
            -q * s / t * state[0] + c * state[1],
        ];
    }
    state[usize::from(pol == TM)]
}

#[test]
fn pml_closed_uniaxial_bases_match_the_box_characteristic_and_are_bi_orthonormal() {
    use lamina_core::StackSide;
    use pml_support::{PROFILE, newton, pml, winding};
    let (gap, depth, a) = (0.1, 0.75, pml_support::A);
    let (core, clad) = ((4.0, 6.25), (2.25, 3.0));
    let window = [1.0 * K0, 2.03 * K0, -0.031 * K0, 1.0 * K0];
    let registration = GuidedOverlapRegistration {
        origins_cm: [0.0; 2],
    };
    let stack = Stack::builder(uniaxial(clad.0, clad.1, 0), uniaxial(clad.0, clad.1, 0))
        .layer(uniaxial(core.0, core.1, 0), cm(2.0 * a))
        .left_termination(pml(
            StackSide::Left,
            gap,
            depth,
            PROFILE,
            WallKind::PerfectElectric,
        ))
        .right_termination(pml(
            StackSide::Right,
            gap,
            depth,
            PROFILE,
            WallKind::PerfectElectric,
        ))
        .finalise();
    for pol in [TE, TM] {
        let request = ModalBasisRequest::new(
            cm(1.0),
            ModeFamily::Polarised(pol),
            ModalWindow::Wavenumber {
                re_cm: [window[0], window[1]],
                im_cm: [window[2], window[3]],
            },
        );
        let basis = modal_basis_in_window_with(&Scatter2::new(), stack.clone(), request).unwrap();
        assert_eq!(
            basis.status,
            BasisStatus::Complete,
            "{:?}",
            basis.unresolved
        );
        let entries = |(o, e): (f64, f64)| channel(stack_frame(o, e, 0), [1.0; 3], pol);
        let tau = depth * PROFILE.mean();
        let regions = [
            (entries(clad), gap + tau),
            (entries(core), Complex::new(2.0 * a, 0.0)),
            (entries(clad), gap + tau),
        ];
        let reference = |beta| box_characteristic(pol, &regions, beta);
        let independent = winding(&reference, window);
        assert_eq!(basis.expected, Some(independent));
        assert_eq!(basis.found(), independent);
        assert!(independent >= 3);
        let mut worst: f64 = 0.0;
        for mode in &basis.modes {
            let beta = mode.point.physical_coordinate();
            let polished = newton(&reference, beta).unwrap();
            worst = worst.max((polished - beta).norm());
        }
        let pairing = basis_pairing_with(&Scatter2::new(), [&basis, &basis], registration).unwrap();
        let n = pairing.unconjugated.nrows();
        let defect = (pairing.unconjugated.clone() - nalgebra::DMatrix::identity(n, n))
            .iter()
            .map(|v| v.norm())
            .fold(0.0, f64::max);
        println!(
            "{pol:?} uniaxial PML basis: {independent} modes, largest |Δβ| = {worst:e} cm^-1, \
             |G − I| = {defect:e}"
        );
        assert!(worst < 1e-9 * K0, "{worst:e}");
        assert!(defect < 1e-8, "{defect:e}");
    }
}
