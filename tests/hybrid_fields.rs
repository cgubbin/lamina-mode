//! Fields, unit power, the phase rule, reciprocity, sensitivities and
//! hybridisation of guided modes of full-tensor stacks.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, ExteriorSampling,
    FieldSampling, LayerSampling, Polarisation, Rotation, Scatter2, Scatter4, Stack, Transfer4,
};
use lamina_materials::PrincipalAxes;
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::TAU;

const K0: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn window(family: ModeFamily, direction: GuidedDirection) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.0),
        family,
        direction,
        abs_beta: [0.0, 3.0 * K0],
        cutoff_margin: 1e-3 * K0,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

fn isotropic(epsilon: f64) -> Region {
    AnalyticalMaterialHandle::new(Constant::dielectric(epsilon)).into()
}

/// A uniaxial region oriented by `rotation` (crystal to stack frame).
fn uniaxial(ordinary: f64, extraordinary: f64, rotation: Rotation<f64>) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        rotation,
    )
}

/// A uniaxial film with its optic axis at (θ, φ) on an isotropic substrate.
fn film(theta: f64, phi: f64) -> ModeStack {
    Stack::builder(isotropic(1.0), isotropic(2.1))
        .layer(
            uniaxial(4.0, 6.25, Rotation::from_optic_axis(theta, phi)),
            cm(0.8),
        )
        .finalise()
}

fn components(profile: &HybridProfile) -> Vec<[Complex; 6]> {
    let q = profile.fields.quantity();
    let (e, h) = (q.electric(), q.magnetic());
    (0..e.x().len())
        .map(|i| [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]])
        .collect()
}

#[test]
fn hybrid_fields_carry_unit_power_are_continuous_and_decay() {
    let stack = film(0.6, 0.7);
    let n = 801;
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::uniform(4, cm(0.6)))
        .layer(0, LayerSampling::uniform_with_boundaries(n, true, true))
        .right_exterior(ExteriorSampling::uniform(4, cm(0.6)));
    let mut worst_power: f64 = 0.0;
    let mut worst_jump: f64 = 0.0;
    let mut worst_im: f64 = 0.0;
    for backend in ["Scatter4", "Transfer4"] {
        let basis = match backend {
            "Scatter4" => enumerate_guided_modes_with(
                Scatter4::new(),
                stack.clone(),
                window(ModeFamily::All, GuidedDirection::Forward),
            ),
            _ => enumerate_guided_modes_with(
                Transfer4::new(),
                stack.clone(),
                window(ModeFamily::All, GuidedDirection::Forward),
            ),
        }
        .unwrap();
        assert!(basis.is_complete() && basis.modes.len() >= 3);
        for point in &basis.modes {
            let profile = match backend {
                "Scatter4" => point.hybrid_power_profile_with(&Scatter4::new(), &sampling),
                _ => point.hybrid_power_profile_with(&Transfer4::new(), &sampling),
            }
            .unwrap();
            assert_eq!(profile.signed_power_w_per_m, 1.0);
            assert_eq!(profile.phase_reference, HybridPhaseReference::EyPlusZ0Hy);
            let f = components(&profile);
            // Samples: left exterior (farthest first), the film, right
            // exterior (nearest first).
            let (left, film, right) = (&f[..4], &f[4..4 + n], &f[4 + n..]);
            // Every component present: a hybrid mode.
            for c in 0..6 {
                assert!(film.iter().any(|v| v[c].norm() > 1e-3));
            }
            // Tangential E_x, E_y, Z0H_x, Z0H_y continuous at both interfaces.
            for (outside, inside) in [(left[3], film[0]), (right[0], film[n - 1])] {
                let scale = inside.iter().map(|v| v.norm()).fold(0.0, f64::max);
                for c in [0, 1, 3, 4] {
                    worst_jump = worst_jump.max((outside[c] - inside[c]).norm() / scale);
                }
            }
            // |ψ| decays into both exteriors.
            let size = |v: &[Complex; 6]| v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            assert!(left.windows(2).all(|w| size(&w[0]) < size(&w[1])));
            assert!(right.windows(2).all(|w| size(&w[0]) > size(&w[1])));
            // Unit power from the profile itself: Simpson over the film and
            // the outgoing tails from the decay of the sampled exteriors are
            // checked by the library; here the film share and the left
            // interface reference.
            let r = left[3][1] + left[3][4];
            assert!(r.im.abs() <= 1e-9 * r.norm() && r.re > 0.0);
            // Unit power, independently: Simpson over 6 cm of each exterior
            // and the film, from the profile's own scaled fields.
            let power = independent_power(point, &profile, backend);
            worst_power = worst_power.max((power - 1.0).abs());
            worst_im = worst_im.max((profile.bilinear_norm_w_per_m.im).abs());
        }
    }
    println!(
        "hybrid fields: largest |P − 1 W/m| {worst_power:e}, largest relative tangential jump \
         {worst_jump:e}, largest |Im N| {worst_im:e} W/m"
    );
    assert!(worst_jump < 1e-9, "{worst_jump:e}");
    assert!(worst_power < 1e-6, "{worst_power:e}");
}

/// `½ Re ∫ (E_y H_z* − E_z H_y*) dz` in W/m of a profile's scaled fields by
/// Simpson's rule on explicit samples, 6 cm into each exterior.
fn independent_power(point: &ModePoint, profile: &HybridProfile, backend: &str) -> f64 {
    let n = 4000;
    let (reach, width) = (6.0, 0.8);
    let distances: Vec<f64> = (0..=n).map(|i| reach * i as f64 / n as f64).collect();
    let offsets: Vec<f64> = (0..=n).map(|i| width * i as f64 / n as f64).collect();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::distances_in(
            distances.iter().copied(),
            lamina_units::LengthUnit::Centimetre,
        ))
        .layer(
            0,
            LayerSampling::offsets_in(offsets, lamina_units::LengthUnit::Centimetre),
        )
        .right_exterior(ExteriorSampling::distances_in(
            distances,
            lamina_units::LengthUnit::Centimetre,
        ));
    let fields = match backend {
        "Scatter4" => point.hybrid_power_profile_with(&Scatter4::new(), &sampling),
        _ => point.hybrid_power_profile_with(&Transfer4::new(), &sampling),
    }
    .unwrap();
    assert_eq!(fields.scale, profile.scale);
    let f = components(&fields);
    let density: Vec<f64> = f
        .iter()
        .map(|v| 0.5 * (v[1] * v[5].conj() - v[2] * v[4].conj()).re)
        .collect();
    let simpson = |values: &[f64], length: f64| {
        let h = length / (values.len() - 1) as f64;
        values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = if i == 0 || i == values.len() - 1 {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                w * v
            })
            .sum::<f64>()
            * h
            / 3.0
    };
    let m = n + 1;
    let total = simpson(&density[..m], reach)
        + simpson(&density[m..2 * m], width)
        + simpson(&density[2 * m..], reach);
    // cm to m, Z0 H to H.
    total * 0.01 / 376.730313412
}

#[test]
fn the_hybrid_phase_rule_reduces_to_the_scalar_rules() {
    let stack: ModeStack = Stack::builder(isotropic(1.0), isotropic(2.25))
        .layer(isotropic(4.0), cm(0.9))
        .finalise();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::uniform(3, cm(0.4)))
        .layer(0, LayerSampling::uniform_with_boundaries(9, true, true))
        .right_exterior(ExteriorSampling::uniform(3, cm(0.4)));
    let all = enumerate_guided_modes(
        stack.clone(),
        window(ModeFamily::All, GuidedDirection::Forward),
    )
    .unwrap();
    let mut worst: f64 = 0.0;
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let scalar = enumerate_guided_modes(
            stack.clone(),
            GuidedWindow {
                abs_beta: all.eligible.unwrap(),
                ..window(pol.into(), GuidedDirection::Forward)
            },
        )
        .unwrap();
        let hybrid: Vec<_> = all
            .modes
            .iter()
            .zip(&all.labels)
            .filter(|(_, l)| **l == ModeLabel::from(pol))
            .map(|(m, _)| m)
            .collect();
        assert_eq!(hybrid.len(), scalar.modes.len());
        for (h, s) in hybrid.iter().zip(&scalar.modes) {
            let a = h
                .hybrid_power_profile_with(&Scatter4::new(), &sampling)
                .unwrap();
            let b = s
                .guided_power_profile_with(Scatter2::new(), &sampling, Default::default())
                .unwrap();
            let q = b.fields.quantity();
            let (e, m) = (q.electric(), q.magnetic());
            let reference: Vec<[Complex; 6]> = (0..e.x().len())
                .map(|i| [e.x()[i], e.y()[i], e.z()[i], m.x()[i], m.y()[i], m.z()[i]])
                .collect();
            let peak = reference
                .iter()
                .flatten()
                .map(|v| v.norm())
                .fold(0.0, f64::max);
            for (x, y) in components(&a).iter().zip(&reference) {
                for c in 0..6 {
                    worst = worst.max((x[c] - y[c]).norm() / peak);
                }
            }
        }
    }
    println!("phase-rule limits: largest relative field difference {worst:e}");
    assert!(worst < 1e-8, "{worst:e}");
}

#[test]
fn forward_and_backward_hybrid_modes_pair_bi_orthogonally() {
    let stack = film(0.6, 0.7);
    let sampling = FieldSampling::new().left_exterior(ExteriorSampling::point(cm(0.0)));
    let solve = |direction| {
        let basis =
            enumerate_guided_modes(stack.clone(), window(ModeFamily::All, direction)).unwrap();
        assert!(basis.is_complete());
        basis
            .modes
            .iter()
            .map(|m| {
                m.hybrid_power_profile_with(&Scatter4::new(), &sampling)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (forward, backward) = (
        solve(GuidedDirection::Forward),
        solve(GuidedDirection::Backward),
    );
    assert_eq!(forward.len(), backward.len());
    let mut ladder: f64 = 0.0;
    for (f, b) in forward.iter().zip(&backward) {
        let (x, y) = (f.point.physical_coordinate(), b.point.physical_coordinate());
        ladder = ladder.max((x.re + y.re).abs() / x.re);
        assert!(f.signed_power_w_per_m > 0.0 && b.signed_power_w_per_m < 0.0);
    }
    let mut off: f64 = 0.0;
    let mut diagonal = f64::INFINITY;
    for (i, f) in forward.iter().enumerate() {
        for (j, b) in backward.iter().enumerate() {
            let c = hybrid_reciprocity_with(&Scatter4::new(), [f, b]).unwrap();
            if i == j {
                diagonal = diagonal.min(c.norm());
            } else {
                off = off.max(c.norm());
            }
        }
    }
    // The z-mirrored stack: layers reversed, optic axis θ → π − θ.
    let mirror: ModeStack = Stack::builder(isotropic(2.1), isotropic(1.0))
        .layer(
            uniaxial(
                4.0,
                6.25,
                Rotation::from_optic_axis(std::f64::consts::PI - 0.6, 0.7),
            ),
            cm(0.8),
        )
        .finalise();
    let mirrored =
        enumerate_guided_modes(mirror, window(ModeFamily::All, GuidedDirection::Forward)).unwrap();
    let mut mirror_ladder: f64 = 0.0;
    for (f, m) in forward.iter().zip(&mirrored.modes) {
        let (x, y) = (f.point.physical_coordinate().re, m.physical_coordinate().re);
        mirror_ladder = mirror_ladder.max((x - y).abs() / x);
    }
    assert_eq!(mirrored.modes.len(), forward.len());
    println!(
        "reciprocity: |β| ladders {ladder:e}, pairing off-diagonal {off:e} against diagonal \
         {diagonal:e} W/m, mirror ladder {mirror_ladder:e}"
    );
    assert!(ladder < 1e-12 && mirror_ladder < 1e-12);
    assert!(off < 1e-8 * diagonal, "{off:e} {diagonal:e}");
}

/// The film with its optic axis turned by `angle` about x from (θ, φ).
fn turned(angle: f64) -> ModeStack {
    let (s, c) = angle.sin_cos();
    let about_x = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
    let base = Rotation::from_optic_axis(0.6, 0.7);
    let b = base.matrix();
    let product: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| about_x[i][k] * b[k][j]).sum())
    });
    Stack::builder(isotropic(1.0), isotropic(2.1))
        .layer(
            uniaxial(4.0, 6.25, Rotation::from_matrix(product).unwrap()),
            cm(0.8),
        )
        .finalise()
}

fn betas(stack: ModeStack) -> Vec<f64> {
    let basis =
        enumerate_guided_modes(stack, window(ModeFamily::All, GuidedDirection::Forward)).unwrap();
    assert!(basis.is_complete());
    basis
        .modes
        .iter()
        .map(|m| m.physical_coordinate().re)
        .collect()
}

#[test]
fn rotation_and_entry_rates_match_central_differences() {
    let stack = turned(0.0);
    let basis = enumerate_guided_modes(
        stack.clone(),
        window(ModeFamily::All, GuidedDirection::Forward),
    )
    .unwrap();
    let h = 1e-5;
    let (plus, minus) = (betas(turned(h)), betas(turned(-h)));
    let rotation = HybridPerturbation::rotation(&stack, K0, 1, [1.0, 0.0, 0.0]).unwrap();
    let mut worst: f64 = 0.0;
    for (i, point) in basis.modes.iter().enumerate() {
        let rate = point
            .hybrid_beta_rate_with(&Scatter4::new(), &rotation)
            .unwrap()
            .beta_rate;
        let fd = (plus[i] - minus[i]) / (2.0 * h);
        worst = worst.max((rate.re - fd).abs() / fd.abs().max(1.0));
        assert!(rate.im.abs() < 1e-9 * rate.norm().max(1.0));
    }
    // A thickness rate.
    let thick = |d: f64| -> ModeStack {
        Stack::builder(isotropic(1.0), isotropic(2.1))
            .layer(
                uniaxial(4.0, 6.25, Rotation::from_optic_axis(0.6, 0.7)),
                cm(d),
            )
            .finalise()
    };
    let (plus, minus) = (betas(thick(0.8 + h)), betas(thick(0.8 - h)));
    let thickness = HybridPerturbation {
        thickness_rates: vec![1.0],
        ..HybridPerturbation::none(&stack)
    };
    let mut worst_thickness: f64 = 0.0;
    for (i, point) in basis.modes.iter().enumerate() {
        let rate = point
            .hybrid_beta_rate_with(&Transfer4::new(), &thickness)
            .unwrap()
            .beta_rate;
        let fd = (plus[i] - minus[i]) / (2.0 * h);
        worst_thickness = worst_thickness.max((rate.re - fd).abs() / fd.abs().max(1.0));
    }
    println!(
        "rates: rotation about x {worst:e}, thickness {worst_thickness:e} (relative to central differences)"
    );
    assert!(worst < 1e-6 && worst_thickness < 1e-6);
}

#[test]
fn the_hybrid_group_index_matches_its_energy_certificate_and_differences() {
    let stack = film(0.6, 0.7);
    let basis = enumerate_guided_modes(
        stack.clone(),
        window(ModeFamily::All, GuidedDirection::Forward),
    )
    .unwrap();
    let at = |k0: f64| {
        let mut w = window(ModeFamily::All, GuidedDirection::Forward);
        w.wavelength = cm(TAU / k0);
        w.abs_beta = [0.0, 3.0 * k0];
        let basis = enumerate_guided_modes(stack.clone(), w).unwrap();
        basis
            .modes
            .iter()
            .map(|m| m.physical_coordinate().re)
            .collect::<Vec<_>>()
    };
    let h = 1e-6;
    let (plus, minus) = (at(K0 * (1.0 + h)), at(K0 * (1.0 - h)));
    let mut worst: f64 = 0.0;
    let mut certificate: f64 = 0.0;
    for (i, point) in basis.modes.iter().enumerate() {
        let group = point.hybrid_group_index_with(&Scatter4::new()).unwrap();
        let fd = (plus[i] - minus[i]) / (2.0 * h * K0);
        worst = worst.max((group.value - fd).abs() / fd);
        certificate = certificate.max(group.relative_discrepancy);
    }
    println!(
        "hybrid group index: against differences {worst:e}, against the energy {certificate:e}"
    );
    assert!(worst < 1e-6 && certificate < 1e-7);
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
    let samples = 20_000;
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
fn rotating_the_optic_axis_opens_an_avoided_crossing_tracked_as_a_subspace() {
    // A uniaxial slab (optic axis z) in a symmetric cladding: TE sees ε_o,
    // TM sees ε_xx = ε_o and ε_zz = ε_e. Tune the half-thickness so the
    // first TE and the third TM modes, both even, are degenerate: turning
    // the axis towards y adds ε_yz, which couples E_y to the even E_z.
    let (o, e, clad) = (4.0, 6.25, 2.1);
    let difference = |a: f64| {
        let te = ladder([1.0, 1.0, o], [1.0, 1.0, clad], a);
        let tm = ladder([o, e, 1.0], [clad, clad, 1.0], a);
        te[0] - tm[2]
    };
    let (mut l, mut h) = (0.5, 0.6);
    assert!(difference(l) * difference(h) < 0.0);
    for _ in 0..60 {
        let m = 0.5 * (l + h);
        if difference(l) * difference(m) <= 0.0 {
            h = m;
        } else {
            l = m;
        }
    }
    let a = 0.5 * (l + h);
    let crossing = ladder([1.0, 1.0, o], [1.0, 1.0, clad], a)[0];
    let slab = |theta: f64| -> ModeStack {
        Stack::builder(isotropic(clad), isotropic(clad))
            .layer(
                uniaxial(
                    o,
                    e,
                    Rotation::from_optic_axis(theta, std::f64::consts::FRAC_PI_2),
                ),
                cm(2.0 * a),
            )
            .finalise()
    };
    // Unrotated: a genuine TE/TM degeneracy, reported as a two-dimensional
    // subspace; every other mode TE or TM.
    let w = window(ModeFamily::All, GuidedDirection::Forward);
    let start = enumerate_guided_modes(slab(0.0), w).unwrap();
    let degenerate: Vec<_> = start
        .unresolved
        .iter()
        .filter_map(|g| match g.cause {
            GuidedLimit::NearDegenerate { count, cell } => Some((count, cell)),
            _ => None,
        })
        .collect();
    assert_eq!(degenerate.len(), 1, "{:?}", start.unresolved);
    let (count, cell) = degenerate[0];
    assert_eq!(count, 2);
    assert!(cell.x_min <= crossing && crossing <= cell.x_max);
    assert!(start.labels.iter().all(|l| *l != ModeLabel::Hybrid));
    let total = start.expected.unwrap();
    // Rotated towards y: hybrid modes, the pair splitting.
    let sampling = FieldSampling::new().left_exterior(ExteriorSampling::point(cm(0.0)));
    let mut previous: Option<Vec<HybridProfile>> = None;
    let mut splits = vec![];
    for theta in [0.01, 0.02, 0.04] {
        let basis = enumerate_guided_modes(slab(theta), w).unwrap();
        assert!(basis.is_complete(), "θ = {theta}: {:?}", basis.unresolved);
        assert_eq!(basis.modes.len(), total);
        assert!(basis.labels.iter().all(|l| *l == ModeLabel::Hybrid));
        let profiles: Vec<_> = basis
            .modes
            .iter()
            .map(|m| {
                m.hybrid_power_profile_with(&Scatter4::new(), &sampling)
                    .unwrap()
            })
            .collect();
        let betas: Vec<f64> = basis
            .modes
            .iter()
            .map(|m| m.physical_coordinate().re)
            .collect();
        // The pair: the two modes nearest the crossing.
        let mut order: Vec<usize> = (0..betas.len()).collect();
        order.sort_by(|&i, &j| {
            (betas[i] - crossing)
                .abs()
                .total_cmp(&(betas[j] - crossing).abs())
        });
        let split = (betas[order[0]] - betas[order[1]]).abs();
        splits.push(split);
        if let Some(earlier) = &previous {
            let tracking =
                hybrid_tracking_with(&Scatter4::new(), earlier, &profiles, 2.0 * split).unwrap();
            // Single modes continue into themselves; the pair as a subspace.
            for (i, j) in &tracking.matches {
                assert_eq!(i, j);
                assert!(tracking.overlaps[*i][*j].norm() > 0.99);
            }
            assert_eq!(tracking.subspaces.len(), 1, "{:?}", tracking.subspaces);
            let (from, to, block) = tracking.subspaces[0];
            assert_eq!(from, to);
            // The block carries the pair's power: unit singular values.
            let weight: f64 = block.iter().flatten().map(|v| v.norm_sqr()).sum();
            assert!((weight - 2.0).abs() < 1e-2, "{weight}");
        }
        previous = Some(profiles);
    }
    println!(
        "avoided crossing at β = {crossing:.9} cm^-1 (half-thickness {a:.9} cm): splits {splits:?} cm^-1 at θ = 0.01, 0.02, 0.04"
    );
    assert!(splits.windows(2).all(|w| w[1] > w[0]));
}
