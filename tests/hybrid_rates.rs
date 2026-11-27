//! Rates of unit-power fields and regional power fractions of hybrid modes
//! along thickness, tensor-entry, rotation and wavelength directions,
//! against central differences of independently enumerated modes, and
//! against the scalar path on a separable stack.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, ExteriorSampling,
    FieldSampling, LayerSampling, Rotation, Scatter2, Scatter4, Stack, Transfer4,
};
use lamina_materials::{ConstantTensor, PrincipalAxes};
use lamina_mode::*;
use lamina_units::{Length, LengthUnit};
use std::f64::consts::{FRAC_PI_2, TAU};

const K0: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn isotropic(epsilon: f64) -> Region {
    AnalyticalMaterialHandle::new(Constant::dielectric(epsilon)).into()
}

/// The rotation by `angle` about x composed with `base`.
fn about_x(angle: f64, base: Rotation<f64>) -> Rotation<f64> {
    let (s, c) = angle.sin_cos();
    let r = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
    let b = base.matrix();
    Rotation::from_matrix(std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| r[i][k] * b[k][j]).sum())
    }))
    .unwrap()
}

/// Every mode of `stack` at wavenumber `k0` with |β| in `[lo, hi]`.
fn modes(stack: ModeStack, k0: f64, abs_beta: [f64; 2]) -> Vec<ModePoint> {
    let basis = enumerate_guided_modes(
        stack,
        GuidedWindow {
            wavelength: cm(TAU / k0),
            family: ModeFamily::All,
            direction: GuidedDirection::Forward,
            abs_beta,
            cutoff_margin: 1e-3 * k0,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        },
    )
    .unwrap();
    assert!(basis.is_complete(), "{:?}", basis.unresolved);
    basis.modes
}

fn beta(point: &ModePoint) -> f64 {
    point.physical_coordinate().re
}

/// The mode of `points` nearest `target` in β.
fn nearest(points: &[ModePoint], target: f64) -> &ModePoint {
    points
        .iter()
        .min_by(|a, b| {
            (beta(a) - target)
                .abs()
                .total_cmp(&(beta(b) - target).abs())
        })
        .unwrap()
}

/// Unit-power fields and regional power fractions of a mode, from the
/// tangent with no direction.
fn still(point: &ModePoint, sampling: &FieldSampling<f64>) -> (Vec<[Complex; 6]>, Vec<f64>) {
    let stack = point.provenance().stack().clone();
    let s = point
        .hybrid_power_field_tangent_with(
            &Scatter4::new(),
            &HybridPerturbation::none(&stack),
            sampling,
            GuidedObservationFrame::MaterialAttached,
        )
        .unwrap();
    (
        s.samples.iter().map(|x| x.field).collect(),
        s.regional_power.iter().map(|r| r.power_w_per_m).collect(),
    )
}

/// The largest discrepancy of `rates` from central differences of
/// `(plus, minus)` over step `2h`, relative to the largest difference;
/// fields and power fractions separately.
#[derive(Debug, Default, Clone, Copy)]
struct Worst {
    fields: f64,
    fractions: f64,
}

fn compare(
    s: &HybridPowerSensitivity,
    plus: &(Vec<[Complex; 6]>, Vec<f64>),
    minus: &(Vec<[Complex; 6]>, Vec<f64>),
    h: f64,
) -> Worst {
    let fd: Vec<[Complex; 6]> = plus
        .0
        .iter()
        .zip(&minus.0)
        .map(|(p, m)| std::array::from_fn(|c| (p[c] - m[c]) / (2.0 * h)))
        .collect();
    let field_scale = fd
        .iter()
        .flatten()
        .chain(s.samples.iter().flat_map(|x| x.field.iter()))
        .map(|v| v.norm())
        .fold(0.0, f64::max);
    let mut worst = Worst::default();
    for (sample, fd) in s.samples.iter().zip(&fd) {
        let tangent = sample.tangent.as_ref().unwrap();
        for c in 0..6 {
            worst.fields = worst.fields.max((tangent[c] - fd[c]).norm() / field_scale);
        }
    }
    let fd: Vec<f64> = plus
        .1
        .iter()
        .zip(&minus.1)
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect();
    let scale = fd.iter().map(|v| v.abs()).fold(1e-3, f64::max);
    for (region, fd) in s.regional_power.iter().zip(&fd) {
        worst.fractions = worst
            .fractions
            .max((region.power_rate_w_per_m - fd).abs() / scale);
    }
    worst
}

// --- A general-tensor film -------------------------------------------------

/// A Hermitian, positive-definite tensor with every entry non-zero, its
/// `xx` and `xy = yx` entries shifted.
fn general(shift: [f64; 2]) -> ConstantTensor<f64> {
    let base = [[4.0, 0.3, 0.2], [0.3, 5.0, 0.15], [0.2, 0.15, 6.25]];
    let mut e = base;
    e[0][0] += shift[0];
    e[0][1] += shift[1];
    e[1][0] += shift[1];
    let one = Complex::ONE;
    let zero = Complex::ZERO;
    let mu = [[one, zero, zero], [zero, one, zero], [zero, zero, one]];
    ConstantTensor::new(e.map(|row| row.map(Complex::from)), mu).unwrap()
}

#[derive(Clone, Copy)]
struct Film {
    thickness: f64,
    shift: [f64; 2],
    angle: f64,
    k0: f64,
}

const FILM: Film = Film {
    thickness: 0.8,
    shift: [0.0; 2],
    angle: 0.0,
    k0: K0,
};

impl Film {
    fn stack(self) -> ModeStack {
        Stack::builder(isotropic(1.0), isotropic(2.1))
            .layer(
                Region::oriented(
                    AnalyticalTensorMaterialHandle::new(general(self.shift)),
                    about_x(self.angle, Rotation::identity()),
                ),
                cm(self.thickness),
            )
            .finalise()
    }

    fn modes(self) -> Vec<ModePoint> {
        modes(self.stack(), self.k0, [0.0, 3.0 * self.k0])
    }
}

/// Samples in both exteriors and across the film, attached to the stack.
fn film_sampling() -> FieldSampling<f64> {
    scaled_sampling(1.0)
}

/// [`film_sampling`] with every length multiplied by `scale`.
fn scaled_sampling(scale: f64) -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::distances_in(
            [0.0, 0.3 * scale],
            LengthUnit::Centimetre,
        ))
        .layer(0, LayerSampling::uniform_with_boundaries(5, true, true))
        .layer(
            0,
            LayerSampling::offsets_in([0.21 * scale], LengthUnit::Centimetre),
        )
        .right_exterior(ExteriorSampling::distances_in(
            [0.0, 0.4 * scale],
            LengthUnit::Centimetre,
        ))
}

/// A named direction, its difference step and the perturbed film.
type FilmCase = (&'static str, HybridPerturbation, f64, fn(f64) -> Film);

#[test]
fn hybrid_film_field_and_fraction_rates_match_central_differences() {
    let base = FILM.modes();
    assert!(base.len() >= 4);
    let stack = FILM.stack();
    let sampling = film_sampling();
    let mut unit_entry = HybridPerturbation::none(&stack);
    unit_entry.tensor_rates[1][0][0][0] = Complex::ONE;
    let mut unit_pair = HybridPerturbation::none(&stack);
    unit_pair.tensor_rates[1][0][0][1] = Complex::ONE;
    unit_pair.tensor_rates[1][0][1][0] = Complex::ONE;
    let cases: [FilmCase; 5] = [
        (
            "thickness",
            HybridPerturbation {
                thickness_rates: vec![1.0],
                ..HybridPerturbation::none(&stack)
            },
            1e-5,
            |h| Film {
                thickness: FILM.thickness + h,
                ..FILM
            },
        ),
        ("ε_xx", unit_entry, 1e-5, |h| Film {
            shift: [h, 0.0],
            ..FILM
        }),
        ("ε_xy + ε_yx", unit_pair, 1e-5, |h| Film {
            shift: [0.0, h],
            ..FILM
        }),
        (
            "rotation about x",
            HybridPerturbation::rotation(&stack, K0, 1, [1.0, 0.0, 0.0]).unwrap(),
            1e-5,
            |h| Film { angle: h, ..FILM },
        ),
        (
            "k0",
            HybridPerturbation::wavenumber(&stack, K0).unwrap(),
            1e-6 * K0,
            |h| Film {
                k0: FILM.k0 + h,
                ..FILM
            },
        ),
    ];
    for (name, direction, h, film) in cases {
        let (plus, minus) = (film(h).modes(), film(-h).modes());
        let mut worst = Worst::default();
        for point in &base {
            let s = point
                .hybrid_power_field_tangent_with(
                    &Scatter4::new(),
                    &direction,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                )
                .unwrap();
            // Transfer4 agrees with Scatter4.
            let t = point
                .hybrid_power_field_tangent_with(
                    &Transfer4::new(),
                    &direction,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                )
                .unwrap();
            for (a, b) in s.samples.iter().zip(&t.samples) {
                let (a, b) = (a.tangent.unwrap(), b.tangent.unwrap());
                for c in 0..6 {
                    assert!((a[c] - b[c]).norm() < 1e-7 * (1.0 + a[c].norm()), "{name}");
                }
            }
            assert_eq!(s.phase_reference, HybridPhaseReference::EyPlusZ0Hy);
            let sum: f64 = s.regional_power.iter().map(|r| r.power_w_per_m).sum();
            let sum_rate: f64 = s.regional_power.iter().map(|r| r.power_rate_w_per_m).sum();
            assert!((sum - 1.0).abs() < 1e-12 && sum_rate.abs() < 1e-9, "{name}");
            let target = beta(point);
            let rate = s.rate.beta_rate.re;
            let (p, m) = (
                nearest(&plus, target + rate * h),
                nearest(&minus, target - rate * h),
            );
            let w = compare(&s, &still(p, &sampling), &still(m, &sampling), h);
            worst.fields = worst.fields.max(w.fields);
            worst.fractions = worst.fractions.max(w.fractions);
        }
        println!(
            "hybrid film, {name}: field rates {:e}, fraction rates {:e} (relative to central differences)",
            worst.fields, worst.fractions
        );
        assert!(
            worst.fields < 1e-6 && worst.fractions < 1e-6,
            "{name}: {worst:?}"
        );
    }
}

/// In the fixed-lab frame a sample keeps its position while the film's far
/// interface moves: layer samples at fixed fractions of the film (moving
/// with it when attached), a right-exterior point measured from the unmoved
/// first interface, and a sample on the moving interface, which has no rate.
#[test]
fn fixed_lab_rates_follow_moving_interfaces() {
    let h = 1e-5;
    let base = FILM.modes();
    let stack = FILM.stack();
    let d = FILM.thickness;
    let z_right = d + 0.4;
    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform_with_boundaries(4, true, false))
        .right_exterior(ExteriorSampling::point(cm(0.4)));
    // The same lab points in a film of thickness `d`.
    let lab = |fractions: &[f64], d: f64| {
        FieldSampling::new()
            .layer(
                0,
                LayerSampling::offsets_in(
                    fractions.iter().map(|f| f * FILM.thickness),
                    LengthUnit::Centimetre,
                ),
            )
            .right_exterior(ExteriorSampling::point(cm(z_right - d)))
    };
    let thickness = HybridPerturbation {
        thickness_rates: vec![1.0],
        ..HybridPerturbation::none(&stack)
    };
    let film = |d| Film {
        thickness: d,
        ..FILM
    };
    let (plus, minus) = (film(d + h).modes(), film(d - h).modes());
    let mut worst: f64 = 0.0;
    let mut moved: f64 = 0.0;
    for point in &base {
        let s = point
            .hybrid_power_field_tangent_with(
                &Scatter4::new(),
                &thickness,
                &sampling,
                GuidedObservationFrame::FixedLab,
            )
            .unwrap();
        let attached = point
            .hybrid_power_field_tangent_with(
                &Scatter4::new(),
                &thickness,
                &sampling,
                GuidedObservationFrame::MaterialAttached,
            )
            .unwrap();
        let fractions: Vec<f64> = s
            .samples
            .iter()
            .filter_map(|sample| match sample.observation {
                GuidedObservation::FixedLab(lamina_core::FieldPosition::Layer {
                    position: lamina_core::ResolvedLayerPosition::Fraction(f),
                    ..
                }) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(fractions.len(), 4);
        // The sample at the fixed first interface moves with neither frame.
        assert_eq!(s.samples[0].tangent, attached.samples[0].tangent);
        for (a, b) in s.samples.iter().zip(&attached.samples).skip(1) {
            let (a, b) = (a.tangent.unwrap(), b.tangent.unwrap());
            moved = moved.max((0..6).map(|c| (a[c] - b[c]).norm()).fold(0.0, f64::max));
        }
        let target = beta(point);
        let rate = s.rate.beta_rate.re;
        let p = still(nearest(&plus, target + rate * h), &lab(&fractions, d + h));
        let m = still(nearest(&minus, target - rate * h), &lab(&fractions, d - h));
        worst = worst.max(compare(&s, &p, &m, h).fields);
        // On the moving interface between the film and the substrate.
        let on = FieldSampling::new().right_exterior(ExteriorSampling::point(cm(0.0)));
        let s = point
            .hybrid_power_field_tangent_with(
                &Scatter4::new(),
                &thickness,
                &on,
                GuidedObservationFrame::FixedLab,
            )
            .unwrap();
        assert_eq!(
            s.samples[0].tangent,
            Err(GuidedObservationError::MovingInterface { interface: 1 })
        );
    }
    println!(
        "hybrid film, fixed-lab thickness rates: {worst:e} (relative to central differences); \
         largest lab − attached difference {moved:.3e}"
    );
    assert!(moved > 1e-1);
    assert!(worst < 1e-6, "{worst:e}");
}

/// Moving the stack's origin through the lab leaves the mode unchanged and
/// moves fixed-lab samples relative to the stack: their rate is `−∂f/∂z`.
#[test]
fn a_moving_origin_shifts_fixed_lab_samples() {
    let h = 1e-6;
    let base = FILM.modes();
    let stack = FILM.stack();
    let at = |shift: f64| {
        FieldSampling::new()
            .left_exterior(ExteriorSampling::point(cm(0.3 + shift)))
            .layer(
                0,
                LayerSampling::offsets_in([0.37 - shift], LengthUnit::Centimetre),
            )
            .right_exterior(ExteriorSampling::point(cm(0.4 - shift)))
    };
    let translation = HybridPerturbation {
        origin_velocity: 1.0,
        ..HybridPerturbation::none(&stack)
    };
    let mut worst: f64 = 0.0;
    for point in &base {
        let s = point
            .hybrid_power_field_tangent_with(
                &Scatter4::new(),
                &translation,
                &at(0.0),
                GuidedObservationFrame::FixedLab,
            )
            .unwrap();
        assert!(s.rate.beta_rate.norm() < 1e-12 * beta(point));
        let rates: Vec<f64> = s
            .regional_power
            .iter()
            .map(|r| r.power_rate_w_per_m)
            .collect();
        assert!(
            rates.iter().all(|r| r.abs() < 1e-9),
            "{rates:?} {:?}",
            s.scale_log_rate
        );
        let w = compare(&s, &still(point, &at(h)), &still(point, &at(-h)), h);
        worst = worst.max(w.fields);
    }
    println!("hybrid film, moving origin: {worst:e} (relative to central differences)");
    assert!(worst < 1e-6, "{worst:e}");
}

// --- An avoided crossing ---------------------------------------------------

/// A uniaxial slab (o = 4, e = 6.25) in a 2.1 cladding whose first TE and
/// third TM modes cross with the optic axis along z (half-thickness
/// 0.547635369 cm); the axis tilted by 0.02 towards y opens the crossing by
/// 5.6e-3 cm⁻¹, then turned by `angle` about x.
fn crossing(angle: f64, thickness: f64) -> ModeStack {
    Stack::builder(isotropic(2.1), isotropic(2.1))
        .layer(
            Region::oriented(
                AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
                    Constant::dielectric(4.0),
                    Constant::dielectric(6.25),
                )),
                about_x(angle, Rotation::from_optic_axis(0.02, FRAC_PI_2)),
            ),
            cm(thickness),
        )
        .finalise()
}

const CROSSING_THICKNESS: f64 = 2.0 * 0.547635369;
const CROSSING_BETA: f64 = 12.342087944;

/// A named direction and the perturbed slab.
type CrossingCase = (&'static str, HybridPerturbation, fn(f64) -> ModeStack);

#[test]
fn avoided_crossing_rates_match_central_differences() {
    let window = [CROSSING_BETA - 0.05, CROSSING_BETA + 0.05];
    let base = modes(crossing(0.0, CROSSING_THICKNESS), K0, window);
    assert_eq!(base.len(), 2, "the pair of the avoided crossing");
    let split = (beta(&base[0]) - beta(&base[1])).abs();
    let stack = crossing(0.0, CROSSING_THICKNESS);
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::distances_in(
            [0.0, 0.3],
            LengthUnit::Centimetre,
        ))
        .layer(0, LayerSampling::uniform_with_boundaries(7, true, true))
        .right_exterior(ExteriorSampling::point(cm(0.3)));
    let h = 1e-6;
    let cases: [CrossingCase; 2] = [
        (
            "rotation about x",
            HybridPerturbation::rotation(&stack, K0, 1, [1.0, 0.0, 0.0]).unwrap(),
            |h| crossing(h, CROSSING_THICKNESS),
        ),
        (
            "thickness",
            HybridPerturbation {
                thickness_rates: vec![1.0],
                ..HybridPerturbation::none(&stack)
            },
            |h| crossing(0.0, CROSSING_THICKNESS + h),
        ),
    ];
    for (name, direction, perturbed) in cases {
        let (plus, minus) = (
            modes(perturbed(h), K0, window),
            modes(perturbed(-h), K0, window),
        );
        assert_eq!((plus.len(), minus.len()), (2, 2));
        let mut worst = Worst::default();
        let mut largest: f64 = 0.0;
        for point in &base {
            let s = point
                .hybrid_power_field_tangent_with(
                    &Scatter4::new(),
                    &direction,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                )
                .unwrap();
            let target = beta(point);
            let rate = s.rate.beta_rate.re;
            let (p, m) = (
                nearest(&plus, target + rate * h),
                nearest(&minus, target - rate * h),
            );
            let w = compare(&s, &still(p, &sampling), &still(m, &sampling), h);
            worst.fields = worst.fields.max(w.fields);
            worst.fractions = worst.fractions.max(w.fractions);
            for sample in &s.samples {
                for v in sample.tangent.unwrap() {
                    largest = largest.max(v.norm());
                }
            }
        }
        println!(
            "avoided crossing (split {split:.3e} cm^-1), {name}: field rates {:e}, fraction \
             rates {:e}, largest field rate {largest:.3e}",
            worst.fields, worst.fractions
        );
        assert!(
            worst.fields < 1e-5 && worst.fractions < 1e-5,
            "{name}: {worst:?}"
        );
    }
}

// --- A separable stack against the scalar path ------------------------------

/// A diagonal (biaxial, principal axes on the stack axes) film on a
/// substrate: TE and TM decouple.
fn separable(scale: f64) -> ModeStack {
    Stack::builder(isotropic(1.0), isotropic(2.1))
        .layer(
            Region::oriented(
                AnalyticalTensorMaterialHandle::new(PrincipalAxes::biaxial(
                    Constant::dielectric(4.0),
                    Constant::dielectric(4.6),
                    Constant::dielectric(5.3),
                )),
                Rotation::identity(),
            ),
            cm(0.8 * scale),
        )
        .finalise()
}

/// At a 1 cm wavelength and at a 1 µm one, where k0 and β are 10⁴ times
/// larger in cm⁻¹.
#[test]
fn separable_hybrid_rates_equal_the_scalar_rates() {
    for scale in [1.0, 1e-4] {
        separable_rates(scale);
    }
}

fn separable_rates(scale: f64) {
    let k0 = K0 / scale;
    let stack = separable(scale);
    let window = |family| GuidedWindow {
        wavelength: cm(scale),
        family,
        direction: GuidedDirection::Forward,
        abs_beta: [0.0, 3.0 * k0],
        cutoff_margin: 1e-3 * k0,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    };
    let all = enumerate_guided_modes(stack.clone(), window(ModeFamily::All)).unwrap();
    assert!(all.is_complete());
    let sampling = scaled_sampling(scale);
    // dλ/dp with dk0/dp = 1: dk0/dλ = −k0²/2π.
    let wavelength_rate = -TAU / (k0 * k0);
    let rates = vec![ConstitutiveRate::default(); 3];
    let mut entry = rates.clone();
    entry[1] = ConstitutiveRate::diagonal([0.0, 1.0, 0.0], [0.0; 3]);
    let mut hybrid_entry = HybridPerturbation::none(&stack);
    hybrid_entry.tensor_rates[1][0][1][1] = Complex::ONE;
    let cases = [
        (
            "thickness",
            GuidedPerturbation::thickness(1, 0).unwrap(),
            HybridPerturbation {
                thickness_rates: vec![1.0],
                ..HybridPerturbation::none(&stack)
            },
        ),
        (
            "ε_yy",
            GuidedPerturbation::new("ε_yy", "1", vec![0.0, 0.0], 0.0, entry).unwrap(),
            hybrid_entry,
        ),
        (
            "k0",
            GuidedPerturbation::new("k0", "1/cm", vec![0.0, 0.0], wavelength_rate, rates).unwrap(),
            HybridPerturbation::wavenumber(&stack, k0).unwrap(),
        ),
    ];
    let mut checked = [0; 2];
    for (name, scalar, hybrid) in cases {
        let (mut worst, mut worst_fractions): (f64, f64) = (0.0, 0.0);
        for (point, label) in all.modes.iter().zip(&all.labels) {
            let polarisation = match label {
                ModeLabel::TransverseElectric => lamina_core::Polarisation::TransverseElectric,
                ModeLabel::TransverseMagnetic => lamina_core::Polarisation::TransverseMagnetic,
                ModeLabel::Hybrid => panic!("a separable stack"),
            };
            checked[polarisation as usize] += 1;
            let family = ModeFamily::Polarised(polarisation);
            let scalar_modes = enumerate_guided_modes(stack.clone(), window(family)).unwrap();
            let scalar_point = nearest(&scalar_modes.modes, beta(point));
            assert!((beta(scalar_point) - beta(point)).abs() < 1e-12 * beta(point));
            let reference = scalar_point
                .guided_beta_sensitivity_with(
                    &Scatter2::new(),
                    &scalar,
                    Default::default(),
                    Default::default(),
                )
                .unwrap()
                .power_field_tangent_with(
                    &Scatter2::new(),
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                    Complex::ONE,
                )
                .unwrap();
            let s = point
                .hybrid_power_field_tangent_with(
                    &Scatter4::new(),
                    &hybrid,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                )
                .unwrap();
            let scale = reference
                .samples
                .iter()
                .flat_map(|x| x.field.iter().chain(x.tangent.as_ref().unwrap()))
                .map(|v| v.norm())
                .fold(0.0, f64::max);
            for (a, b) in s.samples.iter().zip(&reference.samples) {
                let (ta, tb) = (a.tangent.unwrap(), b.tangent.unwrap());
                for c in 0..6 {
                    worst = worst.max((a.field[c] - b.field[c]).norm() / scale);
                    worst = worst.max((ta[c] - tb[c]).norm() / scale);
                }
            }
            let rates = reference
                .regional_power
                .iter()
                .map(|r| r.power_rate_w_per_m.abs())
                .fold(1.0, f64::max);
            for (a, b) in s.regional_power.iter().zip(&reference.regional_power) {
                worst_fractions = worst_fractions.max((a.power_w_per_m - b.power_w_per_m).abs());
                worst_fractions = worst_fractions
                    .max((a.power_rate_w_per_m - b.power_rate_w_per_m).abs() / rates);
            }
        }
        println!(
            "separable film (λ = {scale} cm), {name}: hybrid against scalar path: fields \
             {worst:e}, fractions {worst_fractions:e}"
        );
        assert!(
            worst < 1e-8 && worst_fractions < 1e-8,
            "{name}: {worst:e} {worst_fractions:e}"
        );
    }
    assert!(checked.iter().all(|&c| c > 0), "{checked:?}");
}

/// A TE mode of a full-tensor film whose optic axis lies in xz, solved on
/// its TE channel: a turn about y keeps it TE and has field rates; a turn
/// about x couples TE and TM, which the TE channel cannot follow.
#[test]
fn a_polarised_channel_refuses_directions_that_couple_te_and_tm() {
    let stack = Stack::builder(isotropic(1.0), isotropic(2.1))
        .layer(
            Region::oriented(
                AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
                    Constant::dielectric(4.0),
                    Constant::dielectric(6.25),
                )),
                Rotation::from_optic_axis(0.6, 0.0),
            ),
            cm(0.8),
        )
        .finalise();
    let basis = enumerate_guided_modes(
        stack.clone(),
        GuidedWindow {
            wavelength: cm(1.0),
            family: ModeFamily::Polarised(lamina_core::Polarisation::TransverseElectric),
            direction: GuidedDirection::Forward,
            abs_beta: [0.0, 3.0 * K0],
            cutoff_margin: 1e-3 * K0,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        },
    )
    .unwrap();
    let point = &basis.modes[0];
    let sampling = film_sampling();
    let about = |axis| HybridPerturbation::rotation(&stack, K0, 1, axis).unwrap();
    let tangent = |direction: &HybridPerturbation| {
        point.hybrid_power_field_tangent_with(
            &Scatter4::new(),
            direction,
            &sampling,
            GuidedObservationFrame::MaterialAttached,
        )
    };
    let kept = tangent(&about([0.0, 1.0, 0.0])).unwrap();
    // TE: E_x, E_z and Z0H_y and their rates vanish.
    for sample in &kept.samples {
        let rate = sample.tangent.unwrap();
        for c in [0, 2, 4] {
            assert_eq!(sample.field[c], Complex::ZERO);
            assert_eq!(rate[c], Complex::ZERO);
        }
    }
    assert!(matches!(
        tangent(&about([1.0, 0.0, 0.0])),
        Err(HybridFieldError::CouplesPolarisations { region: 1 })
    ));
}
