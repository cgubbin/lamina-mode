//! Rates of hybrid-mode β along single permittivity entries of a general
//! (full) tensor film, with the rates declared by the material model, against
//! central differences of independently enumerated modes.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Constant, Rotation, Scatter4, Stack,
    Transfer4,
};
use lamina_materials::{ConstantTensor, CrystalAxis, TensorParameter};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::TAU;

const K0: f64 = TAU;

/// A Hermitian, positive-definite film tensor with every entry non-zero.
fn epsilon(shift: [[f64; 3]; 3]) -> [[Complex; 3]; 3] {
    let base = [[4.0, 0.3, 0.2], [0.3, 5.0, 0.15], [0.2, 0.15, 6.25]];
    std::array::from_fn(|i| std::array::from_fn(|j| Complex::from(base[i][j] + shift[i][j])))
}

fn material(shift: [[f64; 3]; 3]) -> ConstantTensor<f64> {
    let one = Complex::ONE;
    let zero = Complex::ZERO;
    let mu = [[one, zero, zero], [zero, one, zero], [zero, zero, one]];
    ConstantTensor::new(epsilon(shift), mu).unwrap()
}

fn stack(shift: [[f64; 3]; 3]) -> ModeStack {
    let isotropic =
        |e: f64| -> Region { AnalyticalMaterialHandle::new(Constant::dielectric(e)).into() };
    Stack::builder(isotropic(1.0), isotropic(2.1))
        .layer(
            Region::oriented(
                AnalyticalTensorMaterialHandle::new(material(shift)),
                Rotation::identity(),
            ),
            Length::centimetres(0.8),
        )
        .finalise()
}

fn modes(shift: [[f64; 3]; 3]) -> Vec<ModePoint> {
    let basis = enumerate_guided_modes(
        stack(shift),
        GuidedWindow {
            wavelength: Length::centimetres(1.0),
            family: ModeFamily::All,
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
    assert!(basis.is_complete());
    basis.modes
}

/// The film's declared rate ∂ε/∂ε_ij (crystal frame = stack frame here) as
/// a direction.
fn entry(row: usize, column: usize) -> HybridPerturbation {
    let axes = [CrystalAxis::A, CrystalAxis::B, CrystalAxis::C];
    let rate = material([[0.0; 3]; 3])
        .permittivity_parameter_rates()
        .into_iter()
        .find(|r| {
            r.parameter
                == TensorParameter::Permittivity {
                    row: axes[row],
                    column: axes[column],
                }
        })
        .unwrap();
    let mut direction = HybridPerturbation::none(&stack([[0.0; 3]; 3]));
    direction.tensor_rates[1][0] = rate.value;
    direction
}

fn sum(a: &HybridPerturbation, b: &HybridPerturbation) -> HybridPerturbation {
    let mut out = a.clone();
    for (region, rates) in out.tensor_rates.iter_mut().enumerate() {
        for f in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    rates[f][i][j] += b.tensor_rates[region][f][i][j];
                }
            }
        }
    }
    out
}

fn rates(direction: &HybridPerturbation, points: &[ModePoint]) -> Vec<Complex> {
    points
        .iter()
        .map(|p| {
            let scatter = p
                .hybrid_beta_rate_with(&Scatter4::new(), direction)
                .unwrap();
            let transfer = p
                .hybrid_beta_rate_with(&Transfer4::new(), direction)
                .unwrap();
            let (a, b) = (scatter.beta_rate, transfer.beta_rate);
            assert!((a - b).norm() < 1e-9 * a.norm().max(1.0), "{a} {b}");
            a
        })
        .collect()
}

#[test]
fn single_entry_rates_match_central_differences() {
    let points = modes([[0.0; 3]; 3]);
    assert!(points.len() >= 4);
    let h = 1e-5;
    // Diagonal entries alone, and the symmetric off-diagonal pair (a
    // Hermitian perturbation, so neighbouring stacks enumerate).
    let cases: [(&str, HybridPerturbation, [[f64; 3]; 3]); 4] = [
        (
            "ε_xx",
            entry(0, 0),
            [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        ),
        (
            "ε_zz",
            entry(2, 2),
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
        ),
        (
            "ε_xy + ε_yx",
            sum(&entry(0, 1), &entry(1, 0)),
            [[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]],
        ),
        (
            "ε_yz + ε_zy",
            sum(&entry(1, 2), &entry(2, 1)),
            [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]],
        ),
    ];
    for (name, direction, unit) in cases {
        let scaled = |s: f64| unit.map(|row| row.map(|v| v * s));
        let (plus, minus) = (modes(scaled(h)), modes(scaled(-h)));
        assert_eq!(plus.len(), points.len());
        assert_eq!(minus.len(), points.len());
        for (i, rate) in rates(&direction, &points).into_iter().enumerate() {
            let fd =
                (plus[i].physical_coordinate().re - minus[i].physical_coordinate().re) / (2.0 * h);
            assert!(
                rate.im.abs() < 1e-9 * rate.norm().max(1.0),
                "{name}: {rate}"
            );
            assert!(
                (rate.re - fd).abs() < 1e-6 * fd.abs().max(1.0),
                "{name}, mode {i}: rate {rate} vs differences {fd}"
            );
        }
    }
}

/// About a lossless (Hermitian) film dβ/dε_ij ∝ ∫ E_i* E_j dz, so the rates
/// of ε_ij and ε_ji are complex conjugates: one entry alone makes the tensor
/// non-Hermitian and β complex; the Hermitian pair moves β along the real
/// axis at the sum of the two.
#[test]
fn off_diagonal_entry_rates_are_conjugate() {
    let points = modes([[0.0; 3]; 3]);
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        let ij = rates(&entry(row, column), &points);
        let ji = rates(&entry(column, row), &points);
        let pair = rates(&sum(&entry(row, column), &entry(column, row)), &points);
        for ((a, b), p) in ij.iter().zip(&ji).zip(&pair) {
            assert!((a - b.conj()).norm() < 1e-9 * p.norm().max(1.0), "{a} {b}");
            assert!(p.im.abs() < 1e-9 * p.norm().max(1.0));
            assert!((a + b - p).norm() < 1e-9 * p.norm().max(1.0), "{a} {b} {p}");
        }
    }
}
