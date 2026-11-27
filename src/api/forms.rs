//! Closed-form integrals of quadratic forms of mode fields over the whole
//! line, through lamina-core's
//! [`directional_forms`](lamina_core::ModalDerivativeBackend::directional_forms):
//! one block exponential per region and form, exact over finite layers, wall
//! gaps and open exteriors, at degenerate layer eigenvalues and at cutoff.
//!
//! Weights act on the six components `f = [E_x, E_y, E_z, Z₀H_x, Z₀H_y,
//! Z₀H_z]`, rows indexing the first operand, which is the conjugated one of
//! a Hermitian form.
use super::*;
use crate::problem::J;
use lamina_core::{
    CanonicalCoordinates, Channel, ConstitutiveTensor, ExteriorBranch, ExteriorWavevectors,
    ModalCharacteristicDirection, ModalDerivativeBackend, ModalForm, ModalFormOperand,
    ScalarAlgebra, SeedJet,
};
use ndarray::arr0;
use std::sync::Arc;

/// A `6 × 6` weight of a form.
pub(super) type Weight = [[Complex; 6]; 6];

/// `Σ f_a[r] W[r][c] f_b[c]` for the bilinear form, `f_a[r]*` for the
/// Hermitian one.
pub(super) fn weight(entries: &[(usize, usize, Complex)]) -> Weight {
    let mut weight = [[Complex::ZERO; 6]; 6];
    for &(row, column, value) in entries {
        weight[row][column] += value;
    }
    weight
}

/// `η_z,a* E_y,b − η_y,a* E_z,b`, Hermitian: `(E_b × η_a*)·x̂`, twice the
/// longitudinal power density when `a = b`.
pub(super) fn flux() -> Weight {
    weight(&[(5, 1, Complex::ONE), (4, 2, -Complex::ONE)])
}

/// `E_y,a η_z,b − E_z,a η_y,b`, bilinear: `(E_a × η_b)·x̂`.
pub(super) fn cross() -> Weight {
    weight(&[(1, 5, Complex::ONE), (2, 4, -Complex::ONE)])
}

/// `E_a† W_e E_b + η_a† W_m η_b` with block weights `[W_e, W_m]`.
pub(super) fn blocks(tensors: &[[[Complex; 3]; 3]; 2]) -> Weight {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| match (row / 3, column / 3) {
            (0, 0) => tensors[0][row][column],
            (1, 1) => tensors[1][row - 3][column - 3],
            _ => Complex::ZERO,
        })
    })
}

pub(super) fn constant(value: Complex) -> J {
    <J as SeedJet>::constant(arr0(value))
}

/// `value + rate·t` as a first-order jet in the direction's parameter.
pub(super) fn seed(value: Complex, rate: Complex) -> J {
    constant(value).add(
        &J::variable(arr0(Complex::ZERO), 0)
            .expect("one jet slot")
            .scale(rate),
    )
}

/// The exterior wavevectors of a point with their rates along a direction.
pub(super) fn exterior_jets(
    signature: &BranchSignature,
    rates: [Option<[Complex; 2]>; 2],
) -> ExteriorWavevectors<J> {
    let side = |scalar: Option<Complex>, pair: Option<[Complex; 2]>, rate: Option<[Complex; 2]>| {
        let rate = rate.unwrap_or([Complex::ZERO; 2]);
        match (pair, scalar) {
            (Some(pair), _) => Some(ExteriorBranch::Pair([
                seed(pair[0], rate[0]),
                seed(pair[1], rate[1]),
            ])),
            (None, Some(kappa)) => Some(ExteriorBranch::Scalar(seed(kappa, rate[0]))),
            (None, None) => None,
        }
    };
    ExteriorWavevectors::from_branches(
        side(signature.left, signature.pairs[0], rates[0]),
        side(signature.right, signature.pairs[1], rates[1]),
    )
}

/// The coordinates, exteriors and zero rates of an accepted point: the
/// direction along which nothing moves.
pub(super) struct Still {
    provenance: Provenance,
    coordinates: CanonicalCoordinates<J>,
    exterior: ExteriorWavevectors<J>,
    thickness: Vec<f64>,
    rates: Vec<[ConstitutiveTensor<Complex>; 2]>,
}

/// A point without accepted branch evidence.
#[derive(Clone, Copy, Debug)]
pub(super) struct NoSignature;

impl Still {
    pub(super) fn new(point: &ModePoint) -> Result<Self, NoSignature> {
        let signature = point.signature().map_err(|_| NoSignature)?;
        let provenance = point.provenance();
        let layers = provenance.stack().len();
        let zero = ConstitutiveTensor::Isotropic(Complex::ZERO);
        Ok(Self {
            provenance,
            coordinates: CanonicalCoordinates::new(
                constant(signature.k0),
                constant(signature.k_parallel),
            ),
            exterior: exterior_jets(signature, [None, None]),
            thickness: vec![0.0; layers],
            rates: vec![[zero.clone(), zero]; layers + 2],
        })
    }

    pub(super) fn direction(
        &self,
    ) -> ModalCharacteristicDirection<'_, Region, [ConstitutiveTensor<Complex>; 2], Channel> {
        ModalCharacteristicDirection {
            stack: self.provenance.stack(),
            coordinates: &self.coordinates,
            exterior: &self.exterior,
            polarisation: self.provenance.channel(),
            thickness_rates: &self.thickness,
            constitutive_rates: &self.rates,
        }
    }
}

/// Each form's integral over each region (left exterior, layers, right
/// exterior), in cm times the fields' units squared.
pub(super) type Integrals = Vec<Vec<Complex>>;

/// The forms of two accepted modes (the same mode when they are the same
/// reference with the same amplitude), each the backend's seed times its
/// amplitude: values region by region, form by form.
pub(super) fn still_forms<B: ModalDerivativeBackend>(
    backend: &B,
    modes: [&ModePoint; 2],
    amplitudes: [Complex; 2],
    forms: &[(bool, &[Weight])],
) -> Result<Result<Integrals, Arc<dyn std::error::Error + Send + Sync>>, NoSignature> {
    let first = Still::new(modes[0])?;
    let same = std::ptr::eq(modes[0], modes[1]) && amplitudes[0] == amplitudes[1];
    let second = match same {
        true => None,
        false => Some(Still::new(modes[1])?),
    };
    let a = ModalFormOperand {
        direction: first.direction(),
        amplitude: amplitudes[0],
    };
    let b = match &second {
        Some(still) => ModalFormOperand {
            direction: still.direction(),
            amplitude: amplitudes[1],
        },
        None => a,
    };
    let forms: Vec<ModalForm<'_>> = forms
        .iter()
        .map(|&(conjugate, weights)| ModalForm { conjugate, weights })
        .collect();
    Ok(backend.directional_forms([a, b], &forms).map(|integrals| {
        integrals
            .iter()
            .map(|regions| regions.iter().map(|value| value.value()[()]).collect())
            .collect()
    }))
}
