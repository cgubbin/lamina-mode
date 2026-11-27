//! Six-component fields of guided modes of any channel: unit longitudinal
//! power, the phase rule, overlaps, the reciprocity pairing, β rates along
//! thickness, tensor-entry, rotation and wavelength directions, and the
//! group index with its energy certificate.
//!
//! Integrals over the whole line are closed forms (see [`super::forms`]):
//! one block exponential per region and form, exact over finite layers, wall
//! gaps and open exteriors; a side closed by a PML is refused.
use super::forms::{self, Weight, exterior_jets, seed};
use super::power::{CM_TO_M_OVER_Z0, PmlIntegral};
use super::*;
use crate::problem::guided::GuidedExteriors;
use crate::problem::material::{self as m, Tensor};
use lamina_core::{
    CanonicalCoordinates, ConstitutiveTensor, FieldSampling, ModalCharacteristicDirection,
    ModalDerivativeBackend, OwnedModeFields, StackSide,
};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum HybridFieldError {
    #[error("accepted branch evidence is unavailable")]
    Branch,
    #[error("needs a guided mode at real k0 and β")]
    Ineligible,
    #[error("field recovery failed")]
    Fields(#[source] Arc<dyn std::error::Error + Send + Sync>),
    /// The exterior wavevectors could not be continued along a direction.
    #[error("the {side:?} exterior cannot be continued along the direction")]
    Exterior { side: StackSide },
    /// Two modes whose stacks differ in their layers.
    #[error("the modes' stacks differ in their layer thicknesses")]
    Mismatched,
    #[error("the power or the phase reference vanishes")]
    IllConditioned,
    #[error(transparent)]
    Termination(#[from] PmlIntegral),
}

/// The left-interface field made real and positive by the phase rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridPhaseReference {
    /// `E_y + Z₀H_y`: `E_y` of a TE mode and `Z₀H_y` of a TM mode.
    EyPlusZ0Hy,
    /// `E_x + Z₀H_x`, where `E_y + Z₀H_y` is below `1e-6` of the tangential
    /// field there.
    ExPlusZ0Hx,
}

/// Fields scaled to unit longitudinal power and the phase rule.
#[derive(Clone, Debug)]
pub struct HybridProfile {
    pub point: ModePoint,
    /// `(E, Z₀H)` at the requested samples.
    pub fields: OwnedModeFields,
    /// `½ Re ∫ (E_y H_z* − E_z H_y*) dz` of the backend's seed, W/m.
    pub seed_power_w_per_m: f64,
    /// ±1 W/m: the sign of the mode's power flow.
    pub signed_power_w_per_m: f64,
    /// Common multiplier of the seed fields.
    pub scale: Complex,
    pub phase_reference: HybridPhaseReference,
    /// The reference of the seed at the left interface.
    pub reference: Complex,
    /// `½ ∫ (E × H)·x̂ dz` of the scaled fields, W/m: real to round-off for
    /// lossless reciprocal modes in a real gauge; recorded, not enforced.
    pub bilinear_norm_w_per_m: Complex,
}

/// Six components `[E_x, E_y, E_z, Z₀H_x, Z₀H_y, Z₀H_z]` at every sample.
fn components(fields: &OwnedModeFields) -> Vec<[Complex; 6]> {
    let q = fields.quantity();
    let (e, h) = (q.electric(), q.magnetic());
    (0..e.x().len())
        .map(|i| [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]])
        .collect()
}

impl ModePoint {
    /// The backend's seed fields at `sampling`, multiplied by `amplitude`.
    fn seed_fields<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &FieldSampling<f64>,
        amplitude: Complex,
    ) -> Result<OwnedModeFields, HybridFieldError> {
        let still = forms::Still::new(self).map_err(|_| HybridFieldError::Branch)?;
        backend
            .directional_fields(still.direction(), sampling, amplitude)
            .map(|fields| fields.fields)
            .map_err(HybridFieldError::Fields)
    }
}

/// The whole-line integrals of `forms` (pairing conjugated or not, and
/// weights) of two modes of stacks with the same layers, each the backend's
/// seed times its amplitude. A side closed by a PML is refused.
fn integrate<B: ModalDerivativeBackend>(
    backend: &B,
    modes: [&ModePoint; 2],
    amplitudes: [Complex; 2],
    integrals: &[(bool, &[Weight])],
) -> Result<Vec<Complex>, HybridFieldError> {
    let provenances = modes.map(ModePoint::provenance);
    let [a, b] = [provenances[0].stack(), provenances[1].stack()];
    for stack in [a, b] {
        crate::problem::walls(stack).map_err(|side| PmlIntegral { side })?;
    }
    let thicknesses = |stack: &ModeStack| -> Vec<f64> {
        stack
            .layers_left_to_right()
            .iter()
            .map(|layer| layer.thickness().as_centimetres())
            .collect()
    };
    if thicknesses(a) != thicknesses(b) {
        return Err(HybridFieldError::Mismatched);
    }
    let regions = forms::still_forms(backend, modes, amplitudes, integrals)
        .map_err(|_| HybridFieldError::Branch)?
        .map_err(HybridFieldError::Fields)?;
    regions
        .iter()
        .map(|shares| {
            let total: Complex = shares.iter().sum();
            match total.re.is_finite() && total.im.is_finite() {
                true => Ok(total),
                false => Err(HybridFieldError::IllConditioned),
            }
        })
        .collect()
}

impl ModePoint {
    /// Scale the fields to `½ Re ∫ (E_y H_z* − E_z H_y*) dz = ±1 W/m` and
    /// make `E_y + Z₀H_y` at the left interface real and positive (or
    /// `E_x + Z₀H_x` where that vanishes); sample them at `sampling`.
    pub fn hybrid_power_profile_with<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &FieldSampling<f64>,
    ) -> Result<HybridProfile, HybridFieldError> {
        let signature = self.signature().map_err(|_| HybridFieldError::Branch)?;
        let (k, beta) = (signature.k0, signature.k_parallel);
        if k.re <= 0.0 || k.im != 0.0 || beta.im.abs() > 1e-9 * beta.norm() {
            return Err(HybridFieldError::Ineligible);
        }
        let one = Complex::ONE;
        // The power and the unconjugated self-pairing from one solve; the
        // pairing is bilinear, so that of the scaled fields is scale² times it.
        let totals = integrate(
            backend,
            [self, self],
            [one, one],
            &[(true, &[forms::flux()]), (false, &[forms::cross()])],
        )?;
        let half = 0.5 * CM_TO_M_OVER_Z0;
        let (power, pairing) = (half * totals[0].re, half * totals[1]);
        if !power.is_finite() || power == 0.0 {
            return Err(HybridFieldError::IllConditioned);
        }
        // The left interface: the first interface of the stack.
        let at = FieldSampling::new().left_exterior(lamina_core::ExteriorSampling::point(
            lamina_units::Length::zero(),
        ));
        let left = components(&self.seed_fields(backend, &at, one)?)[0];
        let tangential = [left[0], left[1], left[3], left[4]]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let (phase_reference, reference) = match left[1] + left[4] {
            r if r.norm() >= 1e-6 * tangential => (HybridPhaseReference::EyPlusZ0Hy, r),
            _ => (HybridPhaseReference::ExPlusZ0Hx, left[0] + left[3]),
        };
        if reference.norm() == 0.0 || !reference.norm().is_finite() {
            return Err(HybridFieldError::IllConditioned);
        }
        let scale = reference.conj() / reference.norm() / power.abs().sqrt();
        let bilinear = scale * scale * pairing;
        Ok(HybridProfile {
            point: self.clone(),
            fields: self.seed_fields(backend, sampling, scale)?,
            seed_power_w_per_m: power,
            signed_power_w_per_m: power.signum(),
            scale,
            phase_reference,
            reference,
            bilinear_norm_w_per_m: bilinear,
        })
    }
}

/// The unconjugated reciprocity pairing
/// `½ ∫ (E_a × H_b − E_b × H_a)·x̂ dz` in W/m of two modes of one stack,
/// each scaled by its [`HybridProfile`] multiplier.
pub fn hybrid_reciprocity_with<B: ModalDerivativeBackend>(
    backend: &B,
    modes: [&HybridProfile; 2],
) -> Result<Complex, HybridFieldError> {
    // (E_a × η_b − E_b × η_a)·x̂.
    let one = Complex::ONE;
    let weight = forms::weight(&[(1, 5, one), (2, 4, -one), (5, 1, -one), (4, 2, one)]);
    let totals = integrate(
        backend,
        [&modes[0].point, &modes[1].point],
        [modes[0].scale, modes[1].scale],
        &[(false, &[weight])],
    )?;
    Ok(0.5 * CM_TO_M_OVER_Z0 * totals[0])
}

/// The power overlap `½ ∫ (E_a × H_b*)·x̂ dz` in W/m of two modes of one
/// stack, each scaled by its [`HybridProfile`] multiplier.
pub fn hybrid_power_overlap_with<B: ModalDerivativeBackend>(
    backend: &B,
    modes: [&HybridProfile; 2],
) -> Result<Complex, HybridFieldError> {
    // (E_a × η_b*)·x̂: the Hermitian flux form with b conjugated.
    let totals = integrate(
        backend,
        [&modes[1].point, &modes[0].point],
        [modes[1].scale, modes[0].scale],
        &[(true, &[forms::flux()])],
    )?;
    Ok(0.5 * CM_TO_M_OVER_Z0 * totals[0])
}

/// A direction in the physical parameters of a stack: layer thickness
/// rates (cm per unit), a k0 rate (cm⁻¹ per unit) and each region's
/// stack-frame `[ε', μ']` (left exterior, layers, right exterior).
#[derive(Clone, Debug)]
pub struct HybridPerturbation {
    pub thickness_rates: Vec<f64>,
    pub k0_rate: f64,
    pub tensor_rates: Vec<[[[Complex; 3]; 3]; 2]>,
}

/// A direction's malformed or unevaluable input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the direction does not match the stack, or its materials cannot be evaluated")]
pub struct HybridDirectionError;

impl HybridPerturbation {
    /// No change.
    pub fn none(stack: &ModeStack) -> Self {
        Self {
            thickness_rates: vec![0.0; stack.len()],
            k0_rate: 0.0,
            tensor_rates: vec![[[[Complex::ZERO; 3]; 3]; 2]; stack.len() + 2],
        }
    }

    /// The rotation of region `region` (0 the left exterior, then the
    /// layers, then the right exterior) by an angle about the unit axis
    /// `axis`: `dε/dθ = Gε − εG`, `dμ/dθ` likewise, `G v = axis × v`.
    pub fn rotation(
        stack: &ModeStack,
        k0: f64,
        region: usize,
        axis: [f64; 3],
    ) -> Result<Self, HybridDirectionError> {
        let (_, material) = m::regions(stack).nth(region).ok_or(HybridDirectionError)?;
        let id = m::region_id(region, stack.len());
        let [values, _] = m::full_tensors(id, material, Complex::new(k0, 0.0))
            .map_err(|_| HybridDirectionError)?;
        let [x, y, z] = axis.map(Complex::from);
        let g = [
            [Complex::ZERO, -z, y],
            [z, Complex::ZERO, -x],
            [-y, x, Complex::ZERO],
        ];
        let product = |a: &Tensor, b: &Tensor| -> Tensor {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
            })
        };
        let mut direction = Self::none(stack);
        direction.tensor_rates[region] = values.map(|t| {
            let (left, right) = (product(&g, &t), product(&t, &g));
            std::array::from_fn(|i| std::array::from_fn(|j| left[i][j] - right[i][j]))
        });
        Ok(direction)
    }

    /// k0 at unit rate, with every region's material dispersion.
    pub fn wavenumber(stack: &ModeStack, k0: f64) -> Result<Self, HybridDirectionError> {
        let mut direction = Self::none(stack);
        direction.k0_rate = 1.0;
        for (index, (id, material)) in m::regions(stack).enumerate() {
            let [_, rates] = m::full_tensors(id, material, Complex::new(k0, 0.0))
                .map_err(|_| HybridDirectionError)?;
            direction.tensor_rates[index] = rates;
        }
        Ok(direction)
    }
}

/// The rate of β along a direction at an accepted point, with its
/// implicit-derivative parts.
#[derive(Clone, Copy, Debug)]
pub struct HybridBetaRate {
    /// cm⁻¹ per parameter unit.
    pub beta_rate: Complex,
    /// ∂F/∂p at fixed β and ∂F/∂β, in the backend's gauge.
    pub characteristic_direction: Complex,
    pub characteristic_beta: Complex,
}

impl ModePoint {
    /// `dβ/dp = −(∂F/∂p)/(∂F/∂β)` through the backend's characteristic,
    /// with the exterior pairs continued along the direction.
    pub fn hybrid_beta_rate_with<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
        direction: &HybridPerturbation,
    ) -> Result<HybridBetaRate, HybridFieldError> {
        let signature = self.signature().map_err(|_| HybridFieldError::Branch)?;
        let provenance = self.provenance();
        let stack = provenance.stack();
        let (k, beta) = (signature.k0, signature.k_parallel);
        if direction.thickness_rates.len() != stack.len()
            || direction.tensor_rates.len() != stack.len() + 2
            || k.im != 0.0
        {
            return Err(HybridFieldError::Ineligible);
        }
        let channel = self.provenance().channel();
        let exteriors =
            GuidedExteriors::new(stack, k.re, channel).map_err(|_| HybridFieldError::Ineligible)?;
        let last = stack.len() + 1;
        let evaluate = |k0_rate: f64,
                        beta_rate: Complex,
                        thickness: &[f64],
                        tensors: &[[[[Complex; 3]; 3]; 2]]|
         -> Result<Complex, HybridFieldError> {
            let rates = exteriors
                .rates(beta, k0_rate, beta_rate, [tensors[0], tensors[last]])
                .ok_or(HybridFieldError::Exterior {
                    side: StackSide::Left,
                })?;
            let coordinates =
                CanonicalCoordinates::new(seed(k, k0_rate.into()), seed(beta, beta_rate));
            let exterior = exterior_jets(signature, rates);
            let constitutive: Vec<_> = tensors
                .iter()
                .map(|pair| pair.map(ConstitutiveTensor::Full))
                .collect();
            backend
                .directional_characteristic(ModalCharacteristicDirection {
                    stack,
                    coordinates: &coordinates,
                    exterior: &exterior,
                    polarisation: channel,
                    thickness_rates: thickness,
                    constitutive_rates: &constitutive,
                })
                .map(|jet| jet.first()[()])
                .map_err(HybridFieldError::Fields)
        };
        let still = HybridPerturbation::none(stack);
        let along_beta = evaluate(
            0.0,
            Complex::ONE,
            &still.thickness_rates,
            &still.tensor_rates,
        )?;
        let along = evaluate(
            direction.k0_rate,
            Complex::ZERO,
            &direction.thickness_rates,
            &direction.tensor_rates,
        )?;
        let beta_rate = -along / along_beta;
        if !beta_rate.re.is_finite() || !beta_rate.im.is_finite() {
            return Err(HybridFieldError::IllConditioned);
        }
        Ok(HybridBetaRate {
            beta_rate,
            characteristic_direction: along,
            characteristic_beta: along_beta,
        })
    }

    /// The group index `dβ/dk0` from the characteristic, and the energy
    /// certificate `½ ∫ (E†ε_d E + η†μ_d η) dz / ∫ Re(E_y η_z* − E_z η_y*) dz`
    /// with the Hermitian dispersive weights `ε_d = ε + k0 ∂ε/∂k0`.
    pub fn hybrid_group_index_with<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
    ) -> Result<HybridGroupIndex, HybridFieldError> {
        let signature = self.signature().map_err(|_| HybridFieldError::Branch)?;
        let provenance = self.provenance();
        let stack = provenance.stack();
        let k = signature.k0.re;
        let direction =
            HybridPerturbation::wavenumber(stack, k).map_err(|_| HybridFieldError::Ineligible)?;
        let derivative = self.hybrid_beta_rate_with(backend, &direction)?.beta_rate;
        // Hermitian dispersive weights per region.
        let weights: Vec<[Tensor; 2]> = m::regions(stack)
            .map(|(id, material)| {
                let [values, rates] = m::full_tensors(id, material, Complex::new(k, 0.0))
                    .map_err(|_| HybridFieldError::Ineligible)?;
                Ok(std::array::from_fn(|f| {
                    let d: Tensor = std::array::from_fn(|i| {
                        std::array::from_fn(|j| values[f][i][j] + k * rates[f][i][j])
                    });
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| 0.5 * (d[i][j] + d[j][i].conj()))
                    })
                }))
            })
            .collect::<Result<_, HybridFieldError>>()?;
        let one = Complex::ONE;
        let energy: Vec<Weight> = weights.iter().map(forms::blocks).collect();
        // Energy and flux from one solve.
        let totals = integrate(
            backend,
            [self, self],
            [one, one],
            &[(true, &energy), (true, &[forms::flux()])],
        )?;
        let certificate = 0.5 * totals[0].re / totals[1].re;
        Ok(HybridGroupIndex {
            value: derivative.re,
            energy: certificate,
            relative_discrepancy: (derivative.re - certificate).abs() / certificate.abs(),
        })
    }
}

/// The group index of a guided mode and its energy certificate.
#[derive(Clone, Copy, Debug)]
pub struct HybridGroupIndex {
    /// `dβ/dk0` from the characteristic.
    pub value: f64,
    /// `c W / P` from the Hermitian dispersive energy and the power.
    pub energy: f64,
    pub relative_discrepancy: f64,
}

/// Earlier and later modes of a near-degenerate pair and the 2×2 block of
/// their power overlaps.
pub type TrackedSubspace = ([usize; 2], [usize; 2], [[Complex; 2]; 2]);

/// How the modes of one basis continue into the next along a parameter
/// path: single modes by their largest power overlap, and near-degenerate
/// pairs as two-dimensional subspaces, never by order within the pair.
#[derive(Clone, Debug)]
pub struct HybridTracking {
    /// For each earlier mode outside a subspace, its continuation.
    pub matches: Vec<(usize, usize)>,
    /// Pairs of earlier and later modes followed as subspaces, with the
    /// 2×2 block of power overlaps between them.
    pub subspaces: Vec<TrackedSubspace>,
    /// The power overlaps, earlier modes by rows.
    pub overlaps: Vec<Vec<Complex>>,
}

/// Follow `earlier` into `later` (unit-power profiles of the same layer
/// structure and exteriors): modes of either basis within `gap` (cm⁻¹) of
/// each other form near-degenerate pairs, followed as subspaces.
pub fn hybrid_tracking_with<B: ModalDerivativeBackend>(
    backend: &B,
    earlier: &[HybridProfile],
    later: &[HybridProfile],
    gap: f64,
) -> Result<HybridTracking, HybridFieldError> {
    let overlaps = earlier
        .iter()
        .map(|a| {
            later
                .iter()
                .map(|b| hybrid_power_overlap_with(backend, [a, b]))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let beta = |p: &HybridProfile| p.point.physical_coordinate().re;
    let pairs = |modes: &[HybridProfile]| -> Vec<[usize; 2]> {
        (1..modes.len())
            .filter(|&i| (beta(&modes[i - 1]) - beta(&modes[i])).abs() <= gap)
            .map(|i| [i - 1, i])
            .collect()
    };
    let mut subspaces = vec![];
    let (mut used_earlier, mut used_later) = (vec![false; earlier.len()], vec![false; later.len()]);
    let block = |e: [usize; 2], l: [usize; 2]| {
        std::array::from_fn(|i| std::array::from_fn(|j| overlaps[e[i]][l[j]]))
    };
    let weight = |b: &[[Complex; 2]; 2]| b.iter().flatten().map(|v| v.norm_sqr()).sum::<f64>();
    // A pair on either side is followed by the pair on the other side that
    // carries the most of its power.
    for (from_later, pair) in pairs(earlier)
        .into_iter()
        .map(|p| (false, p))
        .chain(pairs(later).into_iter().map(|p| (true, p)))
    {
        let other = if from_later {
            earlier.len()
        } else {
            later.len()
        };
        let best = (1..other).map(|j| [j - 1, j]).max_by(|x, y| {
            let (bx, by) = if from_later {
                (block(*x, pair), block(*y, pair))
            } else {
                (block(pair, *x), block(pair, *y))
            };
            weight(&bx).total_cmp(&weight(&by))
        });
        let Some(best) = best else { continue };
        let (e, l) = if from_later {
            (best, pair)
        } else {
            (pair, best)
        };
        if e.iter().any(|&i| used_earlier[i]) || l.iter().any(|&j| used_later[j]) {
            continue;
        }
        e.iter().for_each(|&i| used_earlier[i] = true);
        l.iter().for_each(|&j| used_later[j] = true);
        subspaces.push((e, l, block(e, l)));
    }
    let mut matches = vec![];
    for (i, row) in overlaps.iter().enumerate() {
        if used_earlier[i] {
            continue;
        }
        if let Some(j) = (0..later.len())
            .filter(|&j| !used_later[j])
            .max_by(|&x, &y| row[x].norm().total_cmp(&row[y].norm()))
        {
            used_later[j] = true;
            matches.push((i, j));
        }
    }
    Ok(HybridTracking {
        matches,
        subspaces,
        overlaps,
    })
}
