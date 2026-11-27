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
    /// The direction couples TE and TM in `region` (an `xy`, `yx`, `yz` or
    /// `zy` tensor rate), which a mode of a TE or TM channel cannot follow:
    /// count it on the coupled channel.
    #[error(
        "the direction couples TE and TM in region {region}, which a TE or TM mode cannot follow"
    )]
    CouplesPolarisations { region: usize },
    /// The direction moves the stack frame's origin, and with it the
    /// stack, against its walls.
    #[error("the direction moves the stack against its walls")]
    MovesAgainstWalls,
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

impl HybridPhaseReference {
    /// The rule's reference at the left interface `left`.
    fn of(left: &[Complex; 6]) -> Result<Self, HybridFieldError> {
        let tangential = [left[0], left[1], left[3], left[4]]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let rule = match (left[1] + left[4]).norm() >= 1e-6 * tangential {
            true => Self::EyPlusZ0Hy,
            false => Self::ExPlusZ0Hx,
        };
        let reference = rule.value(left).norm();
        if reference == 0.0 || !reference.is_finite() {
            return Err(HybridFieldError::IllConditioned);
        }
        Ok(rule)
    }

    /// The reference of the six components `f` (or of their tangent).
    fn value(self, f: &[Complex; 6]) -> Complex {
        match self {
            Self::EyPlusZ0Hy => f[1] + f[4],
            Self::ExPlusZ0Hx => f[0] + f[3],
        }
    }
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
        let phase_reference = HybridPhaseReference::of(&left)?;
        let reference = phase_reference.value(&left);
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
    /// The lab-frame velocity of the first interface, the origin of the
    /// stack frame, in cm per unit: it moves fixed-lab samples relative to
    /// the stack and changes nothing else. A walled stack refuses it.
    pub origin_velocity: f64,
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
            origin_velocity: 0.0,
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

/// The jets of a direction at an accepted point: k0, β and the exterior
/// wavevectors with their rates, the layers' thickness rates and every
/// region's full-tensor rates.
struct Moving {
    provenance: Provenance,
    coordinates: CanonicalCoordinates<crate::problem::J>,
    exterior: lamina_core::ExteriorWavevectors<crate::problem::J>,
    thickness: Vec<f64>,
    rates: Vec<[ConstitutiveTensor<Complex>; 2]>,
}

impl Moving {
    fn direction(
        &self,
    ) -> ModalCharacteristicDirection<
        '_,
        Region,
        [ConstitutiveTensor<Complex>; 2],
        lamina_core::Channel,
    > {
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

impl ModePoint {
    /// The direction's jets with β moving at `beta_rate`, the exterior
    /// pairs continued along it.
    fn moving(
        &self,
        k0_rate: f64,
        beta_rate: Complex,
        thickness: &[f64],
        tensors: &[[[[Complex; 3]; 3]; 2]],
    ) -> Result<Moving, HybridFieldError> {
        let signature = self.signature().map_err(|_| HybridFieldError::Branch)?;
        let provenance = self.provenance();
        let stack = provenance.stack();
        let (k, beta) = (signature.k0, signature.k_parallel);
        if thickness.len() != stack.len() || tensors.len() != stack.len() + 2 || k.im != 0.0 {
            return Err(HybridFieldError::Ineligible);
        }
        if provenance.channel() != lamina_core::Channel::Coupled
            && let Some(region) = tensors.iter().position(|pair| {
                pair.iter()
                    .any(|t| [t[0][1], t[1][0], t[1][2], t[2][1]] != [Complex::ZERO; 4])
            })
        {
            return Err(HybridFieldError::CouplesPolarisations { region });
        }
        let exteriors = GuidedExteriors::new(stack, k.re, provenance.channel())
            .map_err(|_| HybridFieldError::Ineligible)?;
        let last = stack.len() + 1;
        let rates = exteriors
            .rates(beta, k0_rate, beta_rate, [tensors[0], tensors[last]])
            .ok_or(HybridFieldError::Exterior {
                side: StackSide::Left,
            })?;
        Ok(Moving {
            coordinates: CanonicalCoordinates::new(seed(k, k0_rate.into()), seed(beta, beta_rate)),
            exterior: exterior_jets(signature, rates),
            thickness: thickness.to_vec(),
            rates: tensors
                .iter()
                .map(|pair| pair.map(ConstitutiveTensor::Full))
                .collect(),
            provenance,
        })
    }

    /// `dβ/dp = −(∂F/∂p)/(∂F/∂β)` through the backend's characteristic,
    /// with the exterior pairs continued along the direction.
    pub fn hybrid_beta_rate_with<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
        direction: &HybridPerturbation,
    ) -> Result<HybridBetaRate, HybridFieldError> {
        let stack = self.provenance().stack().clone();
        if !stack.is_open() && direction.origin_velocity != 0.0 {
            return Err(HybridFieldError::MovesAgainstWalls);
        }
        let evaluate = |moving: Moving| -> Result<Complex, HybridFieldError> {
            backend
                .directional_characteristic(moving.direction())
                .map(|jet| jet.first()[()])
                .map_err(HybridFieldError::Fields)
        };
        let still = HybridPerturbation::none(&stack);
        let along_beta = evaluate(self.moving(
            0.0,
            Complex::ONE,
            &still.thickness_rates,
            &still.tensor_rates,
        )?)?;
        let along = evaluate(self.moving(
            direction.k0_rate,
            Complex::ZERO,
            &direction.thickness_rates,
            &direction.tensor_rates,
        )?)?;
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

/// Unit-power fields of a guided mode of any channel and their rates along
/// a [`HybridPerturbation`], with the regional power fractions and their
/// rates: the coupled-channel counterpart of
/// [`GuidedBetaSensitivity::power_field_tangent_with`].
#[derive(Clone, Debug)]
pub struct HybridPowerSensitivity {
    pub rate: HybridBetaRate,
    pub phase_reference: HybridPhaseReference,
    /// The rate of the logarithm of the unit-power multiplier of the
    /// backend's seed: `−½ P′/P` (with the moving-boundary parts of `P′`)
    /// plus `i` times the rate of the phase rule's phase.
    pub scale_log_rate: Complex,
    /// Fields in the unit-power, phase-rule gauge of
    /// [`ModePoint::hybrid_power_profile_with`] and their rates in the frame.
    pub samples: Vec<GuidedFieldSample>,
    /// Left exterior, finite layers, then right exterior: each region's
    /// power over `|P|` and its rate; they sum to the sign of the power.
    pub regional_power: Vec<RegionalPowerSensitivity>,
}

/// `∂f/∂z` of the six components `f` in a region with stack-frame `[ε, μ]`,
/// from Maxwell's equations at `(k0, β)`: `∇×E = i k0 μ η`,
/// `∇×η = −i k0 ε E`, `∂x = iβ`, `∂y = 0`.
fn normal_gradient(
    f: &[Complex; 6],
    [e, m]: &[[[Complex; 3]; 3]; 2],
    k: Complex,
    beta: Complex,
) -> [Complex; 6] {
    let i = Complex::I;
    let ratio = beta / k;
    let product = |t: &[[Complex; 3]; 3], v: [Complex; 3], row: usize| {
        (0..3).map(|c| t[row][c] * v[c]).sum::<Complex>()
    };
    let (electric, magnetic) = ([f[0], f[1], f[2]], [f[3], f[4], f[5]]);
    let dex = i * beta * f[2] + i * k * product(m, magnetic, 1);
    let dey = -i * k * product(m, magnetic, 0);
    let dhx = i * beta * f[5] - i * k * product(e, electric, 1);
    let dhy = i * k * product(e, electric, 0);
    // The normal components from (εE)_z = −(β/k0) η_y, (μη)_z = (β/k0) E_y.
    let dez = (-ratio * dhy - e[2][0] * dex - e[2][1] * dey) / e[2][2];
    let dhz = (ratio * dey - m[2][0] * dhx - m[2][1] * dhy) / m[2][2];
    [dex, dey, dez, dhx, dhy, dhz]
}

impl ModePoint {
    /// Fields at unit longitudinal power under the phase rule of
    /// [`Self::hybrid_power_profile_with`], their rates along `direction`
    /// in `frame`, and the regional power fractions and their rates.
    ///
    /// The rates follow from the backend's field and form jets along the
    /// direction (β moving at its rate, exterior pairs continued): the rate
    /// of the unit-power multiplier includes the moving ends of layers whose
    /// thickness changes, and the phase rule's rate is that of its reference
    /// at the left interface, the stack frame's origin. A
    /// fixed-lab sample on an interface that moves between regions that
    /// differ has no rate (`MovingInterface`).
    pub fn hybrid_power_field_tangent_with<B: ModalDerivativeBackend>(
        &self,
        backend: &B,
        direction: &HybridPerturbation,
        sampling: &FieldSampling<f64>,
        frame: GuidedObservationFrame,
    ) -> Result<HybridPowerSensitivity, HybridFieldError> {
        let signature = self.signature().map_err(|_| HybridFieldError::Branch)?;
        let (k, beta) = (signature.k0, signature.k_parallel);
        if k.re <= 0.0 || k.im != 0.0 || beta.im.abs() > 1e-9 * beta.norm() {
            return Err(HybridFieldError::Ineligible);
        }
        let provenance = self.provenance();
        let stack = provenance.stack();
        crate::problem::walls(stack).map_err(|side| PmlIntegral { side })?;
        let rate = self.hybrid_beta_rate_with(backend, direction)?;
        let moving = self.moving(
            direction.k0_rate,
            rate.beta_rate,
            &direction.thickness_rates,
            &direction.tensor_rates,
        )?;
        // Regional power and its total rate, the layers' far ends moving.
        let operand = lamina_core::ModalFormOperand {
            direction: moving.direction(),
            amplitude: Complex::ONE,
        };
        let integrals = backend
            .directional_forms(
                [operand, operand],
                &[lamina_core::ModalForm {
                    conjugate: true,
                    weights: &[forms::flux()],
                }],
            )
            .map_err(HybridFieldError::Fields)?;
        let half = 0.5 * CM_TO_M_OVER_Z0;
        let regional: Vec<[f64; 2]> = integrals[0]
            .iter()
            .map(|share| [half * share.value()[()].re, half * share.first()[()].re])
            .collect();
        let total: f64 = regional.iter().map(|p| p[0]).sum();
        let total_rate: f64 = regional.iter().map(|p| p[1]).sum();
        if !total.is_finite() || !total_rate.is_finite() || total == 0.0 {
            return Err(HybridFieldError::IllConditioned);
        }
        let jets = |sampling: &FieldSampling<f64>| {
            backend
                .directional_fields(moving.direction(), sampling, Complex::ONE)
                .map_err(HybridFieldError::Fields)
        };
        // The phase rule's reference and its rate at the left interface.
        let at = FieldSampling::new().left_exterior(lamina_core::ExteriorSampling::point(
            lamina_units::Length::zero(),
        ));
        let left = jets(&at)?;
        let (value, tangent) = (components(&left.fields)[0], components(&left.tangent)[0]);
        let phase_reference = HybridPhaseReference::of(&value)?;
        let (reference, reference_rate) = (
            phase_reference.value(&value),
            phase_reference.value(&tangent),
        );
        let scale = reference.conj() / reference.norm() / total.abs().sqrt();
        let log_rate = Complex::new(-0.5 * total_rate / total, -(reference_rate / reference).im);
        if [scale, log_rate]
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(HybridFieldError::IllConditioned);
        }
        // Region tensors, and the interfaces' lab-frame velocities.
        let tensors: Vec<[[[Complex; 3]; 3]; 2]> = m::regions(stack)
            .map(|(id, material)| {
                m::full_tensors(id, material, k)
                    .map(|[values, _]| values)
                    .map_err(|_| HybridFieldError::Ineligible)
            })
            .collect::<Result<_, _>>()?;
        let origin = direction.origin_velocity;
        let velocities: Vec<f64> = std::iter::once(origin)
            .chain(direction.thickness_rates.iter().scan(origin, |at, rate| {
                *at += rate;
                Some(*at)
            }))
            .collect();
        let recovered = jets(sampling)?;
        let (fields, tangents) = (
            components(&recovered.fields),
            components(&recovered.tangent),
        );
        let fixed_lab = frame == GuidedObservationFrame::FixedLab;
        let samples = recovered
            .fields
            .sampling()
            .positions()
            .iter()
            .zip(fields.into_iter().zip(tangents))
            .map(|(&position, (field, tangent))| {
                let observation = match fixed_lab {
                    true => GuidedObservation::FixedLab(position),
                    false => GuidedObservation::MaterialAttached(position),
                };
                let tangent = match fixed_lab {
                    true => lab_tangent(
                        &position,
                        &field,
                        tangent,
                        stack,
                        &velocities,
                        &tensors,
                        &direction.tensor_rates,
                        k,
                        beta,
                    ),
                    false => Ok(tangent),
                }
                .and_then(|tangent| {
                    let scaled = tangent
                        .iter()
                        .zip(field)
                        .map(|(t, f)| scale * (t + log_rate * f));
                    let scaled: Vec<Complex> = scaled.collect();
                    match scaled.iter().all(|v| v.re.is_finite() && v.im.is_finite()) {
                        true => Ok(std::array::from_fn(|c| scaled[c])),
                        false => Err(GuidedObservationError::NonFinite),
                    }
                });
                GuidedFieldSample {
                    observation,
                    field: field.map(|f| scale * f),
                    tangent,
                }
            })
            .collect();
        let regional_power = regional
            .into_iter()
            .map(|[power, rate]| RegionalPowerSensitivity {
                power_w_per_m: power / total.abs(),
                power_rate_w_per_m: (rate - power * total_rate / total) / total.abs(),
            })
            .collect();
        Ok(HybridPowerSensitivity {
            rate,
            phase_reference,
            scale_log_rate: log_rate,
            samples,
            regional_power,
        })
    }
}

/// The fixed-lab rate of a material-attached sample: its attached rate less
/// its velocity times `∂f/∂z`. A sample on an interface that moves between
/// regions whose tensors or tensor rates differ has none.
#[allow(clippy::too_many_arguments)]
fn lab_tangent(
    position: &lamina_core::FieldPosition<f64>,
    field: &[Complex; 6],
    mut tangent: [Complex; 6],
    stack: &ModeStack,
    velocities: &[f64],
    tensors: &[[[[Complex; 3]; 3]; 2]],
    rates: &[[[[Complex; 3]; 3]; 2]],
    k: Complex,
    beta: Complex,
) -> Result<[Complex; 6], GuidedObservationError> {
    use lamina_core::{FieldPosition, ResolvedLayerPosition};
    let last = velocities.len() - 1;
    let (region, velocity, interface) = match position {
        FieldPosition::LeftExterior { distance } => (
            0,
            velocities[0],
            (distance.as_centimetres() == 0.0).then_some(0),
        ),
        FieldPosition::RightExterior { distance } => (
            last + 1,
            velocities[last],
            (distance.as_centimetres() == 0.0).then_some(last),
        ),
        FieldPosition::Layer { index, position } => {
            let i = index.get();
            let width = stack.layers_left_to_right()[i].thickness().as_centimetres();
            let (velocity, on_left, on_right) = match position {
                ResolvedLayerPosition::FromLeft(d) => {
                    let d = d.as_centimetres();
                    (velocities[i], d == 0.0, d == width)
                }
                ResolvedLayerPosition::FromRight(d) => {
                    let d = d.as_centimetres();
                    (velocities[i + 1], d == width, d == 0.0)
                }
                ResolvedLayerPosition::Fraction(f) => (
                    (1.0 - f) * velocities[i] + f * velocities[i + 1],
                    *f == 0.0,
                    *f == 1.0,
                ),
            };
            let interface = match (on_left, on_right) {
                (true, _) => Some(i),
                (_, true) => Some(i + 1),
                _ => None,
            };
            (i + 1, velocity, interface)
        }
    };
    if let Some(interface) = interface
        && velocities[interface] != 0.0
        && (tensors[interface] != tensors[interface + 1]
            || rates[interface] != rates[interface + 1])
    {
        return Err(GuidedObservationError::MovingInterface { interface });
    }
    if velocity != 0.0 {
        let gradient = normal_gradient(field, &tensors[region], k, beta);
        for (value, gradient) in tangent.iter_mut().zip(gradient) {
            *value -= velocity * gradient;
        }
    }
    Ok(tangent)
}
