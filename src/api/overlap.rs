//! Registered electromagnetic pairing of guided planar modes.

use super::*;
use lamina_core::{
    ExteriorSampling, FieldSampling, LayerSampling, ModalFieldBackend, OwnedModeFields,
};
use lamina_units::Length;

/// Positions of the two leftmost interfaces in a common transverse frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuidedOverlapRegistration {
    pub origins_cm: [f64; 2],
}

/// Hermitian longitudinal power pairing, linear in the first mode.
#[derive(Clone, Debug)]
pub struct GuidedOverlap {
    pub registration: GuidedOverlapRegistration,
    pub power_w_per_m: Complex,
    pub exterior_power_w_per_m: [Complex; 2],
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    /// Change between the last two composite quadrature estimates.
    pub quadrature_change_w_per_m: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum GuidedOverlapError {
    #[error("invalid transverse registration or unrepresentable interface spacing")]
    Registration,
    #[error("expected two accepted modes at a common real vacuum wavelength")]
    IncompatibleWavelength,
    #[error("guided field recovery failed for mode {mode}")]
    Field {
        mode: usize,
        #[source]
        source: GuidedPowerError,
    },
    #[error("overlap quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
    #[error("pairings need stacks open on both sides")]
    ClosedStack,
}

pub(super) fn interface_positions(
    point: &ModePoint,
    origin: f64,
) -> Result<Vec<f64>, GuidedOverlapError> {
    if !origin.is_finite() {
        return Err(GuidedOverlapError::Registration);
    }
    if !point.provenance().stack().is_open() {
        return Err(GuidedOverlapError::ClosedStack);
    }
    let mut positions = vec![origin];
    for layer in point.provenance().stack().iter() {
        let left = *positions.last().expect("initial interface");
        let right = left + layer.thickness().as_centimetres();
        if !right.is_finite() || right <= left {
            return Err(GuidedOverlapError::Registration);
        }
        positions.push(right);
    }
    Ok(positions)
}

fn sample_grid(
    interfaces: &[f64],
    widths: &[f64],
    partition: &[f64],
    panels: usize,
) -> FieldSampling<f64> {
    sample_grid_with_exterior(interfaces, widths, partition, panels, false)
}

pub(super) fn sample_grid_with_exterior(
    interfaces: &[f64],
    widths: &[f64],
    partition: &[f64],
    panels: usize,
    exterior_at_interface: bool,
) -> FieldSampling<f64> {
    let distance = |d| Length::centimetres(if exterior_at_interface { 0.0 } else { d });
    let last = interfaces.len() - 1;
    let mut sampling = FieldSampling::new().left_exterior(ExteriorSampling::point(distance(
        interfaces[0] - partition[0],
    )));
    for interval in partition.windows(2) {
        let left = interval[0];
        let right = interval[1];
        // The left boundary belongs to the region immediately to its right.
        let region = interfaces.partition_point(|&edge| edge <= left);
        let coordinates = (0..=panels).map(|i| {
            if i == panels {
                right
            } else {
                left + (right - left) * (i as f64 / panels as f64)
            }
        });
        sampling = if region == 0 {
            sampling.left_exterior(ExteriorSampling::distances(
                coordinates.map(|z| distance(interfaces[0] - z)),
            ))
        } else if region > last {
            sampling.right_exterior(ExteriorSampling::distances(
                coordinates.map(|z| distance(z - interfaces[last])),
            ))
        } else {
            let width = widths[region - 1];
            sampling.layer(
                region - 1,
                LayerSampling::offsets(coordinates.map(|z| {
                    let offset = if z == interfaces[region] {
                        width
                    } else {
                        (z - interfaces[region - 1]).clamp(0.0, width)
                    };
                    Length::centimetres(offset)
                })),
            )
        };
    }
    sampling.right_exterior(ExteriorSampling::point(distance(
        partition[partition.len() - 1] - interfaces[last],
    )))
}

fn integrand(a: &OwnedModeFields, b: &OwnedModeFields, i: usize, flux: FluxPairing) -> Complex {
    let components = |fields: &OwnedModeFields| {
        let e = fields.quantity().electric();
        let h = fields.quantity().magnetic();
        [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]]
    };
    let a = components(a);
    let b = components(b);
    match flux {
        FluxPairing::Power => pairing(a, b),
        FluxPairing::Reciprocity => {
            (a[1] * b[5] - a[2] * b[4] - b[1] * a[5] + b[2] * a[4]) * super::power::CM_TO_M_OVER_Z0
        }
    }
}

/// Evaluate the power pairing on the union of both structures' interfaces.
pub fn guided_power_overlap_with<B: ModalFieldBackend + Clone>(
    backend: &B,
    modes: [&ModePoint; 2],
    registration: GuidedOverlapRegistration,
    settings: FieldRecoverySettings,
) -> Result<GuidedOverlap, GuidedOverlapError> {
    integrate_guided_flux(backend, modes, registration, settings, FluxPairing::Power)
}

/// Bilinear longitudinal Lorentz flux of two normalized guided modes.
#[derive(Clone, Debug)]
pub struct GuidedReciprocity {
    pub registration: GuidedOverlapRegistration,
    /// Integral of (E₁ × H₂ − E₂ × H₁) along the common propagation axis.
    pub flux_w_per_m: Complex,
    pub exterior_flux_w_per_m: [Complex; 2],
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
}

/// Evaluate the unconjugated, antisymmetric flux using signed unit-power modes.
pub fn guided_reciprocity_with<B: ModalFieldBackend + Clone>(
    backend: &B,
    modes: [&ModePoint; 2],
    registration: GuidedOverlapRegistration,
    settings: FieldRecoverySettings,
) -> Result<GuidedReciprocity, GuidedOverlapError> {
    let result = integrate_guided_flux(
        backend,
        modes,
        registration,
        settings,
        FluxPairing::Reciprocity,
    )?;
    Ok(GuidedReciprocity {
        registration,
        flux_w_per_m: result.power_w_per_m,
        exterior_flux_w_per_m: result.exterior_power_w_per_m,
        partition_cm: result.partition_cm,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.quadrature_change_w_per_m,
    })
}

#[derive(Clone, Copy)]
enum FluxPairing {
    Power,
    Reciprocity,
}

fn integrate_guided_flux<B: ModalFieldBackend + Clone>(
    backend: &B,
    modes: [&ModePoint; 2],
    registration: GuidedOverlapRegistration,
    settings: FieldRecoverySettings,
    flux: FluxPairing,
) -> Result<GuidedOverlap, GuidedOverlapError> {
    use GuidedOverlapError::*;
    let [a, b] = modes;
    let ka = a.signature().map_err(|_| IncompatibleWavelength)?.k0;
    let kb = b.signature().map_err(|_| IncompatibleWavelength)?.k0;
    if [ka, kb]
        .iter()
        .any(|k| !k.re.is_finite() || !k.im.is_finite() || k.re <= 0.0 || k.im.abs() > 1e-9 * k.re)
        || (ka - kb).norm() > 1e-12 * ka.re.max(kb.re)
    {
        return Err(IncompatibleWavelength);
    }
    let interfaces = [
        interface_positions(a, registration.origins_cm[0])?,
        interface_positions(b, registration.origins_cm[1])?,
    ];
    let widths = modes.map(|mode| {
        mode.provenance()
            .stack()
            .iter()
            .map(|layer| layer.thickness().as_centimetres())
            .collect::<Vec<_>>()
    });
    let mut partition: Vec<_> = interfaces.iter().flatten().copied().collect();
    partition.sort_by(f64::total_cmp);
    partition.dedup();
    let roots = modes.map(|point| {
        let sig = point.signature().expect("accepted signature");
        sig.open_exteriors().expect("an open stack").map(|r| r.im)
    });
    let mut previous: Option<Complex> = None;
    for level in 0..11 {
        let panels = 8 << level;
        let intervals = partition.len() - 1;
        let sample_count = intervals
            .checked_mul(panels + 1)
            .and_then(|n| n.checked_add(2))
            .ok_or(Quadrature)?;
        if sample_count > 131072 {
            return Err(Quadrature);
        }
        let recover = |mode: usize| {
            modes[mode]
                .guided_power_profile_with(
                    backend.clone(),
                    &sample_grid(&interfaces[mode], &widths[mode], &partition, panels),
                    settings,
                )
                .map_err(|source| Field { mode, source })
        };
        let a = recover(0)?;
        let b = recover(1)?;
        let value = |i| integrand(&a.fields, &b.fields, i, flux);
        let tails = [
            value(0) / (roots[0][0] + roots[1][0]),
            value(sample_count - 1) / (roots[0][1] + roots[1][1]),
        ];
        let mut total = tails[0] + tails[1];
        for (i, edges) in partition.windows(2).enumerate() {
            let offset = 1 + i * (panels + 1);
            let mut sum = value(offset) + value(offset + panels);
            for j in 1..panels {
                sum += value(offset + j) * if j % 2 == 1 { 4.0 } else { 2.0 };
            }
            total += sum * ((edges[1] - edges[0]) / (3.0 * panels as f64));
        }
        if let Some(previous) = previous {
            let change = (total - previous).norm();
            if change <= 1e-9 * total.norm().max(1.0) {
                return Ok(GuidedOverlap {
                    registration,
                    power_w_per_m: total,
                    exterior_power_w_per_m: tails,
                    partition_cm: partition,
                    sample_count,
                    quadrature_change_w_per_m: change,
                });
            }
        }
        previous = Some(total);
    }
    Err(Quadrature)
}

/// Base pairing and the outcome of differentiating it along a shared parameter.
#[derive(Debug)]
pub struct GuidedOverlapSensitivity {
    pub base: GuidedOverlap,
    /// Power rate in W/m per unit of the shared parameter.
    pub tangent: Result<GuidedOverlapTangent, GuidedOverlapDerivativeError>,
}

#[derive(Clone, Copy, Debug)]
pub struct GuidedOverlapTangent {
    pub power_rate_w_per_m: Complex,
    pub exterior_rates_w_per_m: [Complex; 2],
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
}

#[derive(Debug)]
pub enum GuidedOverlapDerivativeError {
    IncompatibleDirection,
    SeparatingInterfaces {
        position_cm: f64,
        velocities_cm: [f64; 2],
    },
    Mode {
        mode: usize,
        source: GuidedBetaError,
    },
    Fields {
        mode: usize,
        source: GuidedPowerSensitivityError,
    },
    Observation(GuidedObservationError),
    Quadrature,
}

/// Differentiate the registered power pairing. Both directions use the same
/// parameter unit, and their first interface velocities move the origins.
pub fn guided_power_overlap_sensitivity_with<
    B: ModalFieldBackend + lamina_core::ModalDerivativeBackend,
>(
    backend: &B,
    modes: [&ModePoint; 2],
    directions: [&GuidedPerturbation; 2],
    registration: GuidedOverlapRegistration,
    settings: FieldRecoverySettings,
) -> Result<GuidedOverlapSensitivity, GuidedOverlapError> {
    let base = guided_power_overlap_with(backend, modes, registration, settings)?;
    let tangent = differentiate_overlap(backend, modes, directions, &base, settings.amplitude);
    Ok(GuidedOverlapSensitivity { base, tangent })
}

fn differentiate_overlap<B: lamina_core::ModalDerivativeBackend>(
    backend: &B,
    modes: [&ModePoint; 2],
    directions: [&GuidedPerturbation; 2],
    base: &GuidedOverlap,
    amplitude: Complex,
) -> Result<GuidedOverlapTangent, GuidedOverlapDerivativeError> {
    use GuidedOverlapDerivativeError::*;
    if directions[0].units() != directions[1].units()
        || directions[0].wavelength_rate() != directions[1].wavelength_rate()
        || (0..2)
            .any(|i| modes[i].provenance().stack().len() != directions[i].thickness_rates().len())
    {
        return Err(IncompatibleDirection);
    }
    let interfaces = [
        interface_positions(modes[0], base.registration.origins_cm[0]).map_err(|_| Quadrature)?,
        interface_positions(modes[1], base.registration.origins_cm[1]).map_err(|_| Quadrature)?,
    ];
    let partition = &base.partition_cm;
    let speeds = partition
        .iter()
        .map(|edge| {
            let speed = |mode: usize| {
                interfaces[mode]
                    .iter()
                    .position(|z| z == edge)
                    .map(|i| directions[mode].interface_velocities()[i])
            };
            match (speed(0), speed(1)) {
                (Some(a), Some(b)) if a != b => Err(SeparatingInterfaces {
                    position_cm: *edge,
                    velocities_cm: [a, b],
                }),
                (Some(v), _) | (_, Some(v)) => Ok(v),
                _ => unreachable!("partition formed from the two sets of interfaces"),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let recover = |mode: usize| {
        modes[mode]
            .guided_beta_sensitivity_with(
                backend,
                directions[mode],
                Default::default(),
                Default::default(),
            )
            .map_err(|source| Mode { mode, source })
    };
    let sensitivities = [recover(0)?, recover(1)?];
    let decay = modes.map(|point| {
        let sig = point.signature().expect("accepted guided point");
        sig.open_exteriors().expect("an open stack").map(|r| r.im)
    });
    let decay_rate = sensitivities
        .each_ref()
        .map(GuidedBetaSensitivity::exterior_decay_rates);
    let widths = modes.map(|mode| {
        mode.provenance()
            .stack()
            .iter()
            .map(|layer| layer.thickness().as_centimetres())
            .collect::<Vec<_>>()
    });
    let intervals = partition.len() - 1;
    let mut panels = (base.sample_count - 2) / intervals.max(1);
    panels = panels.saturating_sub(1).max(8);
    let mut previous: Option<(Complex, Complex)> = None;
    loop {
        let count = intervals
            .checked_mul(panels + 1)
            .and_then(|n| n.checked_add(2))
            .ok_or(Quadrature)?;
        if count > 131072 {
            return Err(Quadrature);
        }
        let mut mesh = Vec::with_capacity(count);
        mesh.push(speeds[0]);
        for pair in speeds.windows(2) {
            mesh.extend((0..=panels).map(|i| {
                if i == panels {
                    pair[1]
                } else {
                    pair[0] + (pair[1] - pair[0]) * i as f64 / panels as f64
                }
            }));
        }
        mesh.push(speeds[speeds.len() - 1]);
        let fields = |mode: usize| {
            sensitivities[mode]
                .power_field_tangent_on_mesh(
                    backend,
                    &sample_grid(&interfaces[mode], &widths[mode], partition, panels),
                    GuidedObservationFrame::MaterialAttached,
                    amplitude,
                    Some(&mesh),
                )
                .map_err(|source| Fields { mode, source })
        };
        let a = fields(0)?;
        let b = fields(1)?;
        let value = |i: usize| {
            let a = &a.samples[i];
            let b = &b.samples[i];
            let da = a.tangent.map_err(Observation)?;
            let db = b.tangent.map_err(Observation)?;
            Ok((
                pairing(a.field, b.field),
                pairing(da, b.field) + pairing(a.field, db),
            ))
        };
        let tail = |side: usize, sample: usize| {
            let (value, rate) = value(sample)?;
            let sum = decay[0][side] + decay[1][side];
            let sum_rate = decay_rate[0][side] + decay_rate[1][side];
            Ok((value / sum, (rate - value * sum_rate / sum) / sum))
        };
        let (left, left_rate) = tail(0, 0)?;
        let (right, right_rate) = tail(1, count - 1)?;
        let (mut integral, mut rate) = (left + right, left_rate + right_rate);
        for (i, edges) in partition.windows(2).enumerate() {
            let offset = 1 + i * (panels + 1);
            let mut sum = Complex::ZERO;
            let mut sum_rate = Complex::ZERO;
            for j in 0..=panels {
                let (value, rate) = value(offset + j)?;
                let weight = if j == 0 || j == panels {
                    1.0
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                sum += weight * value;
                sum_rate += weight * rate;
            }
            let width = edges[1] - edges[0];
            let width_rate = speeds[i + 1] - speeds[i];
            let scale = 1.0 / (3.0 * panels as f64);
            integral += scale * width * sum;
            rate += scale * (width * sum_rate + width_rate * sum);
        }
        if let Some((previous_integral, previous_rate)) = previous {
            let change = (rate - previous_rate).norm();
            if change <= 1e-9 * rate.norm().max(1.0)
                && (integral - previous_integral).norm() <= 1e-9 * integral.norm().max(1.0)
            {
                return Ok(GuidedOverlapTangent {
                    power_rate_w_per_m: rate,
                    exterior_rates_w_per_m: [left_rate, right_rate],
                    sample_count: count,
                    quadrature_change_w_per_m: change,
                });
            }
        }
        previous = Some((integral, rate));
        panels = panels.checked_mul(2).ok_or(Quadrature)?;
    }
}

pub(super) fn pairing(a: [Complex; 6], b: [Complex; 6]) -> Complex {
    (a[1] * b[5].conj() - a[2] * b[4].conj() + b[1].conj() * a[5] - b[2].conj() * a[4])
        * (0.25 * super::power::CM_TO_M_OVER_Z0)
}
