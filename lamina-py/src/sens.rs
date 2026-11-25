//! Guided-mode sensitivities and the certified group index, in SI.
use crate::records::{self as rec};
use crate::solve::{Failure, M_PER_CM, Solved, Z0_OHM, c2, k0_cm, point, sampling};
use lamina_core::Scatter2;
use lamina_mode::{
    Complex, ConstitutiveRate, GuidedObservationFrame, GuidedPerturbation, ImplicitSlopeSettings,
    LocalDerivativeSettings,
};

/// A perturbation in lamina-mode's units, and the factors that take its
/// rates to SI per parameter unit.
pub(crate) struct Direction {
    pub(crate) perturbation: GuidedPerturbation,
    pub(crate) unit: &'static str,
    /// Multiplies a rate per internal parameter unit to a rate per SI unit.
    pub(crate) per_unit: f64,
}

pub(crate) fn direction(solved: &Solved, parameter: rec::Parameter) -> Result<Direction, Failure> {
    use rec::Parameter::*;
    let input = &solved.record.input;
    let n = input.stack.layers.len();
    let invalid = |what: String| Failure::Invalid(what);
    let mut velocities = vec![0.; n + 1];
    let mut rates = vec![ConstitutiveRate::default(); n + 2];
    let mut wavelength_rate = 0.;
    let region_ok = |r: usize| {
        if r <= n + 1 {
            Ok(r)
        } else {
            Err(invalid(format!("region {r} is outside 0..={}", n + 1)))
        }
    };
    // Geometric rates are per cm internally: to per metre, divide by M_PER_CM.
    let (name, unit, per_unit) = match parameter {
        Thickness { layer } => {
            if layer >= n {
                return Err(invalid(format!("layer {layer} is outside 0..{n}")));
            }
            velocities[layer + 1..].fill(1.);
            ("layer thickness", "m", 1. / M_PER_CM)
        }
        Interface { interface } => {
            if interface > n {
                return Err(invalid(format!("interface {interface} is outside 0..={n}")));
            }
            velocities[interface] = 1.;
            ("interface position", "m", 1. / M_PER_CM)
        }
        Translation => {
            velocities.fill(1.);
            ("rigid translation", "m", 1. / M_PER_CM)
        }
        Wavelength => {
            wavelength_rate = 1.;
            ("vacuum wavelength", "m", 1. / M_PER_CM)
        }
        Permittivity { region } => {
            rates[region_ok(region)?].relative_permittivity = 1.;
            ("relative permittivity", "1", 1.)
        }
        Permeability { region } => {
            rates[region_ok(region)?].relative_permeability = 1.;
            ("relative permeability", "1", 1.)
        }
        SellmeierConstant { region }
        | SellmeierCoefficient { region, .. }
        | SellmeierPoleWavelength { region, .. } => {
            let material = crate::solve::regions(&input.stack)
                .nth(region_ok(region)?)
                .expect("region checked");
            let rec::Material::Sellmeier(m) = material else {
                return Err(invalid(format!(
                    "region {region} is not a Sellmeier material"
                )));
            };
            let model = crate::solve::sellmeier_model(&m.sellmeier)?;
            let model_rates = model.permittivity_parameter_rates(k0_cm(&input.request));
            let term_ok = |term: usize| {
                if term < m.sellmeier.terms.len() {
                    Ok(term)
                } else {
                    Err(invalid(format!(
                        "region {region} has no Sellmeier term {term}"
                    )))
                }
            };
            let (rate, name, unit) = match parameter {
                SellmeierConstant { .. } => (model_rates.constant, "Sellmeier constant", "1"),
                SellmeierCoefficient { term, .. } => (
                    model_rates.coefficients[term_ok(term)?],
                    "Sellmeier coefficient",
                    "1",
                ),
                SellmeierPoleWavelength { term, .. } => {
                    // dε/dλp = dε/dq · dq/dλp, with q = 2π / λp in cm^-1 and λp in m.
                    let term = term_ok(term)?;
                    let pole_m = m.sellmeier.terms[term].pole_wavelength_m;
                    let q_cm = std::f64::consts::TAU / (pole_m / M_PER_CM);
                    (
                        model_rates.pole_wavenumbers[term] * (-q_cm / pole_m),
                        "Sellmeier pole wavelength",
                        "m",
                    )
                }
                _ => unreachable!(),
            };
            rates[region].relative_permittivity = rate;
            (name, unit, 1.)
        }
    };
    // Walls stay fixed in the stack frame, whose origin is the first interface.
    let walled =
        !(input.stack.left_termination.is_open() && input.stack.right_termination.is_open());
    if walled && velocities[0] != 0. {
        return Err(invalid(format!(
            "{name} moves the first interface, and the stack, against its walls"
        )));
    }
    let internal_unit = if per_unit == 1. { "1" } else { "cm" };
    let perturbation =
        GuidedPerturbation::new(name, internal_unit, velocities, wavelength_rate, rates)
            .map_err(|e| invalid(format!("parameter direction: {e:?}")))?;
    Ok(Direction {
        perturbation,
        unit,
        per_unit,
    })
}

fn conditioning(s: &lamina_mode::GuidedBetaSensitivity) -> rec::Conditioning {
    rec::Conditioning {
        characteristic: c2(s.characteristic),
        characteristic_direction: c2(s.characteristic_direction),
        characteristic_beta: c2(s.characteristic_beta),
        denominator_threshold: s.denominator_threshold,
    }
}

pub fn sensitivity(
    solved: &Solved,
    mode_index: usize,
    parameter: rec::Parameter,
    z_m: &[f64],
    frame: rec::Frame,
) -> Result<rec::GuidedSensitivity, Failure> {
    let point = point(solved, mode_index)?;
    let direction = direction(solved, parameter)?;
    let backend = Scatter2::new();
    let eigenvalue = point
        .guided_beta_sensitivity_with(
            &backend,
            &direction.perturbation,
            LocalDerivativeSettings::default(),
            ImplicitSlopeSettings::default(),
        )
        .map_err(|e| Failure::Derivatives(format!("eigenvalue sensitivity: {e:?}")))?;
    // With no requested samples, sample one point for the power bookkeeping and drop it.
    let probe = [0.];
    let requested = if z_m.is_empty() { &probe[..] } else { z_m };
    let frame_internal = match frame {
        rec::Frame::MaterialAttached => GuidedObservationFrame::MaterialAttached,
        rec::Frame::FixedLab => GuidedObservationFrame::FixedLab,
    };
    let power = eigenvalue
        .power_field_tangent_with(
            &backend,
            &sampling(&solved.record.input.stack, requested)?,
            frame_internal,
            Complex::ONE,
        )
        .map_err(|e| Failure::Derivatives(format!("field sensitivity: {e:?}")))?;
    if power.samples.len() != requested.len() {
        return Err(Failure::Derivatives(
            "sample count changed during recovery".into(),
        ));
    }
    let samples = if z_m.is_empty() {
        &power.samples[..0]
    } else {
        &power.samples[..]
    };
    // Components [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz]: H to A/m; rates to per SI unit.
    let split = |values: &[Complex; 6], scale: f64| {
        let e = |i: usize| values[i] * scale;
        let h = |i: usize| values[i + 3] * (scale / Z0_OHM);
        (
            [e(0).re, e(1).re, e(2).re],
            [e(0).im, e(1).im, e(2).im],
            [h(0).re, h(1).re, h(2).re],
            [h(0).im, h(1).im, h(2).im],
        )
    };
    let mut record = rec::GuidedSensitivity {
        schema: rec::SENSITIVITY_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        parameter,
        parameter_unit: direction.unit.to_owned(),
        frame,
        beta_rad_per_m: solved.record.modes[mode_index].beta_rad_per_m,
        dbeta_rad_per_m: c2(eigenvalue.beta_rate * (direction.per_unit / M_PER_CM)),
        conditioning: conditioning(&eigenvalue),
        z_m: z_m.iter().map(|z| z + 0.).collect(),
        electric_real: vec![],
        electric_imag: vec![],
        magnetic_real: vec![],
        magnetic_imag: vec![],
        d_electric_real: vec![],
        d_electric_imag: vec![],
        d_magnetic_real: vec![],
        d_magnetic_imag: vec![],
        sample_failures: vec![],
        power_fractions: vec![],
        power_fraction_rates: vec![],
    };
    for sample in samples {
        let (er, ei, hr, hi) = split(&sample.field, 1.);
        record.electric_real.push(er);
        record.electric_imag.push(ei);
        record.magnetic_real.push(hr);
        record.magnetic_imag.push(hi);
        let (tangent, failure) = match &sample.tangent {
            Ok(t) => (*t, None),
            Err(e) => ([Complex::ZERO; 6], Some(format!("{e:?}"))),
        };
        let (er, ei, hr, hi) = split(&tangent, direction.per_unit);
        record.d_electric_real.push(er);
        record.d_electric_imag.push(ei);
        record.d_magnetic_real.push(hr);
        record.d_magnetic_imag.push(hi);
        record.sample_failures.push(failure);
    }
    // Regional powers are normalised by |P|; their sum is the sign of P.
    let sign: f64 = power
        .regional_power
        .iter()
        .map(|r| r.power_w_per_m)
        .sum::<f64>()
        .signum();
    record.power_fractions = power
        .regional_power
        .iter()
        .map(|r| r.power_w_per_m * sign)
        .collect();
    record.power_fraction_rates = power
        .regional_power
        .iter()
        .map(|r| r.power_rate_w_per_m * sign * direction.per_unit)
        .collect();
    Ok(record)
}

pub fn group_index(
    solved: &Solved,
    mode_index: usize,
    tolerance: f64,
) -> Result<rec::GroupIndex, Failure> {
    let point = point(solved, mode_index)?;
    let group = point
        .guided_group_index_with(
            &Scatter2::new(),
            crate::solve::recovery_settings(&solved.record.input.request),
            tolerance,
        )
        .map_err(|e| match e {
            lamina_mode::GroupIndexError::InvalidTolerance => Failure::Invalid(e.to_string()),
            other => Failure::Derivatives(other.to_string()),
        })?;
    Ok(rec::GroupIndex {
        schema: rec::GROUP_INDEX_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        value: group.value,
        derivative: c2(group.derivative),
        energy_estimate: group.energy_estimate,
        relative_discrepancy: group.relative_discrepancy,
        tolerance: group.tolerance,
        conditioning: conditioning(&group.sensitivity),
    })
}
