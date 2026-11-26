//! Guided-mode sensitivities and the certified group index, in SI.
use crate::records::{self as rec};
use crate::solve::{Failure, M_PER_CM, Solved, Z0_OHM, c2, k0_cm, point, sampling};
use lamina_core::{Scatter2, Scatter4};
use lamina_mode::{
    Complex, ConstitutiveRate, GuidedObservationFrame, GuidedPerturbation, HybridPerturbation,
    ImplicitSlopeSettings, LocalDerivativeSettings,
};

type Tensor = [[Complex; 3]; 3];

/// A parameter direction in lamina-mode's units: interface velocities (cm
/// per unit), a wavelength rate (cm per unit) and each region's stack-frame
/// `[ε', μ']`, with the factors that take rates to SI per parameter unit.
pub(crate) struct Direction {
    /// The direction as scalar rates, when every region's rate is real and
    /// diagonal in the stack frame.
    pub(crate) perturbation: Option<GuidedPerturbation>,
    pub(crate) velocities: Vec<f64>,
    pub(crate) wavelength_rate: f64,
    pub(crate) tensors: Vec<[Tensor; 2]>,
    pub(crate) name: &'static str,
    pub(crate) unit: &'static str,
    /// Multiplies a rate per internal parameter unit to a rate per SI unit.
    pub(crate) per_unit: f64,
}

impl Direction {
    /// The scalar perturbation, or why the direction has none.
    pub(crate) fn scalar(&self) -> Result<&GuidedPerturbation, Failure> {
        self.perturbation.as_ref().ok_or_else(|| {
            Failure::Invalid(format!(
                "{} has an off-diagonal or complex stack-frame rate; only its dβ is available",
                self.name
            ))
        })
    }
}

pub(crate) fn direction(solved: &Solved, parameter: rec::Parameter) -> Result<Direction, Failure> {
    let input = &solved.record.input;
    direction_for(&input.stack, k0_cm(&input.request), parameter)
}

fn unit_entry(i: usize, j: usize) -> Tensor {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            if (r, c) == (i, j) {
                Complex::ONE
            } else {
                Complex::ZERO
            }
        })
    })
}

fn identity(r: Complex) -> Tensor {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { r } else { Complex::ZERO }))
}

fn scaled(t: &Tensor, f: Complex) -> Tensor {
    t.map(|row| row.map(|v| v * f))
}

/// The scalar rate of a stack-frame rate that is real and diagonal.
fn scalar_rate(t: &[Tensor; 2]) -> Option<ConstitutiveRate> {
    let mut entries = [[0.; 3]; 2];
    for (f, tensor) in t.iter().enumerate() {
        for (i, row) in tensor.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                if v.im != 0. || (i != j && v.re != 0.) {
                    return None;
                }
            }
            entries[f][i] = row[i].re;
        }
    }
    Some(ConstitutiveRate::diagonal(entries[0], entries[1]))
}

/// A direction on `stack` at vacuum wavenumber `k0_cm` (cm^-1).
pub(crate) fn direction_for(
    stack: &rec::Stack,
    k0_cm: f64,
    parameter: rec::Parameter,
) -> Result<Direction, Failure> {
    use rec::Parameter::*;
    let n = stack.layers.len();
    let invalid = |what: String| Failure::Invalid(what);
    let zero = [[[Complex::ZERO; 3]; 3]; 2];
    let mut velocities = vec![0.; n + 1];
    let mut tensors = vec![zero; n + 2];
    // Rates of the isotropic parameters keep their scalar form.
    let mut isotropic: Option<(usize, ConstitutiveRate)> = None;
    let mut wavelength_rate = 0.;
    let region_ok = |r: usize| {
        if r <= n + 1 {
            Ok(r)
        } else {
            Err(invalid(format!("region {r} is outside 0..={}", n + 1)))
        }
    };
    let material = |r: usize| -> Result<&rec::Material, Failure> {
        crate::solve::regions(stack)
            .nth(region_ok(r)?)
            .ok_or_else(|| invalid(format!("region {r} does not exist")))
    };
    let wavelength_m = std::f64::consts::TAU / k0_cm * M_PER_CM;
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
            if crate::solve::tabulated(stack) {
                return Err(invalid(
                    "a tabulated material has no wavelength derivative".into(),
                ));
            }
            wavelength_rate = 1.;
            ("vacuum wavelength", "m", 1. / M_PER_CM)
        }
        Permittivity { region, component } | Permeability { region, component } => {
            let field = usize::from(matches!(parameter, Permeability { .. }));
            region_ok(region)?;
            tensors[region][field] = match component {
                None => {
                    let rate = if field == 0 {
                        ConstitutiveRate {
                            relative_permittivity: 1.,
                            ..Default::default()
                        }
                    } else {
                        ConstitutiveRate {
                            relative_permeability: 1.,
                            ..Default::default()
                        }
                    };
                    isotropic = Some((region, rate));
                    identity(Complex::ONE)
                }
                Some([i, j]) if i < 3 && j < 3 => unit_entry(i, j),
                Some(c) => return Err(invalid(format!("component {c:?} is outside 0..3"))),
            };
            if field == 0 {
                ("relative permittivity", "1", 1.)
            } else {
                ("relative permeability", "1", 1.)
            }
        }
        Rotation { region, axis } => {
            let [x, y, z] = match axis {
                rec::Axis::X => [1., 0., 0.],
                rec::Axis::Y => [0., 1., 0.],
                rec::Axis::Z => [0., 0., 1.],
                other => {
                    return Err(invalid(format!(
                        "a rotation turns about a stack axis x, y or z, not {}",
                        other.name()
                    )));
                }
            }
            .map(Complex::from);
            let g = [
                [Complex::ZERO, -z, y],
                [z, Complex::ZERO, -x],
                [-y, x, Complex::ZERO],
            ];
            let values = crate::materials::stack_frame(
                &format!("region {region}"),
                material(region)?,
                wavelength_m,
            )?;
            let product = |a: &Tensor, b: &Tensor| -> Tensor {
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
                })
            };
            tensors[region] = values.map(|t| {
                let (l, r) = (product(&g, &t), product(&t, &g));
                std::array::from_fn(|i| std::array::from_fn(|j| l[i][j] - r[i][j]))
            });
            ("rotation", "rad", 1.)
        }
        SellmeierConstant { region, axis }
        | SellmeierCoefficient { region, axis, .. }
        | SellmeierPoleWavelength { region, axis, .. } => {
            let m = material(region)?;
            let found = crate::materials::models(m)
                .into_iter()
                .find(|(a, _)| *a == axis);
            let Some((_, rec::ScalarModel::Sellmeier { sellmeier, .. })) = found else {
                let on = axis.map_or(String::new(), |a| format!(" on axis {}", a.name()));
                return Err(invalid(format!(
                    "region {region} has no Sellmeier model{on}"
                )));
            };
            let model = crate::materials::sellmeier_model(sellmeier)?;
            let model_rates = model.permittivity_parameter_rates(k0_cm);
            let term_ok = |term: usize| {
                if term < sellmeier.terms.len() {
                    Ok(term)
                } else {
                    Err(invalid(format!(
                        "region {region} has no Sellmeier term {term}"
                    )))
                }
            };
            let (rate, name, unit) = match parameter {
                SellmeierCoefficient { term, .. } => (
                    model_rates.coefficients[term_ok(term)?],
                    "Sellmeier coefficient",
                    "1",
                ),
                SellmeierPoleWavelength { term, .. } => {
                    // dε/dλp = dε/dq · dq/dλp, with q = 2π / λp in cm^-1 and λp in m.
                    let term = term_ok(term)?;
                    let pole_m = sellmeier.terms[term].pole_wavelength_m;
                    let q_cm = std::f64::consts::TAU / (pole_m / M_PER_CM);
                    (
                        model_rates.pole_wavenumbers[term] * (-q_cm / pole_m),
                        "Sellmeier pole wavelength",
                        "m",
                    )
                }
                _ => (model_rates.constant, "Sellmeier constant", "1"),
            };
            let r = Complex::from(rate);
            tensors[region][0] = match (m, axis) {
                (rec::Material::Isotropic { .. }, _) => {
                    isotropic = Some((
                        region,
                        ConstitutiveRate {
                            relative_permittivity: rate,
                            ..Default::default()
                        },
                    ));
                    identity(r)
                }
                (rec::Material::Diagonal { .. }, Some(a)) => {
                    scaled(&unit_entry(a as usize, a as usize), r)
                }
                (rec::Material::Full { rotation, .. }, Some(a)) => {
                    let k = a as usize - 3;
                    crate::materials::rotate(rotation, &scaled(&unit_entry(k, k), r))
                }
                _ => return Err(invalid(format!("region {region} has no axis model"))),
            };
            (name, unit, 1.)
        }
    };
    // Walls stay fixed in the stack frame, whose origin is the first interface.
    let walled = !(stack.left_termination.is_open() && stack.right_termination.is_open());
    if walled && velocities[0] != 0. {
        return Err(invalid(format!(
            "{name} moves the first interface, and the stack, against its walls"
        )));
    }
    let rates: Option<Vec<ConstitutiveRate>> = tensors
        .iter()
        .enumerate()
        .map(|(r, t)| match isotropic {
            Some((region, rate)) if region == r => Some(rate),
            _ => scalar_rate(t),
        })
        .collect();
    let internal_unit = if per_unit == 1. { "1" } else { "cm" };
    let perturbation = rates
        .map(|rates| {
            GuidedPerturbation::new(
                name,
                internal_unit,
                velocities.clone(),
                wavelength_rate,
                rates,
            )
            .map_err(|e| invalid(format!("parameter direction: {e:?}")))
        })
        .transpose()?;
    Ok(Direction {
        perturbation,
        velocities,
        wavelength_rate,
        tensors,
        name,
        unit,
        per_unit,
    })
}

/// The direction as lamina-mode's full-tensor perturbation of `solved`'s
/// stack, with the factor its β rate takes: layer thickness rates from the
/// interface velocities and stack-frame tensor rates, or for the wavelength,
/// unit k0 with every material's dispersion and dk0/dλ = −k0/λ (λ in cm).
fn hybrid_direction(solved: &Solved, d: &Direction) -> Result<(HybridPerturbation, f64), Failure> {
    let input = &solved.record.input;
    if d.wavelength_rate != 0. {
        let stack = crate::solve::stack(&input.stack, input.request.wavelength_m)?;
        let k0 = k0_cm(&input.request);
        let direction = HybridPerturbation::wavenumber(&stack, k0)
            .map_err(|e| Failure::Derivatives(format!("material dispersion: {e}")))?;
        let factor = -k0 / (input.request.wavelength_m / M_PER_CM) * d.wavelength_rate;
        return Ok((direction, factor));
    }
    Ok((
        HybridPerturbation {
            thickness_rates: d.velocities.windows(2).map(|v| v[1] - v[0]).collect(),
            k0_rate: 0.,
            tensor_rates: d.tensors.clone(),
        },
        1.,
    ))
}

/// dβ along a direction of a mode counted by the argument principle, or
/// along a rate that is not real and diagonal: no field or power samples.
fn hybrid_sensitivity(
    solved: &Solved,
    mode_index: usize,
    parameter: rec::Parameter,
    d: &Direction,
    z_m: &[f64],
    frame: rec::Frame,
) -> Result<rec::GuidedSensitivity, Failure> {
    if !z_m.is_empty() {
        return Err(Failure::Invalid(format!(
            "field derivatives along {} are available for TE and TM modes of isotropic and \
             diagonal regions along real diagonal rates only; request no z_m",
            d.name
        )));
    }
    let point = point(solved, mode_index)?;
    let (direction, factor) = hybrid_direction(solved, d)?;
    let rate = point
        .hybrid_beta_rate_with(&Scatter4::new(), &direction)
        .map_err(|e| Failure::Derivatives(format!("eigenvalue sensitivity: {e}")))?;
    let mode = &solved.record.modes[mode_index];
    Ok(rec::GuidedSensitivity {
        schema: rec::SENSITIVITY_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: solved.record.assumption.clone(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        parameter,
        parameter_unit: d.unit.to_owned(),
        frame,
        beta_rad_per_m: mode.beta_rad_per_m,
        dbeta_rad_per_m: c2(rate.beta_rate * (factor * d.per_unit / M_PER_CM)),
        conditioning: rec::Conditioning {
            characteristic: mode.characteristic_value,
            characteristic_direction: c2(rate.characteristic_direction * factor),
            characteristic_beta: c2(rate.characteristic_beta),
            denominator_threshold: None,
        },
        z_m: vec![],
        electric_real: vec![],
        electric_imag: vec![],
        magnetic_real: vec![],
        magnetic_imag: vec![],
        d_electric_real: vec![],
        d_electric_imag: vec![],
        d_magnetic_real: vec![],
        d_magnetic_imag: vec![],
        sample_failures: vec![],
        power_fractions: None,
        power_fraction_rates: None,
    })
}

/// The group index of a mode counted by the argument principle: dβ/dk0
/// from the coupled characteristic, certified against the energy estimate.
fn hybrid_group_index(
    solved: &Solved,
    mode_index: usize,
    tolerance: f64,
) -> Result<rec::GroupIndex, Failure> {
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(Failure::Invalid(
            "the certification tolerance must be finite and positive".into(),
        ));
    }
    let point = point(solved, mode_index)?;
    let input = &solved.record.input;
    let stack = crate::solve::stack(&input.stack, input.request.wavelength_m)?;
    let backend = Scatter4::new();
    let direction = HybridPerturbation::wavenumber(&stack, k0_cm(&input.request))
        .map_err(|e| Failure::Derivatives(format!("material dispersion: {e}")))?;
    let rate = point
        .hybrid_beta_rate_with(&backend, &direction)
        .map_err(|e| Failure::Derivatives(format!("wavelength sensitivity: {e}")))?;
    let group = point
        .hybrid_group_index_with(&backend)
        .map_err(|e| Failure::Derivatives(format!("group index: {e}")))?;
    if group.relative_discrepancy.is_nan() || group.relative_discrepancy > tolerance {
        return Err(Failure::Derivatives(format!(
            "group index not certified: dβ/dk0 = {}, c W / P = {}, relative discrepancy {:e} \
             exceeds {tolerance:e}",
            group.value, group.energy, group.relative_discrepancy
        )));
    }
    Ok(rec::GroupIndex {
        schema: rec::GROUP_INDEX_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: solved.record.assumption.clone(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        value: group.value,
        derivative: c2(rate.beta_rate),
        energy_estimate: group.energy,
        relative_discrepancy: group.relative_discrepancy,
        tolerance,
        conditioning: rec::Conditioning {
            characteristic: solved.record.modes[mode_index].characteristic_value,
            characteristic_direction: c2(rate.characteristic_direction),
            characteristic_beta: c2(rate.characteristic_beta),
            denominator_threshold: None,
        },
    })
}

fn conditioning(s: &lamina_mode::GuidedBetaSensitivity) -> rec::Conditioning {
    rec::Conditioning {
        characteristic: c2(s.characteristic),
        characteristic_direction: c2(s.characteristic_direction),
        characteristic_beta: c2(s.characteristic_beta),
        denominator_threshold: Some(s.denominator_threshold),
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
    let scalar = match &direction.perturbation {
        Some(p) if !solved.coupled() => p,
        _ => return hybrid_sensitivity(solved, mode_index, parameter, &direction, z_m, frame),
    };
    let backend = Scatter2::new();
    let eigenvalue = point
        .guided_beta_sensitivity_with(
            &backend,
            scalar,
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
        assumption: solved.record.assumption.clone(),
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
        power_fractions: None,
        power_fraction_rates: None,
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
    record.power_fractions = Some(
        power
            .regional_power
            .iter()
            .map(|r| r.power_w_per_m * sign)
            .collect(),
    );
    record.power_fraction_rates = Some(
        power
            .regional_power
            .iter()
            .map(|r| r.power_rate_w_per_m * sign * direction.per_unit)
            .collect(),
    );
    Ok(record)
}

pub fn group_index(
    solved: &Solved,
    mode_index: usize,
    tolerance: f64,
) -> Result<rec::GroupIndex, Failure> {
    let point = point(solved, mode_index)?;
    if crate::solve::tabulated(&solved.record.input.stack) {
        return Err(Failure::Invalid(
            "a group index needs the wavelength dispersion a tabulated material lacks".into(),
        ));
    }
    if solved.coupled() {
        return hybrid_group_index(solved, mode_index, tolerance);
    }
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
        assumption: solved.record.assumption.clone(),
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
