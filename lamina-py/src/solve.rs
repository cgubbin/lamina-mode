//! Adapter between the SI records and lamina-mode's internal cm^-1 units.
use crate::records::{self as rec, C2};
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, Constant, ExteriorSampling, FieldSampling,
    LayerSampling, Pml, Polarisation, Scatter2, StretchProfile, Termination, Wall, WallKind,
    stack::StackBuilder,
};
use lamina_mode::{
    AnalyticAssumption, Complex, FieldRecoverySettings, GuidedDirection, GuidedLimit, GuidedWindow,
    ImplicitSlopeRequest, ImplicitSlopeSettings, LocalAnalyticityAssumption,
    LocalDerivativeSettings, PhysicalParameter, RefinementConfig,
};
use lamina_units::{Length, LengthUnit};
use std::f64::consts::TAU;

/// Metres per centimetre.
pub(crate) const M_PER_CM: f64 = 0.01;
/// Impedance of free space, as lamina-mode uses it for guided power.
pub(crate) const Z0_OHM: f64 = 376.730313412;

/// The bindings accept constant real materials only. These are entire
/// functions of frequency and wavenumber, so the analytic declarations that
/// lamina-mode requires hold by construction.
pub const ASSUMPTION: &str = "constant real materials: holomorphic and pole-free in k0 and beta, \
     regular in layer thickness (declared by the bindings)";

/// The declaration for stacks with a Sellmeier material: its poles are
/// checked to lie away from the operating wavelength, so the materials are
/// holomorphic in a neighbourhood of the operating k0 and entire in beta.
pub const DISPERSIVE_ASSUMPTION: &str = "constant or Sellmeier real materials with every pole away \
     from the operating wavelength: holomorphic near the operating k0, pole-free in beta, regular in \
     layer thickness (declared by the bindings)";

/// Minimum relative separation of a Sellmeier pole wavelength from the
/// operating wavelength.
const POLE_SEPARATION: f64 = 1e-6;

pub fn regions(spec: &rec::Stack) -> impl Iterator<Item = &rec::Material> {
    std::iter::once(&spec.left)
        .chain(spec.layers.iter().map(|l| &l.material))
        .chain(std::iter::once(&spec.right))
}

pub fn assumption(spec: &rec::Stack) -> &'static str {
    if regions(spec).all(|m| matches!(m, rec::Material::Constant(_))) {
        ASSUMPTION
    } else {
        DISPERSIVE_ASSUMPTION
    }
}

/// The lamina-materials Sellmeier model of a record, in canonical units.
pub fn sellmeier_model(spec: &rec::Sellmeier) -> Result<lamina_materials::Sellmeier<f64>, Failure> {
    let terms = spec
        .terms
        .iter()
        .map(|t| {
            lamina_materials::SellmeierTerm::from_wavelength_pole(
                t.coefficient,
                lamina_units::Wavelength::centimetres(t.pole_wavelength_m / M_PER_CM),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Failure::Invalid(format!("Sellmeier term: {e:?}")))?;
    lamina_materials::Sellmeier::new(spec.constant, terms)
        .map_err(|e| Failure::Invalid(format!("Sellmeier model: {e:?}")))
}

#[derive(Debug)]
pub enum Failure {
    Invalid(String),
    /// A termination the operation does not support.
    Unsupported(String),
    Enumeration(String),
    Fields(String),
    Derivatives(String),
    /// A pairing integral exceeded its accuracy or sampling budget.
    Integration(String),
}

pub(crate) fn c2(z: Complex) -> C2 {
    [z.re, z.im]
}

pub(crate) fn finite_positive(name: &str, v: f64) -> Result<f64, Failure> {
    if v.is_finite() && v > 0. {
        Ok(v)
    } else {
        Err(Failure::Invalid(format!(
            "{name} must be finite and positive, got {v}"
        )))
    }
}

/// Validate an input and normalise signed zeros, so that equal inputs have
/// one canonical form.
pub fn normalise(mut input: rec::GuidedInput) -> Result<rec::GuidedInput, Failure> {
    rec::check_identity(&input).map_err(Failure::Invalid)?;
    normalise_stack(&mut input.stack, input.request.wavelength_m)?;
    let r = &mut input.request;
    finite_positive("wavelength_m", r.wavelength_m)?;
    finite_positive("cutoff_margin_rad_per_m", r.cutoff_margin_rad_per_m)?;
    finite_positive("phase_tolerance", r.phase_tolerance)?;
    let [lo, hi] = r.abs_beta_rad_per_m;
    if !(lo.is_finite() && hi.is_finite() && lo >= 0. && lo < hi) {
        return Err(Failure::Invalid(format!(
            "abs_beta_rad_per_m must satisfy 0 <= low < high, got [{lo}, {hi}]"
        )));
    }
    // Adding +0.0 maps -0.0 to +0.0 and leaves every other value unchanged.
    r.abs_beta_rad_per_m = [lo + 0., hi + 0.];
    let f = &r.refinement;
    let tolerances = [
        f.residual_atol,
        f.residual_rtol,
        f.native_atol,
        f.native_rtol,
    ];
    if !tolerances.iter().all(|t| t.is_finite() && *t >= 0.)
        || f.residual_atol + f.residual_rtol <= 0.
    {
        return Err(Failure::Invalid(
            "refinement tolerances must be finite and non-negative, with a positive residual limit"
                .into(),
        ));
    }
    Ok(input)
}

/// Validate a stack at `wavelength` and normalise the signed zeros of its
/// wall and PML fields.
pub(crate) fn normalise_stack(stack: &mut rec::Stack, wavelength: f64) -> Result<(), Failure> {
    let material = |name: &str, m: &rec::Material| -> Result<(), Failure> {
        match m {
            rec::Material::Constant(c) => {
                finite_positive(
                    &format!("{name} relative permittivity"),
                    c.relative_permittivity,
                )?;
                finite_positive(
                    &format!("{name} relative permeability"),
                    c.relative_permeability,
                )?;
            }
            rec::Material::Sellmeier(m) => {
                if m.relative_permeability != 1. {
                    return Err(Failure::Invalid(format!(
                        "{name}: Sellmeier materials support relative_permeability 1 only"
                    )));
                }
                if !m.sellmeier.constant.is_finite() {
                    return Err(Failure::Invalid(format!(
                        "{name} Sellmeier constant must be finite"
                    )));
                }
                for (j, t) in m.sellmeier.terms.iter().enumerate() {
                    if !t.coefficient.is_finite() {
                        return Err(Failure::Invalid(format!(
                            "{name} Sellmeier term {j} coefficient must be finite"
                        )));
                    }
                    finite_positive(
                        &format!("{name} Sellmeier term {j} pole_wavelength_m"),
                        t.pole_wavelength_m,
                    )?;
                    if (t.pole_wavelength_m - wavelength).abs() <= POLE_SEPARATION * wavelength {
                        return Err(Failure::Invalid(format!(
                            "{name} Sellmeier term {j} has its pole at the operating wavelength"
                        )));
                    }
                }
            }
        }
        Ok(())
    };
    material("left exterior", &stack.left)?;
    material("right exterior", &stack.right)?;
    for (i, layer) in stack.layers.iter().enumerate() {
        material(&format!("layer {i}"), &layer.material)?;
        finite_positive(&format!("layer {i} thickness_m"), layer.thickness_m)?;
    }
    // Walls and PMLs start on or outside the outermost interface of their
    // side: z ≤ 0 on the left, z ≥ L on the right, with L summed left to right
    // as in `Stack.interfaces_m`.
    let length = total_m(stack);
    for (side, termination, interface) in [
        ("left", &mut stack.left_termination, 0.),
        ("right", &mut stack.right_termination, length),
    ] {
        let (field, z_m) = match termination {
            rec::Termination::Open => continue,
            rec::Termination::Wall { z_m, .. } => ("z_m", z_m),
            rec::Termination::Pml {
                start_z_m,
                thickness_m,
                profile: rec::StretchProfile::Polynomial { order, strength },
                ..
            } => {
                finite_positive(&format!("{side}_termination thickness_m"), *thickness_m)?;
                if !(order.is_finite() && *order >= 0.) {
                    return Err(Failure::Invalid(format!(
                        "{side}_termination profile order must be finite and non-negative, \
                         got {order}"
                    )));
                }
                let [re, im] = *strength;
                if !(re.is_finite() && im.is_finite() && re >= 0. && im > 0.) {
                    return Err(Failure::Invalid(format!(
                        "{side}_termination profile strength must be finite with a \
                         non-negative real and a positive imaginary part, got [{re}, {im}]"
                    )));
                }
                *order += 0.;
                *strength = [re + 0., im];
                ("start_z_m", start_z_m)
            }
        };
        let inside = if side == "left" {
            *z_m > interface
        } else {
            *z_m < interface
        };
        if !z_m.is_finite() || inside {
            return Err(Failure::Invalid(format!(
                "{side}_termination {field} must be finite and on or outside the {side} \
                 interface at {interface} m, got {z_m}"
            )));
        }
        *z_m += 0.;
    }
    Ok(())
}

fn handle(m: &rec::Material) -> Result<AnalyticalMaterialHandle<Complex>, Failure> {
    Ok(match m {
        rec::Material::Constant(c) => AnalyticalMaterialHandle::new(Constant::new(
            c.relative_permittivity,
            c.relative_permeability,
        )),
        rec::Material::Sellmeier(s) => {
            AnalyticalMaterialHandle::new(sellmeier_model(&s.sellmeier)?)
        }
    })
}

/// The last interface, in metres: the layer thicknesses summed left to right.
fn total_m(spec: &rec::Stack) -> f64 {
    spec.layers
        .iter()
        .fold(0., |total, l| total + l.thickness_m)
}

/// The termination closing each side, with its gap from the outermost
/// interface to the wall or PML start in cm; `None` when open.
fn terminations(spec: &rec::Stack) -> [Option<(Termination<f64>, f64)>; 2] {
    let length = total_m(spec);
    let last_cm = spec
        .layers
        .iter()
        .fold(0., |total, l| total + l.thickness_m / M_PER_CM);
    let kind = |kind| match kind {
        rec::WallType::Pec => WallKind::PerfectElectric,
        rec::WallType::Pmc => WallKind::PerfectMagnetic,
    };
    // The gap is formed in metres, where the position was validated, and the
    // right termination placed that gap beyond the layers as summed in cm, so
    // one on the interface stays exactly on it.
    let place = |z_m: f64, right: bool| {
        let gap_cm = if right { z_m - length } else { -z_m } / M_PER_CM;
        let z_cm = if right { last_cm + gap_cm } else { -gap_cm };
        (Length::centimetres(z_cm), gap_cm)
    };
    let close = |termination: &rec::Termination, right: bool| match *termination {
        rec::Termination::Open => None,
        rec::Termination::Wall { kind: wall, z_m } => {
            let (z, gap_cm) = place(z_m, right);
            Some((Wall::new(kind(wall), z).into(), gap_cm))
        }
        rec::Termination::Pml {
            start_z_m,
            thickness_m,
            profile: rec::StretchProfile::Polynomial { order, strength },
            backing,
        } => {
            let (start, gap_cm) = place(start_z_m, right);
            let profile = StretchProfile::polynomial(order, Complex::new(strength[0], strength[1]));
            let thickness = Length::centimetres(thickness_m / M_PER_CM);
            Some((
                Pml::new(start, thickness, profile, kind(backing)).into(),
                gap_cm,
            ))
        }
    };
    [
        close(&spec.left_termination, false),
        close(&spec.right_termination, true),
    ]
}

pub(crate) fn stack(spec: &rec::Stack) -> Result<AnalyticalMaterialStack<Complex>, Failure> {
    let mut builder = StackBuilder::new(handle(&spec.left)?, handle(&spec.right)?);
    for layer in &spec.layers {
        builder = builder.layer(
            handle(&layer.material)?,
            Length::centimetres(layer.thickness_m / M_PER_CM),
        );
    }
    let [left, right] = terminations(spec);
    if let Some((termination, _)) = left {
        builder = builder.left_termination(termination);
    }
    if let Some((termination, _)) = right {
        builder = builder.right_termination(termination);
    }
    Ok(builder.finalise())
}

/// k0 in lamina-mode's internal cm^-1.
/// Field-recovery settings with the characteristic limit scaled to k0.
pub(crate) fn recovery_settings(r: &rec::GuidedRequest) -> FieldRecoverySettings {
    FieldRecoverySettings {
        characteristic_limit: CHARACTERISTIC_RTOL * k0_cm(r),
        relative_boundary_limit: RELATIVE_BOUNDARY_LIMIT,
        ..Default::default()
    }
}

pub(crate) fn k0_cm(r: &rec::GuidedRequest) -> f64 {
    TAU / (r.wavelength_m / M_PER_CM)
}

fn window(r: &rec::GuidedRequest) -> GuidedWindow {
    let f = r.refinement;
    // Lengths become cm and wavenumbers cm^-1: multiply rad/m by m per cm.
    GuidedWindow {
        wavelength: Length::centimetres(r.wavelength_m / M_PER_CM),
        family: match r.polarisation {
            rec::Polarisation::Te => Polarisation::TransverseElectric,
            rec::Polarisation::Tm => Polarisation::TransverseMagnetic,
        }
        .into(),
        direction: match r.direction {
            rec::Direction::Forward => GuidedDirection::Forward,
            rec::Direction::Backward => GuidedDirection::Backward,
        },
        abs_beta: r.abs_beta_rad_per_m.map(|b| b * M_PER_CM),
        cutoff_margin: r.cutoff_margin_rad_per_m * M_PER_CM,
        phase_tolerance: r.phase_tolerance,
        max_subdivisions: r.max_subdivisions,
        refinement: RefinementConfig {
            residual_atol: f.residual_atol,
            residual_rtol: f.residual_rtol,
            residual_scale: k0_cm(r),
            native_atol: f.native_atol,
            native_rtol: f.native_rtol,
            max_updates: f.max_updates,
            max_backtracks: f.max_backtracks,
            max_evaluations: f.max_evaluations,
        },
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

/// Residual limit per unit k0 (cm^-1). The residual floor at accepted roots
/// depends on the stack and the search path, up to about 2e-10 k0 measured;
/// the default leaves a fivefold margin while keeping the seed's boundary
/// defect near 1e-7. Root accuracy is set by the native correction
/// tolerances, which must also pass (docs/decisions/0003).
const RESIDUAL_RTOL: f64 = 1e-9;
/// Field-recovery limit on the seed's relative boundary defect, above the
/// defects that roots accepted at the residual limit leave; every profile
/// records its actual defect (docs/decisions/0003).
const RELATIVE_BOUNDARY_LIMIT: f64 = 1e-5;
/// Field-recovery characteristic limit per unit k0 (cm^-1): lamina-mode's
/// default, 1e-6, at a wavelength of 1 cm.
const CHARACTERISTIC_RTOL: f64 = 1e-6 / TAU;

pub fn default_refinement() -> rec::Refinement {
    let d = RefinementConfig::default();
    rec::Refinement {
        residual_atol: 0.,
        residual_rtol: RESIDUAL_RTOL,
        native_atol: d.native_atol,
        native_rtol: d.native_rtol,
        max_updates: d.max_updates,
        max_backtracks: d.max_backtracks,
        max_evaluations: d.max_evaluations,
    }
}

/// An enumerated basis with the accepted points kept for later sampling.
pub struct Solved {
    pub record: rec::GuidedBasis,
    pub basis: lamina_mode::GuidedBasis,
}

fn gap(g: &lamina_mode::GuidedGap) -> rec::Gap {
    let (cause, detail) = match &g.cause {
        GuidedLimit::EndpointPhase => ("endpoint_phase", None),
        GuidedLimit::PhaseResolution => ("phase_resolution", None),
        GuidedLimit::Budget => ("budget", None),
        GuidedLimit::Correction(failure) => ("correction", Some(format!("{failure:?}"))),
        GuidedLimit::OutsideBracket => ("outside_bracket", None),
        // Argument-principle causes of full-tensor stacks, not in schema 1.
        other => ("hybrid_count", Some(format!("{other:?}"))),
    };
    rec::Gap {
        abs_beta_rad_per_m: g.abs_beta.map(|b| b / M_PER_CM),
        expected: g.expected,
        cause: cause.to_owned(),
        detail,
    }
}

pub fn enumerate(input: rec::GuidedInput) -> Result<Solved, Failure> {
    let input = normalise(input)?;
    let basis = lamina_mode::enumerate_guided_modes(stack(&input.stack)?, window(&input.request))
        .map_err(|e| match e {
        lamina_mode::EnumerationError::Termination { .. } => Failure::Unsupported(e.to_string()),
        e => Failure::Enumeration(e.to_string()),
    })?;
    let k0_cm = k0_cm(&input.request);
    let modes = basis
        .modes
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let beta = point.physical_coordinate();
            let evidence = point.evidence();
            rec::Mode {
                index,
                beta_rad_per_m: c2(beta / M_PER_CM),
                n_eff: c2(beta / k0_cm),
                native_coordinate: c2(evidence.native_coordinate),
                characteristic_value: c2(evidence.value),
                characteristic_derivative: c2(evidence.derivative),
                residual: evidence.value.norm(),
                residual_limit: evidence.residual_limit,
                correction: evidence.correction,
                correction_limit: evidence.correction_limit,
            }
        })
        .collect::<Vec<_>>();
    let completeness = if basis.is_complete() {
        rec::Completeness::Complete
    } else if basis.expected.is_none() {
        rec::Completeness::CountUnknown
    } else {
        rec::Completeness::Partial
    };
    let record = rec::GuidedBasis {
        schema: rec::BASIS_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: assumption(&input.stack).to_owned(),
        input_hash: rec::digest(&input).map_err(Failure::Invalid)?,
        k0_rad_per_m: k0_cm / M_PER_CM,
        light_line_rad_per_m: basis.light_line / M_PER_CM,
        upper_bound_rad_per_m: basis.upper_bound / M_PER_CM,
        eligible_rad_per_m: basis.eligible.map(|w| w.map(|b| b / M_PER_CM)),
        expected: basis.expected,
        found: modes.len(),
        completeness,
        modes,
        unresolved: basis.unresolved.iter().map(gap).collect(),
        input,
    };
    Ok(Solved { record, basis })
}

/// Map z positions (m) to core sampling regions. Consecutive samples in the
/// same region share one request; request order is preserved. A sample on
/// an interface belongs to the region on its +z side.
pub(crate) fn sampling(spec: &rec::Stack, z_m: &[f64]) -> Result<FieldSampling<f64>, Failure> {
    let mut interfaces_cm = vec![0.];
    for layer in &spec.layers {
        interfaces_cm.push(interfaces_cm.last().unwrap() + layer.thickness_m / M_PER_CM);
    }
    let total = *interfaces_cm.last().unwrap();
    // Samples reach a right wall, or a right PML's backing wall.
    let right_gap = terminations(spec)[1].map(|(termination, gap)| match termination {
        Termination::Pml(pml) => gap + pml.thickness().as_centimetres(),
        _ => gap,
    });
    #[derive(PartialEq, Clone, Copy)]
    enum Region {
        Left,
        Layer(usize),
        Right,
    }
    let mut runs: Vec<(Region, Vec<f64>)> = Vec::new();
    for &z in z_m {
        if !z.is_finite() {
            return Err(Failure::Invalid(format!("z_m must be finite, got {z}")));
        }
        for (side, termination, beyond) in [
            (
                "left",
                spec.left_termination,
                (|z, wall| z < wall) as fn(f64, f64) -> bool,
            ),
            ("right", spec.right_termination, |z, wall| z > wall),
        ] {
            let wall = match termination {
                rec::Termination::Wall { z_m, .. } => Some(z_m),
                rec::Termination::Pml {
                    start_z_m,
                    thickness_m,
                    ..
                } => Some(if side == "left" {
                    start_z_m - thickness_m
                } else {
                    start_z_m + thickness_m
                }),
                _ => None,
            };
            if let Some(wall) = wall
                && beyond(z, wall)
            {
                return Err(Failure::Invalid(format!(
                    "z_m {z} lies beyond the {side} wall at {wall}"
                )));
            }
        }
        let z_cm = z / M_PER_CM;
        let (region, offset) = if z_cm < 0. {
            (Region::Left, -z_cm)
        } else if z_cm >= total {
            // Up to the right wall, whose gap was formed in metres.
            let offset = z_cm - total;
            (
                Region::Right,
                right_gap.map_or(offset, |gap| offset.min(gap)),
            )
        } else {
            let i = interfaces_cm.partition_point(|&x| x <= z_cm) - 1;
            (Region::Layer(i), z_cm - interfaces_cm[i])
        };
        match runs.last_mut() {
            Some((r, offsets)) if *r == region => offsets.push(offset),
            _ => runs.push((region, vec![offset])),
        }
    }
    let unit = LengthUnit::Centimetre;
    Ok(runs
        .into_iter()
        .fold(FieldSampling::new(), |s, (region, offsets)| match region {
            Region::Left => s.left_exterior(ExteriorSampling::distances_in(offsets, unit)),
            Region::Layer(i) => s.layer(i, LayerSampling::offsets_in(offsets, unit)),
            Region::Right => s.right_exterior(ExteriorSampling::distances_in(offsets, unit)),
        }))
}

pub(crate) fn point(
    solved: &Solved,
    mode_index: usize,
) -> Result<&lamina_mode::ModePoint, Failure> {
    solved.basis.modes.get(mode_index).ok_or_else(|| {
        Failure::Invalid(format!(
            "mode_index {mode_index} is out of range for {} modes",
            solved.basis.modes.len()
        ))
    })
}

pub fn profile(
    solved: &Solved,
    mode_index: usize,
    z_m: &[f64],
) -> Result<rec::GuidedProfile, Failure> {
    let point = point(solved, mode_index)?;
    let sampling = sampling(&solved.record.input.stack, z_m)?;
    let profile = point
        .guided_power_profile_with(
            Scatter2::new(),
            &sampling,
            recovery_settings(&solved.record.input.request),
        )
        .map_err(|e| Failure::Fields(format!("{e}: {e:?}")))?;
    let normalisation = match &profile.normalization {
        lamina_mode::ProfileNormalization::GuidedPower(n) => rec::PowerNormalisation {
            signed_power_w_per_m: n.signed_power_w_per_m,
            scale: c2(n.scale),
            phase_reference: c2(n.phase_reference),
            seed_power_w_per_m: n.seed_power_w_per_m,
            seed_exterior_power_w_per_m: n.seed_exterior_power_w_per_m,
        },
        _ => return Err(Failure::Fields("profile is not power normalised".into())),
    };
    if profile.depths_cm.len() != z_m.len() {
        return Err(Failure::Fields(
            "sample count changed during recovery".into(),
        ));
    }
    let fields = profile.fields.quantity();
    let split = |v: &lamina_core::VectorField<Complex, ndarray::Ix1>, scale: f64| {
        let (x, y, z) = (v.x(), v.y(), v.z());
        let re = (0..x.len())
            .map(|i| [x[i].re * scale, y[i].re * scale, z[i].re * scale])
            .collect::<Vec<_>>();
        let im = (0..x.len())
            .map(|i| [x[i].im * scale, y[i].im * scale, z[i].im * scale])
            .collect::<Vec<_>>();
        (re, im)
    };
    let (electric_real, electric_imag) = split(fields.electric(), 1.);
    // Core stores Z0*H in V/m; divide by Z0 for A/m.
    let (magnetic_real, magnetic_imag) = split(fields.magnetic(), 1. / Z0_OHM);
    Ok(rec::GuidedProfile {
        schema: rec::PROFILE_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        beta_rad_per_m: solved.record.modes[mode_index].beta_rad_per_m,
        // Request order is preserved, so report the positions as requested.
        z_m: z_m.iter().map(|z| z + 0.).collect(),
        electric_real,
        electric_imag,
        magnetic_real,
        magnetic_imag,
        normalisation,
        relative_boundary_defect: profile.relative_boundary_defect,
    })
}

pub fn derivatives(
    solved: &Solved,
    mode_index: usize,
    layers: &[usize],
) -> Result<rec::LocalDerivatives, Failure> {
    let point = point(solved, mode_index)?;
    let count = solved.record.input.stack.layers.len();
    for (i, &layer) in layers.iter().enumerate() {
        if layer >= count || layers[..i].contains(&layer) {
            return Err(Failure::Invalid(format!(
                "thickness layers must be distinct indices below {count}, got {layers:?}"
            )));
        }
    }
    let mut parameters = vec![PhysicalParameter::K0, PhysicalParameter::Beta];
    parameters.extend(
        layers
            .iter()
            .map(|&i| PhysicalParameter::LayerThickness(i.into())),
    );
    let report = point
        .local_derivatives_with(
            Scatter2::new(),
            &parameters,
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            LocalDerivativeSettings::default(),
        )
        .map_err(|e| Failure::Derivatives(format!("{e:?}")))?;
    // Partials of the characteristic F. Per rad/m: F_k[per cm^-1] * (cm^-1 per rad/m).
    // Per m of thickness: F_d[per cm] * (cm per m).
    let partials = report
        .outcomes
        .iter()
        .map(|outcome| {
            let (parameter, layer, factor, unit) = match outcome.parameter {
                PhysicalParameter::K0 => ("k0", None, M_PER_CM, "characteristic per rad/m"),
                PhysicalParameter::Beta => ("beta", None, M_PER_CM, "characteristic per rad/m"),
                PhysicalParameter::LayerThickness(i) => (
                    "thickness",
                    Some(i.get()),
                    1. / M_PER_CM,
                    "characteristic per m",
                ),
            };
            let (value, held_fixed, failure) = match &outcome.result {
                Ok(p) => (
                    Some(c2(p.value * factor)),
                    Some(p.held_fixed.to_owned()),
                    None,
                ),
                Err(e) => (None, None, Some(e.to_string())),
            };
            rec::Partial {
                parameter: parameter.to_owned(),
                layer,
                value,
                unit: unit.to_owned(),
                held_fixed,
                failure,
            }
        })
        .collect();
    let mut slopes = vec![(ImplicitSlopeRequest::BetaPerK0, None)];
    slopes.extend(
        layers
            .iter()
            .map(|&i| (ImplicitSlopeRequest::BetaPerThickness(i.into()), Some(i))),
    );
    let slopes = slopes
        .into_iter()
        .map(|(request, layer)| {
            // dbeta/dk0 is dimensionless. dbeta/dd: (rad/m per cm^-1) * (cm per m).
            let (name, factor, unit) = match layer {
                None => ("dbeta_dk0", 1., "1"),
                Some(_) => (
                    "dbeta_dthickness",
                    1. / (M_PER_CM * M_PER_CM),
                    "rad/m per m",
                ),
            };
            let (value, failure) =
                match report.implicit_slope(request, ImplicitSlopeSettings::default()) {
                    Ok(s) => (Some(c2(s.value * factor)), None),
                    Err(e) => (None, Some(e.to_string())),
                };
            rec::Slope {
                name: name.to_owned(),
                layer,
                value,
                unit: unit.to_owned(),
                failure,
            }
        })
        .collect();
    let (characteristic, characteristic_failure) = match &report.characteristic {
        Ok(c) => (Some(c2(*c)), None),
        Err(e) => (None, Some(format!("{e:?}"))),
    };
    Ok(rec::LocalDerivatives {
        schema: rec::DERIVATIVES_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: assumption(&solved.record.input.stack).to_owned(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        characteristic,
        characteristic_failure,
        partials,
        slopes,
    })
}
