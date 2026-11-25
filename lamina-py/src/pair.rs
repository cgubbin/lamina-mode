//! Pairings of guided modes with each other and with supplied fields, in SI.
use crate::records as rec;
use crate::sens::{Direction, direction};
use crate::solve::{
    DISPERSIVE_ASSUMPTION, Failure, M_PER_CM, Solved, Z0_OHM, assumption, c2, point,
    recovery_settings,
};
use lamina_core::Scatter2;
use lamina_mode::{
    Complex, ConstitutiveRate, GuidedCouplingError, GuidedOverlapDerivativeError,
    GuidedOverlapError, GuidedOverlapRegistration, GuidedPerturbation, GuidedProjectionError,
    GuidedProjectionInput, ProjectionSide,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// The assumption of a result that uses both stacks.
fn joint_assumption(first: &Solved, second: &Solved) -> String {
    let a = assumption(&first.record.input.stack);
    let b = assumption(&second.record.input.stack);
    if a == DISPERSIVE_ASSUMPTION || b == DISPERSIVE_ASSUMPTION {
        DISPERSIVE_ASSUMPTION
    } else {
        a
    }
    .to_owned()
}

fn overlap_failure(e: GuidedOverlapError) -> Failure {
    match e {
        GuidedOverlapError::Registration => Failure::Invalid(format!(
            "{e}: origins must be finite and the registered interfaces representable"
        )),
        GuidedOverlapError::IncompatibleWavelength => {
            Failure::Invalid(format!("{e}: the two bases must share their wavelength"))
        }
        GuidedOverlapError::Field { .. } => Failure::Fields(format!("{e}: {e:?}")),
        GuidedOverlapError::Quadrature => Failure::Integration(e.to_string()),
        GuidedOverlapError::ClosedStack => Failure::Invalid(e.to_string()),
    }
}

fn registration(origins_m: [f64; 2]) -> GuidedOverlapRegistration {
    GuidedOverlapRegistration {
        origins_cm: origins_m.map(|z| z / M_PER_CM),
    }
}

/// Positions in cm to m, with signed zeros normalised.
fn to_m(values_cm: &[f64]) -> Vec<f64> {
    values_cm.iter().map(|z| z * M_PER_CM + 0.).collect()
}

struct Pair<'a> {
    first: &'a Solved,
    first_mode: usize,
    second: &'a Solved,
    second_mode: usize,
}

impl<'a> Pair<'a> {
    fn new(first: &'a Solved, first_mode: usize, second: &'a Solved, second_mode: usize) -> Self {
        Self {
            first,
            first_mode,
            second,
            second_mode,
        }
    }

    fn points(&self) -> Result<[&'a lamina_mode::ModePoint; 2], Failure> {
        Ok([
            point(self.first, self.first_mode)?,
            point(self.second, self.second_mode)?,
        ])
    }
}

pub fn power_overlap(
    first: &Solved,
    first_mode: usize,
    second: &Solved,
    second_mode: usize,
    origins_m: [f64; 2],
) -> Result<rec::PowerOverlap, Failure> {
    let pair = Pair::new(first, first_mode, second, second_mode);
    let overlap = lamina_mode::guided_power_overlap_with(
        &Scatter2::new(),
        pair.points()?,
        registration(origins_m),
        recovery_settings(&first.record.input.request),
    )
    .map_err(overlap_failure)?;
    Ok(rec::PowerOverlap {
        schema: rec::POWER_OVERLAP_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: joint_assumption(first, second),
        definition: rec::POWER_OVERLAP_DEFINITION.to_owned(),
        first_input_hash: first.record.input_hash.clone(),
        first_mode_index: first_mode,
        second_input_hash: second.record.input_hash.clone(),
        second_mode_index: second_mode,
        origins_m: origins_m.map(|z| z + 0.),
        power_w_per_m: c2(overlap.power_w_per_m),
        exterior_power_w_per_m: overlap.exterior_power_w_per_m.map(c2),
        partition_m: to_m(&overlap.partition_cm),
        sample_count: overlap.sample_count,
        quadrature_change_w_per_m: overlap.quadrature_change_w_per_m,
    })
}

pub fn reciprocity(
    first: &Solved,
    first_mode: usize,
    second: &Solved,
    second_mode: usize,
    origins_m: [f64; 2],
) -> Result<rec::Reciprocity, Failure> {
    let pair = Pair::new(first, first_mode, second, second_mode);
    let flux = lamina_mode::guided_reciprocity_with(
        &Scatter2::new(),
        pair.points()?,
        registration(origins_m),
        recovery_settings(&first.record.input.request),
    )
    .map_err(overlap_failure)?;
    Ok(rec::Reciprocity {
        schema: rec::RECIPROCITY_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: joint_assumption(first, second),
        definition: rec::RECIPROCITY_DEFINITION.to_owned(),
        first_input_hash: first.record.input_hash.clone(),
        first_mode_index: first_mode,
        second_input_hash: second.record.input_hash.clone(),
        second_mode_index: second_mode,
        origins_m: origins_m.map(|z| z + 0.),
        flux_w_per_m: c2(flux.flux_w_per_m),
        exterior_flux_w_per_m: flux.exterior_flux_w_per_m.map(c2),
        partition_m: to_m(&flux.partition_cm),
        sample_count: flux.sample_count,
        quadrature_change_w_per_m: flux.quadrature_change_w_per_m,
    })
}

/// Rebuild a direction with its rates multiplied by `factor` and a new
/// internal unit.
fn rescaled(d: &Direction, unit: &str, factor: f64) -> Result<GuidedPerturbation, Failure> {
    let p = &d.perturbation;
    GuidedPerturbation::new(
        p.parameter(),
        unit,
        p.interface_velocities()
            .iter()
            .map(|v| v * factor)
            .collect(),
        p.wavelength_rate() * factor,
        p.regional_rates()
            .iter()
            .map(|r| r.scaled(factor))
            .collect(),
    )
    .map_err(|e| Failure::Invalid(format!("parameter direction: {e:?}")))
}

/// A direction per stack, both per one internal unit.
fn overlap_directions(
    pair: &Pair<'_>,
    parameters: [Option<rec::Parameter>; 2],
) -> Result<([GuidedPerturbation; 2], &'static str, f64), Failure> {
    let solved = [pair.first, pair.second];
    let mut directions = [None, None];
    for side in 0..2 {
        if let Some(p) = parameters[side] {
            let d = direction(solved[side], p)?;
            // Length parameters are per cm internally, so that rates along a
            // geometric and a material length parameter share one unit.
            let d = if d.unit == "m" && d.perturbation.units() != "cm" {
                Direction {
                    perturbation: rescaled(&d, "cm", M_PER_CM)?,
                    unit: "m",
                    per_unit: 1. / M_PER_CM,
                }
            } else {
                d
            };
            directions[side] = Some(d);
        }
    }
    let (unit, per_unit, internal) = match (&directions[0], &directions[1]) {
        (None, None) => {
            return Err(Failure::Invalid(
                "at least one of first_parameter and second_parameter must be set".into(),
            ));
        }
        (Some(a), Some(b)) => {
            if a.unit != b.unit {
                return Err(Failure::Invalid(format!(
                    "the two parameters must share their unit, got {:?} and {:?}",
                    a.unit, b.unit
                )));
            }
            if a.perturbation.wavelength_rate() != b.perturbation.wavelength_rate() {
                return Err(Failure::Invalid(
                    "the wavelength must vary on both sides or neither".into(),
                ));
            }
            (a.unit, a.per_unit, a.perturbation.units().to_owned())
        }
        (Some(d), None) | (None, Some(d)) => {
            if d.perturbation.wavelength_rate() != 0. {
                return Err(Failure::Invalid(
                    "the wavelength must vary on both sides or neither".into(),
                ));
            }
            (d.unit, d.per_unit, d.perturbation.units().to_owned())
        }
    };
    let fixed = |s: &Solved| {
        let n = s.record.input.stack.layers.len();
        GuidedPerturbation::new(
            "held fixed",
            internal.as_str(),
            vec![0.; n + 1],
            0.,
            vec![ConstitutiveRate::default(); n + 2],
        )
        .map_err(|e| Failure::Invalid(format!("parameter direction: {e:?}")))
    };
    let [a, b] = directions;
    let a = match a {
        Some(d) => d.perturbation,
        None => fixed(pair.first)?,
    };
    let b = match b {
        Some(d) => d.perturbation,
        None => fixed(pair.second)?,
    };
    Ok(([a, b], unit, per_unit))
}

fn derivative_failure(e: &GuidedOverlapDerivativeError) -> String {
    use GuidedOverlapDerivativeError::*;
    let side = |mode: &usize| if *mode == 0 { "first" } else { "second" };
    match e {
        IncompatibleDirection => "the two parameter directions are incompatible".into(),
        SeparatingInterfaces {
            position_cm,
            velocities_cm,
        } => format!(
            "coincident interfaces at z = {} m separate: velocities {} and {} per parameter unit",
            position_cm * M_PER_CM,
            velocities_cm[0],
            velocities_cm[1]
        ),
        Mode { mode, source } => {
            format!("{} mode eigenvalue sensitivity: {source:?}", side(mode))
        }
        Fields { mode, source } => format!("{} mode field sensitivity: {source:?}", side(mode)),
        Observation(source) => format!("field observation: {source:?}"),
        Quadrature => {
            "overlap derivative quadrature exceeded its accuracy or sampling budget".into()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn overlap_sensitivity(
    first: &Solved,
    first_mode: usize,
    second: &Solved,
    second_mode: usize,
    first_parameter: Option<rec::Parameter>,
    second_parameter: Option<rec::Parameter>,
    origins_m: [f64; 2],
) -> Result<rec::OverlapSensitivity, Failure> {
    let pair = Pair::new(first, first_mode, second, second_mode);
    let points = pair.points()?;
    let ([a, b], unit, per_unit) = overlap_directions(&pair, [first_parameter, second_parameter])?;
    let result = lamina_mode::guided_power_overlap_sensitivity_with(
        &Scatter2::new(),
        points,
        [&a, &b],
        registration(origins_m),
        recovery_settings(&first.record.input.request),
    )
    .map_err(overlap_failure)?;
    let base = &result.base;
    let mut record = rec::OverlapSensitivity {
        schema: rec::OVERLAP_SENSITIVITY_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: joint_assumption(first, second),
        definition: rec::OVERLAP_SENSITIVITY_DEFINITION.to_owned(),
        first_input_hash: first.record.input_hash.clone(),
        first_mode_index: first_mode,
        second_input_hash: second.record.input_hash.clone(),
        second_mode_index: second_mode,
        first_parameter,
        second_parameter,
        parameter_unit: unit.to_owned(),
        origins_m: origins_m.map(|z| z + 0.),
        power_w_per_m: c2(base.power_w_per_m),
        exterior_power_w_per_m: base.exterior_power_w_per_m.map(c2),
        partition_m: to_m(&base.partition_cm),
        sample_count: base.sample_count,
        quadrature_change_w_per_m: base.quadrature_change_w_per_m,
        power_rate_w_per_m: None,
        exterior_rates_w_per_m: None,
        rate_sample_count: None,
        rate_quadrature_change_w_per_m: None,
        derivative_failure: None,
    };
    match &result.tangent {
        Ok(t) => {
            record.power_rate_w_per_m = Some(c2(t.power_rate_w_per_m * per_unit));
            record.exterior_rates_w_per_m =
                Some(t.exterior_rates_w_per_m.map(|r| c2(r * per_unit)));
            record.rate_sample_count = Some(t.sample_count);
            record.rate_quadrature_change_w_per_m = Some(t.quadrature_change_w_per_m * per_unit);
        }
        Err(e) => record.derivative_failure = Some(derivative_failure(e)),
    }
    Ok(record)
}

/// Rows-first real and imaginary parts of an n × n matrix.
fn split_matrix(
    n: usize,
    entry: impl Fn(usize, usize) -> Complex,
    scale: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let rows = |part: fn(Complex) -> f64| {
        (0..n)
            .map(|i| (0..n).map(|j| part(entry(i, j) * scale)).collect())
            .collect()
    };
    (rows(|z| z.re), rows(|z| z.im))
}

/// The local coupling of a complete basis along one fixed-wavelength parameter.
pub fn local_coupling(
    solved: &Solved,
    parameter: rec::Parameter,
    gap_rad_per_m: Option<f64>,
) -> Result<rec::LocalCoupling, Failure> {
    let invalid = |m: String| Err(Failure::Invalid(m));
    let record = &solved.record;
    if record.completeness != rec::Completeness::Complete {
        return invalid(
            "local coupling needs a complete basis; this one is partial or its count unknown"
                .into(),
        );
    }
    if solved.basis.modes.is_empty() {
        return invalid("local coupling needs at least one mode".into());
    }
    match parameter {
        rec::Parameter::Wavelength => {
            return invalid(
                "local coupling needs a fixed-wavelength parameter, not the wavelength".into(),
            );
        }
        rec::Parameter::Permeability { .. } => {
            return invalid(
                "local coupling is for nonmagnetic stacks; a permeability parameter is ineligible"
                    .into(),
            );
        }
        _ => {}
    }
    let k0_m = record.k0_rad_per_m;
    let wavelength = record.input.request.wavelength_m;
    for (region, material) in crate::solve::regions(&record.input.stack).enumerate() {
        let (epsilon, mu) = match material {
            rec::Material::Constant(c) => (c.relative_permittivity, c.relative_permeability),
            rec::Material::Sellmeier(m) => {
                let l2 = wavelength * wavelength;
                let epsilon = m.sellmeier.constant
                    + m.sellmeier
                        .terms
                        .iter()
                        .map(|t| {
                            t.coefficient * l2 / (l2 - t.pole_wavelength_m * t.pole_wavelength_m)
                        })
                        .sum::<f64>();
                (epsilon, m.relative_permeability)
            }
        };
        if mu != 1. {
            return invalid(format!(
                "local coupling is for nonmagnetic stacks; region {region} has relative permeability {mu}"
            ));
        }
        if !(epsilon.is_finite() && epsilon > 0.) {
            return invalid(format!(
                "local coupling needs a positive permittivity; region {region} has {epsilon}"
            ));
        }
    }
    let gap = gap_rad_per_m.unwrap_or(GAP_RTOL * k0_m);
    if !(gap.is_finite() && gap >= 0.) {
        return invalid(format!(
            "gap_rad_per_m must be finite and non-negative, got {gap}"
        ));
    }
    let d = direction(solved, parameter)?;
    let coupling = lamina_mode::guided_local_coupling_with(
        &Scatter2::new(),
        &solved.basis,
        &d.perturbation,
        gap * M_PER_CM,
        recovery_settings(&record.input.request),
    )
    .map_err(|e| match e {
        GuidedCouplingError::Ineligible | GuidedCouplingError::Threshold => {
            Failure::Invalid(format!("local coupling: {e}"))
        }
        GuidedCouplingError::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
        GuidedCouplingError::Quadrature => Failure::Integration(e.to_string()),
    })?;
    // K is in cm^-1 per internal unit: to rad/m per SI unit.
    let matrix_scale = d.per_unit / M_PER_CM;
    let (perturbation_matrix_real, perturbation_matrix_imag) = split_matrix(
        coupling.perturbation_matrix.nrows(),
        |i, j| coupling.perturbation_matrix[(i, j)],
        matrix_scale,
    );
    let c = &coupling.coefficients;
    let part = |f: fn(Complex) -> f64| -> Vec<Vec<Option<f64>>> {
        (0..c.nrows())
            .map(|i| {
                (0..c.ncols())
                    .map(|j| c[(i, j)].map(|z| f(z * d.per_unit)))
                    .collect()
            })
            .collect()
    };
    Ok(rec::LocalCoupling {
        schema: rec::LOCAL_COUPLING_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: assumption(&record.input.stack).to_owned(),
        definition: rec::LOCAL_COUPLING_DEFINITION.to_owned(),
        input_hash: record.input_hash.clone(),
        parameter,
        parameter_unit: d.unit.to_owned(),
        gap_rad_per_m: gap,
        perturbation_matrix_real,
        perturbation_matrix_imag,
        coefficients_real: part(|z| z.re),
        coefficients_imag: part(|z| z.im),
        unresolved_groups: coupling.unresolved_groups,
        quadrature_change_rad_per_m: coupling.quadrature_change_per_cm * matrix_scale,
    })
}

/// Default coupling gap per unit k0.
const GAP_RTOL: f64 = 1e-6;

/// A supplied field's sample at one point: [Ex, Ey, Ez] in V/m and [Hx, Hy, Hz]
/// in A/m, or the error that stops the projection.
pub type Sample<'a> = dyn Fn(f64, &'static str) -> Result<[Complex; 6], SampleError> + 'a;

/// Why a supplied-field sample stopped the projection.
pub enum SampleError {
    /// A wrong shape or a non-finite value.
    Invalid(String),
    /// The caller's own failure, returned unchanged.
    Raised(pyo3::PyErr),
}

pub enum ProjectionFailure {
    Failure(Failure),
    Raised(pyo3::PyErr),
}

impl From<Failure> for ProjectionFailure {
    fn from(f: Failure) -> Self {
        Self::Failure(f)
    }
}

pub struct ProjectionRequest<'a> {
    pub support_m: [f64; 2],
    pub interfaces_m: &'a [f64],
    pub origin_m: f64,
    pub rank_rtol: f64,
    pub field_id: Option<String>,
}

/// Project a supplied field onto a complete basis. Samples are memoised by
/// position and side, so each point is evaluated once.
pub fn project(
    solved: &Solved,
    request: ProjectionRequest<'_>,
    field: &Sample<'_>,
) -> Result<rec::FieldProjection, ProjectionFailure> {
    let invalid = |m: String| Err(ProjectionFailure::Failure(Failure::Invalid(m)));
    let record = &solved.record;
    if record.completeness != rec::Completeness::Complete || solved.basis.modes.is_empty() {
        return invalid("projection needs a complete, nonempty basis".into());
    }
    let [lo, hi] = request.support_m;
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return invalid(format!(
            "support_m must be finite with low < high, got [{lo}, {hi}]"
        ));
    }
    if let Some(x) = request.interfaces_m.iter().find(|&&x| !(lo < x && x < hi)) {
        return invalid(format!(
            "interfaces_m must lie strictly inside the support, got {x}"
        ));
    }
    if !request.origin_m.is_finite() {
        return invalid(format!("origin_m must be finite, got {}", request.origin_m));
    }
    if !(0. < request.rank_rtol && request.rank_rtol < 1.) {
        return invalid(format!(
            "rank_rtol must lie in (0, 1), got {}",
            request.rank_rtol
        ));
    }
    // The positions the caller gave map back to their exact values.
    let given: HashMap<u64, f64> = [lo, hi]
        .iter()
        .chain(request.interfaces_m)
        .map(|&z| ((z / M_PER_CM).to_bits(), z))
        .collect();
    let memo: RefCell<HashMap<(u64, u8), [Complex; 6]>> = RefCell::default();
    let stopped: RefCell<Option<SampleError>> = RefCell::default();
    let nan = [Complex::new(f64::NAN, f64::NAN); 6];
    let sample = |z_cm: f64, side: ProjectionSide| -> [Complex; 6] {
        if stopped.borrow().is_some() {
            return nan;
        }
        let (code, name) = match side {
            ProjectionSide::Left => (0, "left"),
            ProjectionSide::Right => (1, "right"),
            ProjectionSide::Interior => (2, "interior"),
        };
        let key = (z_cm.to_bits(), code);
        if let Some(v) = memo.borrow().get(&key) {
            return *v;
        }
        let z_m = given.get(&key.0).copied().unwrap_or(z_cm * M_PER_CM);
        match field(z_m, name) {
            Ok(mut v) => {
                if v.iter().any(|c| !(c.re.is_finite() && c.im.is_finite())) {
                    *stopped.borrow_mut() = Some(SampleError::Invalid(format!(
                        "the supplied field is not finite at z = {z_m} m ({name})"
                    )));
                    return nan;
                }
                for h in &mut v[3..] {
                    *h *= Z0_OHM;
                }
                memo.borrow_mut().insert(key, v);
                v
            }
            Err(e) => {
                *stopped.borrow_mut() = Some(e);
                nan
            }
        }
    };
    let interfaces_cm: Vec<f64> = request.interfaces_m.iter().map(|z| z / M_PER_CM).collect();
    let input = GuidedProjectionInput {
        wavelength_cm: record.input.request.wavelength_m / M_PER_CM,
        support_cm: [lo / M_PER_CM, hi / M_PER_CM],
        interfaces_cm: &interfaces_cm,
        sample: &sample,
    };
    let result = lamina_mode::project_guided_field(
        &solved.basis,
        request.origin_m / M_PER_CM,
        input,
        request.rank_rtol,
        recovery_settings(&record.input.request),
    );
    if let Some(stop) = stopped.into_inner() {
        return Err(match stop {
            SampleError::Invalid(m) => ProjectionFailure::Failure(Failure::Invalid(m)),
            SampleError::Raised(e) => ProjectionFailure::Raised(e),
        });
    }
    let projection = result.map_err(|e| {
        ProjectionFailure::Failure(match e {
            GuidedProjectionError::InvalidInput | GuidedProjectionError::NonFiniteField => {
                Failure::Invalid(format!("projection: {e}"))
            }
            GuidedProjectionError::Overlap(e) => overlap_failure(e),
            GuidedProjectionError::Fields(e) => Failure::Fields(format!("{e}: {e:?}")),
            GuidedProjectionError::Quadrature => Failure::Integration(e.to_string()),
        })
    })?;
    let (gram_real, gram_imag) =
        split_matrix(projection.gram.nrows(), |i, j| projection.gram[(i, j)], 1.);
    let rhs = &projection.right_hand_side;
    let solution = projection.solution.as_ref();
    let coefficients = solution.map(|s| s.coefficients.iter().copied().collect::<Vec<_>>());
    Ok(rec::FieldProjection {
        schema: rec::PROJECTION_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        assumption: assumption(&record.input.stack).to_owned(),
        definition: rec::PROJECTION_DEFINITION.to_owned(),
        input_hash: record.input_hash.clone(),
        field_id: request.field_id,
        support_m: [lo + 0., hi + 0.],
        interfaces_m: request.interfaces_m.iter().map(|z| z + 0.).collect(),
        origin_m: request.origin_m + 0.,
        rank_rtol: request.rank_rtol,
        gram_real,
        gram_imag,
        right_hand_side_real: rhs.iter().map(|z| z.re).collect(),
        right_hand_side_imag: rhs.iter().map(|z| z.im).collect(),
        rank: projection.rank,
        singular_values_w_per_m: projection.singular_values.clone(),
        field_sample_count: memo.into_inner().len(),
        quadrature_change_w_per_m: projection.quadrature_change_w_per_m,
        coefficients_real: coefficients
            .as_ref()
            .map(|c| c.iter().map(|z| z.re).collect()),
        coefficients_imag: coefficients
            .as_ref()
            .map(|c| c.iter().map(|z| z.im).collect()),
        condition_number: solution.map(|s| s.condition_number),
        residual_w_per_m: solution.map(|s| s.residual_w_per_m),
    })
}
