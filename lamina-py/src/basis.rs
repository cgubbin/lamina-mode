//! Complete modal bases in a β window and their pairings, in SI.
use crate::records::{self as rec, Rect2};
use crate::solve::{Failure, M_PER_CM, Z0_OHM, assumption, c2, finite_positive, normalise_stack};
use lamina_core::{Polarisation, Scatter2};
use lamina_mode::{
    Absent, BasisFieldError, BasisLimit, BasisScope, ExteriorSheet, GuidedOverlapRegistration,
    ModalBasisError, ModalBasisRequest, ModalWindow, ModeFamily, Normalisation, PairingError, Rect,
    SignReference, WindowError,
};
use lamina_units::Length;
use std::f64::consts::TAU;

/// A solved basis with its record.
pub struct SolvedBasis {
    pub record: rec::ModalBasis,
    pub basis: lamina_mode::ModalBasis,
}

fn k0_rad_per_m(wavelength_m: f64) -> f64 {
    TAU / wavelength_m
}

/// Validate an input, resolve the default gap and normalise signed zeros,
/// so that equal inputs have one canonical form.
pub fn normalise(mut input: rec::ModalBasisInput) -> Result<rec::ModalBasisInput, Failure> {
    rec::check_identity(&input).map_err(Failure::Invalid)?;
    normalise_stack(&mut input.stack, input.wavelength_m)?;
    finite_positive("wavelength_m", input.wavelength_m)?;
    let (bounds, unit) = match &mut input.window {
        rec::Window::Wavenumber {
            re_rad_per_m,
            im_rad_per_m,
        } => ([re_rad_per_m, im_rad_per_m], "_rad_per_m"),
        rec::Window::Chart { re, im } => ([re, im], ""),
    };
    for (name, bounds) in ["re", "im"].into_iter().zip(bounds) {
        let [lo, hi] = *bounds;
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(Failure::Invalid(format!(
                "window {name}{unit} must be finite and ascending, got [{lo}, {hi}]"
            )));
        }
        *bounds = [lo + 0., hi + 0.];
    }
    let gap = input
        .degeneracy_gap_rad_per_m
        .unwrap_or(1e-6 * k0_rad_per_m(input.wavelength_m));
    if !(gap.is_finite() && gap >= 0.) {
        return Err(Failure::Invalid(format!(
            "degeneracy_gap_rad_per_m must be finite and non-negative, got {gap}"
        )));
    }
    input.degeneracy_gap_rad_per_m = Some(gap + 0.);
    let rtol = input.self_pairing_rtol;
    if !(rtol > 0. && rtol < 1.) {
        return Err(Failure::Invalid(format!(
            "self_pairing_rtol must lie in (0, 1), got {rtol}"
        )));
    }
    Ok(input)
}

/// A rectangle in cm^-1 as rad/m, or in the dimensionless z when `per_cm`
/// is one.
fn rect_in(r: Rect<f64>, per_cm: f64) -> Rect2 {
    [
        [r.x_min / per_cm, r.x_max / per_cm],
        [r.y_min / per_cm, r.y_max / per_cm],
    ]
}

fn rect(r: Rect<f64>) -> Rect2 {
    rect_in(r, M_PER_CM)
}

fn sheet(sheet: ExteriorSheet) -> &'static str {
    match sheet {
        ExteriorSheet::Proper => "proper",
        ExteriorSheet::Improper => "improper",
    }
}

fn scope(scope: BasisScope) -> String {
    match scope {
        BasisScope::Discrete => "discrete",
        _ => "discrete_without_continuum",
    }
    .into()
}

fn window_failure(e: WindowError) -> Failure {
    Failure::Invalid(match e {
        WindowError::Period => "a ChartWindow's imaginary extent must stay below 2π".into(),
        WindowError::ClosedStack => "a ChartWindow needs a stack with an open side".into(),
        _ => e.to_string(),
    })
}

fn cause(cause: &BasisLimit) -> (&'static str, Option<String>) {
    match cause {
        BasisLimit::BoundaryRoot => ("boundary_root", None),
        BasisLimit::NonIntegerCount => ("non_integer_count", None),
        BasisLimit::Budget => ("budget", None),
        BasisLimit::Refinement(failure) => ("refinement", Some(failure.to_string())),
        BasisLimit::Multiplicity { count } => ("multiplicity", Some(format!("{count} roots"))),
        BasisLimit::Cutoff => ("cutoff", None),
        BasisLimit::Scale {
            side,
            log_magnitude,
        } => (
            "scale",
            Some(format!("{side:?} PML scale exp({log_magnitude})")),
        ),
        BasisLimit::BranchPoint { side } => ("branch_point", Some(format!("{side:?}"))),
        _ => ("other", Some(format!("{cause:?}"))),
    }
}

fn status(status: lamina_mode::BasisStatus) -> rec::BasisStatus {
    match status {
        lamina_mode::BasisStatus::Complete => rec::BasisStatus::Complete,
        lamina_mode::BasisStatus::Incomplete => rec::BasisStatus::Incomplete,
        lamina_mode::BasisStatus::CountFailed => rec::BasisStatus::CountFailed,
    }
}

fn basis_failure(e: ModalBasisError) -> Failure {
    match e {
        ModalBasisError::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
        ModalBasisError::Quadrature | ModalBasisError::TailSingular { .. } => {
            Failure::Integration(e.to_string())
        }
        ModalBasisError::OpenSide => {
            Failure::Invalid("a stack with an open side needs a ChartWindow".into())
        }
        ModalBasisError::Window(e) => window_failure(e),
        _ => Failure::Invalid(e.to_string()),
    }
}

pub fn solve(input: rec::ModalBasisInput) -> Result<SolvedBasis, Failure> {
    let input = normalise(input)?;
    let k0_cm = k0_rad_per_m(input.wavelength_m) * M_PER_CM;
    let (window, per_cm) = match input.window {
        rec::Window::Wavenumber {
            re_rad_per_m,
            im_rad_per_m,
        } => (
            ModalWindow::Wavenumber {
                re_cm: re_rad_per_m.map(|b| b * M_PER_CM),
                im_cm: im_rad_per_m.map(|b| b * M_PER_CM),
            },
            M_PER_CM,
        ),
        rec::Window::Chart { re, im } => (ModalWindow::Chart { re, im }, 1.),
    };
    let polarisation = match input.polarisation {
        rec::Polarisation::Te => Polarisation::TransverseElectric,
        rec::Polarisation::Tm => Polarisation::TransverseMagnetic,
    };
    let mut request = ModalBasisRequest::new(
        Length::centimetres(input.wavelength_m / M_PER_CM),
        ModeFamily::Polarised(polarisation),
        window,
    );
    request.degeneracy_gap_cm = input.degeneracy_gap_rad_per_m.map(|g| g * M_PER_CM);
    request.self_pairing_rtol = input.self_pairing_rtol;
    let basis = lamina_mode::modal_basis_in_window_with(
        &Scatter2::new(),
        crate::solve::stack(&input.stack)?,
        request,
    )
    .map_err(basis_failure)?;
    let open_sides = [
        &input.stack.left_termination,
        &input.stack.right_termination,
    ]
    .iter()
    .filter(|t| matches!(t, rec::Termination::Open))
    .count();
    let modes = basis
        .modes
        .iter()
        .enumerate()
        .map(|(index, mode)| {
            let beta = mode.point.physical_coordinate();
            let (normalisation, ratio) = match mode.normalisation {
                Normalisation::Unit { ratio } => ("unit", ratio),
                Normalisation::IllConditioned { ratio } => ("ill_conditioned", ratio),
            };
            // Forward along +x: Re β > 0, or a decay along +x when Re β is zero.
            let forward = if beta.re.abs() > 1e-9 * beta.norm() {
                beta.re > 0.
            } else {
                beta.im > 0.
            };
            let mode_type = match mode.sheets {
                [None, None] => "closed",
                sheets if sheets.contains(&Some(ExteriorSheet::Improper)) => "leaky",
                _ => "guided",
            };
            rec::BasisMode {
                index,
                beta_rad_per_m: c2(beta / M_PER_CM),
                n_eff: c2(beta / k0_cm),
                mode_type: mode_type.into(),
                direction: if forward {
                    rec::Direction::Forward
                } else {
                    rec::Direction::Backward
                },
                sheet: mode.sheets.map(|s| s.map(|s| sheet(s).to_owned())),
                residual: mode.point.evidence().value.norm(),
                self_pairing_raw_w_per_m: c2(mode.self_pairing_raw_w_per_m),
                normalisation: normalisation.into(),
                self_pairing_ratio: ratio,
                sign_reference: match mode.sign_reference {
                    SignReference::FirstInterface => "first_interface",
                    SignReference::WallOnFirstInterface => "wall_on_first_interface",
                }
                .into(),
                sign_margin: mode.sign_margin,
                sign_ambiguous: mode.sign_ambiguous(),
            }
        })
        .collect::<Vec<_>>();
    let record = rec::ModalBasis {
        schema: rec::MODAL_BASIS_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        definitions: rec::Definitions::current(),
        assumption: assumption(&input.stack).to_owned(),
        input_hash: rec::digest(&input).map_err(Failure::Invalid)?,
        k0_rad_per_m: k0_rad_per_m(input.wavelength_m),
        chart: match open_sides {
            0 => "wavenumber",
            1 => "one_exterior",
            _ => "two_exterior",
        }
        .into(),
        window_beta_image_rad_per_m: rect(basis.window_beta_image),
        scope: scope(basis.scope),
        status: status(basis.status),
        expected: basis.expected,
        found: modes.len(),
        unresolved: basis
            .unresolved
            .iter()
            .map(|r| {
                let (cause, detail) = cause(&r.cause);
                rec::UnresolvedRegion {
                    bounds: rect_in(r.bounds, per_cm),
                    beta_image_rad_per_m: rect(r.beta_image),
                    count: r.count,
                    cause: cause.into(),
                    detail,
                }
            })
            .collect(),
        groups: basis.groups.clone(),
        modes,
        input,
    };
    Ok(SolvedBasis { record, basis })
}

/// Normalised fields of one mode at z (m) with the region of each sample.
pub fn profile(
    solved: &SolvedBasis,
    mode_index: usize,
    z_m: &[f64],
) -> Result<rec::ModalProfile, Failure> {
    let mode = solved.basis.modes.get(mode_index).ok_or_else(|| {
        Failure::Invalid(format!(
            "mode_index {mode_index} is out of range for {} modes",
            solved.basis.modes.len()
        ))
    })?;
    let spec = &solved.record.input.stack;
    let sampling = crate::solve::sampling(spec, z_m)?;
    let fields = mode
        .fields_with(Scatter2::new(), &sampling, solved.basis.request.recovery)
        .map_err(|e| match e {
            BasisFieldError::IllConditioned => {
                Failure::Invalid(format!("mode {mode_index} is ill-conditioned: {e}"))
            }
            BasisFieldError::Recovery(_) => Failure::Fields(format!("{e}: {e:?}")),
        })?;
    if fields.len() != z_m.len() {
        return Err(Failure::Fields(
            "sample count changed during recovery".into(),
        ));
    }
    let length = spec.layers.iter().fold(0., |t, l| t + l.thickness_m);
    // Each sample's region, and inside a PML its stretched coordinate z̃ and
    // stretch s(ζ) = 1 + c (ζ/d)^m at the outward depth ζ from the start.
    let sample = |z: f64| {
        let (termination, outward) = if z < 0. {
            (&spec.left_termination, -1.)
        } else if z >= length {
            (&spec.right_termination, 1.)
        } else {
            return ("layer", None);
        };
        match *termination {
            rec::Termination::Open => ("exterior", None),
            rec::Termination::Pml {
                start_z_m,
                thickness_m,
                profile: rec::StretchProfile::Polynomial { order, strength },
                ..
            } if (z - start_z_m) * outward > 0. => {
                let c = lamina_mode::Complex::new(strength[0], strength[1]);
                let depth = ((z - start_z_m) * outward).min(thickness_m);
                let fraction = depth / thickness_m;
                let stretched =
                    depth + c * (thickness_m / (order + 1.)) * fraction.powf(order + 1.);
                let stretch = 1. + c * fraction.powf(order);
                let z = start_z_m + outward * stretched;
                ("pml", Some(([z.re, z.im], [stretch.re, stretch.im])))
            }
            _ => ("gap", None),
        }
    };
    let samples: Vec<_> = z_m.iter().map(|&z| sample(z)).collect();
    let region = samples.iter().map(|(r, _)| (*r).to_owned()).collect();
    let has_pml = [&spec.left_termination, &spec.right_termination]
        .iter()
        .any(|t| matches!(t, rec::Termination::Pml { .. }));
    let pml = has_pml.then(|| rec::PmlSamples {
        fields: rec::PML_FIELDS.to_owned(),
        stretched_z_m: samples.iter().map(|(_, p)| p.map(|p| p.0)).collect(),
        stretch: samples.iter().map(|(_, p)| p.map(|p| p.1)).collect(),
    });
    let part = |range: std::ops::Range<usize>, scale: f64, imaginary: bool| {
        fields
            .iter()
            .map(|s| {
                let v = &s[range.clone()];
                [0, 1, 2].map(|c| scale * if imaginary { v[c].im } else { v[c].re })
            })
            .collect::<Vec<_>>()
    };
    Ok(rec::ModalProfile {
        schema: rec::MODAL_PROFILE_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        definitions: rec::Definitions::current(),
        input_hash: solved.record.input_hash.clone(),
        mode_index,
        beta_rad_per_m: solved.record.modes[mode_index].beta_rad_per_m,
        z_m: z_m.iter().map(|z| z + 0.).collect(),
        region,
        electric_real: part(0..3, 1., false),
        electric_imag: part(0..3, 1., true),
        // Core stores Z0*H in V/m; divide by Z0 for A/m.
        magnetic_real: part(3..6, 1. / Z0_OHM, false),
        magnetic_imag: part(3..6, 1. / Z0_OHM, true),
        pml,
    })
}

fn pairing_failure(e: PairingError) -> Failure {
    match e {
        PairingError::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
        PairingError::Quadrature | PairingError::TailSingular { .. } => {
            Failure::Integration(e.to_string())
        }
        _ => Failure::Invalid(e.to_string()),
    }
}

/// Both pairing matrices between two bases, registered by their first
/// interfaces at `origins_m`.
pub fn pairing(
    first: &SolvedBasis,
    second: &SolvedBasis,
    origins_m: [f64; 2],
) -> Result<rec::BasisPairing, Failure> {
    let registration = GuidedOverlapRegistration {
        origins_cm: origins_m.map(|o| o / M_PER_CM),
    };
    let result = lamina_mode::basis_pairing_with(
        &Scatter2::new(),
        [&first.basis, &second.basis],
        registration,
    )
    .map_err(pairing_failure)?;
    let (n, m) = result.conjugated.shape();
    let rows = |entry: &dyn Fn(usize, usize) -> f64| {
        (0..n)
            .map(|i| (0..m).map(|j| entry(i, j)).collect())
            .collect::<Vec<Vec<f64>>>()
    };
    let (u, c) = (&result.unconjugated, &result.conjugated);
    let conjugated = |part: fn(lamina_mode::Complex) -> f64| {
        (0..n)
            .map(|i| (0..m).map(|j| c[(i, j)].ok().map(part)).collect())
            .collect()
    };
    Ok(rec::BasisPairing {
        schema: rec::BASIS_PAIRING_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        definitions: rec::Definitions::current(),
        first_input_hash: first.record.input_hash.clone(),
        second_input_hash: second.record.input_hash.clone(),
        origins_m: origins_m.map(|o| o + 0.),
        rows: result.modes[0].clone(),
        columns: result.modes[1].clone(),
        unconjugated_real: rows(&|i, j| u[(i, j)].re),
        unconjugated_imag: rows(&|i, j| u[(i, j)].im),
        conjugated_real: conjugated(|v| v.re),
        conjugated_imag: conjugated(|v| v.im),
        conjugated_absent: (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| match c[(i, j)] {
                        Err(Absent::Unbounded { .. }) => Some("unbounded".to_owned()),
                        Ok(_) => None,
                    })
                    .collect()
            })
            .collect(),
        singular_values_w_per_m: result.singular_values,
        leading_condition_numbers: result.leading_condition_numbers,
        self_defects: result.self_defects,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.quadrature_change_w_per_m,
        status: result.status.map(status),
        scope: result.scope.map(scope),
    })
}

/// The chart window covering a β rectangle (rad/m) on stated sheets, one per
/// side ("proper", "improper", or none for a closed side), and the extra
/// area of its β image, rad²/m².
pub fn covering(
    mut spec: rec::Stack,
    wavelength_m: f64,
    beta_rect_rad_per_m: Rect2,
    sheets: [Option<String>; 2],
) -> Result<(Rect2, f64), Failure> {
    finite_positive("wavelength_m", wavelength_m)?;
    normalise_stack(&mut spec, wavelength_m)?;
    let sheets = sheets.map(|s| match s.as_deref() {
        None => Ok(None),
        Some("proper") => Ok(Some(ExteriorSheet::Proper)),
        Some("improper") => Ok(Some(ExteriorSheet::Improper)),
        Some(other) => Err(Failure::Invalid(format!(
            "a sheet is \"proper\", \"improper\" or None, got {other:?}"
        ))),
    });
    let [left, right] = sheets;
    let covering = ModalWindow::covering(
        crate::solve::stack(&spec)?,
        Length::centimetres(wavelength_m / M_PER_CM),
        beta_rect_rad_per_m[0].map(|b| b * M_PER_CM),
        beta_rect_rad_per_m[1].map(|b| b * M_PER_CM),
        [left?, right?],
    )
    .map_err(window_failure)?;
    let ModalWindow::Chart { re, im } = covering.window else {
        return Err(Failure::Invalid(
            "the covering is not a chart window".into(),
        ));
    };
    Ok(([re, im], covering.extra_area_cm2 / (M_PER_CM * M_PER_CM)))
}

/// Rows-first real or imaginary parts of an n × n matrix.
fn rows_of(
    n: usize,
    entry: &dyn Fn(usize, usize) -> lamina_mode::Complex,
    part: fn(lamina_mode::Complex) -> f64,
    scale: f64,
) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| part(entry(i, j) * scale)).collect())
        .collect()
}

fn to_m(values_cm: &[f64]) -> Vec<f64> {
    values_cm.iter().map(|z| z * M_PER_CM + 0.).collect()
}

/// A projection request on a modal basis, in SI.
pub struct BasisProjectionRequest<'a> {
    pub support_m: [f64; 2],
    pub interfaces_m: &'a [f64],
    pub origin_m: f64,
    pub rank_rtol: f64,
    pub field_id: Option<String>,
    /// Left and right continuations beyond the support.
    pub exterior: [Vec<rec::ExteriorWave>; 2],
}

/// Project a supplied field onto a modal basis. Samples are memoised by
/// position and side, so each point is evaluated once.
pub fn project(
    solved: &SolvedBasis,
    request: BasisProjectionRequest<'_>,
    field: &crate::pair::Sample<'_>,
) -> Result<rec::BasisProjection, crate::pair::ProjectionFailure> {
    use crate::pair::{Memo, ProjectionFailure};
    use lamina_mode::{
        BasisProjectionError, BasisProjectionInput, ExteriorWave, GuidedProjectionInput,
        ProjectionSide,
    };
    let invalid = |m: String| Err(ProjectionFailure::Failure(Failure::Invalid(m)));
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
    let spec = &solved.record.input.stack;
    for (side, waves) in request.exterior.iter().enumerate() {
        if waves.is_empty() {
            continue;
        }
        let termination = [&spec.left_termination, &spec.right_termination][side];
        if !termination.is_open() {
            return invalid(format!(
                "an exterior continuation needs an open side; the {} side is closed",
                ["left", "right"][side]
            ));
        }
    }
    let waves = request.exterior.clone().map(|side| {
        side.iter()
            .map(|w| {
                let c = |re: [f64; 3], im: [f64; 3], scale: f64| {
                    [0, 1, 2].map(|k| lamina_mode::Complex::new(re[k], im[k]) * scale)
                };
                let (e, h) = (
                    c(w.electric_real, w.electric_imag, 1.),
                    c(w.magnetic_real, w.magnetic_imag, Z0_OHM),
                );
                ExteriorWave {
                    kappa_cm: lamina_mode::Complex::new(w.kappa_rad_per_m[0], w.kappa_rad_per_m[1])
                        * M_PER_CM,
                    edge: [e[0], e[1], e[2], h[0], h[1], h[2]],
                }
            })
            .collect::<Vec<_>>()
    });
    let memo = Memo::new(field, &[lo, hi], request.interfaces_m);
    let sample = |z_cm: f64, side: ProjectionSide| memo.sample(z_cm, side);
    let interfaces_cm: Vec<f64> = request.interfaces_m.iter().map(|z| z / M_PER_CM).collect();
    let input = BasisProjectionInput {
        field: GuidedProjectionInput {
            wavelength_cm: solved.record.input.wavelength_m / M_PER_CM,
            support_cm: [lo / M_PER_CM, hi / M_PER_CM],
            interfaces_cm: &interfaces_cm,
            sample: &sample,
        },
        exterior: [&waves[0], &waves[1]],
    };
    let result = lamina_mode::basis_projection_with(
        &Scatter2::new(),
        &solved.basis,
        request.origin_m / M_PER_CM,
        input,
        request.rank_rtol,
    );
    let field_sample_count = memo.finish()?;
    let p = result.map_err(|e| {
        ProjectionFailure::Failure(match e {
            BasisProjectionError::InvalidInput => Failure::Invalid(format!(
                "projection: {e}; a support must stay within the walls, and a continuation \
                 needs its edge at or beyond the outermost interface"
            )),
            BasisProjectionError::NonFiniteField => Failure::Invalid(format!("projection: {e}")),
            BasisProjectionError::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
            _ => Failure::Integration(e.to_string()),
        })
    })?;
    let solution = p.solution.as_ref();
    let coefficients = solution.map(|s| s.coefficients.iter().copied().collect::<Vec<_>>());
    let rhs = &p.right_hand_side;
    let n = p.modes.len();
    let gram = |i: usize, j: usize| p.gram[(i, j)];
    Ok(rec::BasisProjection {
        schema: rec::BASIS_PROJECTION_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        definitions: rec::Definitions::current(),
        definition: rec::BASIS_PROJECTION_DEFINITION.to_owned(),
        assumption: assumption(spec).to_owned(),
        input_hash: solved.record.input_hash.clone(),
        field_id: request.field_id,
        support_m: [lo + 0., hi + 0.],
        interfaces_m: request.interfaces_m.iter().map(|z| z + 0.).collect(),
        origin_m: request.origin_m + 0.,
        exterior: request.exterior,
        rank_rtol: request.rank_rtol,
        modes: p.modes.clone(),
        gram_real: rows_of(n, &gram, |z| z.re, 1.),
        gram_imag: rows_of(n, &gram, |z| z.im, 1.),
        right_hand_side_real: rhs.iter().map(|z| z.re).collect(),
        right_hand_side_imag: rhs.iter().map(|z| z.im).collect(),
        rank: p.rank,
        singular_values_w_per_m: p.singular_values.clone(),
        gram_defect: p.gram_defect,
        partition_m: to_m(&p.partition_cm),
        field_sample_count,
        sample_count: p.sample_count,
        quadrature_change_w_per_m: p.quadrature_change_w_per_m,
        coefficients_real: coefficients
            .as_ref()
            .map(|c| c.iter().map(|z| z.re).collect()),
        coefficients_imag: coefficients
            .as_ref()
            .map(|c| c.iter().map(|z| z.im).collect()),
        condition_number: solution.map(|s| s.condition_number),
        equation_residual_w_per_m: solution.map(|s| s.equation_residual_w_per_m),
        field_scale_w_per_m: p.field_scale_w_per_m,
        residual_scale_w_per_m: solution.map(|s| s.residual_scale_w_per_m),
        relative_residual: solution.and_then(|s| s.relative_residual),
        residual_includes_continuum: p.residual_includes_continuum,
        status: status(p.status),
        scope: scope(p.scope),
    })
}

/// The local coupling of a modal basis along one fixed-wavelength parameter.
pub fn local_coupling(
    solved: &SolvedBasis,
    parameter: rec::Parameter,
    gap_rad_per_m: Option<f64>,
) -> Result<rec::BasisLocalCoupling, Failure> {
    use lamina_mode::BasisCouplingError as E;
    let invalid = |m: String| Err(Failure::Invalid(m));
    match parameter {
        rec::Parameter::Wavelength => {
            return invalid(
                "local coupling needs a fixed-wavelength parameter, not the wavelength".into(),
            );
        }
        rec::Parameter::Permeability { .. } => {
            return invalid(
                "local coupling takes permittivity rates only; a permeability parameter is \
                 ineligible"
                    .into(),
            );
        }
        _ => {}
    }
    if let Some(gap) = gap_rad_per_m {
        if !(gap.is_finite() && gap >= 0.) {
            return invalid(format!(
                "gap_rad_per_m must be finite and non-negative, got {gap}"
            ));
        }
    }
    let input = &solved.record.input;
    let k0_cm = k0_rad_per_m(input.wavelength_m) * M_PER_CM;
    let d = crate::sens::direction_for(&input.stack, k0_cm, parameter)?;
    let coupling = lamina_mode::basis_local_coupling_with(
        &Scatter2::new(),
        &solved.basis,
        &d.perturbation,
        gap_rad_per_m.map(|g| g * M_PER_CM),
    )
    .map_err(|e| match e {
        E::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
        E::Quadrature | E::TailSingular { .. } => Failure::Integration(e.to_string()),
        _ => Failure::Invalid(format!("local coupling: {e}")),
    })?;
    // K is in cm^-1 per internal unit: to rad/m per SI unit.
    let matrix_scale = d.per_unit / M_PER_CM;
    let n = coupling.modes.len();
    let k = |i: usize, j: usize| coupling.perturbation_matrix[(i, j)];
    let c = &coupling.coefficients;
    let part = |f: fn(lamina_mode::Complex) -> f64| -> Vec<Vec<Option<f64>>> {
        (0..c.nrows())
            .map(|i| {
                (0..c.ncols())
                    .map(|j| c[(i, j)].map(|z| f(z * d.per_unit)))
                    .collect()
            })
            .collect()
    };
    Ok(rec::BasisLocalCoupling {
        schema: rec::BASIS_LOCAL_COUPLING_SCHEMA.to_owned(),
        schema_version: rec::SCHEMA_VERSION,
        conventions: rec::Conventions::current(),
        definitions: rec::Definitions::current(),
        definition: rec::BASIS_LOCAL_COUPLING_DEFINITION.to_owned(),
        assumption: assumption(&input.stack).to_owned(),
        input_hash: solved.record.input_hash.clone(),
        parameter,
        parameter_unit: d.unit.to_owned(),
        gap_rad_per_m: coupling.gap_cm / M_PER_CM + 0.,
        modes: coupling.modes.clone(),
        perturbation_matrix_real: rows_of(n, &k, |z| z.re, matrix_scale),
        perturbation_matrix_imag: rows_of(n, &k, |z| z.im, matrix_scale),
        coefficients_real: part(|z| z.re),
        coefficients_imag: part(|z| z.im),
        groups: coupling.groups.clone(),
        partition_m: to_m(&coupling.partition_cm),
        sample_count: coupling.sample_count,
        quadrature_change_rad_per_m: coupling.quadrature_change_per_cm * matrix_scale,
        status: status(coupling.status),
        scope: scope(coupling.scope),
    })
}
