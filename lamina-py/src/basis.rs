//! Complete modal bases in a β window and their pairings, in SI.
use crate::records::{self as rec, Rect2};
use crate::solve::{Failure, M_PER_CM, Z0_OHM, assumption, c2, finite_positive, normalise_stack};
use lamina_core::{Polarisation, Scatter2};
use lamina_mode::{
    BasisFieldError, BasisLimit, GuidedOverlapRegistration, ModalBasisError, ModalBasisRequest,
    ModalWindow, ModeFamily, Normalisation, PairingError, Rect, SignReference,
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
    let rec::Window::Wavenumber {
        re_rad_per_m,
        im_rad_per_m,
    } = &mut input.window;
    for (name, bounds) in [("re", re_rad_per_m), ("im", im_rad_per_m)] {
        let [lo, hi] = *bounds;
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(Failure::Invalid(format!(
                "window {name}_rad_per_m must be finite and ascending, got [{lo}, {hi}]"
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

fn rect(r: Rect<f64>) -> Rect2 {
    [
        [r.x_min / M_PER_CM, r.x_max / M_PER_CM],
        [r.y_min / M_PER_CM, r.y_max / M_PER_CM],
    ]
}

fn cause(cause: &BasisLimit) -> (&'static str, Option<String>) {
    match cause {
        BasisLimit::BoundaryRoot => ("boundary_root", None),
        BasisLimit::NonIntegerCount => ("non_integer_count", None),
        BasisLimit::Budget => ("budget", None),
        BasisLimit::Refinement(failure) => ("refinement", Some(failure.to_string())),
        BasisLimit::Multiplicity { count } => ("multiplicity", Some(format!("{count} roots"))),
        BasisLimit::Cutoff => ("cutoff", None),
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
        ModalBasisError::Quadrature => Failure::Integration(e.to_string()),
        ModalBasisError::Termination { .. } => Failure::Unsupported(e.to_string()),
        _ => Failure::Invalid(e.to_string()),
    }
}

pub fn solve(input: rec::ModalBasisInput) -> Result<SolvedBasis, Failure> {
    let input = normalise(input)?;
    let k0_cm = k0_rad_per_m(input.wavelength_m) * M_PER_CM;
    let rec::Window::Wavenumber {
        re_rad_per_m,
        im_rad_per_m,
    } = input.window;
    let window = ModalWindow::Wavenumber {
        re_cm: re_rad_per_m.map(|b| b * M_PER_CM),
        im_cm: im_rad_per_m.map(|b| b * M_PER_CM),
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
            rec::BasisMode {
                index,
                beta_rad_per_m: c2(beta / M_PER_CM),
                n_eff: c2(beta / k0_cm),
                mode_type: "closed".into(),
                direction: if forward {
                    rec::Direction::Forward
                } else {
                    rec::Direction::Backward
                },
                sheet: [None, None],
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
        chart: "wavenumber".into(),
        window_beta_image_rad_per_m: rect(basis.window_beta_image),
        scope: "discrete".into(),
        status: status(basis.status),
        expected: basis.expected,
        found: modes.len(),
        unresolved: basis
            .unresolved
            .iter()
            .map(|r| {
                let (cause, detail) = cause(&r.cause);
                rec::UnresolvedRegion {
                    bounds: rect(r.bounds),
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
    let region = z_m
        .iter()
        .map(|&z| {
            let outside = |termination: &rec::Termination| match termination {
                rec::Termination::Open => "exterior",
                // Bases refuse PML sides, so a closed side here is a wall.
                rec::Termination::Wall { .. } | rec::Termination::Pml { .. } => "gap",
            };
            if z < 0. {
                outside(&spec.left_termination)
            } else if z >= length {
                outside(&spec.right_termination)
            } else {
                "layer"
            }
            .to_owned()
        })
        .collect();
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
    })
}

fn pairing_failure(e: PairingError) -> Failure {
    match e {
        PairingError::Fields { .. } => Failure::Fields(format!("{e}: {e:?}")),
        PairingError::Quadrature => Failure::Integration(e.to_string()),
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
    let present = |v: Vec<Vec<f64>>| {
        v.into_iter()
            .map(|row| row.into_iter().map(Some).collect())
            .collect()
    };
    let (u, c) = (&result.unconjugated, &result.conjugated);
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
        conjugated_real: present(rows(&|i, j| c[(i, j)].re)),
        conjugated_imag: present(rows(&|i, j| c[(i, j)].im)),
        conjugated_absent: vec![vec![None; m]; n],
        singular_values_w_per_m: result.singular_values,
        leading_condition_numbers: result.leading_condition_numbers,
        self_defects: result.self_defects,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.quadrature_change_w_per_m,
        status: result.status.map(status),
        scope: ["discrete".into(), "discrete".into()],
    })
}
