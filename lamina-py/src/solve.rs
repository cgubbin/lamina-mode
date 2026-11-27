//! Adapter between the SI records and lamina-mode's internal cm^-1 units.
use crate::records::{self as rec, C2};
use lamina_core::{
    ConstitutiveClass, ExteriorSampling, FieldSampling, LayerSampling, Pml, Polarisation, Scatter2,
    Scatter4, StretchProfile, Termination, Wall, WallKind, stack::StackBuilder,
};
use lamina_mode::{
    AnalyticAssumption, Complex, FieldRecoverySettings, GuidedDirection, GuidedLimit, GuidedWindow,
    HybridPhaseReference, ImplicitSlopeRequest, ImplicitSlopeSettings, LocalAnalyticityAssumption,
    LocalDerivativeSettings, ModeFamily, ModeLabel, ModeStack, PhysicalParameter, RefinementConfig,
};
use lamina_units::{Length, LengthUnit};
use std::f64::consts::TAU;

/// Metres per centimetre.
pub(crate) const M_PER_CM: f64 = 0.01;
/// Impedance of free space, as lamina-mode uses it for guided power.
pub(crate) const Z0_OHM: f64 = 376.730313412;

/// The declaration for stacks of constant models and tensors. These are
/// entire functions of frequency and wavenumber, so the analytic
/// declarations that lamina-mode requires hold by construction.
pub const ASSUMPTION: &str = "constant materials: holomorphic and pole-free in k0 and beta, \
     regular in layer thickness (declared by the bindings)";

/// The declaration for stacks with a dispersive model: its poles are
/// checked to lie away from the operating wavelength, so the materials are
/// holomorphic in a neighbourhood of the operating k0 and entire in beta.
pub const DISPERSIVE_ASSUMPTION: &str = "constant, Sellmeier, Lorentz or Drude-Lorentz materials \
     with every real pole away from the operating wavelength: holomorphic near the operating k0, \
     pole-free in beta, regular in layer thickness (declared by the bindings)";

/// The declaration for stacks with a tabulated model: it is held at its
/// interpolated value at the operating wavelength, so no derivative in the
/// wavelength is available.
pub const TABULATED_ASSUMPTION: &str = "tabulated materials held constant at their interpolated \
     value at the operating wavelength, other materials as declared for dispersive models: \
     holomorphic and pole-free in beta at the operating k0, regular in layer thickness; no \
     wavelength dispersion of the tabulated materials (declared by the bindings)";

/// Minimum relative separation of a real pole from the operating wavelength.
const POLE_SEPARATION: f64 = 1e-6;

pub fn regions(spec: &rec::Stack) -> impl Iterator<Item = &rec::Material> {
    std::iter::once(&spec.left)
        .chain(spec.layers.iter().map(|l| &l.material))
        .chain(std::iter::once(&spec.right))
}

/// The scalar models of a region material.
pub fn models(m: &rec::Material) -> impl Iterator<Item = &rec::ScalarModel> {
    crate::materials::models(m).into_iter().map(|(_, m)| m)
}

/// `true` when a stack holds a tabulated model.
pub fn tabulated(spec: &rec::Stack) -> bool {
    regions(spec)
        .flat_map(models)
        .any(|m| matches!(m, rec::ScalarModel::Tabulated { .. }))
}

pub fn assumption(spec: &rec::Stack) -> &'static str {
    if tabulated(spec) {
        TABULATED_ASSUMPTION
    } else if regions(spec)
        .flat_map(models)
        .all(|m| matches!(m, rec::ScalarModel::Constant { .. }))
    {
        ASSUMPTION
    } else {
        DISPERSIVE_ASSUMPTION
    }
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

/// Validate one scalar model at `wavelength`.
fn scalar_model(name: &str, m: &rec::ScalarModel, wavelength: f64) -> Result<(), Failure> {
    match m {
        rec::ScalarModel::Constant {
            relative_permittivity,
            relative_permeability,
        } => {
            finite_positive(
                &format!("{name} relative permittivity"),
                *relative_permittivity,
            )?;
            finite_positive(
                &format!("{name} relative permeability"),
                *relative_permeability,
            )?;
        }
        rec::ScalarModel::Sellmeier {
            sellmeier,
            relative_permeability,
        } => {
            if *relative_permeability != 1. {
                return Err(Failure::Invalid(format!(
                    "{name}: Sellmeier models support relative_permeability 1 only"
                )));
            }
            if !sellmeier.constant.is_finite() {
                return Err(Failure::Invalid(format!(
                    "{name} Sellmeier constant must be finite"
                )));
            }
            for (j, t) in sellmeier.terms.iter().enumerate() {
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
        rec::ScalarModel::Lorentz {
            epsilon_infinity,
            oscillators,
            relative_permeability,
        }
        | rec::ScalarModel::DrudeLorentz {
            epsilon_infinity,
            oscillators,
            relative_permeability,
            ..
        } => {
            if *relative_permeability != 1. {
                return Err(Failure::Invalid(format!(
                    "{name}: Lorentz and Drude-Lorentz models support relative_permeability 1 only"
                )));
            }
            if !epsilon_infinity.is_finite() {
                return Err(Failure::Invalid(format!(
                    "{name} epsilon_infinity must be finite"
                )));
            }
            if let rec::ScalarModel::DrudeLorentz {
                plasma_rad_per_m,
                damping_rad_per_m,
                ..
            } = m
            {
                for (field, v) in [
                    ("plasma_rad_per_m", plasma_rad_per_m),
                    ("damping_rad_per_m", damping_rad_per_m),
                ] {
                    if !(v.is_finite() && *v >= 0.) {
                        return Err(Failure::Invalid(format!(
                            "{name} {field} must be finite and non-negative, got {v}"
                        )));
                    }
                }
            }
            let k0 = std::f64::consts::TAU / wavelength;
            for (j, o) in oscillators.iter().enumerate() {
                let values = [o.delta_epsilon, o.resonance_rad_per_m, o.damping_rad_per_m];
                if !values.iter().all(|v| v.is_finite() && *v >= 0.) || o.resonance_rad_per_m == 0.
                {
                    return Err(Failure::Invalid(format!(
                        "{name} oscillator {j} needs finite delta_epsilon >= 0, resonance > 0 \
                         and damping >= 0"
                    )));
                }
                if o.damping_rad_per_m == 0.
                    && (o.resonance_rad_per_m - k0).abs() <= POLE_SEPARATION * k0
                {
                    return Err(Failure::Invalid(format!(
                        "{name} oscillator {j} has its pole at the operating wavelength"
                    )));
                }
            }
        }
        rec::ScalarModel::Tabulated {
            wavelength_m,
            relative_permeability,
            ..
        } => {
            finite_positive(
                &format!("{name} relative permeability"),
                *relative_permeability,
            )?;
            if !wavelength_m.iter().all(|l| l.is_finite() && *l > 0.)
                || wavelength_m.windows(2).any(|w| w[1] <= w[0])
            {
                return Err(Failure::Invalid(format!(
                    "{name}: tabulated wavelength_m must be finite, positive and ascending"
                )));
            }
        }
    }
    crate::materials::model(name, m, wavelength).map(|_| ())
}

/// Validate a stack at `wavelength` and normalise the signed zeros of its
/// wall and PML fields.
pub(crate) fn normalise_stack(stack: &mut rec::Stack, wavelength: f64) -> Result<(), Failure> {
    let material = |name: &str, m: &mut rec::Material| -> Result<(), Failure> {
        for (axis, model) in crate::materials::models(m) {
            let name = match axis {
                Some(axis) => format!("{name} axis {}", axis.name()),
                None => name.to_owned(),
            };
            scalar_model(&name, model, wavelength)?;
        }
        if let rec::Material::Full { material, rotation } = m {
            // Adding +0.0 maps -0.0 to +0.0, so equal matrices hash equally.
            *rotation = rotation.map(|row| row.map(|v| v + 0.));
            crate::materials::rotation(name, rotation)?;
            if let rec::TensorMaterial::Tensor {
                relative_permittivity,
                relative_permeability,
            } = material
            {
                for t in [relative_permittivity, relative_permeability] {
                    *t = t.map(|row| row.map(|[re, im]| [re + 0., im + 0.]));
                }
            }
        }
        crate::materials::region(name, m, wavelength).map(|_| ())
    };
    material("left exterior", &mut stack.left)?;
    material("right exterior", &mut stack.right)?;
    for (i, layer) in stack.layers.iter_mut().enumerate() {
        material(&format!("layer {i}"), &mut layer.material)?;
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

/// The lamina-core stack of a record at the operating wavelength (m).
pub(crate) fn stack(spec: &rec::Stack, wavelength_m: f64) -> Result<ModeStack, Failure> {
    let region = |name: &str, m| crate::materials::region(name, m, wavelength_m);
    let mut builder = StackBuilder::new(
        region("left exterior", &spec.left)?,
        region("right exterior", &spec.right)?,
    );
    for (i, layer) in spec.layers.iter().enumerate() {
        builder = builder.layer(
            region(&format!("layer {i}"), &layer.material)?,
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
        family: family(r.family),
        direction: match r.direction {
            rec::Direction::Forward => GuidedDirection::Forward,
            rec::Direction::Backward => GuidedDirection::Backward,
        },
        abs_beta: r.abs_beta_rad_per_m.map(|b| b * M_PER_CM),
        cutoff_margin: r.cutoff_margin_rad_per_m * M_PER_CM,
        phase_tolerance: r.phase_tolerance,
        max_subdivisions: r.max_subdivisions,
        refinement: RefinementConfig {
            residual_atol: Some(f.residual_atol),
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

/// lamina-mode's family of a request.
pub(crate) fn family(f: rec::Family) -> ModeFamily {
    match f {
        rec::Family::Te => Polarisation::TransverseElectric.into(),
        rec::Family::Tm => Polarisation::TransverseMagnetic.into(),
        rec::Family::All => ModeFamily::All,
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

impl Solved {
    /// `true` when the modes were counted by the argument principle and
    /// solved through a 4×4 backend.
    pub(crate) fn coupled(&self) -> bool {
        self.basis.hybrid.is_some()
    }
}

/// Modes whose |β| differ by no more than this, per unit k0, share a
/// degenerate group.
pub(crate) const DEGENERACY_RTOL: f64 = 1e-6;

/// A rectangle in cm^-1 as rad/m.
pub(crate) fn rect_m(r: lamina_mode::Rect<f64>) -> rec::Rect2 {
    [
        [r.x_min / M_PER_CM, r.x_max / M_PER_CM],
        [r.y_min / M_PER_CM, r.y_max / M_PER_CM],
    ]
}

fn gap(g: &lamina_mode::GuidedGap) -> rec::Gap {
    let (cause, detail) = match &g.cause {
        GuidedLimit::EndpointPhase => ("endpoint_phase", None),
        GuidedLimit::PhaseResolution => ("phase_resolution", None),
        GuidedLimit::Budget => ("budget", None),
        GuidedLimit::Correction(failure) => ("correction", Some(format!("{failure:?}"))),
        GuidedLimit::OutsideBracket => ("outside_bracket", None),
        GuidedLimit::CountUnknown => ("count_unknown", None),
        GuidedLimit::Region(cause) => ("region", Some(format!("{cause:?}"))),
        GuidedLimit::NonRealRoot { beta } => (
            "non_real_root",
            Some(format!(
                "beta = [{}, {}] rad/m",
                beta.re / M_PER_CM,
                beta.im / M_PER_CM
            )),
        ),
        GuidedLimit::NearDegenerate { count, .. } => {
            ("near_degenerate", Some(format!("{count} roots")))
        }
        GuidedLimit::Cell(limit) => ("cell", Some(format!("{limit:?}"))),
        GuidedLimit::SeparableCount { counts } => (
            "separable_count",
            Some(format!("TE and TM counts {counts:?}")),
        ),
    };
    rec::Gap {
        abs_beta_rad_per_m: g.abs_beta.map(|b| b / M_PER_CM),
        expected: g.expected,
        cause: cause.to_owned(),
        detail,
    }
}

fn label(l: ModeLabel) -> rec::Label {
    match l {
        ModeLabel::TransverseElectric => rec::Label::Te,
        ModeLabel::TransverseMagnetic => rec::Label::Tm,
        ModeLabel::Hybrid => rec::Label::Hybrid,
    }
}

/// The last segment of a backend's type name.
fn backend_name(name: &str) -> String {
    name.rsplit("::").next().unwrap_or(name).to_owned()
}

fn count_evidence(e: &lamina_mode::HybridCountEvidence) -> rec::CountEvidence {
    rec::CountEvidence {
        channel: match e.channel.polarisation() {
            Some(Polarisation::TransverseElectric) => "TE",
            Some(Polarisation::TransverseMagnetic) => "TM",
            None => "coupled",
        }
        .to_owned(),
        rectangles_rad_per_m: e.rectangles.iter().map(|&r| rect_m(r)).collect(),
        count_backends: e.count_backends.iter().map(|n| backend_name(n)).collect(),
        region: e.region.as_ref().map(|r| rec::CountRegion {
            half_height_rad_per_m: r.half_height / M_PER_CM,
            exterior_separation: r.exterior_separation,
            pair_margin: r.pair_margin,
            gauge_conditioning: r.gauge_conditioning,
            samples: r.samples,
        }),
        endpoint_shift_rad_per_m: e.endpoint_shift.map(|s| s / M_PER_CM),
        off_family: e.off_family.clone(),
        separable_counts: e.separable_counts,
    }
}

/// Group indices of modes, in order, whose |β| lie within `gap` of a
/// neighbour; `None` for a mode alone.
fn degenerate_groups(beta: &[f64], gap: f64) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = (0..beta.len()).collect();
    order.sort_by(|&a, &b| beta[a].total_cmp(&beta[b]));
    let mut groups = vec![None; beta.len()];
    let mut next = 0;
    for pair in order.windows(2) {
        let [a, b] = [pair[0], pair[1]];
        if (beta[b] - beta[a]).abs() <= gap {
            let group = *groups[a].get_or_insert_with(|| {
                next += 1;
                next - 1
            });
            groups[b] = Some(group);
        }
    }
    // Number groups by their first mode in basis order.
    let mut renumber = vec![None; next];
    let mut count = 0;
    for g in groups.iter_mut().flatten() {
        *g = *renumber[*g].get_or_insert_with(|| {
            count += 1;
            count - 1
        });
    }
    groups
}

fn enumeration_failure(e: lamina_mode::EnumerationError) -> Failure {
    use lamina_core::{ConstitutiveError, StackRegion};
    use lamina_mode::EnumerationError as E;
    match e {
        E::Constitutive(ConstitutiveError::Unsupported {
            region: region @ (StackRegion::LeftExterior | StackRegion::RightExterior),
            ..
        }) => Failure::Invalid(format!(
            "anisotropic open exteriors are not supported yet: the {} exterior couples TE and \
             TM (an ε or μ entry xy, yx, yz or zy); an open exterior must be isotropic, \
             diagonal or have its principal axes in the xz plane and along y",
            if region == StackRegion::LeftExterior {
                "left"
            } else {
                "right"
            }
        )),
        E::Termination { .. } => Failure::Unsupported(e.to_string()),
        E::FamilyNotInvariant { .. } | E::Tensor { .. } | E::InvalidRequest => {
            Failure::Invalid(e.to_string())
        }
        e => Failure::Enumeration(e.to_string()),
    }
}

/// Each region as solved: its class and separability as lamina-core derives
/// them, and its stack-frame tensors at the operating wavelength.
pub(crate) fn solved_regions(
    spec: &rec::Stack,
    solved: &ModeStack,
    wavelength_m: f64,
) -> Result<Vec<rec::SolvedRegion>, Failure> {
    let core = std::iter::once(solved.left_exterior())
        .chain(solved.iter().map(|layer| layer.material()))
        .chain(std::iter::once(solved.right_exterior()));
    regions(spec)
        .zip(core)
        .enumerate()
        .map(|(i, (material, region))| {
            let [epsilon, mu] =
                crate::materials::stack_frame(&format!("region {i}"), material, wavelength_m)?;
            let pairs = |t: [[Complex; 3]; 3]| t.map(|row| row.map(c2));
            Ok(rec::SolvedRegion {
                class: match region.class() {
                    ConstitutiveClass::Isotropic => "isotropic",
                    ConstitutiveClass::Diagonal => "diagonal",
                    ConstitutiveClass::Full => "full",
                }
                .to_owned(),
                separable: region.is_separable(),
                relative_permittivity: pairs(epsilon),
                relative_permeability: pairs(mu),
            })
        })
        .collect()
}

pub fn enumerate(input: rec::GuidedInput) -> Result<Solved, Failure> {
    let input = normalise(input)?;
    let modes_stack = stack(&input.stack, input.request.wavelength_m)?;
    let regions = solved_regions(&input.stack, &modes_stack, input.request.wavelength_m)?;
    if input.request.family != rec::Family::All
        && let Some(i) = regions.iter().position(|r| !r.separable)
    {
        return Err(Failure::Invalid(format!(
            "region {i} couples TE and TM: request family \"all\""
        )));
    }
    let basis = lamina_mode::enumerate_guided_modes(modes_stack, window(&input.request))
        .map_err(enumeration_failure)?;
    let k0_cm = k0_cm(&input.request);
    let gap_m = DEGENERACY_RTOL * k0_cm / M_PER_CM;
    let groups = degenerate_groups(
        &basis
            .modes
            .iter()
            .map(|p| p.physical_coordinate().re.abs() / M_PER_CM)
            .collect::<Vec<_>>(),
        gap_m,
    );
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
                label: basis
                    .labels
                    .get(index)
                    .copied()
                    .map_or(rec::Label::Hybrid, label),
                degenerate_group: groups[index],
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
        count_method: if basis.hybrid.is_some() {
            "argument_principle"
        } else {
            "oscillation"
        }
        .to_owned(),
        count_evidence: basis.hybrid.as_ref().map(count_evidence),
        degeneracy_gap_rad_per_m: gap_m,
        regions,
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

/// The phase references a profile names.
pub(crate) const EY_PLUS_Z0_HY: &str = "Ey+Z0Hy";
pub(crate) const EX_PLUS_Z0_HX: &str = "Ex+Z0Hx";

/// Refuse an operation on modes counted by the argument principle.
pub(crate) fn scalar_only(solved: &Solved, what: &str) -> Result<(), Failure> {
    if solved.coupled() {
        return Err(Failure::Invalid(format!(
            "{what} needs TE or TM modes of a stack of isotropic and diagonal regions"
        )));
    }
    Ok(())
}

pub fn profile(
    solved: &Solved,
    mode_index: usize,
    z_m: &[f64],
) -> Result<rec::GuidedProfile, Failure> {
    let point = point(solved, mode_index)?;
    let sampling = sampling(&solved.record.input.stack, z_m)?;
    let (fields, normalisation, defect) = if solved.coupled() {
        let profile = point
            .hybrid_power_profile_with(&Scatter4::new(), &sampling)
            .map_err(|e| Failure::Fields(format!("{e}: {e:?}")))?;
        let normalisation = rec::PowerNormalisation {
            signed_power_w_per_m: profile.signed_power_w_per_m,
            scale: c2(profile.scale),
            phase_reference: c2(profile.reference),
            phase_reference_component: match profile.phase_reference {
                HybridPhaseReference::EyPlusZ0Hy => EY_PLUS_Z0_HY,
                HybridPhaseReference::ExPlusZ0Hx => EX_PLUS_Z0_HX,
            }
            .to_owned(),
            bilinear_norm_w_per_m: Some(c2(profile.bilinear_norm_w_per_m)),
            seed_power_w_per_m: profile.seed_power_w_per_m,
            seed_exterior_power_w_per_m: None,
        };
        (profile.fields, normalisation, None)
    } else {
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
                phase_reference_component: EY_PLUS_Z0_HY.to_owned(),
                bilinear_norm_w_per_m: None,
                seed_power_w_per_m: n.seed_power_w_per_m,
                seed_exterior_power_w_per_m: Some(n.seed_exterior_power_w_per_m),
            },
            _ => return Err(Failure::Fields("profile is not power normalised".into())),
        };
        if profile.depths_cm.len() != z_m.len() {
            return Err(Failure::Fields(
                "sample count changed during recovery".into(),
            ));
        }
        (
            profile.fields,
            normalisation,
            Some(profile.relative_boundary_defect),
        )
    };
    let fields = fields.quantity();
    if fields.electric().x().len() != z_m.len() {
        return Err(Failure::Fields(
            "sample count changed during recovery".into(),
        ));
    }
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
        relative_boundary_defect: defect,
    })
}

pub fn derivatives(
    solved: &Solved,
    mode_index: usize,
    layers: &[usize],
) -> Result<rec::LocalDerivatives, Failure> {
    let point = point(solved, mode_index)?;
    scalar_only(solved, "local derivatives")?;
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
    let analyticity =
        LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness;
    let settings = LocalDerivativeSettings::default();
    // The characteristic of the backend the modes were counted with.
    let report = point
        .local_derivatives_with(Scatter2::new(), &parameters, analyticity, settings)
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
