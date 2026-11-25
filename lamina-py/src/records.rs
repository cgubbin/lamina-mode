//! The serialised form of inputs and results: schema version 1.
//!
//! Every record names its schema and version and carries the conventions it
//! was produced under. Serialisation is JSON with fields in declaration order,
//! no whitespace and shortest round-trip floats, so equal records serialise to
//! equal bytes and their SHA-256 digests can key caches.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;

pub const INPUT_SCHEMA: &str = "lamina.guided_input";
pub const BASIS_SCHEMA: &str = "lamina.guided_basis";
pub const PROFILE_SCHEMA: &str = "lamina.guided_profile";
pub const DERIVATIVES_SCHEMA: &str = "lamina.local_derivatives";
pub const SENSITIVITY_SCHEMA: &str = "lamina.guided_sensitivity";
pub const GROUP_INDEX_SCHEMA: &str = "lamina.group_index";
pub const POWER_OVERLAP_SCHEMA: &str = "lamina.guided_power_overlap";
pub const OVERLAP_SENSITIVITY_SCHEMA: &str = "lamina.guided_overlap_sensitivity";
pub const RECIPROCITY_SCHEMA: &str = "lamina.guided_reciprocity";
pub const LOCAL_COUPLING_SCHEMA: &str = "lamina.guided_local_coupling";
pub const PROJECTION_SCHEMA: &str = "lamina.guided_projection";
pub const MODAL_BASIS_INPUT_SCHEMA: &str = "lamina.modal_basis_input";
pub const MODAL_BASIS_SCHEMA: &str = "lamina.modal_basis";
pub const MODAL_PROFILE_SCHEMA: &str = "lamina.modal_profile";
pub const BASIS_PAIRING_SCHEMA: &str = "lamina.basis_pairing";

/// The pairing each record kind evaluates. A record's `definition` must equal
/// its kind's, as its conventions must equal the current ones.
pub const POWER_OVERLAP_DEFINITION: &str =
    "(1/4) ∫ (E1 × H2* + E2* × H1) · x dz, linear in the first mode";
pub const OVERLAP_SENSITIVITY_DEFINITION: &str = "d/dp of (1/4) ∫ (E1 × H2* + E2* × H1) · x dz, \
     linear in the first mode, each mode along its own parameter direction";
pub const RECIPROCITY_DEFINITION: &str = "∫ (E1 × H2 − E2 × H1) · x dz, unconjugated";
pub const LOCAL_COUPLING_DEFINITION: &str = "K_ij = (ω ε0 / 4) [∫ (∂ε/∂p) E_i* · E_j dz \
     + Σ_k v_k ((ε_k− − ε_k+) E_i,t* · E_j,t − (1/ε_k− − 1/ε_k+) D_i,z* D_j,z / ε0²)], \
     v_k the velocity of interface k; C_ij = s K_ij / (β_j − β_i), s = +1 forward and −1 backward; \
     row i the receiving mode, column j the differentiated mode";
pub const PROJECTION_DEFINITION: &str = "G c = b with G_ij = (1/4) ∫ (E_j × H_i* + E_i* × H_j) · x dz \
     and b_i = (1/4) ∫ (E × H_i* + E_i* × H) · x dz over the support, E and H the supplied field";

/// A complex number as `[real, imaginary]`.
pub type C2 = [f64; 2];

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Conventions {
    pub length: String,
    pub wavenumber: String,
    pub electric_field: String,
    pub magnetic_field: String,
    pub power: String,
    pub stack_normal: String,
    pub propagation_axis: String,
    pub invariant_axis: String,
    pub origin: String,
    pub time_convention: String,
    pub polarisation: String,
    pub normalisation: String,
    pub interface_samples: String,
}

impl Conventions {
    pub fn current() -> Self {
        let s = |v: &str| v.to_owned();
        Self {
            length: s("m"),
            wavenumber: s("rad/m"),
            electric_field: s("V/m"),
            magnetic_field: s("A/m"),
            power: s("W per m of invariant width"),
            stack_normal: s("z"),
            propagation_axis: s("x"),
            invariant_axis: s("y"),
            origin: s("z = 0 at the first interface; layers run towards +z"),
            time_convention: s("exp(+i*beta*x - i*omega*t)"),
            polarisation: s("relative to the stack normal: TE has Ey, Hx, Hz; TM has Hy, Ex, Ez"),
            normalisation: s("unit signed longitudinal power per m of invariant width; \
                 left-interface Ey (TE) or Z0*Hy (TM) real and positive"),
            interface_samples: s("a sample on an interface takes the limit from +z"),
        }
    }
}

/// A layer or exterior material. Untagged, with the constant material first,
/// so records written before the Sellmeier variant existed keep their form
/// and hash (docs/decisions/0005).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Material {
    Constant(ConstantMaterial),
    Sellmeier(SellmeierMaterial),
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConstantMaterial {
    pub relative_permittivity: f64,
    pub relative_permeability: f64,
}

/// ε(λ) = constant + Σ Bⱼ λ² / (λ² − λⱼ²), with a constant permeability.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SellmeierMaterial {
    pub sellmeier: Sellmeier,
    pub relative_permeability: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sellmeier {
    pub constant: f64,
    pub terms: Vec<SellmeierTerm>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SellmeierTerm {
    pub coefficient: f64,
    pub pole_wavelength_m: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub material: Material,
    pub thickness_m: f64,
}

/// How one side of the stack ends. Internally tagged by `"kind"`; an open
/// side is omitted from the record, and an explicit `{"kind": "open"}` reads
/// as the same open side, so open stacks keep their form and hash
/// (docs/decisions/0009).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Termination {
    #[default]
    Open,
    /// A perfect electric or magnetic wall at absolute z in the stack frame.
    Wall {
        #[serde(rename = "type")]
        kind: WallType,
        z_m: f64,
    },
    /// A wall-backed perfectly matched layer starting at absolute z in the
    /// stack frame and extending outward by its thickness.
    Pml {
        start_z_m: f64,
        thickness_m: f64,
        profile: StretchProfile,
        backing: WallType,
    },
}

/// The complex stretch of a PML, s(ζ) = 1 + c (ζ/d)^m. Internally tagged by
/// `"kind"`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StretchProfile {
    Polynomial { order: f64, strength: C2 },
}

impl Termination {
    pub fn is_open(&self) -> bool {
        *self == Self::Open
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WallType {
    Pec,
    Pmc,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stack {
    pub left: Material,
    pub layers: Vec<Layer>,
    pub right: Material,
    #[serde(default, skip_serializing_if = "Termination::is_open")]
    pub left_termination: Termination,
    #[serde(default, skip_serializing_if = "Termination::is_open")]
    pub right_termination: Termination,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarisation {
    #[serde(rename = "TE")]
    Te,
    #[serde(rename = "TM")]
    Tm,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Refinement {
    /// Absolute residual limit, in the characteristic's gauge.
    pub residual_atol: f64,
    /// Residual limit relative to k0. The characteristic is linear in
    /// wavenumber, so this limit means the same at every device scale.
    pub residual_rtol: f64,
    pub native_atol: f64,
    pub native_rtol: f64,
    pub max_updates: usize,
    pub max_backtracks: usize,
    pub max_evaluations: usize,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuidedRequest {
    pub wavelength_m: f64,
    pub polarisation: Polarisation,
    pub direction: Direction,
    /// Open interval of |beta|.
    pub abs_beta_rad_per_m: [f64; 2],
    pub cutoff_margin_rad_per_m: f64,
    /// In cycles of pi, as in lamina-mode.
    pub phase_tolerance: f64,
    pub max_subdivisions: usize,
    pub refinement: Refinement,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuidedInput {
    pub schema: String,
    pub schema_version: u32,
    pub stack: Stack,
    pub request: GuidedRequest,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Completeness {
    /// One accepted root per counted state in the eligible window.
    Complete,
    /// A count exists but some states are unresolved.
    Partial,
    /// An endpoint phase was undecidable, so no count exists.
    CountUnknown,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Mode {
    pub index: usize,
    pub beta_rad_per_m: C2,
    pub n_eff: C2,
    pub native_coordinate: C2,
    pub characteristic_value: C2,
    pub characteristic_derivative: C2,
    /// |characteristic_value| at the accepted root.
    pub residual: f64,
    pub residual_limit: f64,
    pub correction: Option<f64>,
    pub correction_limit: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gap {
    pub abs_beta_rad_per_m: [f64; 2],
    pub expected: Option<usize>,
    pub cause: String,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuidedBasis {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub input: GuidedInput,
    pub input_hash: String,
    pub k0_rad_per_m: f64,
    pub light_line_rad_per_m: f64,
    pub upper_bound_rad_per_m: f64,
    pub eligible_rad_per_m: Option<[f64; 2]>,
    pub expected: Option<usize>,
    pub found: usize,
    pub completeness: Completeness,
    pub modes: Vec<Mode>,
    pub unresolved: Vec<Gap>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PowerNormalisation {
    pub signed_power_w_per_m: f64,
    /// Common multiplier applied to the backend seed fields.
    pub scale: C2,
    /// Primary transverse field at the left interface before normalisation.
    pub phase_reference: C2,
    pub seed_power_w_per_m: f64,
    pub seed_exterior_power_w_per_m: [f64; 2],
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuidedProfile {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub input_hash: String,
    pub mode_index: usize,
    pub beta_rad_per_m: C2,
    pub z_m: Vec<f64>,
    /// Per sample, the (x, y, z) components.
    pub electric_real: Vec<[f64; 3]>,
    pub electric_imag: Vec<[f64; 3]>,
    pub magnetic_real: Vec<[f64; 3]>,
    pub magnetic_imag: Vec<[f64; 3]>,
    pub normalisation: PowerNormalisation,
    pub relative_boundary_defect: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Partial {
    /// "k0", "beta" or "thickness".
    pub parameter: String,
    pub layer: Option<usize>,
    pub value: Option<C2>,
    pub unit: String,
    pub held_fixed: Option<String>,
    pub failure: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Slope {
    /// "dbeta_dk0" or "dbeta_dthickness".
    pub name: String,
    pub layer: Option<usize>,
    pub value: Option<C2>,
    pub unit: String,
    pub failure: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LocalDerivatives {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub input_hash: String,
    pub mode_index: usize,
    pub characteristic: Option<C2>,
    pub characteristic_failure: Option<String>,
    pub partials: Vec<Partial>,
    pub slopes: Vec<Slope>,
}

/// A parameter of the stack or its materials. Region indices run left
/// exterior (0), layers (1..=n), right exterior (n + 1); interface indices run
/// 0..=n from the left. See docs/decisions/0006 for units.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Parameter {
    Thickness { layer: usize },
    Interface { interface: usize },
    Translation,
    Wavelength,
    Permittivity { region: usize },
    Permeability { region: usize },
    SellmeierConstant { region: usize },
    SellmeierCoefficient { region: usize, term: usize },
    SellmeierPoleWavelength { region: usize, term: usize },
}

/// Conditioning of a simple root's implicit derivative.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Conditioning {
    pub characteristic: C2,
    pub characteristic_direction: C2,
    pub characteristic_beta: C2,
    pub denominator_threshold: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    MaterialAttached,
    FixedLab,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GuidedSensitivity {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub input_hash: String,
    pub mode_index: usize,
    pub parameter: Parameter,
    pub parameter_unit: String,
    pub frame: Frame,
    pub beta_rad_per_m: C2,
    /// dβ/dp in rad/m per parameter unit.
    pub dbeta_rad_per_m: C2,
    pub conditioning: Conditioning,
    pub z_m: Vec<f64>,
    pub electric_real: Vec<[f64; 3]>,
    pub electric_imag: Vec<[f64; 3]>,
    pub magnetic_real: Vec<[f64; 3]>,
    pub magnetic_imag: Vec<[f64; 3]>,
    /// Derivatives per parameter unit; zero where `sample_failures` names a cause.
    pub d_electric_real: Vec<[f64; 3]>,
    pub d_electric_imag: Vec<[f64; 3]>,
    pub d_magnetic_real: Vec<[f64; 3]>,
    pub d_magnetic_imag: Vec<[f64; 3]>,
    pub sample_failures: Vec<Option<String>>,
    /// Left exterior, layers, right exterior.
    pub power_fractions: Vec<f64>,
    pub power_fraction_rates: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GroupIndex {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub input_hash: String,
    pub mode_index: usize,
    pub value: f64,
    pub derivative: C2,
    pub energy_estimate: f64,
    pub relative_discrepancy: f64,
    pub tolerance: f64,
    pub conditioning: Conditioning,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PowerOverlap {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub definition: String,
    pub first_input_hash: String,
    pub first_mode_index: usize,
    pub second_input_hash: String,
    pub second_mode_index: usize,
    /// Each basis's first interface in a common frame.
    pub origins_m: [f64; 2],
    pub power_w_per_m: C2,
    /// Analytic tails beyond the outermost interfaces: left, right.
    pub exterior_power_w_per_m: [C2; 2],
    pub partition_m: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OverlapSensitivity {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub definition: String,
    pub first_input_hash: String,
    pub first_mode_index: usize,
    pub second_input_hash: String,
    pub second_mode_index: usize,
    /// None holds that stack fixed.
    pub first_parameter: Option<Parameter>,
    pub second_parameter: Option<Parameter>,
    pub parameter_unit: String,
    pub origins_m: [f64; 2],
    pub power_w_per_m: C2,
    pub exterior_power_w_per_m: [C2; 2],
    pub partition_m: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
    /// W/m per parameter unit; None where `derivative_failure` names a cause.
    pub power_rate_w_per_m: Option<C2>,
    pub exterior_rates_w_per_m: Option<[C2; 2]>,
    pub rate_sample_count: Option<usize>,
    pub rate_quadrature_change_w_per_m: Option<f64>,
    pub derivative_failure: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Reciprocity {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub definition: String,
    pub first_input_hash: String,
    pub first_mode_index: usize,
    pub second_input_hash: String,
    pub second_mode_index: usize,
    pub origins_m: [f64; 2],
    pub flux_w_per_m: C2,
    pub exterior_flux_w_per_m: [C2; 2],
    pub partition_m: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LocalCoupling {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub definition: String,
    pub input_hash: String,
    pub parameter: Parameter,
    pub parameter_unit: String,
    /// Modes whose |β| differ by no more than this share a group.
    pub gap_rad_per_m: f64,
    /// Rows first, in rad/m per parameter unit.
    pub perturbation_matrix_real: Vec<Vec<f64>>,
    pub perturbation_matrix_imag: Vec<Vec<f64>>,
    /// Rows first, per parameter unit; null on the diagonal and within a group.
    pub coefficients_real: Vec<Vec<Option<f64>>>,
    pub coefficients_imag: Vec<Vec<Option<f64>>>,
    pub unresolved_groups: Vec<Vec<usize>>,
    pub quadrature_change_rad_per_m: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FieldProjection {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub assumption: String,
    pub definition: String,
    pub input_hash: String,
    /// Caller-supplied identity of the field; null when unidentified.
    pub field_id: Option<String>,
    /// In the supplied field's coordinate.
    pub support_m: [f64; 2],
    pub interfaces_m: Vec<f64>,
    /// The basis's first interface in the supplied field's coordinate.
    pub origin_m: f64,
    pub rank_rtol: f64,
    pub gram_real: Vec<Vec<f64>>,
    pub gram_imag: Vec<Vec<f64>>,
    pub right_hand_side_real: Vec<f64>,
    pub right_hand_side_imag: Vec<f64>,
    pub rank: usize,
    pub singular_values_w_per_m: Vec<f64>,
    /// Distinct samples of the supplied field.
    pub field_sample_count: usize,
    pub quadrature_change_w_per_m: f64,
    /// Present at full numerical rank only.
    pub coefficients_real: Option<Vec<f64>>,
    pub coefficients_imag: Option<Vec<f64>>,
    pub condition_number: Option<f64>,
    pub residual_w_per_m: Option<f64>,
}

/// The definitions every modal basis, profile and basis pairing carries.
/// Parsing rejects any other text (docs/decisions/0011).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Definitions {
    pub unconjugated_pairing: String,
    pub conjugated_pairing: String,
    pub normalisation: String,
    pub sign_rule: String,
    pub ordering: String,
    pub degenerate_basis: String,
}

impl Definitions {
    pub fn current() -> Self {
        let s = |v: &str| v.to_owned();
        Self {
            unconjugated_pairing: s("(1/4) ∫ (E1 × H2 + E2 × H1) · x dz̃; dz̃ = s dz in a PML; \
                 open tails by analytic continuation"),
            conjugated_pairing: s("(1/4) ∫ (E1 × H2* + E2* × H1) · x dz, real coordinate, \
                 PML equivalent-medium fields"),
            normalisation: s("unit unconjugated self-pairing, 1 W/m"),
            sign_rule: s(
                "Re(Ey + Z0 Hy) > 0 at the first interface; at a wall on it \
                 Re(Ey + Z0 Hy + i(Z0 Hx − Ex)) > 0; ties by Im",
            ),
            ordering: s("decreasing Re β², then decreasing Im β², then decreasing Re β"),
            degenerate_basis: s(
                "unconjugated Löwdin within groups, principal G^(-1/2), then the sign rule",
            ),
        }
    }
}

/// A window of the complex β plane. Internally tagged by `"kind"`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Window {
    /// A β rectangle on the direct chart: ascending real and imaginary bounds.
    Wavenumber {
        re_rad_per_m: [f64; 2],
        im_rad_per_m: [f64; 2],
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModalBasisInput {
    pub schema: String,
    pub schema_version: u32,
    pub stack: Stack,
    pub wavelength_m: f64,
    pub polarisation: Polarisation,
    pub window: Window,
    /// Resolved on validation: 1e-6 k0 when not given.
    pub degeneracy_gap_rad_per_m: Option<f64>,
    pub self_pairing_rtol: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BasisStatus {
    Complete,
    Incomplete,
    CountFailed,
}

/// A rectangle as `[[re_low, re_high], [im_low, im_high]]`.
pub type Rect2 = [[f64; 2]; 2];

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnresolvedRegion {
    /// In the chart's coordinate: rad/m on the wavenumber chart.
    pub bounds: Rect2,
    pub beta_image_rad_per_m: Rect2,
    pub count: Option<usize>,
    pub cause: String,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BasisMode {
    pub index: usize,
    pub beta_rad_per_m: C2,
    pub n_eff: C2,
    pub mode_type: String,
    pub direction: Direction,
    /// Per side: the sheet of an open side, null for a closed one.
    pub sheet: [Option<String>; 2],
    /// |characteristic| at the accepted root.
    pub residual: f64,
    pub self_pairing_raw_w_per_m: C2,
    /// "unit" or "ill_conditioned".
    pub normalisation: String,
    /// |⟨u, u⟩| over the bounded-region field scale.
    pub self_pairing_ratio: f64,
    /// "first_interface" or "wall_on_first_interface".
    pub sign_reference: String,
    pub sign_margin: Option<f64>,
    pub sign_ambiguous: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModalBasis {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub definitions: Definitions,
    pub assumption: String,
    pub input: ModalBasisInput,
    pub input_hash: String,
    pub k0_rad_per_m: f64,
    pub chart: String,
    pub window_beta_image_rad_per_m: Rect2,
    pub scope: String,
    pub status: BasisStatus,
    pub expected: Option<usize>,
    pub found: usize,
    pub unresolved: Vec<UnresolvedRegion>,
    /// Near-degenerate groups of two or more modes.
    pub groups: Vec<Vec<usize>>,
    pub modes: Vec<BasisMode>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModalProfile {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub definitions: Definitions,
    pub input_hash: String,
    pub mode_index: usize,
    pub beta_rad_per_m: C2,
    pub z_m: Vec<f64>,
    /// Per sample: "layer", "gap" or "exterior".
    pub region: Vec<String>,
    pub electric_real: Vec<[f64; 3]>,
    pub electric_imag: Vec<[f64; 3]>,
    pub magnetic_real: Vec<[f64; 3]>,
    pub magnetic_imag: Vec<[f64; 3]>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BasisPairing {
    pub schema: String,
    pub schema_version: u32,
    pub conventions: Conventions,
    pub definitions: Definitions,
    pub first_input_hash: String,
    pub second_input_hash: String,
    pub origins_m: [f64; 2],
    /// Mode indices of the rows (first basis) and columns (second basis).
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    /// Rows first, in W/m.
    pub unconjugated_real: Vec<Vec<f64>>,
    pub unconjugated_imag: Vec<Vec<f64>>,
    /// Rows first, in W/m; null where the entry is absent, with its cause.
    pub conjugated_real: Vec<Vec<Option<f64>>>,
    pub conjugated_imag: Vec<Vec<Option<f64>>>,
    pub conjugated_absent: Vec<Vec<Option<String>>>,
    pub singular_values_w_per_m: Vec<f64>,
    pub leading_condition_numbers: Vec<f64>,
    pub self_defects: [f64; 2],
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
    pub status: [BasisStatus; 2],
    pub scope: [String; 2],
}

/// Records with a schema identity that can be checked after deserialisation.
pub trait Record: Serialize + for<'de> Deserialize<'de> {
    const SCHEMA: &'static str;
    fn identity(&self) -> (&str, u32);
    /// Conventions, where the record carries them.
    fn conventions(&self) -> Option<&Conventions> {
        None
    }
    /// Further checks beyond schema and conventions.
    fn check(&self) -> Result<(), String> {
        Ok(())
    }
}

macro_rules! record {
    ($ty:ty, $schema:expr, conventions) => {
        impl Record for $ty {
            const SCHEMA: &'static str = $schema;
            fn identity(&self) -> (&str, u32) {
                (&self.schema, self.schema_version)
            }
            fn conventions(&self) -> Option<&Conventions> {
                Some(&self.conventions)
            }
        }
    };
}

impl Record for GuidedInput {
    const SCHEMA: &'static str = INPUT_SCHEMA;
    fn identity(&self) -> (&str, u32) {
        (&self.schema, self.schema_version)
    }
}

impl Record for GuidedBasis {
    const SCHEMA: &'static str = BASIS_SCHEMA;
    fn identity(&self) -> (&str, u32) {
        (&self.schema, self.schema_version)
    }
    fn conventions(&self) -> Option<&Conventions> {
        Some(&self.conventions)
    }
    fn check(&self) -> Result<(), String> {
        check_identity(&self.input)?;
        if self.input_hash != digest(&self.input)? {
            return Err("input_hash does not match the input".into());
        }
        if self.found != self.modes.len() {
            return Err("found does not match the number of modes".into());
        }
        Ok(())
    }
}

record!(GuidedProfile, PROFILE_SCHEMA, conventions);
record!(LocalDerivatives, DERIVATIVES_SCHEMA, conventions);
record!(GuidedSensitivity, SENSITIVITY_SCHEMA, conventions);
record!(GroupIndex, GROUP_INDEX_SCHEMA, conventions);

macro_rules! pairing_record {
    ($ty:ty, $schema:expr, $definition:expr) => {
        impl Record for $ty {
            const SCHEMA: &'static str = $schema;
            fn identity(&self) -> (&str, u32) {
                (&self.schema, self.schema_version)
            }
            fn conventions(&self) -> Option<&Conventions> {
                Some(&self.conventions)
            }
            fn check(&self) -> Result<(), String> {
                if self.definition != $definition {
                    return Err(format!("definition differs from {}'s", $schema));
                }
                Ok(())
            }
        }
    };
}

pairing_record!(PowerOverlap, POWER_OVERLAP_SCHEMA, POWER_OVERLAP_DEFINITION);
pairing_record!(
    OverlapSensitivity,
    OVERLAP_SENSITIVITY_SCHEMA,
    OVERLAP_SENSITIVITY_DEFINITION
);
pairing_record!(Reciprocity, RECIPROCITY_SCHEMA, RECIPROCITY_DEFINITION);
pairing_record!(
    LocalCoupling,
    LOCAL_COUPLING_SCHEMA,
    LOCAL_COUPLING_DEFINITION
);
pairing_record!(FieldProjection, PROJECTION_SCHEMA, PROJECTION_DEFINITION);

impl Record for ModalBasisInput {
    const SCHEMA: &'static str = MODAL_BASIS_INPUT_SCHEMA;
    fn identity(&self) -> (&str, u32) {
        (&self.schema, self.schema_version)
    }
}

fn check_definitions(definitions: &Definitions, schema: &str) -> Result<(), String> {
    if *definitions != Definitions::current() {
        return Err(format!("definitions differ from {schema}'s"));
    }
    Ok(())
}

impl Record for ModalBasis {
    const SCHEMA: &'static str = MODAL_BASIS_SCHEMA;
    fn identity(&self) -> (&str, u32) {
        (&self.schema, self.schema_version)
    }
    fn conventions(&self) -> Option<&Conventions> {
        Some(&self.conventions)
    }
    fn check(&self) -> Result<(), String> {
        check_definitions(&self.definitions, Self::SCHEMA)?;
        check_identity(&self.input)?;
        if self.input_hash != digest(&self.input)? {
            return Err("input_hash does not match the input".into());
        }
        if self.found != self.modes.len() {
            return Err("found does not match the number of modes".into());
        }
        Ok(())
    }
}

macro_rules! defined_record {
    ($ty:ty, $schema:expr) => {
        impl Record for $ty {
            const SCHEMA: &'static str = $schema;
            fn identity(&self) -> (&str, u32) {
                (&self.schema, self.schema_version)
            }
            fn conventions(&self) -> Option<&Conventions> {
                Some(&self.conventions)
            }
            fn check(&self) -> Result<(), String> {
                check_definitions(&self.definitions, Self::SCHEMA)
            }
        }
    };
}

defined_record!(ModalProfile, MODAL_PROFILE_SCHEMA);
defined_record!(BasisPairing, BASIS_PAIRING_SCHEMA);

pub fn check_identity<R: Record>(record: &R) -> Result<(), String> {
    let (schema, version) = record.identity();
    if schema != R::SCHEMA {
        return Err(format!("expected schema {}, found {schema}", R::SCHEMA));
    }
    if version != SCHEMA_VERSION {
        return Err(format!(
            "schema version {version} is not supported (this build reads {SCHEMA_VERSION})"
        ));
    }
    if let Some(conventions) = record.conventions() {
        if *conventions != Conventions::current() {
            return Err("conventions differ from this schema version's".into());
        }
    }
    record.check()
}

pub fn to_canonical<R: Record>(record: &R) -> Result<String, String> {
    serde_json::to_string(record).map_err(|e| e.to_string())
}

/// Parse, check and re-serialise, so any accepted text maps to one form.
pub fn canonicalise<R: Record>(json: &str) -> Result<(R, String), String> {
    let record: R = serde_json::from_str(json).map_err(|e| e.to_string())?;
    check_identity(&record)?;
    let canonical = to_canonical(&record)?;
    Ok((record, canonical))
}

/// SHA-256 of the canonical form, as lowercase hex.
pub fn digest<R: Record>(record: &R) -> Result<String, String> {
    let canonical = to_canonical(record)?;
    let hash = Sha256::digest(canonical.as_bytes());
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}
