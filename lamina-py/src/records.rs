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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stack {
    pub left: Material,
    pub layers: Vec<Layer>,
    pub right: Material,
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
