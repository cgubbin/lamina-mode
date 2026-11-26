//! Physical-input mode searches with explicit strategies and finite coverage.
//!
//! Use [`PropagationProblem`] at fixed real k0, [`ConformalSpectralProblem`]
//! for constant exteriors, or [`FrequencyProblem`] for an analytical core stack
//! with exterior dispersion. Scatter2 is the default; `with_backend` accepts
//! core backends. Analytic declarations and radiation choices remain explicit.
//! [`Plan::discover`] permits inspection before refinement; [`Plan::solve`]
//! returns an owned report. Estimates, acceptance, interpretation and coverage
//! are separate evidence. Accepted points also support owned arbitrary-amplitude field postprocessing; no QNM normalization is computed.
//!
//! Analytic declarations cannot be omitted, even through convenience execution.
//!
//! ```compile_fail
//! use lamina_mode::{FrequencyProblem, FrequencyDomainInput};
//! fn missing_declaration(problem: &FrequencyProblem, input: FrequencyDomainInput) {
//!     let _ = problem.domain(input); // an explicit AnalyticAssumption is required
//! }
//! ```
mod basis;
mod coupling;
mod derivatives;
mod diagnostic;
mod fields;
mod group;
mod guided;
mod hybrid;
mod hybrid_fields;
mod integration;
mod normalization;
mod open;
mod overlap;
mod pairing;
mod power;
mod profile;
mod projection;
mod qnm_design;
mod qnm_poles;
mod qnm_scattering;
mod sensitivity;
mod tracking;
pub use basis::*;
pub use coupling::*;
pub use derivatives::*;
pub use fields::*;
pub use group::*;
pub use guided::*;
pub use hybrid::*;
pub use hybrid_fields::*;
pub use normalization::*;
pub use open::*;
pub use overlap::*;
pub use pairing::*;
pub use power::*;
pub use profile::*;
pub use projection::*;
pub use qnm_design::*;
pub use qnm_poles::*;
pub use qnm_scattering::*;
pub use sensitivity::*;
pub use tracking::*;
mod branch;
mod report;
mod solver;
pub use branch::*;
pub use diagnostic::*;
pub use report::*;
pub use solver::*;

pub use crate::analytic::CoordinateKind;
pub use crate::conformal::chart::{ConformalChartConfig, CoordinateSheet};
pub use crate::conformal::in_plane::region::InPlaneSearchExtent;
pub use crate::conformal::wavenumber::region::WavenumberSearchExtent;
pub use crate::conformal::{ConformalRegion, ConformalRegionError};
pub use crate::problem::frequency::{
    ContinuationConfig, ExclusionReason, ExteriorAnalyticAssumption, ExteriorFailure,
    ExteriorReferenceValues, FrequencyDomainInput, FrequencyError, FrequencyEvidence,
    FrequencyExclusion, FrequencyReference, FrequencyRegion,
};
pub use crate::problem::guided::{GuidedRegionError, GuidedRegionEvidence};
pub use crate::problem::material::{ConstantExterior, InvalidConstantExterior, ModeStack, Region};
pub use crate::problem::orchestration::{
    Ambiguity, BranchSignature, CoverageExclusion, DEFAULT_ZERO_TOLERANCE, DiscoveryConfig,
    DuplicateGroup, ExteriorSector, MatchTolerance, PlanError, PlanGeometry,
    ReconciliationDiagnostic,
};
pub use crate::problem::physical::{
    BoundaryReason, Classification, Evolution, ExteriorBehaviour, Flux, Interpretation,
    InterpretationConfig, ModeType, PhysicalData, PhysicalDetails, RadiationQualification, Sign,
    Unsupported,
};
pub use crate::problem::radiation::{RadiationChoice, RadiationTolerance};
pub use crate::problem::{AnalyticAssumption, ReferenceSelection, SearchCoordinate};
pub use crate::refinement::RefinementWork;
pub use argloc::{
    ArgumentLeaf, Rect, RunSummary, SearchTarget, SingularPointEstimate, SingularPointEstimateKind,
    Termination,
};
/// Complex f64 coordinates. Physical wavenumbers use cm^-1.
pub type Complex = crate::problem::C;

/// Conformal real-anchor data. The native coordinate is dimensionless.
#[derive(Clone, Copy, Debug)]
pub struct RadiationReference {
    pub native_coordinate: Complex,
    /// Real varying physical wavenumber, cm^-1.
    pub coordinate: f64,
    /// One choice per open side; `None` exactly on a side closed by a wall or a PML.
    pub choices: [Option<RadiationChoice>; 2],
    pub tolerance: RadiationTolerance,
}
impl From<RadiationReference> for crate::problem::radiation::RadiationReference {
    fn from(v: RadiationReference) -> Self {
        Self {
            z: v.native_coordinate,
            coordinate: v.coordinate,
            choices: v.choices,
            tolerance: v.tolerance,
        }
    }
}
/// Per-candidate numerical controls. Absolute correction is in the native units
/// recorded by the domain: dimensionless conformal z or frequency k0 in cm^-1.
///
/// The characteristic is linear in wavenumber, so an absolute residual means
/// something different at each length scale. `residual_atol: None` (the
/// default) takes [`DEFAULT_RESIDUAL_ATOL`] at a 1 cm wavelength, scaled with
/// the problem's wavenumber, as [`ModalBasisRequest::new`] does: `1e-10 k0/2π`
/// for a fixed k0 and the largest |k0| (or |β|) of the domains in k0. A value
/// is used as given.
#[derive(Clone, Copy, Debug)]
pub struct RefinementConfig {
    pub residual_atol: Option<f64>,
    pub residual_rtol: f64,
    pub residual_scale: f64,
    pub native_atol: f64,
    pub native_rtol: f64,
    pub max_updates: usize,
    pub max_backtracks: usize,
    pub max_evaluations: usize,
}
/// The absolute residual limit of the defaults at a 1 cm wavelength.
pub const DEFAULT_RESIDUAL_ATOL: f64 = 1e-10;
/// The multiple of a 1 cm wavelength's wavenumber, `|k| / 2π`, that scales
/// the default absolute tolerances; 1 when `wavenumber` is zero or not finite.
pub(crate) fn tolerance_scale(wavenumber: f64) -> f64 {
    let scale = wavenumber.abs() / std::f64::consts::TAU;
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}
impl Default for RefinementConfig {
    fn default() -> Self {
        let internal = crate::refinement::RefinementConfig::default();
        Self {
            residual_atol: None,
            residual_rtol: internal.residual_rtol,
            residual_scale: internal.residual_scale,
            native_atol: internal.z_atol,
            native_rtol: internal.z_rtol,
            max_updates: internal.max_updates,
            max_backtracks: internal.max_backtracks,
            max_evaluations: internal.max_evaluations,
        }
    }
}
impl RefinementConfig {
    /// The absolute residual limit at `wavenumber` (cm^-1): the given one, or
    /// the scaled default.
    pub fn residual_atol_at(&self, wavenumber: f64) -> f64 {
        self.residual_atol
            .unwrap_or(DEFAULT_RESIDUAL_ATOL * tolerance_scale(wavenumber))
    }
    /// The controls with the default residual limit resolved at `wavenumber`.
    pub(crate) fn resolve(self, wavenumber: f64) -> crate::refinement::RefinementConfig {
        crate::refinement::RefinementConfig {
            residual_atol: self.residual_atol_at(wavenumber),
            residual_rtol: self.residual_rtol,
            residual_scale: self.residual_scale,
            z_atol: self.native_atol,
            z_rtol: self.native_rtol,
            max_updates: self.max_updates,
            max_backtracks: self.max_backtracks,
            max_evaluations: self.max_evaluations,
        }
    }
}
impl From<crate::refinement::RefinementConfig> for RefinementConfig {
    fn from(v: crate::refinement::RefinementConfig) -> Self {
        Self {
            residual_atol: Some(v.residual_atol),
            residual_rtol: v.residual_rtol,
            residual_scale: v.residual_scale,
            native_atol: v.z_atol,
            native_rtol: v.z_rtol,
            max_updates: v.max_updates,
            max_backtracks: v.max_backtracks,
            max_evaluations: v.max_evaluations,
        }
    }
}
/// Duplicate observation matching. Native absolute units follow the domain;
/// physical and exterior absolute tolerances are cm^-1, relative ones dimensionless.
#[derive(Clone, Copy, Debug)]
pub struct MatchConfig {
    pub native: MatchTolerance,
    pub physical: MatchTolerance,
    pub exterior: MatchTolerance,
}
impl Default for MatchConfig {
    fn default() -> Self {
        let v = crate::problem::orchestration::MatchConfig::default();
        Self {
            native: v.conformal,
            physical: v.physical,
            exterior: v.exterior,
        }
    }
}
impl From<MatchConfig> for crate::problem::orchestration::MatchConfig {
    fn from(v: MatchConfig) -> Self {
        Self {
            conformal: v.native,
            physical: v.physical,
            exterior: v.exterior,
        }
    }
}

// Public cause types for matching diagnostics without private module imports.
pub use crate::conformal::chart::ConformalChartError;
pub use crate::conformal::in_plane::region::InPlaneRegionError;
pub use crate::conformal::in_plane::{InPlaneMapError, InPlaneSearchError};
pub use crate::conformal::wavenumber::region::WavenumberRegionError;
pub use crate::conformal::wavenumber::{WavenumberMapError, WavenumberSearchError};
pub use crate::problem::ProblemError as ConstructionError;
pub use crate::problem::direct::DirectChartError;
pub use crate::problem::physical::InputError as InterpretationError;
pub use crate::problem::radiation::RadiationReferenceError;
pub use crate::refinement::RefinementInputError;

/// Physical radiation verification is separate from coordinate branch selection.
#[derive(Clone, Debug, PartialEq)]
pub enum RadiationProvenance {
    Unspecified,
    Verified(VerifiedRadiationReference),
}
/// Opaque verified evidence built only by checked domain construction.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRadiationReference {
    inner: crate::problem::radiation::VerifiedRadiationReference,
}
impl VerifiedRadiationReference {
    pub fn reference(&self) -> RadiationReference {
        let r = self.inner.reference();
        RadiationReference {
            native_coordinate: r.z,
            coordinate: r.coordinate,
            choices: r.choices,
            tolerance: r.tolerance,
        }
    }
    pub fn physical_coordinates(&self) -> (f64, f64) {
        self.inner.physical_coordinates()
    }
    pub fn epsilon(&self) -> [f64; 2] {
        self.inner.epsilon()
    }
    pub fn observed(&self) -> [Option<Complex>; 2] {
        self.inner.observed()
    }
    pub fn all_outgoing(&self) -> bool {
        self.inner.all_outgoing()
    }
}
impl From<&crate::problem::radiation::RadiationProvenance> for RadiationProvenance {
    fn from(value: &crate::problem::radiation::RadiationProvenance) -> Self {
        match value {
            crate::problem::radiation::RadiationProvenance::Unspecified => Self::Unspecified,
            crate::problem::radiation::RadiationProvenance::Verified(inner) => {
                Self::Verified(VerifiedRadiationReference {
                    inner: inner.clone(),
                })
            }
        }
    }
}

/// Explicit numerical strategy; none is selected from material samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Propagation,
    ConformalSpectral,
    DirectFrequency,
}
