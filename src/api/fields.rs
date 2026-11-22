//! Owned arbitrary-amplitude mode fields; root acceptance remains unchanged.
use super::*;
use lamina_core::{FieldSampling, ModalFieldBackend, OwnedModeFields, UnscaledFieldError};
#[derive(Clone, Copy, Debug)]
pub struct FieldRecoverySettings {
    pub amplitude: Complex,
    /// Dimensionless relative outgoing boundary defect limit.
    pub relative_boundary_limit: f64,
    /// Absolute fresh characteristic limit, in the core backend gauge.
    pub characteristic_limit: f64,
}
impl Default for FieldRecoverySettings {
    fn default() -> Self {
        Self {
            amplitude: Complex::new(1., 0.),
            relative_boundary_limit: 1e-8,
            characteristic_limit: 1e-6,
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum FieldRecoveryError {
    #[error("normalization reevaluation failed")]
    Normalization(#[source] lamina_core::PoleNormalizationError),
    #[error("normalization source changed after evaluation")]
    NormalizationChanged,
    #[error("invalid field amplitude or residual settings")]
    InvalidSettings,
    #[error("accepted branch evidence is unavailable")]
    Branch(#[source] Diagnostic),
    #[error("core homogeneous field recovery failed")]
    Core(#[source] UnscaledFieldError),
    #[error("field boundary defect {defect} exceeds {limit}")]
    BoundaryDefect { defect: f64, limit: f64 },
    #[error("fresh characteristic magnitude {residual} exceeds {limit}")]
    Characteristic { residual: f64, limit: f64 },
}
/// Postprocessing failure owns its originating accepted point and source cause.
#[derive(Debug, thiserror::Error)]
#[error("field recovery failed: {cause}")]
pub struct FieldRecoveryFailure {
    pub point: ModePoint,
    #[source]
    pub cause: FieldRecoveryError,
}
#[derive(Clone, Debug)]
pub struct ModeFields {
    pub point: ModePoint,
    pub fields: OwnedModeFields,
    pub source_backend: &'static str,
    pub recovery_backend: &'static str,
    /// Backend seed gauge; E and H are multiplied together by amplitude.
    pub gauge: &'static str,
    pub settings: FieldRecoverySettings,
    pub source_evidence: NumericalEvidence,
    pub characteristic: Complex,
    pub relative_boundary_defect: f64,
    pub seed_residual: Complex,
    pub seed_scale: f64,
}
impl ModePoint {
    /// Recover fields at the stored physical point with core sampling conventions.
    /// No discovery, refinement, principal-root replacement or QNM normalization.
    pub fn recover_fields_with<B: ModalFieldBackend>(
        &self,
        backend: B,
        sampling: &FieldSampling<f64>,
        settings: FieldRecoverySettings,
    ) -> Result<ModeFields, Box<FieldRecoveryFailure>> {
        let attempt = || {
            if ![
                settings.amplitude.re,
                settings.amplitude.im,
                settings.relative_boundary_limit,
                settings.characteristic_limit,
            ]
            .into_iter()
            .all(f64::is_finite)
                || settings.amplitude.norm() == 0.
                || settings.relative_boundary_limit < 0.
                || settings.characteristic_limit < 0.
            {
                return Err(FieldRecoveryError::InvalidSettings);
            }
            let signature = self
                .signature()
                .map_err(|e| FieldRecoveryError::Branch(e.clone()))?;
            let provenance = self.provenance();
            let result = backend
                .unscaled_fields(
                    provenance.stack(),
                    signature.k0,
                    signature.k_parallel,
                    [signature.left, signature.right],
                    provenance.polarisation(),
                    sampling,
                    settings.amplitude,
                )
                .map_err(FieldRecoveryError::Core)?;
            if result.relative_boundary_defect > settings.relative_boundary_limit {
                return Err(FieldRecoveryError::BoundaryDefect {
                    defect: result.relative_boundary_defect,
                    limit: settings.relative_boundary_limit,
                });
            }
            if result.characteristic.norm() > settings.characteristic_limit {
                return Err(FieldRecoveryError::Characteristic {
                    residual: result.characteristic.norm(),
                    limit: settings.characteristic_limit,
                });
            }
            Ok(ModeFields {
                point: self.clone(),
                fields: result.fields,
                source_backend: self.backend(),
                recovery_backend: std::any::type_name::<B>(),
                gauge: "core backend homogeneous seed",
                settings,
                source_evidence: self.evidence(),
                characteristic: result.characteristic,
                relative_boundary_defect: result.relative_boundary_defect,
                seed_residual: result.seed_residual,
                seed_scale: result.seed_scale,
            })
        };
        attempt().map_err(|cause| {
            Box::new(FieldRecoveryFailure {
                point: self.clone(),
                cause,
            })
        })
    }
}
