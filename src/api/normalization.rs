//! Separately fallible simple-pole normalization of accepted points.
//!
//! Local analytic declarations cannot be omitted:
//! ```compile_fail
//! use lamina_mode::{ModePoint, PoleVariable, PoleNormalizationSettings, FieldRecoverySettings};
//! fn missing(point: &ModePoint) {
//!     point.normalize_with(lamina_core::Scatter2::new(), PoleVariable::BetaAtFixedK0,
//!         PoleNormalizationSettings::default(), FieldRecoverySettings::default());
//! }
//! ```
use super::*;
use lamina_core::{
    FieldSampling, ModalNormalizationBackend, PoleNormalizationError, PoleNormalizationInput,
};
pub use lamina_core::{PoleNormalizationData, PoleNormalizationSettings, PoleVariable};
#[derive(Debug, thiserror::Error)]
pub enum NormalizationError {
    #[error("pole variable does not match the source problem's held-fixed coordinate")]
    IncompatibleVariable,
    #[error("branch evidence unavailable")]
    Branch(#[source] Diagnostic),
    #[error("normalization backend failed")]
    Core(#[source] PoleNormalizationError),
    #[error("field recovery evidence exceeds normalization limits")]
    Residual,
    #[error("invalid recovery settings")]
    InvalidSettings,
}
#[derive(Debug, thiserror::Error)]
#[error("mode normalization failed: {cause}")]
pub struct NormalizationFailure {
    pub point: ModePoint,
    #[source]
    pub cause: NormalizationError,
}
#[derive(Clone, Debug)]
pub struct NormalizedMode<B> {
    pub point: ModePoint,
    pub normalization: PoleNormalizationData,
    pub local_assumption: AnalyticAssumption,
    pub source_backend: &'static str,
    pub normalization_backend: &'static str,
    backend: B,
    recovery: FieldRecoverySettings,
}
impl ModePoint {
    /// Explicit local analytic declaration is required even if the source search
    /// had a declaration. Scalar unit-delta pole convention; not a 3D mode volume.
    pub fn normalize_with<B: ModalNormalizationBackend + Clone>(
        &self,
        backend: B,
        variable: PoleVariable,
        local_assumption: AnalyticAssumption,
        settings: PoleNormalizationSettings,
        recovery: FieldRecoverySettings,
    ) -> Result<NormalizedMode<B>, Box<NormalizationFailure>> {
        let attempt = || {
            match (self.provenance().direction(), variable) {
                (SearchCoordinate::InPlane { .. }, PoleVariable::BetaAtFixedK0)
                | (SearchCoordinate::Spectral { .. }, PoleVariable::K0AtFixedBeta) => {}
                _ => return Err(NormalizationError::IncompatibleVariable),
            }
            if [
                recovery.characteristic_limit,
                recovery.relative_boundary_limit,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || v < 0.)
            {
                return Err(NormalizationError::InvalidSettings);
            }
            let sig = self
                .signature()
                .map_err(|e| NormalizationError::Branch(e.clone()))?;
            let input = PoleNormalizationInput {
                k0: sig.k0,
                beta: sig.k_parallel,
                kappa: [sig.left, sig.right],
                polarisation: self.provenance().polarisation(),
                variable,
            };
            let normalization = backend
                .pole_normalization(self.provenance().stack(), input, settings)
                .map_err(NormalizationError::Core)?;
            if normalization.characteristic.norm() > recovery.characteristic_limit
                || normalization.relative_boundary_defect > recovery.relative_boundary_limit
            {
                return Err(NormalizationError::Residual);
            }
            Ok(NormalizedMode {
                point: self.clone(),
                normalization,
                local_assumption,
                source_backend: self.backend(),
                normalization_backend: std::any::type_name::<B>(),
                backend: backend.clone(),
                recovery,
            })
        };
        attempt().map_err(|cause| {
            Box::new(NormalizationFailure {
                point: self.clone(),
                cause,
            })
        })
    }
}
impl<B: ModalNormalizationBackend + Clone> NormalizedMode<B> {
    /// Sample normalized E/H with one common scale, preserving core positions.
    /// Re-evaluates normalization to detect changes in the retained material.
    pub fn fields(
        &self,
        sampling: &FieldSampling<f64>,
    ) -> Result<ModeFields, Box<FieldRecoveryFailure>> {
        let current = self
            .backend
            .pole_normalization(
                self.point.provenance().stack(),
                self.normalization.input,
                self.normalization.settings,
            )
            .map_err(|e| {
                Box::new(FieldRecoveryFailure {
                    point: self.point.clone(),
                    cause: FieldRecoveryError::Normalization(e),
                })
            })?;
        if (current.total - self.normalization.total).norm()
            > 1e-8 * self.normalization.contribution_scale
        {
            return Err(Box::new(FieldRecoveryFailure {
                point: self.point.clone(),
                cause: FieldRecoveryError::NormalizationChanged,
            }));
        }
        self.point.recover_fields_with(
            self.backend.clone(),
            sampling,
            FieldRecoverySettings {
                amplitude: self.normalization.settings.amplitude * self.normalization.scale,
                ..self.recovery
            },
        )
    }
}
