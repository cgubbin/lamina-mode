//! Owned physical samples for consumers of planar modal fields.
//! Coordinates use exp(+i β x − i ω t), transverse z increasing from left to
//! right and impedance-scaled magnetic components as in core field recovery.
use super::*;
use lamina_core::{
    FieldPosition, FieldSampling, ModalFieldBackend, ModalNormalizationBackend, OwnedModeFields,
    Polarisation, ResolvedLayerPosition,
};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ProfileNormalization {
    BackendSeed {
        convention: &'static str,
        amplitude: Complex,
    },
    ScalarPole(Box<PoleNormalizationData>),
    GuidedPower(GuidedPowerNormalization),
}

/// An owned snapshot of the fields and their physical interpretation. The source
/// material handles retain the same material stability obligation as the point.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ModalProfile {
    /// Physical k0, beta and both exterior transverse wavevectors, in cm^-1.
    pub wavevectors: BranchSignature,
    pub polarisation: Polarisation,
    pub stack: ModeStack,
    /// First interface at zero. In a two-half-space system this contains [0].
    pub interfaces_cm: Vec<f64>,
    /// Same order as fields.sampling().positions(), including one-sided limits.
    pub depths_cm: Vec<f64>,
    pub fields: OwnedModeFields,
    pub normalization: ProfileNormalization,
    pub interpretation: Result<PhysicalDetails, Diagnostic>,
    pub evidence: NumericalEvidence,
    pub relative_boundary_defect: f64,
}
pub(super) fn snapshot(
    recovered: ModeFields,
    normalization: ProfileNormalization,
) -> Result<ModalProfile, Box<FieldRecoveryFailure>> {
    let point = &recovered.point;
    let wavevectors = *point.signature().map_err(|err| {
        Box::new(FieldRecoveryFailure {
            point: point.clone(),
            cause: FieldRecoveryError::Branch(err.clone()),
        })
    })?;
    let source = point.provenance();
    let polarisation = source.polarisation().ok_or_else(|| {
        Box::new(FieldRecoveryFailure {
            point: point.clone(),
            cause: FieldRecoveryError::CoupledChannel,
        })
    })?;
    let stack = source.stack().clone();
    let mut interfaces = vec![0.];
    for layer in stack.layers_left_to_right() {
        interfaces.push(interfaces.last().unwrap() + layer.thickness().as_centimetres());
    }
    let depths = recovered
        .fields
        .sampling()
        .positions()
        .iter()
        .map(|&position| match position {
            FieldPosition::LeftExterior { distance } => -distance.as_centimetres(),
            FieldPosition::RightExterior { distance } => {
                interfaces.last().unwrap() + distance.as_centimetres()
            }
            FieldPosition::Layer { index, position } => {
                let i = index.get();
                match position {
                    ResolvedLayerPosition::FromLeft(d) => interfaces[i] + d.as_centimetres(),
                    ResolvedLayerPosition::FromRight(d) => interfaces[i + 1] - d.as_centimetres(),
                    ResolvedLayerPosition::Fraction(f) => {
                        interfaces[i]
                            + f * stack.layers_left_to_right()[i].thickness().as_centimetres()
                    }
                }
            }
        })
        .collect();
    Ok(ModalProfile {
        wavevectors,
        polarisation,
        stack,
        interfaces_cm: interfaces,
        depths_cm: depths,
        fields: recovered.fields,
        normalization,
        interpretation: point.describe(Default::default()),
        evidence: recovered.source_evidence,
        relative_boundary_defect: recovered.relative_boundary_defect,
    })
}
impl ModePoint {
    pub fn sample_profile_with<B: ModalFieldBackend<Region>>(
        &self,
        backend: B,
        sampling: &FieldSampling<f64>,
        settings: FieldRecoverySettings,
    ) -> Result<ModalProfile, Box<FieldRecoveryFailure>> {
        let recovered = self.recover_fields_with(backend, sampling, settings)?;
        let normalization = ProfileNormalization::BackendSeed {
            convention: recovered.gauge,
            amplitude: settings.amplitude,
        };
        snapshot(recovered, normalization)
    }
}
impl<B: ModalNormalizationBackend<Region> + Clone> NormalizedMode<B> {
    pub fn sample_profile(
        &self,
        sampling: &FieldSampling<f64>,
    ) -> Result<ModalProfile, Box<FieldRecoveryFailure>> {
        snapshot(
            self.fields(sampling)?,
            ProfileNormalization::ScalarPole(Box::new(self.normalization.clone())),
        )
    }
}
