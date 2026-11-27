//! Longitudinal power per metre of invariant width for a lossless dielectric guide.
use super::forms;
use super::*;
use crate::problem::material::{self, Channel};
use lamina_core::{
    ExteriorSampling, FieldSampling, LayerSampling, ModalDerivativeBackend, ModalFieldBackend,
    Polarisation,
};
use lamina_units::Length;

// Core stores Z0 H; convert flux to W/m² and transverse cm to metres.
pub(super) const CM_TO_M_OVER_Z0: f64 = 0.01 / 376.730313412;

#[derive(Clone, Debug)]
pub struct GuidedPowerNormalization {
    pub seed_power_w_per_m: f64,
    pub seed_exterior_power_w_per_m: [f64; 2],
    /// Common multiplier relative to the supplied field recovery amplitude.
    pub scale: Complex,
    /// Signed power of the returned profile in W/m.
    pub signed_power_w_per_m: f64,
    /// Primary transverse field at the left interface prior to normalization.
    pub phase_reference: Complex,
}
#[derive(Debug, thiserror::Error)]
pub enum GuidedPowerError {
    #[error(
        "requires a lossless dielectric guided mode with real frequency and propagation constant"
    )]
    Ineligible,
    #[error("guided field phase or power is ill-conditioned")]
    IllConditioned,
    #[error("the power integral failed")]
    Integrals(#[source] std::sync::Arc<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Termination(#[from] PmlIntegral),
    #[error(transparent)]
    Recovery(#[from] Box<FieldRecoveryFailure>),
}
/// A transverse integral was requested through a side closed by a PML, where
/// the fields live on a complex coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("transverse integrals need open or walled sides; the {side:?} side is closed by a PML")]
pub struct PmlIntegral {
    pub side: lamina_core::StackSide,
}
/// Samples of a stack at its interfaces and the centre of every layer:
/// the left interface first.
pub(super) fn interface_sampling(stack: &ModeStack) -> FieldSampling<f64> {
    let interface = || ExteriorSampling::point(Length::centimetres(0.));
    (0..stack.len())
        .fold(
            FieldSampling::new().left_exterior(interface()),
            |sampling, layer| {
                sampling.layer(layer, LayerSampling::uniform_with_boundaries(3, true, true))
            },
        )
        .right_exterior(interface())
}

impl ModePoint {
    /// Set the magnitude of longitudinal power to one W/m and the left
    /// interface Ey (TE) or Z0 Hy (TM) to positive real. The input sampling
    /// controls only the returned profile; the power is integrated in closed
    /// form. A side closed by a PML is refused.
    pub fn guided_power_profile_with<B>(
        &self,
        backend: B,
        sampling: &FieldSampling<f64>,
        settings: FieldRecoverySettings,
    ) -> Result<ModalProfile, GuidedPowerError>
    where
        B: ModalFieldBackend<Region> + ModalDerivativeBackend + Clone,
    {
        use GuidedPowerError::*;
        let sig = self.signature().map_err(|_| Ineligible)?;
        let k = sig.k0;
        let beta = sig.k_parallel;
        let tol = 1e-9;
        if k.re <= 0.
            || k.im.abs() > tol * k.re
            || beta.im.abs() > tol * beta.re.abs()
            || sig
                .exteriors()
                .iter()
                .flatten()
                .any(|r| r.im <= tol * k.re || r.re.abs() > tol * r.im)
        {
            return Err(Ineligible);
        }
        let prov = self.provenance();
        let polarisation = prov.polarisation().ok_or(Ineligible)?;
        let stack = prov.stack();
        let at = Complex::new(k.re, 0.);
        // Lossless in the entries this polarisation sees.
        let lossless = |(region, m): (lamina_core::StackRegion, &Region)| {
            material::tensors(region, m, at).is_ok_and(|tensors| {
                Channel::of(&tensors, polarisation)
                    .entries()
                    .iter()
                    .all(|v| v.re.is_finite() && v.re > 0. && v.im == 0.)
            })
        };
        if !material::regions(stack).all(lossless) {
            return Err(Ineligible);
        }
        crate::problem::walls(stack).map_err(|side| PmlIntegral { side })?;
        let amplitude = settings.amplitude;
        let integrals = forms::still_forms(
            &backend,
            [self, self],
            [amplitude, amplitude],
            &[(true, &[forms::flux()])],
        )
        .map_err(|_| Ineligible)?
        .map_err(Integrals)?;
        let regions: Vec<f64> = integrals[0]
            .iter()
            .map(|share| 0.5 * CM_TO_M_OVER_Z0 * share.re)
            .collect();
        let total: f64 = regions.iter().sum();
        if !total.is_finite() || total == 0. {
            return Err(IllConditioned);
        }
        // The phase reference at the left interface, against the primary
        // field's size at the interfaces and layer centres.
        let mode =
            self.recover_fields_with(backend.clone(), &interface_sampling(stack), settings)?;
        let fields = mode.fields.quantity();
        let primary = match polarisation {
            Polarisation::TransverseElectric => fields.electric().y(),
            Polarisation::TransverseMagnetic => fields.magnetic().y(),
        };
        let reference = primary[0];
        let peak = primary.iter().map(|z| z.norm()).fold(0., f64::max);
        if reference.norm() <= 1e-10 * peak || !peak.is_finite() {
            return Err(IllConditioned);
        }
        let scale = reference.conj() / reference.norm() / total.abs().sqrt();
        let result = self.recover_fields_with(
            backend,
            sampling,
            FieldRecoverySettings {
                amplitude: settings.amplitude * scale,
                ..settings
            },
        )?;
        Ok(super::profile::snapshot(
            result,
            ProfileNormalization::GuidedPower(GuidedPowerNormalization {
                seed_power_w_per_m: total,
                seed_exterior_power_w_per_m: [regions[0], regions[regions.len() - 1]],
                scale,
                signed_power_w_per_m: total.signum(),
                phase_reference: reference,
            }),
        )?)
    }
}
