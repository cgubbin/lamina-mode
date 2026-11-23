//! Longitudinal power per metre of invariant width for a lossless dielectric guide.
use super::*;
use lamina_core::{
    EvaluateMeromorphicMaterial, ExteriorSampling, FieldSampling, LayerSampling, ModalFieldBackend,
    Polarisation, Scalar,
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
    #[error("finite-layer power quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
    #[error(transparent)]
    Recovery(#[from] Box<FieldRecoveryFailure>),
}
impl ModePoint {
    /// Set the magnitude of longitudinal power to one W/m and the left
    /// interface Ey (TE) or Z0 Hy (TM) to positive real. The input sampling
    /// controls only the returned profile, independently of power integration.
    pub fn guided_power_profile_with<B: ModalFieldBackend + Clone>(
        &self,
        backend: B,
        sampling: &FieldSampling<f64>,
        settings: FieldRecoverySettings,
    ) -> Result<ModalProfile, GuidedPowerError> {
        use GuidedPowerError::*;
        let sig = self.signature().map_err(|_| Ineligible)?;
        let k = sig.k0;
        let beta = sig.k_parallel;
        let tol = 1e-9;
        if k.re <= 0.
            || k.im.abs() > tol * k.re
            || beta.im.abs() > tol * beta.re.abs()
            || [sig.left, sig.right]
                .iter()
                .any(|r| r.im <= tol * k.re || r.re.abs() > tol * r.im)
        {
            return Err(Ineligible);
        }
        let prov = self.provenance();
        let stack = prov.stack();
        let material_index = |mat: &lamina_core::AnalyticalMaterialHandle<Complex>| {
            let at = Scalar(Complex::new(k.re, 0.));
            let e = mat.evaluate_relative_permittivity_complex(at);
            let m = mat.evaluate_relative_permeability_complex(at);
            if [e, m]
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite() || v.re <= 0. || v.im != 0.)
            {
                return Err(Ineligible);
            }
            Ok(e.re * m.re)
        };
        for exterior in [stack.left_exterior(), stack.right_exterior()] {
            material_index(exterior)?;
        }
        let mut grid = Vec::new();
        for layer in stack.layers_left_to_right() {
            let d = layer.thickness().as_centimetres();
            let phase = d
                * (material_index(layer.material())? * k.re * k.re - beta.re * beta.re)
                    .abs()
                    .sqrt();
            if !phase.is_finite() || phase > 4096. {
                return Err(Quadrature);
            }
            grid.push((d, 2 * (phase.ceil() as usize).max(4)));
        }
        let mut prev = None;
        for level in 0..8 {
            let edge = || ExteriorSampling::point(Length::centimetres(0.));
            let mut integration = FieldSampling::new().left_exterior(edge());
            let mut budget = 2;
            for (i, &(_, base)) in grid.iter().enumerate() {
                let count = (base << level) + 1;
                budget += count;
                if budget > 131072 {
                    return Err(Quadrature);
                }
                integration =
                    integration.layer(i, LayerSampling::uniform_with_boundaries(count, true, true));
            }
            integration = integration.right_exterior(edge());
            let mode = self.recover_fields_with(backend.clone(), &integration, settings)?;
            let fields = mode.fields.quantity();
            let e = fields.electric();
            let h = fields.magnetic();
            let sx: Vec<_> = (0..e.y().len())
                .map(|i| 0.5 * (e.y()[i] * h.z()[i].conj() - e.z()[i] * h.y()[i].conj()).re)
                .collect();
            let tails = [
                sx[0] / (2. * sig.left.im) * CM_TO_M_OVER_Z0,
                sx[sx.len() - 1] / (2. * sig.right.im) * CM_TO_M_OVER_Z0,
            ];
            let mut total = tails[0] + tails[1];
            let mut offset = 1;
            for &(d, base) in &grid {
                let n = base << level;
                let weighted = sx[offset]
                    + sx[offset + n]
                    + (1..n)
                        .map(|j| sx[offset + j] * if j % 2 == 1 { 4. } else { 2. })
                        .sum::<f64>();
                total += CM_TO_M_OVER_Z0 * d * weighted / (3. * n as f64);
                offset += n + 1;
            }
            if !total.is_finite() || total == 0. {
                return Err(IllConditioned);
            }
            if prev.is_some_and(|p: f64| (p - total).abs() <= 1e-9 * total.abs()) {
                let primary = match prov.polarisation() {
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
                return Ok(super::profile::snapshot(
                    result,
                    ProfileNormalization::GuidedPower(GuidedPowerNormalization {
                        seed_power_w_per_m: total,
                        seed_exterior_power_w_per_m: tails,
                        scale,
                        signed_power_w_per_m: total.signum(),
                        phase_reference: reference,
                    }),
                )?);
            }
            prev = Some(total);
        }
        Err(Quadrature)
    }
}
