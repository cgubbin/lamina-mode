//! Longitudinal power per metre of invariant width for a lossless dielectric guide.
use super::*;
use crate::problem::material::{self, Channel};
use lamina_core::{
    ExteriorSampling, FieldSampling, LayerSampling, ModalFieldBackend, Polarisation,
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
/// A composite-Simpson grid over the transverse profile of a guided mode,
/// region by region from the left exterior to the right. Each finite layer
/// and each wall gap is sampled uniformly with both ends, its intervals set by
/// its transverse phase; an open exterior is sampled at its interface only,
/// for its analytic tail.
pub(super) struct ProfileGrid {
    /// Width (cm) and base intervals per region; `None` for an open exterior.
    regions: Vec<Option<(f64, usize)>>,
}

/// One region of a [`ProfileGrid`] at one refinement level.
pub(super) struct Span {
    /// Left exterior 0, layers 1..=n, right exterior n + 1.
    pub region: usize,
    /// Index of the region's first sample.
    pub first: usize,
    /// Simpson intervals; `None` for an open exterior's interface sample.
    pub intervals: Option<usize>,
    pub width: f64,
}

impl Span {
    /// The region's last sample: the interface of an open exterior.
    pub fn last(&self) -> usize {
        self.first + self.intervals.unwrap_or(0)
    }

    /// Composite-Simpson mean of `f` over the region's samples.
    pub fn mean(&self, f: impl Fn(usize) -> f64) -> f64 {
        let n = self.intervals.expect("a sampled region");
        let weighted: f64 = (0..=n)
            .map(|j| {
                let weight = if j == 0 || j == n {
                    1.
                } else if j % 2 == 1 {
                    4.
                } else {
                    2.
                };
                weight * f(self.first + j)
            })
            .sum();
        weighted / (3. * n as f64)
    }

    /// Composite Simpson of `f` over the region's samples.
    pub fn simpson(&self, f: impl Fn(usize) -> f64) -> f64 {
        self.width * self.mean(f)
    }
}

impl ProfileGrid {
    /// Base intervals are `per_radian` times the region's transverse phase
    /// for `polarisation`, rounded up and at least `min_radians`. `None` past
    /// 4096 radians or for a region that cannot be evaluated. A PML side is
    /// refused.
    pub(super) fn new(
        stack: &ModeStack,
        polarisation: Polarisation,
        k: Complex,
        beta: Complex,
        per_radian: usize,
        min_radians: usize,
    ) -> Result<Option<Self>, PmlIntegral> {
        let layers = stack.len();
        let intervals = |index: usize, region: &Region, width: f64| {
            let tensors = material::tensors(material::region_id(index, layers), region, k).ok()?;
            let kappa_squared = Channel::of(&tensors, polarisation).kappa_squared(k, beta);
            let phase = width * kappa_squared.norm().sqrt();
            (phase.is_finite() && phase <= 4096.)
                .then(|| (width, per_radian * (phase.ceil() as usize).max(min_radians)))
        };
        let [left, right] = crate::problem::walls(stack).map_err(|side| PmlIntegral { side })?;
        let gap = |wall: Option<(lamina_core::WallKind, f64)>, index, material| match wall {
            None => Some(None),
            Some((_, width)) => intervals(index, material, width).map(Some),
        };
        let regions = std::iter::once(gap(left, 0, stack.left_exterior()))
            .chain(
                stack
                    .layers_left_to_right()
                    .iter()
                    .enumerate()
                    .map(|(i, layer)| {
                        intervals(i + 1, layer.material(), layer.thickness().as_centimetres())
                            .map(Some)
                    }),
            )
            .chain(std::iter::once(gap(
                right,
                layers + 1,
                stack.right_exterior(),
            )))
            .collect::<Option<_>>();
        Ok(regions.map(|regions| Self { regions }))
    }

    /// The regions at refinement `level`, each with twice the intervals of the last.
    pub(super) fn spans(&self, level: u32) -> Vec<Span> {
        let mut first = 0;
        self.regions
            .iter()
            .enumerate()
            .map(|(region, sampled)| {
                let intervals = sampled.map(|(_, base)| base << level);
                let span = Span {
                    region,
                    first,
                    intervals,
                    width: sampled.map_or(0., |(width, _)| width),
                };
                first = span.last() + 1;
                span
            })
            .collect()
    }

    /// The sampling of [`Self::spans`] at `level`; `None` past the sample budget.
    pub(super) fn sampling(&self, level: u32) -> Option<FieldSampling<f64>> {
        let spans = self.spans(level);
        if spans.last()?.last() >= 131072 {
            return None;
        }
        let exterior = |span: &Span| match span.intervals {
            None => ExteriorSampling::point(Length::centimetres(0.)),
            Some(n) => ExteriorSampling::uniform(n + 1, Length::centimetres(span.width)),
        };
        let (last, layers) = spans.split_last()?;
        let (left, layers) = layers.split_first()?;
        let sampling = layers.iter().enumerate().fold(
            FieldSampling::new().left_exterior(exterior(left)),
            |sampling, (i, span)| {
                let points = span.intervals.expect("a layer") + 1;
                sampling.layer(
                    i,
                    LayerSampling::uniform_with_boundaries(points, true, true),
                )
            },
        );
        Some(sampling.right_exterior(exterior(last)))
    }
}

impl ModePoint {
    /// Set the magnitude of longitudinal power to one W/m and the left
    /// interface Ey (TE) or Z0 Hy (TM) to positive real. The input sampling
    /// controls only the returned profile, independently of power integration.
    pub fn guided_power_profile_with<B: ModalFieldBackend<Region> + Clone>(
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
            || sig
                .exteriors()
                .iter()
                .flatten()
                .any(|r| r.im <= tol * k.re || r.re.abs() > tol * r.im)
        {
            return Err(Ineligible);
        }
        let prov = self.provenance();
        let stack = prov.stack();
        let at = Complex::new(k.re, 0.);
        // Lossless in the entries this polarisation sees.
        let lossless = |(region, m): (lamina_core::StackRegion, &Region)| {
            material::tensors(region, m, at).is_ok_and(|tensors| {
                Channel::of(&tensors, prov.polarisation())
                    .entries()
                    .iter()
                    .all(|v| v.re.is_finite() && v.re > 0. && v.im == 0.)
            })
        };
        if !material::regions(stack).all(lossless) {
            return Err(Ineligible);
        }
        let grid =
            ProfileGrid::new(stack, prov.polarisation(), k, beta, 2, 4)?.ok_or(Quadrature)?;
        let mut prev = None;
        for level in 0..8 {
            let integration = grid.sampling(level).ok_or(Quadrature)?;
            let mode = self.recover_fields_with(backend.clone(), &integration, settings)?;
            let fields = mode.fields.quantity();
            let e = fields.electric();
            let h = fields.magnetic();
            let sx: Vec<_> = (0..e.y().len())
                .map(|i| 0.5 * (e.y()[i] * h.z()[i].conj() - e.z()[i] * h.y()[i].conj()).re)
                .collect();
            let decay = sig.exteriors().map(|r| r.map(|r| r.im));
            let regions: Vec<_> = grid
                .spans(level)
                .iter()
                .map(|span| match span.intervals {
                    None => {
                        let side = usize::from(span.region > 0);
                        sx[span.first] / (2. * decay[side].expect("open")) * CM_TO_M_OVER_Z0
                    }
                    Some(_) => CM_TO_M_OVER_Z0 * span.simpson(|i| sx[i]),
                })
                .collect();
            let total: f64 = regions.iter().sum();
            if !total.is_finite() || total == 0. {
                return Err(IllConditioned);
            }
            if prev.is_some_and(|p: f64| (p - total).abs() <= 1e-9 * total.abs()) {
                let primary = match prov.polarisation() {
                    Polarisation::TransverseElectric => fields.electric().y(),
                    Polarisation::TransverseMagnetic => fields.magnetic().y(),
                };
                let reference = primary[grid.spans(level)[0].last()];
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
                        seed_exterior_power_w_per_m: [regions[0], regions[regions.len() - 1]],
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
