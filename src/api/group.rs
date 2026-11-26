//! Certified group index of lossless guided modes.
//!
//! The group index `n_g = c dβ/dω = dβ/dk₀` is evaluated twice, independently:
//!
//! - as the total derivative of β along the root as the vacuum wavelength
//!   changes, with every material's dispersion included;
//! - as the energy velocity ratio `c W / P`, from the time-averaged stored
//!   energy per unit length `W`, with the dispersive weights `ε_ii + k₀ ∂ε_ii/∂k₀`
//!   and `μ_ii + k₀ ∂μ_ii/∂k₀` of each field component, and the longitudinal
//!   power `P`.
//!
//! For a lossless guided mode the two agree; the result is certified only if
//! they do, within a stated tolerance, and the derivative is real. The value
//! is signed like β: positive for forward modes.
use super::*;
use crate::problem::material::{self, Channel, Principal};

/// A group index and the evidence that certifies it.
#[derive(Clone, Debug)]
pub struct GroupIndex {
    /// The certified group index, `Re(dβ/dk₀)`, signed like β.
    pub value: f64,
    /// The total derivative dβ/dk₀ along the root, including dispersion.
    pub derivative: Complex,
    /// The independent estimate `c W / P` from the stored-energy integral.
    pub energy_estimate: f64,
    /// `|derivative − energy_estimate| / |energy_estimate|`.
    pub relative_discrepancy: f64,
    /// The tolerance the discrepancy was certified against.
    pub tolerance: f64,
    /// The wavelength sensitivity behind `derivative`, with its conditioning.
    pub sensitivity: GuidedBetaSensitivity,
}

#[derive(Debug, thiserror::Error)]
pub enum GroupIndexError {
    #[error("the certification tolerance must be finite and positive")]
    InvalidTolerance,
    #[error("requires a lossless guided mode in materials with real dispersion")]
    Ineligible,
    #[error("the wavelength sensitivity failed: {0:?}")]
    Sensitivity(GuidedBetaError),
    #[error(transparent)]
    Recovery(#[from] Box<FieldRecoveryFailure>),
    #[error("the stored-energy quadrature did not converge")]
    Quadrature,
    #[error(transparent)]
    Termination(#[from] super::PmlIntegral),
    #[error(
        "group index not certified: dβ/dk0 = {derivative}, c W / P = {energy_estimate}, \
         relative discrepancy {relative_discrepancy:e} exceeds {tolerance:e}"
    )]
    Uncertified {
        derivative: Complex,
        energy_estimate: f64,
        relative_discrepancy: f64,
        tolerance: f64,
    },
}

impl ModePoint {
    /// Evaluate and certify the group index of a lossless guided mode. The
    /// field-recovery settings apply to the stored-energy integral; their
    /// characteristic limit must suit the device scale, as for unit-power
    /// profiles.
    pub fn guided_group_index_with<B>(
        &self,
        backend: &B,
        settings: FieldRecoverySettings,
        tolerance: f64,
    ) -> Result<GroupIndex, GroupIndexError>
    where
        B: lamina_core::ModalDerivativeBackend + lamina_core::ModalFieldBackend<Region> + Clone,
    {
        use GroupIndexError::*;
        if !tolerance.is_finite() || tolerance <= 0. {
            return Err(InvalidTolerance);
        }
        let sig = *self.signature().map_err(|_| Ineligible)?;
        let (k, beta) = (sig.k0, sig.k_parallel);
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
        let provenance = self.provenance();
        let stack = provenance.stack();
        let layers = stack.len();

        // Total derivative along the root as the wavelength changes.
        let direction = GuidedPerturbation::new(
            "vacuum wavelength",
            "cm",
            vec![0.; layers + 1],
            1.,
            vec![ConstitutiveRate::default(); layers + 2],
        )
        .map_err(|_| Ineligible)?;
        let sensitivity = self
            .guided_beta_sensitivity_with(
                backend,
                &direction,
                LocalDerivativeSettings::default(),
                ImplicitSlopeSettings::default(),
            )
            .map_err(Sensitivity)?;
        let k0_rate = direction.k0_rate(k.re).map_err(|_| Ineligible)?;
        let derivative = sensitivity.beta_rate / k0_rate;

        // Dispersive energy weights per region and component: left exterior,
        // layers, right exterior. The entries the polarisation sees must be real.
        let polarisation = provenance.polarisation().ok_or(Ineligible)?;
        let at = Complex::new(k.re, 0.);
        let dispersive = |value: Principal<Complex>, rate: Principal<Complex>| match (value, rate) {
            (Principal::Isotropic(v), Principal::Isotropic(d)) => {
                Principal::Isotropic(v + k.re * d)
            }
            _ => {
                let (v, d) = (value.entries(), rate.entries());
                Principal::Diagonal(std::array::from_fn(|i| v[i] + k.re * d[i]))
            }
        };
        let weights = material::regions(stack)
            .map(|(region, m)| {
                let [values, rates] =
                    material::tensors_and_rates(region, m, at).map_err(|_| Ineligible)?;
                let weights = [0, 1].map(|i| dispersive(values[i], rates[i]));
                let real = |z: &Complex| z.re.is_finite() && z.im == 0.;
                let seen = [values, weights].map(|t| Channel::of(&t, polarisation).entries());
                if !seen.iter().flatten().all(real) {
                    return Err(Ineligible);
                }
                Ok(weights.map(|w| w.map(|z| z.re)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Composite Simpson in each layer and wall gap, and analytic tails of
        // open exteriors. The integrands oscillate at twice the transverse
        // wavenumber, so start at eight intervals per radian. Simpson's error
        // falls as h^4, so each halving is Richardson-extrapolated, and the
        // result is accepted when successive extrapolations agree.
        let grid = super::power::ProfileGrid::new(stack, polarisation, k, beta, 8, 2)?
            .ok_or(Quadrature)?;
        let decay = sig.exteriors().map(|r| r.map(|r| 2. * r.im));
        let mut previous: Option<(f64, f64)> = None;
        let mut previous_estimate: Option<f64> = None;
        for level in 0..8 {
            let sampling = grid.sampling(level).ok_or(Quadrature)?;
            let recovered = self.recover_fields_with(backend.clone(), &sampling, settings)?;
            let fields = recovered.fields.quantity();
            let (e, h) = (fields.electric(), fields.magnetic());
            let flux =
                |i: usize| 0.5 * (e.y()[i] * h.z()[i].conj() - e.z()[i] * h.y()[i].conj()).re;
            let energy = |i: usize, [we, wm]: [Principal<f64>; 2]| {
                let e2 = [
                    e.x()[i].norm_sqr(),
                    e.y()[i].norm_sqr(),
                    e.z()[i].norm_sqr(),
                ];
                let h2 = [
                    h.x()[i].norm_sqr(),
                    h.y()[i].norm_sqr(),
                    h.z()[i].norm_sqr(),
                ];
                0.25 * (we.weigh(e2) + wm.weigh(h2))
            };
            let (mut power, mut stored) = (0., 0.);
            for span in grid.spans(level) {
                let weight = weights[span.region];
                match span.intervals {
                    None => {
                        let decay = decay[usize::from(span.region > 0)].expect("open");
                        power += flux(span.first) / decay;
                        stored += energy(span.first, weight) / decay;
                    }
                    Some(_) => {
                        power += span.simpson(flux);
                        stored += span.simpson(|i| energy(i, weight));
                    }
                }
            }
            let extrapolated = previous.map(|(p, w)| {
                let richardson = |fine: f64, coarse: f64| fine + (fine - coarse) / 15.;
                richardson(stored, w) / richardson(power, p)
            });
            previous = Some((power, stored));
            let Some(ratio) = extrapolated else {
                continue;
            };
            if !ratio.is_finite() {
                return Err(Quadrature);
            }
            let settled =
                previous_estimate.is_some_and(|p| (p - ratio).abs() <= 1e-12 * ratio.abs());
            previous_estimate = Some(ratio);
            if settled {
                let relative_discrepancy = (derivative - ratio).norm() / ratio.abs();
                if relative_discrepancy > tolerance {
                    return Err(Uncertified {
                        derivative,
                        energy_estimate: ratio,
                        relative_discrepancy,
                        tolerance,
                    });
                }
                return Ok(GroupIndex {
                    value: derivative.re,
                    derivative,
                    energy_estimate: ratio,
                    relative_discrepancy,
                    tolerance,
                    sensitivity,
                });
            }
        }
        Err(Quadrature)
    }
}
