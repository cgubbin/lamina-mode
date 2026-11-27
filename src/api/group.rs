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
    #[error(
        "requires real, positive entries where the polarisation sees them: {entry} of {} is {value}",
        super::guided::region_name(*region)
    )]
    IneligibleEntry {
        region: lamina_core::StackRegion,
        entry: TensorEntry,
        value: Complex,
    },
    #[error("the wavelength sensitivity failed: {0:?}")]
    Sensitivity(GuidedBetaError),
    #[error("the stored-energy and power integrals failed")]
    Integrals(#[source] std::sync::Arc<dyn std::error::Error + Send + Sync>),
    #[error("the power vanishes")]
    IllConditioned,
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
    /// stored energy and the power are closed-form integrals of the mode's
    /// fields; of the field-recovery settings only the amplitude enters, and
    /// it cancels in their ratio.
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
            .map_err(|e| match e {
                GuidedBetaError::IneligibleEntry {
                    region,
                    entry,
                    value,
                } => IneligibleEntry {
                    region,
                    entry,
                    value,
                },
                other => Sensitivity(other),
            })?;
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
                let names = TensorEntry::channel(polarisation);
                for entries in seen {
                    if let Some((entry, value)) =
                        names.into_iter().zip(entries).find(|(_, v)| !real(v))
                    {
                        return Err(IneligibleEntry {
                            region,
                            entry,
                            value,
                        });
                    }
                }
                Ok(weights.map(|w| w.map(|z| z.re)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Stored energy ¼ ∫ (E† ε_d E + η† μ_d η) dz and power
        // ½ Re ∫ (E_y η_z* − E_z η_y*) dz in closed form, from one solve.
        crate::problem::walls(stack).map_err(|side| super::PmlIntegral { side })?;
        let energy: Vec<super::forms::Weight> = weights
            .iter()
            .map(|pair| {
                let diagonal = |w: &Principal<f64>| {
                    let entries = w.entries();
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| match i == j {
                            true => Complex::from(0.25 * entries[i]),
                            false => Complex::ZERO,
                        })
                    })
                };
                super::forms::blocks(&[diagonal(&pair[0]), diagonal(&pair[1])])
            })
            .collect();
        let amplitude = settings.amplitude;
        let integrals = super::forms::still_forms(
            backend,
            [self, self],
            [amplitude, amplitude],
            &[(true, &energy), (true, &[super::forms::flux()])],
        )
        .map_err(|_| Ineligible)?
        .map_err(Integrals)?;
        let [stored, power] = [0, 1].map(|form| integrals[form].iter().sum::<Complex>().re);
        let ratio = stored / (0.5 * power);
        if !ratio.is_finite() {
            return Err(IllConditioned);
        }
        let relative_discrepancy = (derivative - ratio).norm() / ratio.abs();
        if relative_discrepancy > tolerance {
            return Err(Uncertified {
                derivative,
                energy_estimate: ratio,
                relative_discrepancy,
                tolerance,
            });
        }
        Ok(GroupIndex {
            value: derivative.re,
            derivative,
            energy_estimate: ratio,
            relative_discrepancy,
            tolerance,
            sensitivity,
        })
    }
}
