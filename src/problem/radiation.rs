//! Real-anchor radiation verification, separate from the coordinate square root.
//! The anchor must be strictly inside the requested chart. Verification establishes
//! the selected analytic continuation, not material analyticity or global coverage.
use super::material::{Channel, exterior_channels};
use super::*;
use lamina_core::ScalarAlgebra;
use ndarray::arr0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiationChoice {
    OutgoingOrDecaying,
    IncomingOrGrowing,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiationReference {
    pub z: C,
    /// Real varying physical coordinate, in cm^-1.
    pub coordinate: f64,
    /// One choice per open side; `None` exactly on a side closed by a wall or a PML.
    pub choices: [Option<RadiationChoice>; 2],
    pub tolerance: RadiationTolerance,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiationTolerance {
    pub absolute: f64,
    pub relative: f64,
    /// Positive wavevector cutoff band, in cm^-1.
    pub cutoff: f64,
}
impl Default for RadiationTolerance {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-10,
            cutoff: 1e-10,
        }
    }
}
impl RadiationTolerance {
    fn valid(self) -> bool {
        [self.absolute, self.relative, self.cutoff]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
            && self.cutoff > 0.0
    }
}
/// Only the problem-owned constructor can establish this evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRadiationReference {
    reference: RadiationReference,
    k0: f64,
    beta: f64,
    epsilon: [f64; 2],
    observed: [Option<C>; 2],
}
impl VerifiedRadiationReference {
    pub fn reference(&self) -> RadiationReference {
        self.reference
    }
    pub fn physical_coordinates(&self) -> (f64, f64) {
        (self.k0, self.beta)
    }
    pub fn epsilon(&self) -> [f64; 2] {
        self.epsilon
    }
    pub fn observed(&self) -> [Option<C>; 2] {
        self.observed
    }
    pub fn all_outgoing(&self) -> bool {
        self.reference
            .choices
            .iter()
            .flatten()
            .all(|choice| *choice == RadiationChoice::OutgoingOrDecaying)
    }
}
#[derive(Clone, Debug, PartialEq, Default)]
pub enum RadiationProvenance {
    #[default]
    Unspecified,
    Verified(VerifiedRadiationReference),
}
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RadiationReferenceError<E: std::error::Error + 'static> {
    #[error("invalid radiation tolerance")]
    InvalidTolerance,
    #[error(
        "radiation reference must be finite, real in physical coordinates, and strictly inside the requested chart with k0>0"
    )]
    InvalidReference,
    #[error("exterior {side} requires an isotropic channel with positive real epsilon and mu=1")]
    UnsupportedMaterial { side: usize },
    #[error("exterior {side} is within the cutoff boundary band")]
    Cutoff { side: usize },
    #[error("exterior {side} does not match the requested radiation branch")]
    BranchMismatch {
        side: usize,
        expected: C,
        observed: C,
    },
    #[error("non-finite radiation reference data")]
    NonFinite,
    #[error("radiation chart construction failed: {0}")]
    Chart(#[source] ConformalChartError<E>),
}
pub(super) fn positive_dielectric(e: C, m: C) -> bool {
    e.re.is_finite() && e.im == 0.0 && e.re > 0.0 && m == C::new(1.0, 0.0)
}
pub(super) fn finite(c: C) -> bool {
    c.re.is_finite() && c.im.is_finite()
}

impl<M, S> ModalProblem<M, S>
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    S: ModeDeterminantSolver<J> + Clone,
{
    /// Verify a real anchor without moving the rectangle or flipping exterior
    /// roots. Cutoffs, unsupported materials and branch mismatches fail before
    /// discovery. The reference defines analytic continuation within this chart;
    /// it does not certify a causal sheet on arbitrary complex contours.
    pub fn domain_with_radiation_reference(
        &self,
        requested: ConformalRegion<f64>,
        reference: RadiationReference,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<ProblemDomain<M, S>, RadiationReferenceError<RegionError<M>>> {
        use RadiationReferenceError as Err;
        let tol = reference.tolerance;
        if !tol.valid() {
            return Err(Err::InvalidTolerance);
        }
        let (xmin, xmax) = requested.real_bounds();
        let (ymin, ymax) = requested.imaginary_bounds();
        let (k0, beta) = match self.context.coordinate {
            SearchCoordinate::InPlane { fixed_k0 } => (fixed_k0, reference.coordinate),
            SearchCoordinate::Spectral { fixed_k_parallel } => {
                (reference.coordinate, fixed_k_parallel)
            }
        };
        if !finite(reference.z)
            || !k0.is_finite()
            || !beta.is_finite()
            || k0 <= 0.0
            || !(reference.z.re > xmin
                && reference.z.re < xmax
                && reference.z.im > ymin
                && reference.z.im < ymax)
        {
            return Err(Err::InvalidReference);
        }
        if reference.choices.map(|c| c.is_some()) != self.context.open_sides() {
            return Err(Err::InvalidReference);
        }
        let pol = self.context.polarisation;
        let channels = exterior_channels(&self.context.stack, C::new(k0, 0.0), pol)
            .map_err(|_| Err::NonFinite)?;
        let expected = anchor_wavevectors(channels, pol, k0, beta, reference.choices, tol)?;
        let mut domain = self
            .domain(
                requested,
                ReferenceSelection::Coordinate {
                    reference: reference.z,
                    coordinate: C::new(reference.coordinate, 0.0),
                },
                config,
                assumption,
            )
            .map_err(Err::Chart)?;
        let point = domain
            .search
            .dispersion()
            .chart()
            .map(J::lift_constant(arr0(reference.z)))
            .map_err(Err::Chart)?;
        let actual_k0 = point.coordinates().vacuum_angular_wavenumber().value()[()];
        let actual_beta = point.coordinates().parallel_angular_wavenumber().value()[()];
        let value = |a: &J| a.value()[()];
        let exterior = point.exterior_wavevectors();
        let observed = [
            exterior
                .left()
                .and_then(lamina_core::ExteriorBranch::scalar)
                .map(value),
            exterior
                .right()
                .and_then(lamina_core::ExteriorBranch::scalar)
                .map(value),
        ];
        let scale = expected
            .iter()
            .flatten()
            .map(|v| v.norm())
            .fold(k0.abs().max(beta.abs()), f64::max);
        let limit = tol.absolute + tol.relative * scale;
        if !limit.is_finite()
            || ![actual_k0, actual_beta]
                .into_iter()
                .chain(observed.into_iter().flatten())
                .all(finite)
        {
            return Err(Err::NonFinite);
        }
        if (actual_k0 - C::new(k0, 0.0)).norm() > limit
            || (actual_beta - C::new(beta, 0.0)).norm() > limit
        {
            return Err(Err::InvalidReference);
        }
        for side in 0..2 {
            let (Some(expected), Some(observed)) = (expected[side], observed[side]) else {
                continue;
            };
            // Reject even if an excessively loose comparison band spans both signs.
            if (observed - expected).norm() > limit
                || (observed + expected).norm() <= (observed - expected).norm()
            {
                return Err(Err::BranchMismatch {
                    side,
                    expected,
                    observed,
                });
            }
        }
        domain.provenance.radiation = RadiationProvenance::Verified(VerifiedRadiationReference {
            reference,
            k0,
            beta,
            epsilon: channels.map(|c| c.factors(pol).0.re),
            observed,
        });
        Ok(domain)
    }
}

/// The exterior wavevector each open side's choice selects at a real anchor;
/// a closed side, with no choice, has none.
fn anchor_wavevectors<E: std::error::Error + 'static>(
    channels: [Channel; 2],
    pol: Polarisation,
    k0: f64,
    beta: f64,
    choices: [Option<RadiationChoice>; 2],
    tolerance: RadiationTolerance,
) -> Result<[Option<C>; 2], RadiationReferenceError<E>> {
    use RadiationReferenceError as Err;
    let mut selected = [None; 2];
    for (side, (channel, choice)) in channels.into_iter().zip(choices).enumerate() {
        let Some(choice) = choice else { continue };
        let (e, m) = channel.factors(pol);
        if !channel.entries().into_iter().all(finite) {
            return Err(Err::NonFinite);
        }
        if channel.scale().is_some() || !positive_dielectric(e, m) {
            return Err(Err::UnsupportedMaterial { side });
        }
        let q = e.re * k0 * k0 - beta * beta;
        let magnitude = q.abs().sqrt();
        if !magnitude.is_finite() {
            return Err(Err::NonFinite);
        }
        if magnitude <= tolerance.cutoff {
            return Err(Err::Cutoff { side });
        }
        let outgoing = if q > 0.0 {
            C::new(magnitude, 0.0)
        } else {
            C::new(0.0, magnitude)
        };
        selected[side] = Some(match choice {
            RadiationChoice::OutgoingOrDecaying => outgoing,
            RadiationChoice::IncomingOrGrowing => -outgoing,
        });
    }
    Ok(selected)
}

/// Direct-frequency anchor selection uses actual material values at this point;
/// it does not establish that the exterior is constant elsewhere.
pub(super) fn frequency_reference(
    context: &ProblemContext,
    k0: f64,
    choices: [RadiationChoice; 2],
    tolerance: RadiationTolerance,
) -> Result<VerifiedRadiationReference, RadiationReferenceError<std::convert::Infallible>> {
    use RadiationReferenceError as Err;
    if !tolerance.valid() {
        return Err(Err::InvalidTolerance);
    }
    let SearchCoordinate::Spectral {
        fixed_k_parallel: beta,
    } = context.coordinate
    else {
        return Err(Err::InvalidReference);
    };
    if !k0.is_finite() || k0 <= 0.0 || !beta.is_finite() {
        return Err(Err::InvalidReference);
    }
    let pol = context.polarisation;
    let channels =
        exterior_channels(&context.stack, C::new(k0, 0.0), pol).map_err(|_| Err::NonFinite)?;
    let choices = choices.map(Some);
    let observed = anchor_wavevectors(channels, pol, k0, beta, choices, tolerance)?;
    Ok(VerifiedRadiationReference {
        reference: RadiationReference {
            z: C::new(k0, 0.0),
            coordinate: k0,
            choices,
            tolerance,
        },
        k0,
        beta,
        epsilon: channels.map(|c| c.factors(pol).0.re),
        observed,
    })
}

#[cfg(test)]
mod tests;
