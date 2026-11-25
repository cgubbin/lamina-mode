//! The direct β chart of a stack closed on both sides, by walls or by
//! wall-backed PMLs. With no open side the characteristic carries no exterior
//! wavevector: it is entire and even in β at fixed k0, so β itself, in cm^-1,
//! is the native coordinate and needs no continuation, sheet or radiation
//! reference.
//!
//! A PML side enters the characteristic normalised by exp(iκ_ref τ), with τ
//! its complex thickness and κ_ref = k0 √(εμ) its material's wavevector at
//! β = 0, which leaves the scale exp(−Im(κ_ref τ) + |Im(κ τ)|), κ² = εμk0² − β².
//! A rectangle on which that scale cannot be represented is refused.
use super::frequency::{contains, halo};
use super::*;
use crate::{conformal::ConformalPoint, conformal::in_plane::ComplexInPlaneMap};
use lamina_core::{CanonicalCoordinates, ExteriorWavevectors, ScalarAlgebra};

pub type DirectDomain<S> = ProblemDomain<ComplexInPlaneMap<J>, S, DirectChart>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DirectChartError {
    #[error("a direct β domain needs a stack closed on both sides")]
    OpenSide,
    #[error("β lies outside the chart region")]
    OutsideRegion,
    #[error("cannot construct a finite β evaluation halo")]
    InvalidMargin,
    #[error(
        "the {side:?} PML normalisation scale reaches exp({log_magnitude}) on the β rectangle, \
         outside the representable range"
    )]
    PmlScale { side: StackSide, log_magnitude: i64 },
    #[error(transparent)]
    Constitutive(#[from] ConstitutiveError),
}

/// Largest accepted magnitude of the log-scale of the characteristic. The
/// search divides by the characteristic, squaring its magnitude, and doubles
/// span about e^±709, so the scale stays within e^±300, leaving room for the
/// rest of the stack.
const PML_LOG_SCALE_LIMIT: f64 = 300.0;
/// Samples per edge of the rectangle for the largest PML scale.
const PML_SCALE_SAMPLES: usize = 256;

/// Refuse a rectangle on which the PML sides' log-scale
/// Σ [−Im(κ_ref τ) + |Im(κ τ)|] leaves the representable range. |Im(κ τ)|
/// does not depend on the root and is subharmonic in β, so the largest value
/// lies on the boundary, sampled here, and the least is at least Σ −Im(κ_ref τ).
/// The refusal names the side with the largest share.
fn check_pml_scale(
    stack: &ModeStack,
    polarisation: Polarisation,
    k0: f64,
    region: ConformalRegion<f64>,
) -> Result<(), DirectChartError> {
    let pmls = pml_sides(stack, polarisation, k0)?;
    let (x0, x1) = region.real_bounds();
    let (y0, y1) = region.imaginary_bounds();
    let corners = [
        C::new(x0, y0),
        C::new(x1, y0),
        C::new(x1, y1),
        C::new(x0, y1),
    ];
    let boundary = (0..4).flat_map(|edge| {
        let (a, b) = (corners[edge], corners[(edge + 1) % 4]);
        (0..PML_SCALE_SAMPLES).map(move |i| a + (b - a) * (i as f64 / PML_SCALE_SAMPLES as f64))
    });
    let total = |shares: &[(StackSide, f64)]| shares.iter().map(|share| share.1).sum::<f64>();
    // Each side's share where the boundary samples peak, then the lower bounds.
    let peak = boundary
        .map(|beta| {
            pmls.iter()
                .map(|pml| (pml.side, pml.share(beta)))
                .collect::<Vec<_>>()
        })
        .max_by(|a, b| total(a).total_cmp(&total(b)));
    let lower = pmls.iter().map(|pml| (pml.side, pml.floor())).collect();
    for shares in peak.into_iter().chain([lower]) {
        let log_scale = total(&shares);
        if let Some(&(side, _)) = shares.iter().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            && (!log_scale.is_finite() || log_scale.abs() > PML_LOG_SCALE_LIMIT)
        {
            return Err(DirectChartError::PmlScale {
                side,
                log_magnitude: if log_scale.is_nan() {
                    i64::MAX
                } else {
                    log_scale.round() as i64
                },
            });
        }
    }
    Ok(())
}

/// A PML side: its exterior's channel, with κ² = s (χ k0² − β²), and its
/// complex thickness τ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PmlSide {
    side: StackSide,
    /// χ k0², the reference κ² before the channel's scale.
    chi_k0: C,
    scale: Option<C>,
    tau: C,
}

impl PmlSide {
    /// κ² = s (χ k0² − β²).
    fn kappa_squared(&self, beta: C) -> C {
        let q = self.chi_k0 - beta * beta;
        self.scale.map_or(q, |s| s * q)
    }

    /// The side's share −Im(κ_ref τ) of the least log-scale, κ_ref = κ(β = 0).
    fn floor(&self) -> f64 {
        let reference = self.scale.map_or(self.chi_k0, |s| s * self.chi_k0);
        -(reference.sqrt() * self.tau).im
    }

    /// The side's share −Im(κ_ref τ) + |Im(κ τ)| of the log-scale at β.
    fn share(&self, beta: C) -> f64 {
        self.floor() + (self.kappa_squared(beta).sqrt() * self.tau).im.abs()
    }
}

/// Each PML side of `stack` for `polarisation` at real `k0`.
pub(crate) fn pml_sides(
    stack: &ModeStack,
    polarisation: Polarisation,
    k0: f64,
) -> Result<Vec<PmlSide>, ConstitutiveError> {
    let channels = material::exterior_channels(stack, C::new(k0, 0.0), polarisation)?;
    Ok([StackSide::Left, StackSide::Right]
        .into_iter()
        .zip(pmls(stack))
        .zip(channels)
        .filter_map(|((side, pml), channel)| {
            pml.map(|(_, tau)| PmlSide {
                side,
                chi_k0: channel.chi() * k0 * k0,
                scale: channel.scale(),
                tau,
            })
        })
        .collect())
}

/// The log-scale Σ [−Im(κ_ref τ) + |Im(κ τ)|] of the characteristic at β:
/// zero without PMLs.
pub(crate) fn pml_log_scale(sides: &[PmlSide], beta: C) -> f64 {
    sides.iter().map(|side| side.share(beta)).sum()
}

/// The least log-scale Σ −Im(κ_ref τ) of the characteristic over every β:
/// where the PML sides' transfer is of order one, near the rotated
/// continuum, the characteristic is of order exp of this. Zero without PMLs.
pub(crate) fn pml_log_floor(sides: &[PmlSide]) -> f64 {
    sides.iter().map(PmlSide::floor).sum()
}

/// A β rectangle in cm^-1 at fixed real k0.
#[derive(Clone, Debug)]
pub struct DirectChart {
    region: ConformalRegion<f64>,
    k0: f64,
}

impl AnalyticChart<J> for DirectChart {
    type Error = DirectChartError;
    fn region(&self) -> &ConformalRegion<f64> {
        &self.region
    }
    fn reference(&self) -> C {
        let (x0, x1) = self.region.real_bounds();
        let (y0, y1) = self.region.imaginary_bounds();
        C::new(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    }
    fn reference_coordinate(&self) -> C {
        self.reference()
    }
    fn coordinate_kind(&self) -> CoordinateKind {
        CoordinateKind::Wavenumber
    }
    fn coordinate(&self, beta: C) -> Result<C, DirectChartError> {
        if contains(self.region, beta) {
            Ok(beta)
        } else {
            Err(DirectChartError::OutsideRegion)
        }
    }
    fn map(&self, beta: J) -> Result<ConformalPoint<J>, DirectChartError> {
        self.coordinate(beta.value()[()])?;
        Ok(ConformalPoint::new(
            CanonicalCoordinates::new(beta.constant(C::new(self.k0, 0.0)), beta),
            ExteriorWavevectors::closed(),
        ))
    }
    fn with_evaluation_margin(&self) -> Result<Self, DirectChartError> {
        Ok(Self {
            region: halo(self.region).ok_or(DirectChartError::InvalidMargin)?,
            k0: self.k0,
        })
    }
}

impl<S: ModeDeterminantSolver<J> + Clone> ModalProblem<ComplexInPlaneMap<J>, S> {
    /// Search a β rectangle (cm^-1) of a stack closed on both sides, by walls
    /// or PMLs. The declaration covers the characteristic on the whole
    /// evaluation halo. A rectangle on which a PML's normalisation scale
    /// cannot be represented is refused.
    pub fn direct_domain(
        &self,
        bounds: ConformalRegion<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<DirectDomain<S>, DirectChartError> {
        if self.context.open_sides() != [false; 2] {
            return Err(DirectChartError::OpenSide);
        }
        let chart = DirectChart {
            region: bounds,
            k0: self.map.k0().value()[()].re,
        };
        let reference = chart.reference();
        let k0 = chart.k0;
        let search = SingleChartModalSearch::new(ConformalDispersion::<_, _, C, _>::with_chart(
            self.solver.clone(),
            chart,
        ))?;
        check_pml_scale(
            &self.context.stack,
            self.context.polarisation,
            k0,
            *search.dispersion().chart().region(),
        )?;
        let provenance = DomainProvenance {
            domain_identity: Arc::new(()),
            context: self.context.clone(),
            requested: bounds,
            evaluation: *search.dispersion().chart().region(),
            selection: ReferenceSelection::Coordinate {
                reference,
                coordinate: reference,
            },
            reference,
            reference_coordinate: reference,
            assumption,
            radiation: radiation::RadiationProvenance::Unspecified,
            coordinate_kind: CoordinateKind::Wavenumber,
            frequency: None,
        };
        Ok(ProblemDomain { search, provenance })
    }
}
