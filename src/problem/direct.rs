//! The direct β chart of a stack closed by walls on both sides. With no open
//! side the characteristic carries no exterior wavevector: it is entire and
//! even in β at fixed k0, so β itself, in cm^-1, is the native coordinate and
//! needs no continuation, sheet or radiation reference.
use super::frequency::{contains, halo};
use super::*;
use crate::{conformal::ConformalPoint, conformal::in_plane::ComplexInPlaneMap};
use lamina_core::{CanonicalCoordinates, ExteriorWavevectors, ScalarAlgebra};

pub type DirectDomain<S> = ProblemDomain<ComplexInPlaneMap<J>, S, DirectChart>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DirectChartError {
    #[error("a direct β domain needs a stack closed by walls on both sides")]
    OpenSide,
    #[error("β lies outside the chart region")]
    OutsideRegion,
    #[error("cannot construct a finite β evaluation halo")]
    InvalidMargin,
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
    /// Search a β rectangle (cm^-1) of a stack closed by walls on both sides.
    /// The declaration covers the characteristic on the whole evaluation halo.
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
        let search = SingleChartModalSearch::new(ConformalDispersion::<_, _, C, _>::with_chart(
            self.solver.clone(),
            chart,
        ))?;
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
