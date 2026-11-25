//! Bound internal mode problems. In-plane construction samples the actual stack
//! exteriors at fixed k0. Conformal spectral construction assembles a stack from
//! constant exterior descriptors and analytical finite layers. Independent maps
//! and exterior replacements are deliberately absent from this interface.
//! Direct frequency construction reuses the supplied stack with actual exterior
//! dispersion through fallible frequency charts.
//!
//! Fixed coordinates use cm^-1 and retain the raw maps' real-coordinate limits.
//! Material behavior must remain unchanged during a problem's lifetime: cloning
//! an erased handle cannot freeze arbitrary user-defined interior mutable state.
//! This is an internal construction layer, not the public physical-input API.
use crate::analytic::{AnalyticChart, CoordinateKind};
use crate::dispersion::ChartDispersionError;
pub mod direct;
pub mod frequency;
pub(crate) mod material;
pub mod orchestration;
pub mod physical;
pub mod radiation;
use crate::{
    conformal::{
        ConformalRegion, ValidateConformalRegion,
        chart::{
            ConformalChart, ConformalChartConfig, ConformalChartError, CoordinateSheet,
            LinearCoordinateMap,
        },
        in_plane::{ComplexInPlaneMap, InPlaneMapError},
        wavenumber::{ComplexWavenumberMap, WavenumberMapError},
    },
    dispersion::{
        CompiledPlaneWaveModeSolver, ConformalDispersion, DispersionError, ModeDeterminantSolver,
    },
    search::{ModalSearchError, ModalSearchResult, SingleChartModalSearch},
};
use argloc::{ArgumentConfig, FunctionBundle};
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialStack, Backend, ComplexPlane, Layer, ModeJet1,
    Polarisation, Stack, StackCompileError, Termination, WallKind,
};
use lamina_units::Length;
pub use material::ConstantExterior;
use nalgebra::Complex;
use ndarray::Ix0;
use std::sync::Arc;
pub type C = Complex<f64>;
pub type J = ModeJet1<C, Ix0>;
pub(crate) type Compiled<B> = CompiledPlaneWaveModeSolver<J, AnalyticalMaterialHandle<C>, B>;
pub type InPlaneProblem<B> = ModalProblem<ComplexInPlaneMap<J>, Compiled<B>>;
pub type SpectralProblem<B> = ModalProblem<ComplexWavenumberMap<J>, Compiled<B>>;
type RegionError<M> = <M as ValidateConformalRegion<f64>>::Error;
type OpticalError<M, S> = DispersionError<RegionError<M>, <S as ModeDeterminantSolver<J>>::Error>;
type SearchError<M, S> = ModalSearchError<C, OpticalError<M, S>>;
type NativeOpticalError<H, S> =
    ChartDispersionError<<H as AnalyticChart<J>>::Error, <S as ModeDeterminantSolver<J>>::Error>;
type NativeSearchError<H, S> = ModalSearchError<C, NativeOpticalError<H, S>>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchCoordinate {
    InPlane { fixed_k0: f64 },
    Spectral { fixed_k_parallel: f64 },
}
/// Caller assertion, never a material certificate or a conclusion from samples.
/// The characteristic must be holomorphic and pole-free on a neighborhood of
/// the closed domain containing the entire evaluation halo. This includes all
/// materials at fixed k0 (in-plane) or throughout mapped k0 (spectral), consistent
/// exterior branches, and validity of the chosen characteristic representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyticAssumption {
    CallerAssumedHolomorphicPoleFree,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceSelection {
    Sheet(CoordinateSheet),
    Coordinate { reference: C, coordinate: C },
}
#[derive(Debug)]
pub struct ProblemContext {
    pub(crate) constants: Option<[ConstantExterior; 2]>,
    stack: AnalyticalMaterialStack<C>,
    polarisation: Polarisation,
    coordinate: SearchCoordinate,
}
impl ProblemContext {
    pub fn stack(&self) -> &AnalyticalMaterialStack<C> {
        &self.stack
    }
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }
    pub fn coordinate(&self) -> SearchCoordinate {
        self.coordinate
    }
    /// Whether each side, left then right, is open rather than closed by a wall.
    pub(crate) fn open_sides(&self) -> [bool; 2] {
        open_sides(&self.stack)
    }
}
pub(crate) fn open_sides<M>(stack: &Stack<M, f64>) -> [bool; 2] {
    walls(stack).map(|wall| wall.is_none())
}
/// Each side's wall and its gap from the outermost interface in cm, `None`
/// when open. The right gap is formed as lamina-core validates it, in the
/// wall's unit when the layers share it, so a wall on the interface has none.
pub(crate) fn walls<M>(stack: &Stack<M, f64>) -> [Option<(WallKind, f64)>; 2] {
    let gap = |termination: &Termination<f64>, right: bool| match termination {
        Termination::Open => None,
        Termination::Wall(wall) => {
            let position = wall.position();
            let unit = position.unit();
            let gap = if !right {
                -position.as_centimetres()
            } else if stack.iter().all(|layer| layer.thickness().unit() == unit) {
                let total = stack.iter().fold(0.0, |t, l| t + l.thickness().value());
                Length::new(position.value() - total, unit).as_centimetres()
            } else {
                let total = stack
                    .iter()
                    .fold(0.0, |t, l| t + l.thickness().as_centimetres());
                position.as_centimetres() - total
            };
            Some((wall.kind(), gap.max(0.0)))
        }
        _ => unimplemented!("a side is open or closed by a wall"),
    };
    [
        gap(stack.left_termination(), false),
        gap(stack.right_termination(), true),
    ]
}
#[derive(Debug, thiserror::Error)]
pub enum ProblemError {
    #[error("in-plane map construction failed: {0}")]
    InPlane(#[from] InPlaneMapError),
    #[error("spectral map construction failed: {0}")]
    Spectral(#[from] WavenumberMapError),
    #[error("optical stack compilation failed: {0}")]
    Stack(#[from] StackCompileError<f64>),
    #[error("this strategy needs a stack open on both sides: a wall termination closes one")]
    ClosedSide,
}
#[derive(Clone, Debug, thiserror::Error)]
pub enum ExecutionError<E: std::error::Error + 'static> {
    #[error("domain belongs to another physical problem")]
    ForeignDomain,
    #[error("bound optical execution failed: {0}")]
    Optical(#[source] Box<E>),
}
#[derive(Clone, Debug)]
pub struct ModalProblem<M, S> {
    context: Arc<ProblemContext>,
    map: M,
    solver: S,
}

impl<B> InPlaneProblem<B>
where
    B: Backend<J, ComplexPlane>,
{
    pub fn in_plane(
        stack: AnalyticalMaterialStack<C>,
        fixed_k0: f64,
        polarisation: Polarisation,
        backend: B,
    ) -> Result<Self, ProblemError> {
        if !fixed_k0.is_finite() || fixed_k0 == 0.0 {
            return Err(InPlaneMapError::InvalidWavenumber.into());
        }
        // Only open sides carry a branch: with one open side, both map
        // constants are that side's material (r = 0).
        let open = open_sides(&stack);
        let factors = radiation::exterior_factors(&stack, C::new(fixed_k0, 0.0));
        let [(el, ml), (er, mr)] = match open {
            [true, false] => [factors[0]; 2],
            [false, true] => [factors[1]; 2],
            _ => factors,
        };
        let map = ComplexInPlaneMap::new(fixed_k0, el, ml, er, mr)?.with_open_sides(open);
        let solver = Compiled::compile(&stack, backend, polarisation)?;
        Ok(Self {
            context: Arc::new(ProblemContext {
                constants: None,
                stack,
                polarisation,
                coordinate: SearchCoordinate::InPlane { fixed_k0 },
            }),
            map,
            solver,
        })
    }
}
impl<B> SpectralProblem<B>
where
    B: Backend<J, ComplexPlane>,
{
    /// Only constant descriptors can supply exteriors; arbitrary analytical
    /// layers remain dispersive. No full-stack-plus-replacement-constants form.
    pub fn spectral(
        left: ConstantExterior,
        right: ConstantExterior,
        layers: impl IntoIterator<Item = Layer<AnalyticalMaterialHandle<C>, f64>>,
        fixed_k_parallel: f64,
        polarisation: Polarisation,
        backend: B,
    ) -> Result<Self, ProblemError> {
        let map = ComplexWavenumberMap::new(
            fixed_k_parallel,
            left.epsilon(),
            left.mu(),
            right.epsilon(),
            right.mu(),
        )?;
        let mut builder = Stack::builder(
            AnalyticalMaterialHandle::new(left),
            AnalyticalMaterialHandle::new(right),
        );
        for layer in layers {
            let (material, thickness) = layer.into_parts();
            builder = builder.layer(material, thickness);
        }
        let stack = builder.finalise();
        let solver = Compiled::compile(&stack, backend, polarisation)?;
        Ok(Self {
            context: Arc::new(ProblemContext {
                constants: Some([left, right]),
                stack,
                polarisation,
                coordinate: SearchCoordinate::Spectral { fixed_k_parallel },
            }),
            map,
            solver,
        })
    }
}
#[derive(Clone, Debug)]
pub struct DomainProvenance {
    domain_identity: Arc<()>,
    context: Arc<ProblemContext>,
    requested: ConformalRegion<f64>,
    evaluation: ConformalRegion<f64>,
    selection: ReferenceSelection,
    reference: C,
    reference_coordinate: C,
    assumption: AnalyticAssumption,
    radiation: radiation::RadiationProvenance,
    coordinate_kind: CoordinateKind,
    frequency: Option<frequency::FrequencyEvidence>,
}
impl DomainProvenance {
    pub fn frequency(&self) -> Option<&frequency::FrequencyEvidence> {
        self.frequency.as_ref()
    }
    pub fn coordinate_kind(&self) -> CoordinateKind {
        self.coordinate_kind
    }
    pub fn same_domain(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.domain_identity, &other.domain_identity)
    }
    pub fn problem(&self) -> &ProblemContext {
        &self.context
    }
    pub fn belongs_to<M, S>(&self, problem: &ModalProblem<M, S>) -> bool {
        Arc::ptr_eq(&self.context, &problem.context)
    }
    pub fn requested(&self) -> ConformalRegion<f64> {
        self.requested
    }
    pub fn evaluation(&self) -> ConformalRegion<f64> {
        self.evaluation
    }
    pub fn selection(&self) -> ReferenceSelection {
        self.selection
    }
    pub fn reference(&self) -> C {
        self.reference
    }
    pub fn reference_coordinate(&self) -> C {
        self.reference_coordinate
    }
    pub fn radiation(&self) -> &radiation::RadiationProvenance {
        &self.radiation
    }
    pub fn assumption(&self) -> AnalyticAssumption {
        self.assumption
    }
}
#[derive(Clone, Debug)]
pub struct ProblemDomain<M, S, H = ConformalChart<M, J>> {
    search: SingleChartModalSearch<S, M, C, H>,
    provenance: DomainProvenance,
}
impl<M, S, H> ProblemDomain<M, S, H> {
    pub fn provenance(&self) -> &DomainProvenance {
        &self.provenance
    }
}
#[derive(Debug)]
pub struct ProblemSearchResult {
    result: ModalSearchResult<C>,
    provenance: DomainProvenance,
}
impl ProblemSearchResult {
    pub fn result(&self) -> &ModalSearchResult<C> {
        &self.result
    }
    pub fn provenance(&self) -> &DomainProvenance {
        &self.provenance
    }
    pub fn into_parts(self) -> (ModalSearchResult<C>, DomainProvenance) {
        (self.result, self.provenance)
    }
}
impl<M, S> ModalProblem<M, S>
where
    M: LinearCoordinateMap<J> + ValidateConformalRegion<f64> + Clone,
    S: ModeDeterminantSolver<J> + Clone,
{
    pub fn domain(
        &self,
        requested: ConformalRegion<f64>,
        selection: ReferenceSelection,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<ProblemDomain<M, S>, ConformalChartError<RegionError<M>>> {
        if self.context.open_sides() == [false; 2] {
            return Err(ConformalChartError::ClosedStack);
        }
        let chart = match selection {
            ReferenceSelection::Sheet(sheet) => {
                ConformalChart::new(self.map.clone(), requested, sheet, config)?
            }
            ReferenceSelection::Coordinate {
                reference,
                coordinate,
            } => ConformalChart::with_reference_coordinate(
                self.map.clone(),
                requested,
                reference,
                coordinate,
                config,
            )?,
        };
        let search =
            SingleChartModalSearch::new(ConformalDispersion::new(self.solver.clone(), chart))?;
        let chart = search.dispersion().chart();
        let provenance = DomainProvenance {
            domain_identity: Arc::new(()),
            context: self.context.clone(),
            requested,
            evaluation: *chart.region(),
            selection,
            reference: chart.reference(),
            reference_coordinate: chart.reference_coordinate(),
            assumption,
            radiation: radiation::RadiationProvenance::Unspecified,
            coordinate_kind: chart.coordinate_kind(),
            frequency: None,
        };
        Ok(ProblemDomain { search, provenance })
    }
}
impl<M, S> ModalProblem<M, S>
where
    M: Clone,
    S: ModeDeterminantSolver<J> + Clone,
{
    pub fn context(&self) -> &ProblemContext {
        &self.context
    }
    pub fn evaluate<H: AnalyticChart<J>>(
        &self,
        domain: &ProblemDomain<M, S, H>,
        z: C,
    ) -> Result<FunctionBundle<C>, ExecutionError<NativeOpticalError<H, S>>> {
        if !domain.provenance.belongs_to(self) {
            return Err(ExecutionError::ForeignDomain);
        }
        domain
            .search
            .dispersion()
            .evaluate(z)
            .map_err(|source| ExecutionError::Optical(Box::new(source)))
    }
    /// Optional numerical acceptance; discovery and coverage remain unchanged.
    pub fn refine<'a, H: AnalyticChart<J>>(
        &self,
        domain: &ProblemDomain<M, S, H>,
        discovery: &'a ProblemSearchResult,
        config: crate::refinement::RefinementConfig,
    ) -> Result<
        crate::refinement::RefinementReport<'a, ExecutionError<NativeOpticalError<H, S>>>,
        crate::refinement::RefinementInputError,
    > {
        use crate::refinement::{RefinementInputError, RefinementReport, refine_candidate};
        let validated = config.validate()?;
        if !domain.provenance.belongs_to(self) || !discovery.provenance.belongs_to(self) {
            return Err(RefinementInputError::ForeignProblem);
        }
        if !domain.provenance.same_domain(&discovery.provenance) {
            return Err(RefinementInputError::ForeignDomain);
        }
        let outcomes = discovery
            .result
            .estimates()
            .iter()
            .enumerate()
            .map(|(index, estimate)| {
                refine_candidate(
                    estimate.estimate(),
                    index,
                    &domain.provenance,
                    validated,
                    |z| self.evaluate(domain, z),
                    |z| {
                        domain
                            .search
                            .dispersion()
                            .chart()
                            .coordinate(z)
                            .map_err(|error| {
                                ExecutionError::Optical(Box::new(ChartDispersionError::Chart(
                                    error,
                                )))
                            })
                    },
                )
            })
            .collect();
        Ok(RefinementReport::new(discovery, config, outcomes))
    }
    /// Results remain estimates, including unresolved clusters. Ownership and
    /// construction errors precede search; neither implies root acceptance.
    pub fn search<H: AnalyticChart<J>>(
        &self,
        domain: &ProblemDomain<M, S, H>,
        config: ArgumentConfig<f64>,
    ) -> Result<ProblemSearchResult, ExecutionError<NativeSearchError<H, S>>> {
        if !domain.provenance.belongs_to(self) {
            return Err(ExecutionError::ForeignDomain);
        }
        let result = domain
            .search
            .search(config)
            .map_err(|source| ExecutionError::Optical(Box::new(source)))?;
        Ok(ProblemSearchResult {
            result,
            provenance: domain.provenance.clone(),
        })
    }
}
#[cfg(test)]
mod tests;

impl<M: Clone, S: ModeDeterminantSolver<J> + Clone> ModalProblem<M, S> {
    pub(crate) fn corrected_point<H: AnalyticChart<J>>(
        &self,
        domain: &ProblemDomain<M, S, H>,
        native: C,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<
        (
            crate::refinement::AcceptedRoot,
            orchestration::BranchSignature,
            argloc::Rect<f64>,
        ),
        crate::api::BranchFailure,
    > {
        use crate::api::BranchFailure;
        use lamina_core::ScalarAlgebra;
        let chart = domain.search.dispersion().chart();
        let bounds = domain.provenance.requested();
        let (x_min, x_max) = bounds.real_bounds();
        let (y_min, y_max) = bounds.imaginary_bounds();
        let enclosure = argloc::Rect {
            x_min,
            x_max,
            y_min,
            y_max,
        };
        let outcome = crate::refinement::correct_seed(
            native,
            enclosure,
            0,
            &domain.provenance,
            config,
            |z| self.evaluate(domain, z),
            |z| {
                chart
                    .coordinate(z)
                    .map_err(|e| ExecutionError::Optical(Box::new(ChartDispersionError::Chart(e))))
            },
        );
        let crate::refinement::CandidateStatus::Accepted(root) = outcome.status() else {
            return Err(BranchFailure::Correction(format!("{:?}", outcome.status())));
        };
        let point = chart
            .map(J::lift_constant(ndarray::arr0(root.evidence().z)))
            .map_err(|e| BranchFailure::Correction(e.to_string()))?;
        Ok((
            (**root).clone(),
            orchestration::BranchSignature::of(&point),
            enclosure,
        ))
    }
}
