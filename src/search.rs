//! Modal searches over one fallible analytic chart (conformal or direct frequency).
//! Native coordinate bounds and tolerances follow the bound domain provenance.
//!
//! This module is the first layer that combines:
//!
//! - a [`ConformalDispersion`];
//! - its validated conformal search region;
//! - the argument-principle zero finder;
//! - conversion of conformal roots back to branch-consistent physical
//!   coordinates.
//!
//! One [`SingleChartModalSearch`] represents exactly one analytic conformal
//! chart. It does not know whether that chart arose from an in-plane or
//! wavenumber search.
//!
//! Checked optical failures retain their error sources. Successful results
//! preserve estimates, multiplicities, unresolved leaves, summary, and termination.
//! A cluster centroid is not an individual mode, and budget termination is not
//! a completeness claim. Material-domain analyticity remains the caller's duty.
//! Errors do not manufacture partial successes absent from argloc's result.
//!
//! Higher-level search orchestration is built above this layer:
//!
//! ```text
//! complex kx:
//!     one chart -> one SingleChartModalSearch
//!
//! complex k0:
//!     lower chart -> one SingleChartModalSearch
//!     upper chart -> one SingleChartModalSearch
//! ```

use crate::analytic::AnalyticChart;
use crate::conformal::chart::ConformalChart;
use crate::dispersion::ChartDispersionError;
use argloc::{
    ArgumentConfig, ArgumentLeaf, ArgumentResult, ComplexFunction, FindSingularitiesError,
    FunctionBundle, LocatableComplex, LocatableFloat, Rect, RunSummary, ScalerError,
    SingularPointEstimate, Termination, find_zeros,
};

use lamina_core::{ComplexScalar, ModeJet1};

use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FromPrimitive, float::FloatCore};
use std::sync::{Arc, Mutex};

/// A single-chart search failure: the chart or the determinant solver failed.
type ChartSearchError<C, ChartError, SolverError> =
    ModalSearchError<C, ChartDispersionError<ChartError, SolverError>>;

use crate::{
    conformal::ConformalRegion,
    dispersion::{ConformalDispersion, ModeDeterminantSolver},
    mode::ConformalMode,
};

#[derive(Debug, thiserror::Error)]
pub enum ModalSearchError<C, E>
where
    C: ComplexField,
    C::RealField: LocatableFloat<C>,
{
    #[error("failed to convert native search region into APM domain: {0}")]
    Domain(#[source] ScalerError<C::RealField>),

    #[error("argument-principle modal search failed: {engine}")]
    Search {
        engine: FindSingularitiesError<C, E>,
        #[source]
        source: Option<E>,
    },

    #[error("failed to construct a physical mode from an APM root: {0}")]
    Mode(#[source] E),
}

/// The upstream oracle error retains evaluation failures structurally, but some
/// engine variants do not expose them through Error::source. Preserve both the
/// full engine error and the last failing optical evaluation as a typed cause.
#[derive(Clone)]
struct TrackedFunction<F: ComplexFunction> {
    inner: F,
    failure: Arc<Mutex<Option<F::Error>>>,
}
impl<F: ComplexFunction> ComplexFunction for TrackedFunction<F>
where
    F::Error: Clone,
{
    type Complex = F::Complex;
    type Error = F::Error;
    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        let result = self.inner.bundle(z);
        *self
            .failure
            .lock()
            .expect("evaluation error observer poisoned") = result.as_ref().err().cloned();
        result
    }
}

/// An APM singular-point estimate together with its physical modal
/// interpretation.
#[derive(Debug)]
pub struct ModalEstimate<C>
where
    C: ComplexField,
{
    estimate: SingularPointEstimate<C>,
    mode: ConformalMode<C>,
}

impl<C> ModalEstimate<C>
where
    C: ComplexField + Copy,
{
    pub(crate) fn new(estimate: SingularPointEstimate<C>, mode: ConformalMode<C>) -> Self {
        Self { estimate, mode }
    }

    /// Original estimate produced by the argument-principle search.
    pub fn estimate(&self) -> &SingularPointEstimate<C> {
        &self.estimate
    }

    /// Branch-consistent interpretation of an estimate; not a validated root.
    pub fn mode(&self) -> &ConformalMode<C> {
        &self.mode
    }

    pub fn into_parts(self) -> (SingularPointEstimate<C>, ConformalMode<C>) {
        (self.estimate, self.mode)
    }
}

/// Result of searching one analytic conformal chart for modal zeros.
#[derive(Debug)]
pub struct ModalSearchResult<C>
where
    C: ComplexField,
{
    estimates: Vec<ModalEstimate<C>>,
    leaves: Vec<ArgumentLeaf<C::RealField>>,
    summary: RunSummary<C::RealField>,
    termination: Termination,
    subdivisions: usize,
    inconsistent: Vec<InconsistentWinding<C::RealField>>,
}

/// A split cell whose quadrants count a different number of zeros than the
/// cell: evidence of a pole or other singularity inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InconsistentWinding<R> {
    pub bounds: Rect<R>,
    pub whole: isize,
    pub parts: isize,
}

impl<C> ModalSearchResult<C>
where
    C: ComplexField + Copy,
{
    pub(crate) fn new(
        estimates: Vec<ModalEstimate<C>>,
        leaves: Vec<ArgumentLeaf<C::RealField>>,
        summary: RunSummary<C::RealField>,
        termination: Termination,
    ) -> Self {
        Self {
            estimates,
            leaves,
            summary,
            termination,
            subdivisions: 0,
            inconsistent: Vec::new(),
        }
    }

    /// Cells split to separate clustered roots, or a chart split after a
    /// numerical failure.
    pub fn subdivisions(&self) -> usize {
        self.subdivisions
    }

    /// Split cells whose quadrants' counts disagree with the cell's.
    pub fn inconsistent(&self) -> &[InconsistentWinding<C::RealField>] {
        &self.inconsistent
    }

    pub fn estimates(&self) -> &[ModalEstimate<C>] {
        &self.estimates
    }

    /// Iterate over physical interpretations of estimates, including clusters.
    pub fn modes(&self) -> impl ExactSizeIterator<Item = &ConformalMode<C>> {
        self.estimates.iter().map(ModalEstimate::mode)
    }

    pub fn leaves(&self) -> &[ArgumentLeaf<C::RealField>] {
        &self.leaves
    }

    pub fn summary(&self) -> &RunSummary<C::RealField> {
        &self.summary
    }

    pub fn termination(&self) -> &Termination {
        &self.termination
    }

    pub fn len(&self) -> usize {
        self.estimates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.estimates.is_empty()
    }

    pub fn into_estimates(self) -> Vec<ModalEstimate<C>> {
        self.estimates
    }
}

#[derive(Clone, Debug)]
pub struct SingleChartModalSearch<S, M, C, H = ConformalChart<M, ModeJet1<C, Ix0>>>
where
    C: ComplexField + Copy,
{
    dispersion: ConformalDispersion<S, M, C, H>,
    search_region: ConformalRegion<C::RealField>,
}

impl<S, M, C, H> SingleChartModalSearch<S, M, C, H>
where
    C: ComplexField + Copy,
{
    pub fn search_region(&self) -> &ConformalRegion<C::RealField> {
        &self.search_region
    }

    pub fn dispersion(&self) -> &ConformalDispersion<S, M, C, H> {
        &self.dispersion
    }

    pub fn into_dispersion(self) -> ConformalDispersion<S, M, C, H> {
        self.dispersion
    }
}

impl<S, M, C, R, H> SingleChartModalSearch<S, M, C, H>
where
    C: ComplexScalar<RealField = R>,
    R: Float + FromPrimitive + FloatCore,
    M: Clone,
    H: AnalyticChart<ModeJet1<C, Ix0>>,
{
    /// Keep the exact requested contour separately from the validated numerical
    /// evaluation margin. This does not certify a pole-free material domain.
    pub fn new(dispersion: ConformalDispersion<S, M, C, H>) -> Result<Self, H::Error> {
        let search_region = *dispersion.chart().region();
        let (solver, chart) = dispersion.into_parts();
        let evaluation = chart.with_evaluation_margin()?;
        Ok(Self {
            dispersion: ConformalDispersion::with_chart(solver, evaluation),
            search_region,
        })
    }
}

impl<S, M, C, R, H> SingleChartModalSearch<S, M, C, H>
where
    C: LocatableComplex<R> + ComplexScalar<RealField = R>,
    R: LocatableFloat<C>,
    M: Clone,
    H: AnalyticChart<ModeJet1<C, Ix0>>,
    S: ModeDeterminantSolver<ModeJet1<C, Ix0>> + Clone,
    ConformalRegion<R>: TryInto<Rect<R>, Error = ScalerError<R>>,
{
    pub fn search(
        &self,
        config: ArgumentConfig<R>,
    ) -> Result<ModalSearchResult<C>, ChartSearchError<C, H::Error, S::Error>> {
        let domain = self
            .search_region
            .try_into()
            .map_err(ModalSearchError::Domain)?;
        self.search_cell(domain, &config)
    }

    fn interpret(
        &self,
        result: ArgumentResult<C>,
    ) -> Result<ModalSearchResult<C>, ChartSearchError<C, H::Error, S::Error>> {
        let ArgumentResult {
            points,
            leaves,
            summary,
            termination,
        } = result;

        let mut estimates = Vec::with_capacity(points.len());

        for estimate in points {
            let mode = self
                .dispersion
                .mode_at(estimate.location)
                .map_err(ModalSearchError::Mode)?;

            estimates.push(ModalEstimate::new(estimate, mode));
        }

        Ok(ModalSearchResult::new(
            estimates,
            leaves,
            summary,
            termination,
        ))
    }

    /// argloc over one rectangle of this chart, inside its evaluation halo.
    fn search_cell(
        &self,
        cell: Rect<R>,
        config: &ArgumentConfig<R>,
    ) -> Result<ModalSearchResult<C>, ChartSearchError<C, H::Error, S::Error>> {
        let failure = Arc::new(Mutex::new(None));
        let function = TrackedFunction {
            inner: self.dispersion.clone(),
            failure: failure.clone(),
        };
        let result = find_zeros(function, cell, config.clone()).map_err(|engine| {
            ModalSearchError::Search {
                engine,
                source: failure
                    .lock()
                    .expect("evaluation error observer poisoned")
                    .take(),
            }
        })?;
        self.interpret(result)
    }

    /// Search every quadrant of `cell`, splitting at the first of a few
    /// off-centre fractions at which every quadrant's search succeeds.
    #[allow(clippy::type_complexity)]
    fn quadrants(
        &self,
        cell: Rect<R>,
        config: &ArgumentConfig<R>,
    ) -> Result<Vec<ModalSearchResult<C>>, ChartSearchError<C, H::Error, S::Error>> {
        let mut last = None;
        for fraction in [0.4637, 0.5381, 0.4129] {
            let f = R::from_f64(fraction).expect("representable fraction");
            let x = cell.x_min + f * (cell.x_max - cell.x_min);
            let y = cell.y_min + f * (cell.y_max - cell.y_min);
            let parts = [
                (cell.x_min, x, cell.y_min, y),
                (x, cell.x_max, cell.y_min, y),
                (cell.x_min, x, y, cell.y_max),
                (x, cell.x_max, y, cell.y_max),
            ];
            let searched: Result<Vec<_>, _> = parts
                .into_iter()
                .map(|(x_min, x_max, y_min, y_max)| {
                    self.search_cell(
                        Rect {
                            x_min,
                            x_max,
                            y_min,
                            y_max,
                        },
                        config,
                    )
                })
                .collect();
            match searched {
                Ok(parts) => return Ok(parts),
                Err(error) => last = Some(error),
            }
        }
        Err(last.expect("at least one split"))
    }

    /// As [`Self::search`], then split every converged leaf that holds
    /// several roots into quadrants searched again, until each holds at most
    /// one, the leaf is below 1e-9 of the chart's larger side, or
    /// `max_subdivisions` splits are spent. A chart whose search fails with
    /// no optical evaluation failure is split once the same way. Quadrant
    /// counts that do not add up to their cell's are recorded.
    pub fn search_subdivided(
        &self,
        config: ArgumentConfig<R>,
        max_subdivisions: usize,
    ) -> Result<ModalSearchResult<C>, ChartSearchError<C, H::Error, S::Error>> {
        let domain: Rect<R> = self
            .search_region
            .try_into()
            .map_err(ModalSearchError::Domain)?;
        let mut budget = max_subdivisions;
        let mut result = match self.search_cell(domain, &config) {
            Ok(result) => result,
            Err(ModalSearchError::Search { source: None, .. }) if budget > 0 => {
                budget -= 1;
                let mut parts = self.quadrants(domain, &config)?.into_iter();
                let mut merged = parts.next().expect("four quadrants");
                for part in parts {
                    merged.absorb(part);
                }
                merged.subdivisions += 1;
                merged
            }
            Err(error) => return Err(error),
        };
        let floor = R::from_f64(1e-9).expect("representable")
            * Float::max(domain.x_max - domain.x_min, domain.y_max - domain.y_min);
        let tolerance = config.quad_tree.target_score;
        let mut queue: Vec<Rect<R>> = result.clusters(tolerance);
        while let Some(cell) = queue.pop() {
            let size = Float::max(cell.x_max - cell.x_min, cell.y_max - cell.y_min);
            if budget == 0 || size <= floor {
                continue;
            }
            let Ok(parts) = self.quadrants(cell, &config) else {
                continue;
            };
            budget -= 1;
            let whole = result.remove_leaf(cell);
            let parts_count: isize = parts
                .iter()
                .flat_map(|p| p.leaves.iter())
                .map(|leaf| leaf.data.root_count)
                .sum();
            if parts_count != whole {
                result.inconsistent.push(InconsistentWinding {
                    bounds: cell,
                    whole,
                    parts: parts_count,
                });
            }
            for part in parts {
                queue.extend(part.clusters(tolerance));
                result.absorb(part);
            }
            result.subdivisions += 1;
        }
        Ok(result)
    }
}

impl<C> ModalSearchResult<C>
where
    C: ComplexField + Copy,
    C::RealField: Float,
{
    /// Bounds of converged leaves whose estimate holds several roots.
    fn clusters(&self, tolerance: C::RealField) -> Vec<Rect<C::RealField>> {
        self.estimates
            .iter()
            .map(ModalEstimate::estimate)
            .filter(|e| e.multiplicity > 1)
            .map(|e| e.enclosure)
            .filter(|bounds| {
                self.leaves.iter().any(|leaf| {
                    leaf.bounds == *bounds
                        && leaf.data.score.is_finite()
                        && leaf.data.score <= tolerance
                })
            })
            .collect()
    }

    /// Remove the leaf with `bounds` and its estimates, returning its count.
    fn remove_leaf(&mut self, bounds: Rect<C::RealField>) -> isize {
        let mut count = 0;
        self.leaves.retain(|leaf| {
            let keep = leaf.bounds != bounds;
            if !keep {
                count += leaf.data.root_count;
            }
            keep
        });
        self.estimates.retain(|e| e.estimate().enclosure != bounds);
        count
    }

    /// Add another search's leaves, estimates and work to this one; a
    /// termination other than convergence is kept.
    fn absorb(&mut self, other: Self) {
        self.estimates.extend(other.estimates);
        self.leaves.extend(other.leaves);
        self.summary.iterations += other.summary.iterations;
        self.summary.elapsed += other.summary.elapsed;
        self.summary.best_measure = match (self.summary.best_measure, other.summary.best_measure) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        if self.termination == Termination::Converged {
            self.termination = other.termination;
        }
        self.subdivisions += other.subdivisions;
        self.inconsistent.extend(other.inconsistent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use argloc::ArgumentConfig;
    use lamina_core::{AnalyticalMaterialStack, Constant, ModeJet1, Polarisation, Scatter2, Stack};
    use lamina_units::Length;
    use ndarray::Ix0;
    use num_complex::Complex64;

    use crate::{
        conformal::{
            chart::{ConformalChart, ConformalChartConfig, CoordinateSheet},
            in_plane::{ComplexInPlaneMap, region::InPlaneSearchExtent},
        },
        dispersion::{CompiledPlaneWaveModeSolver, ConformalDispersion},
    };

    type C = Complex64;
    type J = ModeJet1<C, Ix0>;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    const WAVELENGTH_UM: f64 = 1.0;
    const CORE_THICKNESS_UM: f64 = 0.283916;

    fn known_te_mode_stack() -> AnalyticalMaterialStack<Complex64> {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
            .analytical_layer(
                Constant::dielectric(4.0),
                Length::micrometres(CORE_THICKNESS_UM),
            )
            .finalise()
    }

    #[test]
    fn single_chart_search_recovers_known_asymmetric_te_slab_mode() {
        // -----------------------------------------------------------------
        // Physical reference problem.
        // -----------------------------------------------------------------

        let wavelength_cm = WAVELENGTH_UM * 1e-4;
        let k0 = 2.0 * std::f64::consts::PI / wavelength_cm;
        let expected_kx = 1.75 * k0;

        let max_kx = 1.90 * k0;

        // -----------------------------------------------------------------
        // Compile the real optical determinant.
        // -----------------------------------------------------------------

        let stack = known_te_mode_stack();

        let solver = CompiledPlaneWaveModeSolver::<J, _, _>::compile(
            &stack,
            Scatter2::new(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

        // -----------------------------------------------------------------
        // Construct the canonical in-plane conformal chart.
        // -----------------------------------------------------------------

        //
        // Exterior media:
        //
        //     left:  epsilon = 1,    mu = 1
        //     right: epsilon = 2.25, mu = 1
        //
        let map =
            ComplexInPlaneMap::<J>::new(k0, c(1.0, 0.0), c(1.0, 0.0), c(2.25, 0.0), c(1.0, 0.0))
                .unwrap();

        let region = map
            .search_region(InPlaneSearchExtent {
                max_parallel_wavenumber: max_kx,

                imaginary_margin: 0.05,
            })
            .unwrap();

        let chart = ConformalChart::new(
            map,
            region.into_inner(),
            CoordinateSheet::Principal,
            ConformalChartConfig::default(),
        )
        .unwrap();

        let dispersion = ConformalDispersion::new(solver, chart);

        // -----------------------------------------------------------------
        // Run the single-chart modal search.
        // -----------------------------------------------------------------

        let requested = *dispersion.chart().region();
        let search = SingleChartModalSearch::new(dispersion).unwrap();
        assert_eq!(*search.search_region(), requested);

        let result = search.search(ArgumentConfig::new(1.0e-6)).unwrap();

        assert!(
            !result.is_empty(),
            "expected the engineered TE slab mode to be found",
        );

        // -----------------------------------------------------------------
        // Identify the returned mode nearest the independently known
        // physical solution.
        // -----------------------------------------------------------------

        let target = c(expected_kx, 0.0);

        let mode = result
            .modes()
            .min_by(|a, b| {
                let da = (a.coordinate() - target).modulus();

                let db = (b.coordinate() - target).modulus();

                da.partial_cmp(&db).unwrap()
            })
            .expect("modal search returned no physical modes");

        // -----------------------------------------------------------------
        // Independent physical regression.
        // -----------------------------------------------------------------

        assert_relative_eq!(
            mode.coordinate().real(),
            expected_kx,
            epsilon = 1.0e-5 * expected_kx,
            max_relative = 1.0e-5,
        );

        assert_relative_eq!(
            mode.coordinate().imaginary(),
            0.0,
            epsilon = 1.0e-6 * expected_kx,
        );

        // -----------------------------------------------------------------
        // The converted physical result must still correspond to a true
        // determinant zero.
        // -----------------------------------------------------------------

        assert!(
            mode.residual().modulus() < 1.0e-7,
            "known slab mode has non-small determinant residual: \
             z={:?}, kx={:?}, F={:?}",
            mode.z(),
            mode.coordinate(),
            mode.residual(),
        );
    }
}
