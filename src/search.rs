//! Modal searches over a single analytic conformal chart.
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

use argloc::{
    ArgumentConfig, ArgumentLeaf, ArgumentResult, ComplexFunction, FindSingularitiesError,
    LocatableComplex, LocatableFloat, Rect, RunSummary, ScalerError, SingularPointEstimate,
    Termination, find_zeros,
};

use lamina_core::{ComplexScalar, ModeJet1, ScalarAlgebra, SeedJet};

use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FromPrimitive, float::FloatCore};
use std::fmt::{Debug, Display};

use crate::{
    conformal::{ConformalRegion, ValidateConformalRegion, chart::LinearCoordinateMap},
    dispersion::{ConformalDispersion, DispersionError, ModeDeterminantSolver},
    mode::ConformalMode,
};

#[derive(Debug, thiserror::Error)]
pub enum ModalSearchError<C, E>
where
    C: ComplexField,
    C::RealField: LocatableFloat<C>,
{
    #[error("failed to convert conformal search region into APM domain: {0}")]
    Domain(#[source] ScalerError<C::RealField>),

    #[error("argument-principle modal search failed: {0}")]
    Search(#[source] FindSingularitiesError<C, E>),

    #[error("failed to construct a physical mode from an APM root: {0}")]
    Mode(#[source] E),
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

    /// Branch-consistent physical mode obtained from that estimate.
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
        }
    }

    pub fn estimates(&self) -> &[ModalEstimate<C>] {
        &self.estimates
    }

    /// Iterate over the physical modes returned by this search.
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
pub struct SingleChartModalSearch<S, M, C>
where
    C: ComplexField + Copy,
{
    dispersion: ConformalDispersion<S, M, C>,
}

impl<S, M, C> SingleChartModalSearch<S, M, C>
where
    C: ComplexField + Copy,
{
    pub fn new(dispersion: ConformalDispersion<S, M, C>) -> Self {
        Self { dispersion }
    }

    pub fn dispersion(&self) -> &ConformalDispersion<S, M, C> {
        &self.dispersion
    }

    pub fn into_dispersion(self) -> ConformalDispersion<S, M, C> {
        self.dispersion
    }
}

impl<S, M, C, R> SingleChartModalSearch<S, M, C>
where
    C: LocatableComplex<R> + ComplexScalar<RealField = R>,
    R: LocatableFloat<C>,
    M: LinearCoordinateMap<ModeJet1<C, Ix0>> + ValidateConformalRegion<R> + Clone,
    S: ModeDeterminantSolver<ModeJet1<C, Ix0>> + Clone,
    ConformalRegion<R>: TryInto<Rect<R>, Error = ScalerError<R>>,
{
    pub fn search(
        &self,
        config: ArgumentConfig<R>,
    ) -> Result<
        ModalSearchResult<C>,
        ModalSearchError<C, DispersionError<<M as ValidateConformalRegion<R>>::Error, S::Error>>,
    > {
        let domain = self
            .dispersion
            .chart()
            .region()
            .clone()
            .try_into()
            .map_err(ModalSearchError::Domain)?;

        dbg!(self.dispersion.chart().region());
        dbg!(&domain);

        let result = find_zeros(self.dispersion.clone(), domain, config)
            .map_err(ModalSearchError::Search)?;

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
            Scatter2::default(),
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

        let search = SingleChartModalSearch::new(dispersion);

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
