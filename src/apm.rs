//! Argument-principle adapter for conformal modal dispersion functions.
//!
//! [`ConformalDispersion`] exposes the modal determinant
//!
//! ```text
//! F(z)
//! ```
//!
//! and its conformal derivative
//!
//! ```text
//! F'(z).
//! ```
//!
//! The argument principle operates on the logarithmic derivative
//!
//! ```text
//! F'(z) / F(z).
//! ```
//!
//! This module provides the thin adapter between those two interfaces. It does
//! not perform contour integration, winding-number calculation, subdivision,
//! or root localisation; those operations belong to the APM implementation.
//!
//! One [`ConformalArgumentPrinciple`] corresponds to exactly one analytic
//! [`ConformalChart`]. A larger physical modal search may therefore compile
//! into multiple adapters. In particular, the canonical complex-wavenumber
//! search uses separate upper and lower conformal charts.

use crate::analytic::AnalyticChart;
use crate::dispersion::ChartDispersionError;
use lamina_core::{ComplexScalar, ModeJet1};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FromPrimitive};

use crate::dispersion::{ConformalDispersion, DispersionError, ModeDeterminantSolver};

use argloc::{ComplexFunction, FunctionBundle};

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ConformalArgumentError<RegionError, SolverError> {
    #[error("modal dispersion evaluation failed: {0}")]
    Dispersion(#[from] DispersionError<RegionError, SolverError>),

    #[error("modal determinant vanishes at an argument-principle evaluation point")]
    ZeroDeterminant,

    #[error("modal logarithmic derivative is non-finite")]
    NonFiniteLogDerivative,
}

#[derive(Clone, Debug)]
pub struct ConformalArgumentPrinciple<S, M, C>
where
    C: ComplexField + Copy,
{
    dispersion: ConformalDispersion<S, M, C>,
}

impl<S, M, C> ConformalArgumentPrinciple<S, M, C>
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

    pub fn chart(&self) -> &crate::conformal::chart::ConformalChart<M, ModeJet1<C, Ix0>> {
        self.dispersion.chart()
    }
}

impl<S, M, C, R, H> ComplexFunction for ConformalDispersion<S, M, C, H>
where
    C: ComplexScalar<RealField = R> + Copy,
    R: Float + FromPrimitive + Copy,
    H: AnalyticChart<ModeJet1<C, Ix0>>,
    S: ModeDeterminantSolver<ModeJet1<C, Ix0>>,
{
    type Complex = C;

    type Error = ChartDispersionError<H::Error, S::Error>;

    fn bundle(&self, z: C) -> Result<FunctionBundle<C>, Self::Error> {
        ConformalDispersion::evaluate(self, z)
    }
}
