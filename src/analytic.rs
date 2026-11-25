//! Fallible native-coordinate chart interface shared by conformal and direct
//! frequency searches. Branch integration errors must remain errors, never NaN
//! sentinels hidden inside an infallible raw map.
use crate::conformal::{
    ConformalPoint, ConformalRegion, ValidateConformalRegion,
    chart::{ConformalChart, ConformalChartError, LinearCoordinateMap},
};
use lamina_core::{ComplexScalar, ScalarAlgebra};
use nalgebra::ComplexField;
use ndarray::Ix0;
use num_traits::{Float, FromPrimitive, float::FloatCore};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateKind {
    Conformal,
    Frequency,
    /// β itself, for a stack closed on both sides, by walls or PMLs.
    Wavenumber,
}
impl CoordinateKind {
    pub fn units(self) -> &'static str {
        match self {
            Self::Conformal => "dimensionless",
            Self::Frequency | Self::Wavenumber => "cm^-1",
        }
    }
}
pub trait AnalyticChart<A>: Clone
where
    A: ScalarAlgebra<Dimension = Ix0>,
    A::Scalar: ComplexField,
{
    type Error: std::error::Error + Clone + Send + Sync + 'static;
    fn region(&self) -> &ConformalRegion<<A::Scalar as ComplexField>::RealField>;
    fn reference(&self) -> A::Scalar;
    fn reference_coordinate(&self) -> A::Scalar;
    fn coordinate_kind(&self) -> CoordinateKind;
    fn coordinate(&self, z: A::Scalar) -> Result<A::Scalar, Self::Error>;
    fn map(&self, z: A) -> Result<ConformalPoint<A>, Self::Error>;
    fn with_evaluation_margin(&self) -> Result<Self, Self::Error>;
}
impl<M, A, R> AnalyticChart<A> for ConformalChart<M, A>
where
    A: ScalarAlgebra<Dimension = Ix0> + Clone,
    A::Scalar: ComplexScalar<RealField = R> + Copy,
    R: Float + FloatCore + FromPrimitive + Copy,
    M: LinearCoordinateMap<A> + ValidateConformalRegion<R> + Clone,
{
    type Error = ConformalChartError<M::Error>;
    fn region(&self) -> &ConformalRegion<R> {
        self.region()
    }
    fn reference(&self) -> A::Scalar {
        self.reference()
    }
    fn reference_coordinate(&self) -> A::Scalar {
        self.reference_coordinate()
    }
    fn coordinate_kind(&self) -> CoordinateKind {
        CoordinateKind::Conformal
    }
    fn coordinate(&self, z: A::Scalar) -> Result<A::Scalar, Self::Error> {
        self.coordinate(z)
    }
    fn map(&self, z: A) -> Result<ConformalPoint<A>, Self::Error> {
        self.map(z)
    }
    fn with_evaluation_margin(&self) -> Result<Self, Self::Error> {
        self.with_evaluation_margin()
    }
}
