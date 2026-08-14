use std::fmt::Debug;
use std::marker::PhantomData;

use argloc::{ComplexFunction, Rect};
use lamina_core::{
    ComplexScalar, CoordinateInput, Coordinates, DifferentiableMeromorphicMaterial,
    InPlaneCoordinate, Parameter, PlaneWaveEvaluator, Polarisation, SpectralCoordinate, Stack,
    backend::Scatter2,
};
use lamina_units::InverseLengthUnit;
use nalgebra::ComplexField;
use ndarray::{Ix0, arr0};
use num_traits::{Float, FloatConst, FromPrimitive};

struct ComplexAngularWavenumberProblem<'a, M, C>
where
    C: ComplexField,
{
    in_plane_angular_wavenumber: C::RealField,
    stack: &'a Stack<M, C::RealField>,
    polarisation: Polarisation,
    domain: Rect<C::RealField>,
    evaluator: &'a PlaneWaveEvaluator<Scatter2>,
    complex: PhantomData<C>,
}

impl<'a, M, C> ComplexAngularWavenumberProblem<'a, M, C>
where
    C: ComplexField,
    C::RealField: Copy,
{
    fn coordinate_input(&self, vacuum_angular_wavenumber: C) -> CoordinateInput<C, Ix0> {
        CoordinateInput::point(
            Coordinates::new(
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
            ),
            vacuum_angular_wavenumber,
            C::from_real(self.in_plane_angular_wavenumber),
        )
        .unwrap()
    }

    /// Convert z to physical vacuum_angular_wavenumber
    fn conformal_mapping(&self, z: C) -> C {
        todo!()
    }
}

impl<'a, M, C> ComplexFunction for ComplexAngularWavenumberProblem<'a, M, C>
where
    C: ComplexScalar,
    C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
    M: DifferentiableMeromorphicMaterial<Real = C::RealField> + Clone,
{
    type Complex = C;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        let vacuum_angular_wavenumber = self.conformal_mapping(z);

        let input = self.coordinate_input(vacuum_angular_wavenumber);

        let solution = self
            .evaluator
            .evaluate_modal(input, &self.stack, self.polarisation)
            .unwrap();

        let (determinant, _) = solution.determinant().into_parts();

        determinant.into_inner()[()]
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let vacuum_angular_wavenumber = self.conformal_mapping(z);

        let input = self.coordinate_input(vacuum_angular_wavenumber);

        let solution = self
            .evaluator
            .evaluate_modal_first(input, &self.stack, self.polarisation, Parameter::Spectral)
            .unwrap();

        let (_, derivatives) = solution.determinant().into_parts();

        derivatives.into_first().into_inner()[()]
    }
}
