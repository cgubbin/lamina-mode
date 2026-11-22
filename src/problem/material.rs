//! Constant exterior construction evidence. Erased handles and sampled zero
//! derivatives do not prove constancy, so spectral constructors take this value.
use super::C;
use lamina_core::{
    ComplexScalar, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    Material, MeromorphicMaterial, Sampled,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantExterior {
    epsilon: C,
    mu: C,
}
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("constant exterior factors must be finite and nonzero")]
pub struct InvalidConstantExterior;
impl ConstantExterior {
    pub fn new(epsilon: C, mu: C) -> Result<Self, InvalidConstantExterior> {
        if [epsilon, mu]
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite() || *v == C::new(0.0, 0.0))
        {
            return Err(InvalidConstantExterior);
        }
        Ok(Self { epsilon, mu })
    }
    pub fn epsilon(&self) -> C {
        self.epsilon
    }
    pub fn mu(&self) -> C {
        self.mu
    }
}
fn convert<T: ComplexScalar<RealField = f64>>(v: C) -> T {
    T::from_real(v.re) + T::i() * T::from_real(v.im)
}
impl Material for ConstantExterior {
    type Real = f64;
    fn relative_permittivity<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| convert(self.epsilon))
    }
    fn relative_permeability<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| convert(self.mu))
    }
}
impl DifferentiableMaterial for ConstantExterior {
    fn relative_permittivity_derivative<I, T>(&self, k: I, _: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(0.0))
    }
    fn relative_permeability_derivative<I, T>(&self, k: I, _: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(0.0))
    }
}
impl MeromorphicMaterial for ConstantExterior {
    fn relative_permittivity_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| convert(self.epsilon))
    }
    fn relative_permeability_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| convert(self.mu))
    }
}
impl DifferentiableMeromorphicMaterial for ConstantExterior {
    fn relative_permittivity_complex_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| T::from_real(0.0))
    }
    fn relative_permeability_complex_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| T::from_real(0.0))
    }
}
