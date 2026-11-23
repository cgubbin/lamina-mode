#![allow(dead_code)]
use lamina_core::{
    ComplexScalar, Constant, DerivativeOrder, DifferentiableMaterial,
    DifferentiableMeromorphicMaterial, Material, MeromorphicMaterial, Sampled,
};

/// Linear dispersive material used for testing.
///
/// ε(k₀) = ε₀ + slope · k₀
/// μ(k₀) = μ₀ + slope · k₀
#[derive(Clone, Debug)]
pub struct LinearDispersion {
    pub epsilon0: f64,
    pub epsilon_slope: f64,
    pub epsilon_loss_slope: f64,
    pub lossless_at: f64,
    pub mu0: f64,
    pub mu_slope: f64,
}

impl LinearDispersion {
    pub fn new(epsilon0: f64, epsilon_slope: f64, mu0: f64, mu_slope: f64) -> Self {
        Self {
            epsilon0,
            epsilon_slope,
            epsilon_loss_slope: 0.0,
            lossless_at: 0.0,
            mu0,
            mu_slope,
        }
    }
}

pub fn vacuum() -> Constant<f64> {
    Constant::new(1.0, 1.0)
}

pub fn constant(epsilon: f64, mu: f64) -> Constant<f64> {
    Constant::new(epsilon, mu)
}

pub fn linear(epsilon0: f64, epsilon_slope: f64, mu0: f64, mu_slope: f64) -> LinearDispersion {
    LinearDispersion::new(epsilon0, epsilon_slope, mu0, mu_slope)
}

impl Material for LinearDispersion {
    type Real = f64;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| {
            C::from_real(self.epsilon0)
                + C::from_real(x) * C::from_real(self.epsilon_slope)
                + C::i() * C::from_real((x - self.lossless_at) * self.epsilon_loss_slope)
        })
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| C::from_real(self.mu0) + C::from_real(x) * C::from_real(self.mu_slope))
    }
}

impl DifferentiableMaterial for LinearDispersion {
    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        match order {
            DerivativeOrder::First => wavenumber.map(|_| {
                C::from_real(self.epsilon_slope) + C::i() * C::from_real(self.epsilon_loss_slope)
            }),
            DerivativeOrder::Second => wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        match order {
            DerivativeOrder::First => wavenumber.map(|_| C::from_real(self.mu_slope)),
            DerivativeOrder::Second => wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => wavenumber.map(|_| C::zero()),
        }
    }
}

impl MeromorphicMaterial for LinearDispersion {
    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|x| {
            C::from_real(self.epsilon0)
                + x * C::from_real(self.epsilon_slope)
                + C::i()
                    * (x - C::from_real(self.lossless_at))
                    * C::from_real(self.epsilon_loss_slope)
        })
    }

    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|x| C::from_real(self.mu0) + x * C::from_real(self.mu_slope))
    }
}

impl DifferentiableMeromorphicMaterial for LinearDispersion {
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|_| {
                C::from_real(self.epsilon_slope) + C::i() * C::from_real(self.epsilon_loss_slope)
            }),
            DerivativeOrder::Second => vacuum_wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|_| C::from_real(self.mu_slope)),
            DerivativeOrder::Second => vacuum_wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }
}
/// Validation-only complex constants: freeze values without their derivatives.
#[derive(Clone, Debug)]
pub struct Frozen {
    pub epsilon: num_complex::Complex64,
    pub mu: num_complex::Complex64,
}
impl Frozen {
    fn value<C: ComplexScalar<RealField = f64>>(v: num_complex::Complex64) -> C {
        C::from_real(v.re) + C::i() * C::from_real(v.im)
    }
}
impl Material for Frozen {
    type Real = f64;
    fn relative_permittivity<I, C>(&self, k: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = f64>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| Self::value(self.epsilon))
    }
    fn relative_permeability<I, C>(&self, k: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = f64>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| Self::value(self.mu))
    }
}
impl MeromorphicMaterial for Frozen {
    fn relative_permittivity_complex<I, C>(&self, k: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| Self::value(self.epsilon))
    }
    fn relative_permeability_complex<I, C>(&self, k: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| Self::value(self.mu))
    }
}
impl DifferentiableMaterial for Frozen {
    fn relative_permittivity_derivative<I, C>(&self, k: I, _: DerivativeOrder) -> I::Mapped<C>
    where
        I: Sampled<Elem = f64>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| C::from_real(0.))
    }
    fn relative_permeability_derivative<I, C>(&self, k: I, _: DerivativeOrder) -> I::Mapped<C>
    where
        I: Sampled<Elem = f64>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| C::from_real(0.))
    }
}
impl DifferentiableMeromorphicMaterial for Frozen {
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| C::from_real(0.))
    }
    fn relative_permeability_complex_derivative<I, C>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = f64>,
    {
        k.map(|_| C::from_real(0.))
    }
}
