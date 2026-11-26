//! Scalar models and region materials of a record, as lamina-core regions.
//!
//! A scalar model is one of lamina-materials' dispersion models, or a
//! tabulated permittivity evaluated once at the operating wavelength and held
//! constant there. One dispatching type implements lamina-core's material
//! traits, so the same model serves an isotropic region and each axis of a
//! principal-axis tensor material.
use crate::records::{self as rec, C2};
use crate::solve::{Failure, M_PER_CM};
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, ComplexScalar, Constant,
    DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material,
    MeromorphicMaterial, Rotation, Sampled,
};
use lamina_materials::{
    ConstantTensor, DrudeLorentz, Extrapolation, Lorentz, LorentzOscillator, PrincipalAxes,
    Sellmeier, Tabulated,
};
use lamina_mode::{Complex, Region};
use lamina_units::VacuumAngularWavenumber;
use std::f64::consts::TAU;

/// A scalar model in lamina-core's units (k0 in cm^-1).
#[derive(Clone, Debug)]
pub enum Model {
    Constant(Constant<f64>),
    Sellmeier(Sellmeier<f64>),
    Lorentz(Lorentz<f64>),
    DrudeLorentz(DrudeLorentz<f64>),
    /// A tabulated permittivity at the operating wavelength, and the
    /// permeability, held constant.
    Frozen {
        epsilon: Complex,
        mu: f64,
    },
}

fn constant<I: Sampled<Elem = E>, E, C: ComplexScalar<RealField = f64>>(
    k0: I,
    value: Complex,
) -> I::Mapped<C> {
    k0.map(|_| C::from_parts(value.re, value.im))
}

impl Model {
    /// `[ε, μ]` of a frozen model; zeros otherwise (never read).
    fn frozen(&self) -> [Complex; 2] {
        match self {
            Model::Frozen { epsilon, mu } => [*epsilon, Complex::from(*mu)],
            _ => [Complex::ZERO; 2],
        }
    }
}

macro_rules! dispatch {
    ($self:ident, $k0:ident, $method:ident $(, $order:ident)?; frozen => $frozen:expr) => {
        match $self {
            Model::Constant(m) => m.$method($k0 $(, $order)?),
            Model::Sellmeier(m) => m.$method($k0 $(, $order)?),
            Model::Lorentz(m) => m.$method($k0 $(, $order)?),
            Model::DrudeLorentz(m) => m.$method($k0 $(, $order)?),
            Model::Frozen { .. } => constant($k0, $frozen),
        }
    };
}

impl Material for Model {
    type Real = f64;

    fn relative_permittivity<I, C>(&self, k0: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        dispatch!(self, k0, relative_permittivity; frozen => self.frozen()[0])
    }

    fn relative_permeability<I, C>(&self, k0: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        dispatch!(self, k0, relative_permeability; frozen => self.frozen()[1])
    }
}

impl DifferentiableMaterial for Model {
    fn relative_permittivity_derivative<I, C>(&self, k0: I, order: DerivativeOrder) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        dispatch!(self, k0, relative_permittivity_derivative, order; frozen => Complex::ZERO)
    }

    fn relative_permeability_derivative<I, C>(&self, k0: I, order: DerivativeOrder) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        dispatch!(self, k0, relative_permeability_derivative, order; frozen => Complex::ZERO)
    }
}

impl MeromorphicMaterial for Model {
    fn relative_permittivity_complex<I, C>(&self, k0: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = C>,
    {
        dispatch!(self, k0, relative_permittivity_complex; frozen => self.frozen()[0])
    }

    fn relative_permeability_complex<I, C>(&self, k0: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = C>,
    {
        dispatch!(self, k0, relative_permeability_complex; frozen => self.frozen()[1])
    }
}

impl DifferentiableMeromorphicMaterial for Model {
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        k0: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = C>,
    {
        dispatch!(self, k0, relative_permittivity_complex_derivative, order; frozen => Complex::ZERO)
    }

    fn relative_permeability_complex_derivative<I, C>(
        &self,
        k0: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = C>,
    {
        dispatch!(self, k0, relative_permeability_complex_derivative, order; frozen => Complex::ZERO)
    }
}

/// An angular wavenumber in rad/m as lamina-units' quantity.
fn wavenumber(rad_per_m: f64) -> VacuumAngularWavenumber<f64> {
    VacuumAngularWavenumber::inverse_centimetres(rad_per_m * M_PER_CM)
}

fn oscillators(name: &str, o: &[rec::Oscillator]) -> Result<Vec<LorentzOscillator<f64>>, Failure> {
    o.iter()
        .enumerate()
        .map(|(j, o)| {
            LorentzOscillator::from_delta_epsilon(
                o.delta_epsilon,
                wavenumber(o.resonance_rad_per_m),
                wavenumber(o.damping_rad_per_m),
            )
            .map_err(|e| Failure::Invalid(format!("{name} oscillator {j}: {e:?}")))
        })
        .collect()
}

/// The lamina-materials Sellmeier model of a record, in canonical units.
pub fn sellmeier_model(spec: &rec::Sellmeier) -> Result<Sellmeier<f64>, Failure> {
    let terms = spec
        .terms
        .iter()
        .map(|t| {
            lamina_materials::SellmeierTerm::from_wavelength_pole(
                t.coefficient,
                lamina_units::Wavelength::centimetres(t.pole_wavelength_m / M_PER_CM),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Failure::Invalid(format!("Sellmeier term: {e:?}")))?;
    Sellmeier::new(spec.constant, terms)
        .map_err(|e| Failure::Invalid(format!("Sellmeier model: {e:?}")))
}

/// The model of a record at the operating wavelength (m). `name` names the
/// region and axis in errors.
pub fn model(name: &str, m: &rec::ScalarModel, wavelength_m: f64) -> Result<Model, Failure> {
    Ok(match m {
        rec::ScalarModel::Constant {
            relative_permittivity,
            relative_permeability,
        } => Model::Constant(Constant::new(
            *relative_permittivity,
            *relative_permeability,
        )),
        rec::ScalarModel::Sellmeier { sellmeier, .. } => {
            Model::Sellmeier(sellmeier_model(sellmeier)?)
        }
        rec::ScalarModel::Lorentz {
            epsilon_infinity,
            oscillators: o,
            ..
        } => Model::Lorentz(
            Lorentz::new(*epsilon_infinity, oscillators(name, o)?)
                .map_err(|e| Failure::Invalid(format!("{name} Lorentz model: {e:?}")))?,
        ),
        rec::ScalarModel::DrudeLorentz {
            epsilon_infinity,
            plasma_rad_per_m,
            damping_rad_per_m,
            oscillators: o,
            ..
        } => Model::DrudeLorentz(
            DrudeLorentz::from_plasma_wavenumber(
                *epsilon_infinity,
                wavenumber(*plasma_rad_per_m),
                wavenumber(*damping_rad_per_m),
                oscillators(name, o)?,
            )
            .map_err(|e| Failure::Invalid(format!("{name} Drude-Lorentz model: {e:?}")))?,
        ),
        rec::ScalarModel::Tabulated {
            wavelength_m: table,
            relative_permittivity,
            relative_permeability,
        } => {
            if table.len() != relative_permittivity.len() {
                return Err(Failure::Invalid(format!(
                    "{name}: tabulated wavelength_m and relative_permittivity differ in length"
                )));
            }
            // Ascending wavelengths are descending wavenumbers: reverse both.
            let k0_cm: Vec<f64> = table.iter().rev().map(|l| TAU / (l / M_PER_CM)).collect();
            let values: Vec<Complex> = relative_permittivity
                .iter()
                .rev()
                .map(|&[re, im]| Complex::new(re, im))
                .collect();
            let tabulated = Tabulated::new(k0_cm.into(), values.into(), Extrapolation::Reject)
                .map_err(|e| Failure::Invalid(format!("{name}: tabulated model: {e:?}")))?;
            let epsilon: Complex = tabulated
                .relative_permittivity(lamina_core::Scalar(TAU / (wavelength_m / M_PER_CM)));
            if !(epsilon.re.is_finite() && epsilon.im.is_finite()) {
                return Err(Failure::Invalid(format!(
                    "{name}: the operating wavelength lies outside the tabulated range"
                )));
            }
            Model::Frozen {
                epsilon,
                mu: *relative_permeability,
            }
        }
    })
}

/// The models of a region material, with the axis each sits on (`None` for
/// an isotropic region).
pub fn models(m: &rec::Material) -> Vec<(Option<rec::Axis>, &rec::ScalarModel)> {
    use rec::Axis::*;
    match m {
        rec::Material::Isotropic { model } => vec![(None, model)],
        rec::Material::Diagonal { x, y, z } => vec![(Some(X), x), (Some(Y), y), (Some(Z), z)],
        rec::Material::Full {
            material: rec::TensorMaterial::Principal { a, b, c },
            ..
        } => vec![(Some(A), a), (Some(B), b), (Some(C), c)],
        rec::Material::Full {
            material: rec::TensorMaterial::Tensor { .. },
            ..
        } => vec![],
    }
}

fn axis_name(name: &str, axis: Option<rec::Axis>) -> String {
    match axis {
        None => name.to_owned(),
        Some(a) => format!("{name} axis {}", a.name()),
    }
}

/// A validated rotation.
pub fn rotation(name: &str, r: &[[f64; 3]; 3]) -> Result<Rotation<f64>, Failure> {
    Rotation::from_matrix(*r).map_err(|e| Failure::Invalid(format!("{name} rotation: {e:?}")))
}

fn tensor(t: &[[C2; 3]; 3]) -> [[Complex; 3]; 3] {
    t.map(|row| row.map(|[re, im]| Complex::new(re, im)))
}

/// The lamina-core region of a material at the operating wavelength.
pub fn region(name: &str, m: &rec::Material, wavelength_m: f64) -> Result<Region, Failure> {
    let model = |axis, s| model(&axis_name(name, axis), s, wavelength_m);
    Ok(match m {
        rec::Material::Isotropic { model: s } => {
            AnalyticalMaterialHandle::new(model(None, s)?).into()
        }
        rec::Material::Diagonal { x, y, z } => Region::oriented(
            AnalyticalTensorMaterialHandle::new(PrincipalAxes::biaxial(
                model(Some(rec::Axis::X), x)?,
                model(Some(rec::Axis::Y), y)?,
                model(Some(rec::Axis::Z), z)?,
            )),
            Rotation::identity(),
        ),
        rec::Material::Full {
            material: rec::TensorMaterial::Principal { a, b, c },
            rotation: r,
        } => Region::oriented(
            AnalyticalTensorMaterialHandle::new(PrincipalAxes::biaxial(
                model(Some(rec::Axis::A), a)?,
                model(Some(rec::Axis::B), b)?,
                model(Some(rec::Axis::C), c)?,
            )),
            rotation(name, r)?,
        ),
        rec::Material::Full {
            material:
                rec::TensorMaterial::Tensor {
                    relative_permittivity,
                    relative_permeability,
                },
            rotation: r,
        } => Region::oriented(
            AnalyticalTensorMaterialHandle::new(
                ConstantTensor::new(tensor(relative_permittivity), tensor(relative_permeability))
                    .map_err(|e| Failure::Invalid(format!("{name} tensor: {e:?}")))?,
            ),
            rotation(name, r)?,
        ),
    })
}

/// `R T Rᵀ`.
pub fn rotate(r: &[[f64; 3]; 3], t: &[[Complex; 3]; 3]) -> [[Complex; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut sum = Complex::ZERO;
            for k in 0..3 {
                for l in 0..3 {
                    sum += r[i][k] * t[k][l] * r[j][l];
                }
            }
            sum
        })
    })
}

fn diagonal(d: [Complex; 3]) -> [[Complex; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { d[i] } else { Complex::ZERO }))
}

/// Stack-frame `[ε, μ]` of a material at the operating wavelength.
pub fn stack_frame(
    name: &str,
    m: &rec::Material,
    wavelength_m: f64,
) -> Result<[[[Complex; 3]; 3]; 2], Failure> {
    let k0 = lamina_core::Scalar(TAU / (wavelength_m / M_PER_CM));
    let values = |s: &rec::ScalarModel| -> Result<[Complex; 2], Failure> {
        let model = model(name, s, wavelength_m)?;
        Ok([
            model.relative_permittivity(k0),
            model.relative_permeability(k0),
        ])
    };
    let principal = |models: [&rec::ScalarModel; 3]| -> Result<[[[Complex; 3]; 3]; 2], Failure> {
        let v = [values(models[0])?, values(models[1])?, values(models[2])?];
        Ok([0, 1].map(|f| diagonal([v[0][f], v[1][f], v[2][f]])))
    };
    Ok(match m {
        rec::Material::Isotropic { model: s } => {
            let [e, u] = values(s)?;
            [diagonal([e; 3]), diagonal([u; 3])]
        }
        rec::Material::Diagonal { x, y, z } => principal([x, y, z])?,
        rec::Material::Full {
            material: rec::TensorMaterial::Principal { a, b, c },
            rotation: r,
        } => principal([a, b, c])?.map(|t| rotate(r, &t)),
        rec::Material::Full {
            material:
                rec::TensorMaterial::Tensor {
                    relative_permittivity,
                    relative_permeability,
                },
            rotation: r,
        } => [
            rotate(r, &tensor(relative_permittivity)),
            rotate(r, &tensor(relative_permeability)),
        ],
    })
}
