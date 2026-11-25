//! Physical parameter directions for guided-mode sensitivities.

use super::Complex;
use lamina_core::{
    AnalyticalMaterialStack, DerivativeOrder, EvaluateDifferentiableMeromorphicMaterial,
    FieldPosition, Scalar,
};

/// An explicit constitutive rate at fixed vacuum wavelength.
/// Both relative material properties are dimensionless.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConstitutiveRate {
    pub relative_permittivity: f64,
    pub relative_permeability: f64,
}

/// An invalid physical parameter direction or selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionError {
    EmptyParameter,
    InvalidGeometry,
    InvalidRegion,
    InvalidRate,
    LossInducingRate,
    /// Moves the first interface, and with it the stack, against its walls.
    MovesAgainstWalls,
}

impl ConstitutiveRate {
    /// Converts a real index rate to a permittivity rate for a nonmagnetic dielectric.
    pub fn nonmagnetic_index(index: f64, rate: f64) -> Result<Self, DirectionError> {
        let epsilon_rate = 2.0 * index * rate;
        if index <= 0.0 || !epsilon_rate.is_finite() {
            return Err(DirectionError::InvalidRate);
        }
        Ok(Self {
            relative_permittivity: epsilon_rate,
            relative_permeability: 0.0,
        })
    }
}

/// A real perturbation of a planar stack with a fixed number of layers.
///
/// Rates are per unit of the named parameter. Interface velocities and vacuum
/// wavelength rates use centimetres. Interface zero is at the base profile's
/// local origin. Its velocity is free and determines the motion of that origin
/// in the lab frame. Regional rates are ordered left exterior, layers, right exterior.
#[derive(Clone, Debug, PartialEq)]
pub struct GuidedPerturbation {
    parameter: String,
    units: String,
    interface_velocities: Vec<f64>,
    wavelength_rate: f64,
    regional_rates: Vec<ConstitutiveRate>,
}

impl GuidedPerturbation {
    pub fn new(
        parameter: impl Into<String>,
        units: impl Into<String>,
        interface_velocities: Vec<f64>,
        wavelength_rate: f64,
        regional_rates: Vec<ConstitutiveRate>,
    ) -> Result<Self, DirectionError> {
        let parameter = parameter.into();
        let units = units.into();
        if parameter.trim().is_empty() || units.trim().is_empty() {
            return Err(DirectionError::EmptyParameter);
        }
        if interface_velocities.is_empty() || regional_rates.len() != interface_velocities.len() + 1
        {
            return Err(DirectionError::InvalidGeometry);
        }
        if !wavelength_rate.is_finite()
            || interface_velocities.iter().any(|v| !v.is_finite())
            || interface_velocities
                .windows(2)
                .any(|v| !(v[1] - v[0]).is_finite())
            || regional_rates.iter().any(|v| {
                !v.relative_permittivity.is_finite() || !v.relative_permeability.is_finite()
            })
        {
            return Err(DirectionError::InvalidRate);
        }
        Ok(Self {
            parameter,
            units,
            interface_velocities,
            wavelength_rate,
            regional_rates,
        })
    }

    /// Varies a single thickness in centimetres, with the left interface anchored.
    pub fn thickness(layer_count: usize, layer: usize) -> Result<Self, DirectionError> {
        if layer >= layer_count {
            return Err(DirectionError::InvalidRegion);
        }
        let mut velocities = vec![0.0; layer_count + 1];
        velocities[layer + 1..].fill(1.0);
        Self::new(
            format!("layer {layer} thickness"),
            "cm",
            velocities,
            0.0,
            vec![ConstitutiveRate::default(); layer_count + 2],
        )
    }

    pub fn parameter(&self) -> &str {
        &self.parameter
    }

    pub fn units(&self) -> &str {
        &self.units
    }

    pub fn interface_velocities(&self) -> &[f64] {
        &self.interface_velocities
    }

    pub fn thickness_rates(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.interface_velocities.windows(2).map(|v| v[1] - v[0])
    }

    pub fn wavelength_rate(&self) -> f64 {
        self.wavelength_rate
    }

    /// Vacuum wavenumber rate in inverse centimetres per parameter unit.
    pub fn k0_rate(&self, k0: f64) -> Result<f64, DirectionError> {
        let rate = -(k0 / std::f64::consts::TAU) * self.wavelength_rate * k0;
        if !k0.is_finite() || k0 <= 0.0 || !rate.is_finite() {
            return Err(DirectionError::InvalidRate);
        }
        Ok(rate)
    }

    /// Combines fixed-wavelength rates and dispersion in left-to-right region order.
    pub fn total_regional_rates(
        &self,
        stack: &AnalyticalMaterialStack<Complex>,
        k0: f64,
    ) -> Result<Vec<ConstitutiveRate>, DirectionError> {
        if stack.layers_left_to_right().len() != self.thickness_rates().len() {
            return Err(DirectionError::InvalidGeometry);
        }
        let rate = self.k0_rate(k0)?;
        if rate == 0.0 {
            return Ok(self.regional_rates.clone());
        }
        let coordinate = Scalar(Complex::new(k0, 0.0));
        let order = DerivativeOrder::First;
        super::power::region_materials(stack)
            .zip(&self.regional_rates)
            .map(|(material, explicit)| {
                let epsilon = material
                    .evaluate_relative_permittivity_complex_derivative(coordinate, order)
                    * rate;
                let mu = material
                    .evaluate_relative_permeability_complex_derivative(coordinate, order)
                    * rate;
                let relative_permittivity = explicit.relative_permittivity + epsilon.re;
                let relative_permeability = explicit.relative_permeability + mu.re;
                if [
                    relative_permittivity,
                    relative_permeability,
                    epsilon.im,
                    mu.im,
                ]
                .iter()
                .any(|v| !v.is_finite())
                {
                    return Err(DirectionError::InvalidRate);
                }
                if epsilon.im != 0.0 || mu.im != 0.0 {
                    return Err(DirectionError::LossInducingRate);
                }
                Ok(ConstitutiveRate {
                    relative_permittivity,
                    relative_permeability,
                })
            })
            .collect()
    }

    pub fn regional_rates(&self) -> &[ConstitutiveRate] {
        &self.regional_rates
    }
}

/// A base field observation and its motion under the parameter direction.
/// At an interface the region in the position selects the limiting side.
#[derive(Clone, Debug, PartialEq)]
pub enum GuidedObservation {
    /// Keeps the base transverse position stationary in the lab frame.
    FixedLab(FieldPosition<f64>),
    /// Follows the selected interface offset or layer fraction.
    MaterialAttached(FieldPosition<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_motion_maps_to_thickness_rates() {
        let rates = vec![ConstitutiveRate::default(); 4];
        let shift =
            GuidedPerturbation::new("interface", "cm", vec![0.0, 1.0, 0.0], 0.0, rates.clone())
                .unwrap();
        let translation =
            GuidedPerturbation::new("translation", "cm", vec![1.0; 3], 0.0, rates).unwrap();
        let thickness = GuidedPerturbation::thickness(2, 0).unwrap();
        assert_eq!(shift.thickness_rates().collect::<Vec<_>>(), [1.0, -1.0]);
        assert_eq!(
            translation.thickness_rates().collect::<Vec<_>>(),
            [0.0, 0.0]
        );
        assert_eq!(thickness.thickness_rates().collect::<Vec<_>>(), [1.0, 0.0]);
        assert_eq!(
            GuidedPerturbation::thickness(2, 2),
            Err(DirectionError::InvalidRegion)
        );
    }
}

/// Conditioning evidence and total propagation-constant derivative at an accepted point.
#[derive(Clone, Debug)]
pub struct GuidedBetaSensitivity {
    point: super::ModePoint,
    direction: GuidedPerturbation,
    /// In inverse centimetres per parameter unit.
    pub beta_rate: Complex,
    pub characteristic: Complex,
    pub characteristic_direction: Complex,
    pub characteristic_beta: Complex,
    pub denominator_threshold: f64,
    coordinates: lamina_core::CanonicalCoordinates<DirectionalJet>,
    exterior: lamina_core::ExteriorWavevectors<DirectionalJet>,
    thickness_rates: Vec<f64>,
    material_rates: Vec<(Complex, Complex)>,
}

type DirectionalJet = lamina_core::ModeJet1<Complex, ndarray::Ix0>;

impl GuidedBetaSensitivity {
    pub fn point(&self) -> &super::ModePoint {
        &self.point
    }

    pub fn direction(&self) -> &GuidedPerturbation {
        &self.direction
    }

    /// Unscaled field derivatives at material-attached samples, in the
    /// selected backend's homogeneous gauge.
    pub fn attached_field_tangent_with<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &lamina_core::FieldSampling<f64>,
        amplitude: Complex,
    ) -> Result<lamina_core::DirectionalModeFields, GuidedBetaError> {
        let provenance = self.point.provenance();
        backend
            .directional_fields(
                lamina_core::ModalCharacteristicDirection {
                    stack: provenance.stack(),
                    coordinates: &self.coordinates,
                    exterior: &self.exterior,
                    polarisation: provenance.polarisation(),
                    thickness_rates: &self.thickness_rates,
                    constitutive_rates: &self.material_rates,
                },
                sampling,
                amplitude,
            )
            .map_err(GuidedBetaError::Evaluation)
    }
}

/// A failed sensitivity request. The borrowed base point remains available.
#[derive(Clone, Debug)]
pub enum GuidedBetaError {
    InvalidSettings,
    Direction(DirectionError),
    Ineligible,
    ExteriorCutoff {
        side: usize,
        magnitude: f64,
    },
    ExteriorRelation {
        side: usize,
        relative_defect: f64,
    },
    Evaluation(std::sync::Arc<dyn std::error::Error + Send + Sync>),
    NonSimpleRoot {
        denominator: Complex,
        threshold: f64,
    },
    NonFinite,
}

impl super::ModePoint {
    pub fn guided_beta_sensitivity_with<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        direction: &GuidedPerturbation,
        exterior_settings: super::LocalDerivativeSettings,
        slope_settings: super::ImplicitSlopeSettings,
    ) -> Result<GuidedBetaSensitivity, GuidedBetaError> {
        use lamina_core::{
            CanonicalCoordinates, EvaluateMeromorphicMaterial, ExteriorWavevectors,
            ModalCharacteristicDirection, ModeJet1, ScalarAlgebra, SeedJet,
        };
        use ndarray::{Ix0, arr0};
        type J = ModeJet1<Complex, Ix0>;
        use GuidedBetaError::*;

        exterior_settings.validate().map_err(|_| InvalidSettings)?;
        slope_settings.validate().map_err(|_| InvalidSettings)?;
        let sig = self.signature().map_err(|_| Ineligible)?;
        let k = sig.k0;
        let beta = sig.k_parallel;
        let roots = sig.exteriors();
        let tol = 1e-9;
        if k.re <= 0.0
            || k.im.abs() > tol * k.re
            || beta.re == 0.0
            || beta.im.abs() > tol * beta.re.abs()
        {
            return Err(Ineligible);
        }
        for (side, root) in roots.iter().enumerate() {
            let Some(root) = root else { continue };
            let magnitude = root.norm();
            if magnitude <= exterior_settings.exterior_kappa_cutoff {
                return Err(ExteriorCutoff { side, magnitude });
            }
            if root.im <= 0.0 || root.re.abs() > tol * root.im {
                return Err(Ineligible);
            }
        }

        let provenance = self.provenance();
        let stack = provenance.stack();
        // Walls stay fixed in the stack frame, whose origin is the first
        // interface: moving it would move the stack against the walls.
        if !stack.is_open() && direction.interface_velocities()[0] != 0.0 {
            return Err(Direction(DirectionError::MovesAgainstWalls));
        }
        let rates = direction
            .total_regional_rates(stack, k.re)
            .map_err(Direction)?;
        let at = Scalar(Complex::new(k.re, 0.0));
        let values: Vec<_> = super::power::region_materials(stack)
            .map(|m| {
                (
                    m.evaluate_relative_permittivity_complex(at),
                    m.evaluate_relative_permeability_complex(at),
                )
            })
            .collect();
        if values.iter().any(|&(e, m)| {
            [e, m]
                .iter()
                .any(|v| !v.re.is_finite() || v.re <= 0.0 || v.im != 0.0)
        }) {
            return Err(Ineligible);
        }
        let exterior_regions = [0, values.len() - 1];
        for (side, (root, &region)) in roots.iter().zip(&exterior_regions).enumerate() {
            let Some(root) = *root else { continue };
            let (e, m) = values[region];
            let q = e * m * k * k - beta * beta;
            let square = root * root;
            let relative_defect =
                (square - q).norm() / square.norm().max(q.norm()).max(f64::MIN_POSITIVE);
            if !relative_defect.is_finite()
                || relative_defect > exterior_settings.branch_relation_rtol
            {
                return Err(ExteriorRelation {
                    side,
                    relative_defect,
                });
            }
        }

        let seed = |value: Complex, rate: Complex| {
            // A zero-valued independent variable preserves the base value exactly.
            <J as SeedJet>::constant(arr0(value)).add(
                &J::variable(arr0(Complex::ZERO), 0)
                    .expect("one jet slot")
                    .scale(rate),
            )
        };
        let coordinate_jets =
            |k_rate: f64, beta_rate: Complex, constitutive_rates: &[(Complex, Complex)]| {
                let [left, right] = std::array::from_fn(|side| {
                    let root = roots[side]?;
                    let region = exterior_regions[side];
                    let (e, m) = values[region];
                    let (de, dm) = constitutive_rates[region];
                    let q_rate = (de * m + e * dm) * k * k + 2.0 * e * m * k * k_rate
                        - 2.0 * beta * beta_rate;
                    Some(seed(root, q_rate / (2.0 * root)))
                });
                (
                    CanonicalCoordinates::new(seed(k, k_rate.into()), seed(beta, beta_rate)),
                    ExteriorWavevectors::from_options(left, right),
                )
            };
        let evaluate = |k_rate: f64,
                        beta_rate: f64,
                        thickness_rates: &[f64],
                        constitutive_rates: &[(Complex, Complex)]| {
            let (coordinates, exterior) =
                coordinate_jets(k_rate, beta_rate.into(), constitutive_rates);
            backend
                .directional_characteristic(ModalCharacteristicDirection {
                    stack,
                    coordinates: &coordinates,
                    exterior: &exterior,
                    polarisation: provenance.polarisation(),
                    thickness_rates,
                    constitutive_rates,
                })
                .map_err(Evaluation)
        };

        let beta_jet = evaluate(
            0.0,
            1.0,
            &vec![0.0; stack.len()],
            &vec![(Complex::ZERO, Complex::ZERO); rates.len()],
        )?;
        let denominator = beta_jet.first()[()];
        let threshold = slope_settings.denominator_atol
            + slope_settings.denominator_rtol * slope_settings.denominator_reference_scale;
        if !threshold.is_finite() {
            return Err(InvalidSettings);
        }
        if denominator.norm() <= threshold {
            return Err(NonSimpleRoot {
                denominator,
                threshold,
            });
        }
        let thickness_rates: Vec<_> = direction.thickness_rates().collect();
        let material_rates: Vec<_> = rates
            .iter()
            .map(|rate| {
                (
                    rate.relative_permittivity.into(),
                    rate.relative_permeability.into(),
                )
            })
            .collect();
        let k_rate = direction.k0_rate(k.re).map_err(Direction)?;
        let numerator = evaluate(k_rate, 0.0, &thickness_rates, &material_rates)?.first()[()];
        let beta_rate = -numerator / denominator;
        if !beta_rate.re.is_finite() || !beta_rate.im.is_finite() {
            return Err(NonFinite);
        }
        let (coordinates, exterior) = coordinate_jets(k_rate, beta_rate, &material_rates);
        Ok(GuidedBetaSensitivity {
            point: self.clone(),
            direction: direction.clone(),
            beta_rate,
            characteristic: beta_jet.value()[()],
            characteristic_direction: numerator,
            characteristic_beta: denominator,
            denominator_threshold: threshold,
            coordinates,
            exterior,
            thickness_rates,
            material_rates,
        })
    }
}

/// Tangent at one sample in the declared frame. Components are ordered
/// `[Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz]` in the core field convention.
#[derive(Clone, Debug)]
pub struct GuidedFieldSample {
    pub observation: GuidedObservation,
    pub field: [Complex; 6],
    pub tangent: Result<[Complex; 6], GuidedObservationError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidedObservationError {
    MovingInterface { interface: usize },
    NonFinite,
}

/// Frame shared by a sampling request. Region and side come from the sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidedObservationFrame {
    FixedLab,
    MaterialAttached,
}

impl GuidedBetaSensitivity {
    /// Recover unscaled field derivatives in the selected frame.
    pub fn field_tangent_with<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &lamina_core::FieldSampling<f64>,
        frame: GuidedObservationFrame,
        amplitude: Complex,
    ) -> Result<Vec<GuidedFieldSample>, GuidedBetaError> {
        self.field_tangent_on_mesh(backend, sampling, frame, amplitude, None)
    }

    fn field_tangent_on_mesh<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &lamina_core::FieldSampling<f64>,
        frame: GuidedObservationFrame,
        amplitude: Complex,
        mesh_velocities: Option<&[f64]>,
    ) -> Result<Vec<GuidedFieldSample>, GuidedBetaError> {
        use lamina_core::{
            EvaluateMeromorphicMaterial, OwnedModeFields, Polarisation, ResolvedLayerPosition,
        };
        fn components(fields: &OwnedModeFields, index: usize) -> [Complex; 6] {
            let fields = fields.quantity();
            let e = fields.electric();
            let h = fields.magnetic();
            [
                e.x()[index],
                e.y()[index],
                e.z()[index],
                h.x()[index],
                h.y()[index],
                h.z()[index],
            ]
        }

        let recovered = self.attached_field_tangent_with(backend, sampling, amplitude)?;
        let provenance = self.point.provenance();
        let stack = provenance.stack();
        let velocities = self.direction.interface_velocities();
        let last = velocities.len() - 1;
        let sig = self
            .point
            .signature()
            .map_err(|_| GuidedBetaError::Ineligible)?;
        let k = sig.k0;
        let beta = sig.k_parallel;
        let materials: Vec<_> = super::power::region_materials(stack)
            .map(|material| {
                (
                    material.evaluate_relative_permittivity_complex(Scalar(k)),
                    material.evaluate_relative_permeability_complex(Scalar(k)),
                )
            })
            .collect();
        let fixed_lab = frame == GuidedObservationFrame::FixedLab;
        Ok(recovered
            .fields
            .sampling()
            .positions()
            .iter()
            .enumerate()
            .map(|(sample, &position)| {
                let observation = if fixed_lab {
                    GuidedObservation::FixedLab(position)
                } else {
                    GuidedObservation::MaterialAttached(position)
                };
                let field = components(&recovered.fields, sample);
                let mut tangent = components(&recovered.tangent, sample);
                let result = (|| {
                    if !fixed_lab && mesh_velocities.is_none() {
                        return Ok(tangent);
                    }
                    let (region, velocity, interface) = match position {
                        FieldPosition::LeftExterior { distance } => (
                            0,
                            velocities[0],
                            (distance.as_centimetres() == 0.0).then_some(0),
                        ),
                        FieldPosition::RightExterior { distance } => (
                            last + 1,
                            velocities[last],
                            (distance.as_centimetres() == 0.0).then_some(last),
                        ),
                        FieldPosition::Layer { index, position } => {
                            let i = index.get();
                            let width =
                                stack.layers_left_to_right()[i].thickness().as_centimetres();
                            let (velocity, on_left, on_right) = match position {
                                ResolvedLayerPosition::FromLeft(d) => {
                                    let d = d.as_centimetres();
                                    (velocities[i], d == 0.0, d == width)
                                }
                                ResolvedLayerPosition::FromRight(d) => {
                                    let d = d.as_centimetres();
                                    (velocities[i + 1], d == width, d == 0.0)
                                }
                                ResolvedLayerPosition::Fraction(f) => (
                                    (1.0 - f) * velocities[i] + f * velocities[i + 1],
                                    f == 0.0,
                                    f == 1.0,
                                ),
                            };
                            let interface = if on_left {
                                Some(i)
                            } else {
                                on_right.then_some(i + 1)
                            };
                            (i + 1, velocity, interface)
                        }
                    };
                    let target_velocity = mesh_velocities.map_or(0.0, |rates| rates[sample]);
                    if let Some(interface) = interface
                        && velocities[interface] != target_velocity
                        && (materials[interface] != materials[interface + 1]
                            || self.material_rates[interface] != self.material_rates[interface + 1])
                    {
                        return Err(GuidedObservationError::MovingInterface { interface });
                    }
                    let (e, m) = materials[region];
                    let q = e * m * k * k - beta * beta;
                    let i = Complex::I;
                    let zero = Complex::ZERO;
                    let gradient = match provenance.polarisation() {
                        Polarisation::TransverseElectric => {
                            let dy = -i * k * m * field[3];
                            [
                                zero,
                                dy,
                                zero,
                                -i * q * field[1] / (k * m),
                                zero,
                                beta * dy / (k * m),
                            ]
                        }
                        Polarisation::TransverseMagnetic => {
                            let dy = i * k * e * field[0];
                            [
                                i * q * field[4] / (k * e),
                                zero,
                                -beta * dy / (k * e),
                                zero,
                                dy,
                                zero,
                            ]
                        }
                    };
                    for (value, gradient) in tangent.iter_mut().zip(gradient) {
                        *value += (target_velocity - velocity) * gradient;
                    }
                    if tangent
                        .iter()
                        .any(|v| !v.re.is_finite() || !v.im.is_finite())
                    {
                        return Err(GuidedObservationError::NonFinite);
                    }
                    Ok(tangent)
                })();
                GuidedFieldSample {
                    observation,
                    field,
                    tangent: result,
                }
            })
            .collect())
    }
}

/// Signed longitudinal power per metre of invariant width and its rate.
#[derive(Clone, Copy, Debug)]
pub struct RegionalPowerSensitivity {
    pub power_w_per_m: f64,
    pub power_rate_w_per_m: f64,
}

/// Owned fields in the signed unit-power, positive left-interface phase gauge.
#[derive(Clone, Debug)]
pub struct GuidedPowerSensitivity {
    pub eigenvalue: GuidedBetaSensitivity,
    pub samples: Vec<GuidedFieldSample>,
    /// Left exterior, finite layers, then right exterior.
    pub regional_power: Vec<RegionalPowerSensitivity>,
}

#[derive(Clone, Debug)]
pub enum GuidedPowerSensitivityError {
    Fields(GuidedBetaError),
    SingularPhase,
    IllConditionedPower,
    Quadrature,
}

fn phase_scale(
    reference: Complex,
    rate: Complex,
    peak: f64,
) -> Result<(Complex, f64), GuidedPowerSensitivityError> {
    if !peak.is_finite() || reference.norm() <= 1e-10 * peak {
        return Err(GuidedPowerSensitivityError::SingularPhase);
    }
    Ok((reference.conj() / reference.norm(), -(rate / reference).im))
}

#[cfg(test)]
#[test]
fn vanishing_left_primary_rejects_phase_normalization() {
    assert!(matches!(
        phase_scale(Complex::ZERO, Complex::ONE, 1.0),
        Err(GuidedPowerSensitivityError::SingularPhase)
    ));
}

fn flux_and_rate(sample: &GuidedFieldSample) -> [f64; 2] {
    let [_, ey, ez, _, hy, hz] = sample.field;
    let [_, dey, dez, _, dhy, dhz] = sample.tangent.expect("attached sample has a tangent");
    [
        0.5 * (ey * hz.conj() - ez * hy.conj()).re,
        0.5 * (dey * hz.conj() + ey * dhz.conj() - dez * hy.conj() - ez * dhy.conj()).re,
    ]
}

impl GuidedBetaSensitivity {
    /// Reconstruct signed unit-power fields and differentiate the power and phase scale.
    pub fn power_field_tangent_with<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &lamina_core::FieldSampling<f64>,
        frame: GuidedObservationFrame,
        amplitude: Complex,
    ) -> Result<GuidedPowerSensitivity, GuidedPowerSensitivityError> {
        self.power_field_tangent_on_mesh(backend, sampling, frame, amplitude, None)
    }

    /// Rates of the exterior decay constants; the pairings that use them
    /// take open stacks only.
    pub(super) fn exterior_decay_rates(&self) -> [f64; 2] {
        [self.exterior.left(), self.exterior.right()]
            .map(|root| root.expect("an open side").first()[()].im)
    }

    pub(super) fn power_field_tangent_on_mesh<B: lamina_core::ModalDerivativeBackend>(
        &self,
        backend: &B,
        sampling: &lamina_core::FieldSampling<f64>,
        frame: GuidedObservationFrame,
        amplitude: Complex,
        mesh_velocities: Option<&[f64]>,
    ) -> Result<GuidedPowerSensitivity, GuidedPowerSensitivityError> {
        use super::power::CM_TO_M_OVER_Z0;
        use GuidedPowerSensitivityError::*;
        use lamina_core::Polarisation;

        let provenance = self.point.provenance();
        let stack = provenance.stack();
        let k = self.coordinates.vacuum_angular_wavenumber().value()[()];
        let beta = self.coordinates.parallel_angular_wavenumber().value()[()];
        let grid = super::power::ProfileGrid::new(stack, k, beta, 2, 4).ok_or(Quadrature)?;
        let roots = [self.exterior.left(), self.exterior.right()];
        // Walls stay fixed: the right gap narrows as the layers thicken.
        let width_rates: Vec<_> = std::iter::once(0.0)
            .chain(self.thickness_rates.iter().copied())
            .chain(std::iter::once(-self.thickness_rates.iter().sum::<f64>()))
            .collect();
        let mut previous: Option<Vec<[f64; 2]>> = None;
        for level in 0..8 {
            let integration = grid.sampling(level).ok_or(Quadrature)?;
            let fields = self
                .field_tangent_with(
                    backend,
                    &integration,
                    GuidedObservationFrame::MaterialAttached,
                    amplitude,
                )
                .map_err(Fields)?;
            let spans = grid.spans(level);
            let regional: Vec<_> = spans
                .iter()
                .map(|span| {
                    let at = |i: usize| flux_and_rate(&fields[i]);
                    match span.intervals {
                        // An open exterior's tail, analytic in its decay constant.
                        None => {
                            let root = roots[usize::from(span.region > 0)].expect("open");
                            let [flux, rate] = at(span.first);
                            let decay = root.value()[()].im;
                            let power = CM_TO_M_OVER_Z0 * flux / (2.0 * decay);
                            [
                                power,
                                CM_TO_M_OVER_Z0 * rate / (2.0 * decay)
                                    - power * root.first()[()].im / decay,
                            ]
                        }
                        // A layer is sampled at fixed fractions. A gap is sampled
                        // at fixed distances from its interface, and only the
                        // right gap changes width, at its fixed wall.
                        Some(_) => {
                            let flux = span.mean(|i| at(i)[0]);
                            let edge = if span.region == spans.len() - 1 {
                                at(span.last())[0]
                            } else {
                                flux
                            };
                            [
                                CM_TO_M_OVER_Z0 * span.width * flux,
                                CM_TO_M_OVER_Z0
                                    * (span.width * span.mean(|i| at(i)[1])
                                        + width_rates[span.region] * edge),
                            ]
                        }
                    }
                })
                .collect();
            let total: f64 = regional.iter().map(|p| p[0]).sum();
            let total_rate: f64 = regional.iter().map(|p| p[1]).sum();
            if !total.is_finite() || !total_rate.is_finite() || total == 0.0 {
                return Err(IllConditionedPower);
            }
            // Every region is converged against a common scale: powers against
            // the total power, rates against the largest regional rate. A region
            // whose rate nearly cancels needs no more relative accuracy than the
            // fractions' rates carry.
            let largest_rate =
                |regions: &[[f64; 2]]| regions.iter().map(|p| p[1].abs()).fold(0.0, f64::max);
            let converged = previous.as_ref().is_some_and(|prev| {
                let rate_scale = total
                    .abs()
                    .max(largest_rate(prev))
                    .max(largest_rate(&regional));
                prev.iter().zip(&regional).all(|(a, b)| {
                    (a[0] - b[0]).abs() <= 1e-9 * total.abs()
                        && (a[1] - b[1]).abs() <= 1e-9 * rate_scale
                })
            });
            if !converged {
                previous = Some(regional);
                continue;
            }
            let primary = match provenance.polarisation() {
                Polarisation::TransverseElectric => 1,
                Polarisation::TransverseMagnetic => 4,
            };
            let left = &fields[spans[0].last()];
            let reference = left.field[primary];
            let reference_rate = left.tangent.expect("attached tangent")[primary];
            let peak = fields
                .iter()
                .map(|sample| sample.field[primary].norm())
                .fold(0.0, f64::max);
            let (phase, phase_rate) = phase_scale(reference, reference_rate, peak)?;
            let scale = phase / total.abs().sqrt();
            let log_rate = Complex::new(-0.5 * total_rate / total, phase_rate);
            if [scale, log_rate]
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
            {
                return Err(IllConditionedPower);
            }
            let mut samples = self
                .field_tangent_on_mesh(backend, sampling, frame, amplitude, mesh_velocities)
                .map_err(Fields)?;
            for sample in &mut samples {
                if let Ok(tangent) = &mut sample.tangent {
                    for (derivative, field) in tangent.iter_mut().zip(sample.field) {
                        *derivative = scale * (*derivative + log_rate * field);
                    }
                }
                for field in &mut sample.field {
                    *field *= scale;
                }
            }
            let regional_power = regional
                .into_iter()
                .map(|[power, rate]| RegionalPowerSensitivity {
                    power_w_per_m: power / total.abs(),
                    power_rate_w_per_m: (rate - power * total_rate / total) / total.abs(),
                })
                .collect();
            return Ok(GuidedPowerSensitivity {
                eigenvalue: self.clone(),
                samples,
                regional_power,
            });
        }
        Err(Quadrature)
    }
}
