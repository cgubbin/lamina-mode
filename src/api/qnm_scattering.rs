//! Port-resolved scattering residues of normalized planar frequency poles.

use super::*;
use lamina_core::{
    AnalyticalMaterialStack, EvaluateMeromorphicMaterial, ExteriorSampling, FieldSampling,
    ModalNormalizationBackend, Polarisation, Scalar,
};
use lamina_units::Length;

/// The primary channel field at each exterior interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QnmChannelField {
    ElectricY,
    ImpedanceScaledMagneticY,
}

/// Laurent residue of the two-port field-amplitude scattering matrix in k₀.
/// Rows are outgoing and columns incoming, both ordered left then right.
#[derive(Clone, Debug)]
pub struct QnmScatteringResidue {
    pub point: ModePoint,
    pub normalization: PoleNormalizationData,
    pub signature: BranchSignature,
    pub channel_field: QnmChannelField,
    pub reference_planes_cm: [f64; 2],
    pub exterior_admittances_per_cm: [Complex; 2],
    pub residue_per_cm: [[Complex; 2]; 2],
}

impl QnmScatteringResidue {
    /// Residue in angular-frequency coordinates, in inverse seconds.
    pub fn angular_frequency_residue(&self) -> [[Complex; 2]; 2] {
        self.residue_per_cm
            .map(|row| row.map(|entry| entry * 2.99792458e10))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QnmScatteringError {
    #[error("requires a normalized simple frequency pole at fixed real in-plane wavevector")]
    IncompatiblePole,
    #[error("exterior channel admittance or residue is singular or nonfinite")]
    InvalidChannel,
    #[error(transparent)]
    Fields(#[from] Box<FieldRecoveryFailure>),
}

impl<B: ModalNormalizationBackend + Clone> NormalizedMode<B> {
    /// Compute the port residues on the accepted exterior sheets. Channel
    /// amplitudes use Ey for TE and Z0 Hy for TM, with exp(−iωt).
    pub fn scattering_residue(&self) -> Result<QnmScatteringResidue, QnmScatteringError> {
        use QnmScatteringError::*;
        let signature = *self.point.signature().map_err(|_| IncompatiblePole)?;
        if self.normalization.input.variable != PoleVariable::K0AtFixedBeta
            || !signature.k_parallel.re.is_finite()
            || !signature.k0.re.is_finite()
            || !signature.k0.im.is_finite()
            || signature.k_parallel.im != 0.0
            || signature.k0.re <= 0.0
            || signature.k0.im >= 0.0
        {
            return Err(IncompatiblePole);
        }
        let provenance = self.point.provenance();
        let stack = provenance.stack();
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(Length::zero()))
            .right_exterior(ExteriorSampling::point(Length::zero()));
        let profile = self.sample_profile(&sampling)?;
        let fields = profile.fields.quantity();
        let (channel_field, primary) = match provenance.polarisation() {
            Polarisation::TransverseElectric => (QnmChannelField::ElectricY, fields.electric().y()),
            Polarisation::TransverseMagnetic => (
                QnmChannelField::ImpedanceScaledMagneticY,
                fields.magnetic().y(),
            ),
        };
        let materials = [stack.left_exterior(), stack.right_exterior()];
        let roots = [signature.left, signature.right];
        let exterior_admittances_per_cm = std::array::from_fn(|i| {
            let coordinate = Scalar(signature.k0);
            let factor = match channel_field {
                QnmChannelField::ElectricY => {
                    materials[i].evaluate_relative_permeability_complex(coordinate)
                }
                QnmChannelField::ImpedanceScaledMagneticY => {
                    materials[i].evaluate_relative_permittivity_complex(coordinate)
                }
            };
            roots[i] / factor
        });
        // An incident primary amplitude a imposes 2iκa/f in the boundary
        // forcing. The normalized scalar Green residue is ψ_i ψ_j.
        let residue_per_cm = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                2.0 * Complex::I * exterior_admittances_per_cm[j] * primary[i] * primary[j]
            })
        });
        let finite = |z: Complex| z.re.is_finite() && z.im.is_finite();
        if exterior_admittances_per_cm
            .iter()
            .any(|&z| !finite(z) || z == Complex::ZERO)
            || residue_per_cm.iter().flatten().any(|&z| !finite(z))
        {
            return Err(InvalidChannel);
        }
        Ok(QnmScatteringResidue {
            point: self.point.clone(),
            normalization: self.normalization.clone(),
            signature,
            channel_field,
            reference_planes_cm: [0.0, profile.interfaces_cm.last().copied().unwrap_or(0.0)],
            exterior_admittances_per_cm,
            residue_per_cm,
        })
    }
}

/// Positive constant isotropic exterior parameters declared over the full
/// pole-to-window continuation. The implementation checks sampled values.
#[derive(Clone, Copy, Debug)]
pub struct QnmConstantChannels {
    pub epsilon: [f64; 2],
    pub mu: [f64; 2],
}

/// Real-frequency approximation controls, in the vacuum wavenumber coordinate.
#[derive(Clone, Copy, Debug)]
pub struct QnmWindowSettings {
    pub k0_per_cm: [f64; 2],
    pub regular_degree: usize,
    pub validation_points: usize,
    pub amplitude_atol: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QnmWindowEvidence {
    /// Real training-node error. Absent for contour reconstruction.
    pub fit_max_amplitude_error: Option<f64>,
    pub direct_construction_samples: usize,
    pub held_out_samples: usize,
    pub held_out_max_phase_error_rad: f64,
    pub held_out_max_reciprocity_defect: f64,
    pub held_out_max_unitarity_defect: f64,
    pub held_out_max_passivity_excess: f64,
    pub contour_max_refinement_change: Option<f64>,
    pub held_out_max_amplitude_error: f64,
    pub held_out_max_pole_only_error: f64,
    /// Maximum error in outgoing to incoming power ratio over validation samples.
    pub held_out_max_power_ratio_error: f64,
    /// Maximum deviation from unit outgoing to incoming power ratio.
    pub held_out_max_flux_balance_defect: f64,
    pub meets_amplitude_tolerance: bool,
}

#[derive(Clone, Debug)]
pub struct QnmScatteringWindow {
    poles: Vec<QnmScatteringResidue>,
    settings: QnmWindowSettings,
    channels: ScatteringChannels,
    pub evidence: QnmWindowEvidence,
    beta_per_cm: f64,
    channel_field: QnmChannelField,
    regular: RegularContribution,
}

#[derive(Debug, thiserror::Error)]
pub enum QnmWindowError {
    #[error("invalid window controls or incompatible poles")]
    InvalidInput,
    #[error("declared exterior parameters disagree with the stack or pole branches")]
    ChannelMismatch,
    #[error("the window includes a nonpropagating or grazing exterior channel")]
    ExteriorThreshold,
    #[error("direct scattering returned an unusable amplitude")]
    DirectScattering,
    #[error("invalid Cauchy disk or quadrature controls")]
    Contour,
    #[error("dispersive reconstruction requires a real outgoing reference")]
    RadiationReference,
    #[error("the requested window or contour lies outside the declared continuation domain")]
    ContinuationDomain,
    #[error("dispersive frequency domain setup failed")]
    Domain(#[source] Diagnostic),
    #[error(transparent)]
    Continuation(#[from] FrequencyError),
    #[error("regular contribution fit failed")]
    Fit,
}

type ChannelMatrix = [[Complex; 2]; 2];

/// Cauchy quadrature in a disk centred at the real window midpoint.
/// The caller declares that the pole-subtracted scattering matrix is
/// holomorphic on and inside this disk, with constant exterior parameters.
#[derive(Clone, Copy, Debug)]
pub struct QnmContourSettings {
    pub radius_per_cm: f64,
    /// Even total count, including the nested half-size grid.
    pub samples: usize,
}

#[derive(Clone, Debug)]
enum RegularContribution {
    Chebyshev(Vec<ChannelMatrix>),
    Cauchy(Vec<(Complex, Complex, ChannelMatrix)>),
}

impl QnmScatteringWindow {
    pub fn poles(&self) -> &[QnmScatteringResidue] {
        &self.poles
    }
    pub fn settings(&self) -> QnmWindowSettings {
        self.settings
    }
    pub fn constant_channels(&self) -> Option<QnmConstantChannels> {
        match self.channels {
            ScatteringChannels::Constant { parameters, .. } => Some(parameters),
            ScatteringChannels::Continued { .. } => None,
        }
    }

    pub fn continuation_evidence(&self) -> Option<&FrequencyEvidence> {
        match &self.channels {
            ScatteringChannels::Constant { .. } => None,
            ScatteringChannels::Continued { chart, .. } => Some(chart.evidence()),
        }
    }

    pub fn continuation_region(&self) -> Option<FrequencyRegion> {
        match &self.channels {
            ScatteringChannels::Constant { .. } => None,
            ScatteringChannels::Continued { region, .. } => Some(*region),
        }
    }

    pub fn continuation_assumption(&self) -> Option<AnalyticAssumption> {
        match &self.channels {
            ScatteringChannels::Constant { .. } => None,
            ScatteringChannels::Continued { assumption, .. } => Some(*assumption),
        }
    }

    /// Evaluate complex field amplitudes on the declared real window.
    pub fn amplitudes(&self, k0_per_cm: f64) -> Option<[[Complex; 2]; 2]> {
        let [lo, hi] = self.settings.k0_per_cm;
        if !(lo..=hi).contains(&k0_per_cm) {
            return None;
        }
        let x = (2.0 * (k0_per_cm - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0);
        let pole = principal_parts(&self.poles, k0_per_cm.into());
        let regular = match &self.regular {
            RegularContribution::Chebyshev(coefficients) => {
                let basis = chebyshev(x, coefficients.len());
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        coefficients
                            .iter()
                            .zip(&basis)
                            .map(|(entry, t)| entry[i][j] * t)
                            .sum()
                    })
                })
            }
            RegularContribution::Cauchy(nodes) => cauchy(nodes, k0_per_cm, 1),
        };
        let matrix = std::array::from_fn(|i| std::array::from_fn(|j| pole[i][j] + regular[i][j]));
        matrix
            .iter()
            .flatten()
            .all(|v| v.re.is_finite() && v.im.is_finite())
            .then_some(matrix)
    }
    /// Reflected and transmitted flux fractions in matrix order. Diagonal
    /// entries are reflectances, off-diagonal entries are transmittances.
    pub fn power_ratios(&self, k0_per_cm: f64) -> Option<[[f64; 2]; 2]> {
        let amplitudes = self.amplitudes(k0_per_cm)?;
        let (_, weights) = self
            .channels
            .at(k0_per_cm.into(), self.beta_per_cm, self.channel_field)
            .ok()?;
        Some(std::array::from_fn(|i| {
            std::array::from_fn(|j| weights[i] / weights[j] * amplitudes[i][j].norm_sqr())
        }))
    }
}

fn chebyshev(x: f64, count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(count);
    let (mut previous, mut current) = (1.0, x);
    for _ in 0..count {
        values.push(previous);
        (previous, current) = (current, 2.0 * x * current - previous);
    }
    values
}

pub(super) fn principal_parts(poles: &[QnmScatteringResidue], k: Complex) -> ChannelMatrix {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            poles
                .iter()
                .map(|pole| pole.residue_per_cm[i][j] / (k - pole.signature.k0))
                .sum()
        })
    })
}

/// Fit a finite-window regular contribution using direct Scatter2 samples.
/// The caller declares that the positive exterior parameters are constant
/// between the poles and the real window. The pole set remains caller-selected.
pub fn reconstruct_qnm_scattering_window(
    poles: Vec<QnmScatteringResidue>,
    channels: QnmConstantChannels,
    settings: QnmWindowSettings,
) -> Result<QnmScatteringWindow, QnmWindowError> {
    let reference = poles
        .first()
        .ok_or(QnmWindowError::InvalidInput)?
        .point
        .clone();
    reconstruct(
        ScatteringReference::from_point(&reference)?,
        poles,
        ExteriorRequest::Constant(channels),
        settings,
        None,
    )
}

/// Fit the same regular basis with an empty pole set.
pub fn reconstruct_direct_scattering_window(
    stack: &AnalyticalMaterialStack<Complex>,
    beta_per_cm: f64,
    polarisation: Polarisation,
    channels: QnmConstantChannels,
    settings: QnmWindowSettings,
) -> Result<QnmScatteringWindow, QnmWindowError> {
    reconstruct(
        ScatteringReference {
            stack: stack.clone(),
            beta_per_cm,
            polarisation,
        },
        Vec::new(),
        ExteriorRequest::Constant(channels),
        settings,
        None,
    )
}

/// Reconstruct the regular scattering contribution by nested Cauchy quadrature.
pub fn reconstruct_qnm_scattering_contour(
    reference: &ModePoint,
    poles: Vec<QnmScatteringResidue>,
    channels: QnmConstantChannels,
    settings: QnmWindowSettings,
    contour: QnmContourSettings,
) -> Result<QnmScatteringWindow, QnmWindowError> {
    reconstruct(
        ScatteringReference::from_point(reference)?,
        poles,
        ExteriorRequest::Constant(channels),
        settings,
        Some(contour),
    )
}

fn cauchy(nodes: &[(Complex, Complex, ChannelMatrix)], k: f64, stride: usize) -> ChannelMatrix {
    let count = nodes.len() / stride;
    let mut result = [[Complex::ZERO; 2]; 2];
    for &(node, radial, remainder) in nodes.iter().step_by(stride) {
        let weight = -radial / (k - node) / count as f64;
        for i in 0..2 {
            for j in 0..2 {
                result[i][j] += weight * remainder[i][j];
            }
        }
    }
    result
}

struct ScatteringReference {
    stack: AnalyticalMaterialStack<Complex>,
    beta_per_cm: f64,
    polarisation: Polarisation,
}

impl ScatteringReference {
    fn from_point(point: &ModePoint) -> Result<Self, QnmWindowError> {
        let signature = point
            .signature()
            .map_err(|_| QnmWindowError::InvalidInput)?;
        if !matches!(
            point.provenance().direction(),
            SearchCoordinate::Spectral { .. }
        ) || signature.k_parallel.im != 0.0
        {
            return Err(QnmWindowError::InvalidInput);
        }
        Ok(Self {
            stack: point.provenance().stack().clone(),
            beta_per_cm: signature.k_parallel.re,
            polarisation: point.provenance().polarisation(),
        })
    }
}

/// Use meromorphic exterior models with a common outgoing continuation
/// region. The supplied real reference must select outgoing on both sides.
pub fn reconstruct_dispersive_qnm_scattering(
    reference: &ModePoint,
    poles: Vec<QnmScatteringResidue>,
    domain: FrequencyDomainInput,
    assumption: AnalyticAssumption,
    settings: QnmWindowSettings,
    contour: Option<QnmContourSettings>,
) -> Result<QnmScatteringWindow, QnmWindowError> {
    reconstruct(
        ScatteringReference::from_point(reference)?,
        poles,
        ExteriorRequest::Continued { domain, assumption },
        settings,
        contour,
    )
}

enum ExteriorRequest {
    Constant(QnmConstantChannels),
    Continued {
        domain: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    },
}

use crate::problem::frequency::FrequencyChart;
type ExteriorMaterials = [lamina_core::AnalyticalMaterialHandle<Complex>; 2];

#[derive(Clone, Debug)]
enum ScatteringChannels {
    Constant {
        parameters: QnmConstantChannels,
        materials: ExteriorMaterials,
    },
    Continued {
        chart: Box<FrequencyChart>,
        region: FrequencyRegion,
        assumption: AnalyticAssumption,
        materials: ExteriorMaterials,
    },
}

impl ScatteringChannels {
    fn at(
        &self,
        k: Complex,
        beta: f64,
        field: QnmChannelField,
    ) -> Result<([Complex; 2], [f64; 2]), QnmWindowError> {
        use QnmWindowError::*;
        let materials = match self {
            Self::Constant { materials, .. } | Self::Continued { materials, .. } => materials,
        };
        let e = materials
            .each_ref()
            .map(|m| m.evaluate_relative_permittivity_complex(Scalar(k)));
        let mu = materials
            .each_ref()
            .map(|m| m.evaluate_relative_permeability_complex(Scalar(k)));
        let roots = match self {
            Self::Constant { parameters, .. } => {
                if (0..2).any(|i| {
                    !channel_close(e[i], parameters.epsilon[i].into())
                        || !channel_close(mu[i], parameters.mu[i].into())
                }) {
                    return Err(ChannelMismatch);
                }
                std::array::from_fn(|i| {
                    (parameters.epsilon[i] * parameters.mu[i] * k * k - beta * beta).sqrt()
                })
            }
            Self::Continued { chart, region, .. } => {
                let bounds = region.bounds();
                let (lo, hi) = bounds.real_bounds();
                let (bottom, top) = bounds.imaginary_bounds();
                if !(lo..=hi).contains(&k.re) || !(bottom..=top).contains(&k.im) {
                    return Err(ContinuationDomain);
                }
                [
                    chart.continuation_at(0, k)?.value,
                    chart.continuation_at(1, k)?.value,
                ]
            }
        };
        let factor = match field {
            QnmChannelField::ElectricY => mu,
            QnmChannelField::ImpedanceScaledMagneticY => e,
        };
        let flux = std::array::from_fn(|i| (roots[i] / factor[i]).re);
        if roots
            .iter()
            .any(|r| !r.re.is_finite() || !r.im.is_finite() || *r == Complex::ZERO)
            || k.im == 0.0
                && (e
                    .iter()
                    .chain(&mu)
                    .any(|m| !m.re.is_finite() || m.re <= 0.0 || m.im.abs() > 1e-9 * m.re)
                    || roots
                        .iter()
                        .any(|r| r.re <= 0.0 || r.im.abs() > 1e-9 * r.re)
                    || flux.iter().any(|w| !w.is_finite() || *w <= 0.0))
        {
            return Err(ExteriorThreshold);
        }
        Ok((roots, flux))
    }
}

fn channel_close(a: Complex, b: Complex) -> bool {
    (a - b).norm() <= 1e-9 * b.norm().max(f64::MIN_POSITIVE)
}

fn reconstruct(
    reference: ScatteringReference,
    poles: Vec<QnmScatteringResidue>,
    exterior_request: ExteriorRequest,
    settings: QnmWindowSettings,
    contour: Option<QnmContourSettings>,
) -> Result<QnmScatteringWindow, QnmWindowError> {
    use QnmWindowError::*;
    use lamina_core::{
        CanonicalCoordinates, ComplexPlaneEvaluator, ExteriorWavevectors, IncidentSide, ModeJet1,
        Scatter2, SeedJet,
    };
    use ndarray::{Ix0, arr0};
    use std::f64::consts::PI;
    type J = ModeJet1<Complex, Ix0>;

    let ScatteringReference {
        stack,
        beta_per_cm,
        polarisation: pol,
    } = reference;
    if !beta_per_cm.is_finite() {
        return Err(InvalidInput);
    }
    let beta = Complex::from(beta_per_cm);
    let [lo, hi] = settings.k0_per_cm;
    if !(lo.is_finite() && hi.is_finite() && 0.0 < lo && lo < hi)
        || settings.regular_degree > 24
        || !(3..=4096).contains(&settings.validation_points)
        || !settings.amplitude_atol.is_finite()
        || settings.amplitude_atol <= 0.0
    {
        return Err(InvalidInput);
    }
    let field = match pol {
        Polarisation::TransverseElectric => QnmChannelField::ElectricY,
        Polarisation::TransverseMagnetic => QnmChannelField::ImpedanceScaledMagneticY,
    };
    let half_width = (hi - lo) * 0.5;
    let midpoint = lo + half_width;
    let radius = contour.map_or(0.0, |settings| settings.radius_per_cm);
    if let Some(contour) = contour
        && (!radius.is_finite()
            || radius <= half_width
            || !(16..=16384).contains(&contour.samples)
            || contour.samples % 2 != 0)
    {
        return Err(Contour);
    }
    let materials = [
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
    ];
    let channels = match exterior_request {
        ExteriorRequest::Constant(parameters) => {
            if parameters
                .epsilon
                .iter()
                .chain(&parameters.mu)
                .any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err(InvalidInput);
            }
            let leftmost = if contour.is_some() {
                midpoint - radius
            } else {
                lo
            };
            if (0..2).any(|i| {
                let n = parameters.epsilon[i] * parameters.mu[i];
                !n.is_finite() || leftmost <= beta_per_cm.abs() / n.sqrt()
            }) {
                return Err(ExteriorThreshold);
            }
            ScatteringChannels::Constant {
                parameters,
                materials,
            }
        }
        ExteriorRequest::Continued { domain, assumption } => {
            use crate::problem::frequency::FrequencyProblem;
            if !matches!(
                domain.reference,
                FrequencyReference::RealRadiation {
                    choices: [
                        RadiationChoice::OutgoingOrDecaying,
                        RadiationChoice::OutgoingOrDecaying
                    ],
                    ..
                }
            ) {
                return Err(RadiationReference);
            }
            let region = domain.bounds;
            let bounds = region.bounds();
            let (left, right) = bounds.real_bounds();
            let (bottom, top) = bounds.imaginary_bounds();
            if left > lo.min(midpoint - radius)
                || right < hi.max(midpoint + radius)
                || bottom > -radius
                || top < radius
            {
                return Err(ContinuationDomain);
            }
            let problem =
                FrequencyProblem::frequency(stack.clone(), beta_per_cm, pol, Scatter2::new())
                    .map_err(|e| Domain(Diagnostic::new(Stage::Construction, e)))?;
            let domain = problem.frequency_domain(domain, assumption)?;
            let chart = domain.frequency_chart().clone();
            let reference = chart.evidence().reference.frequency;
            let continued = ScatteringChannels::Continued {
                chart: Box::new(chart),
                region,
                assumption,
                materials,
            };
            continued.at(reference, beta_per_cm, field)?;
            continued
        }
    };
    for (i, pole) in poles.iter().enumerate() {
        if pole.point.provenance().stack() != &stack
            || pole.point.provenance().polarisation() != pol
            || pole.signature.k_parallel != beta
            || poles[..i]
                .iter()
                .any(|other| channel_close(pole.signature.k0, other.signature.k0))
        {
            return Err(InvalidInput);
        }
        let (roots, _) = channels.at(pole.signature.k0, beta_per_cm, field)?;
        if !channel_close(pole.signature.left, roots[0])
            || !channel_close(pole.signature.right, roots[1])
        {
            return Err(ChannelMismatch);
        }
    }
    let evaluator = ComplexPlaneEvaluator::<J, _, _>::compile(&stack, Scatter2::new())
        .map_err(|_| DirectScattering)?;
    let seed = |value| <J as SeedJet>::constant(arr0(value));
    let complex_direct = |k: Complex| -> Result<_, QnmWindowError> {
        let (roots, flux) = channels.at(k, beta_per_cm, field)?;
        let result = evaluator
            .solve(
                &CanonicalCoordinates::new(seed(k), seed(beta)),
                &ExteriorWavevectors::new(seed(roots[0]), seed(roots[1])),
                pol,
            )
            .map_err(|_| DirectScattering)?;
        let l = result.amplitudes(IncidentSide::Left);
        let r = result.amplitudes(IncidentSide::Right);
        let values = [
            [l.reflection().value()[()], r.transmission().value()[()]],
            [l.transmission().value()[()], r.reflection().value()[()]],
        ];
        if values
            .iter()
            .flatten()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(DirectScattering);
        }
        Ok((values, flux))
    };
    let (regular, training, direct_construction_samples) = if let Some(contour) = contour {
        let nodes = (0..contour.samples)
            .map(|i| {
                let radial =
                    radius * (Complex::I * (2.0 * PI * i as f64 / contour.samples as f64)).exp();
                let node = midpoint + radial;
                let (target, _) = complex_direct(node)?;
                let principal = principal_parts(&poles, node);
                let remainder = std::array::from_fn(|i| {
                    std::array::from_fn(|j| target[i][j] - principal[i][j])
                });
                if remainder
                    .iter()
                    .flatten()
                    .any(|v| !v.re.is_finite() || !v.im.is_finite())
                {
                    return Err(Contour);
                }
                Ok((node, radial, remainder))
            })
            .collect::<Result<Vec<_>, _>>()?;
        (
            RegularContribution::Cauchy(nodes),
            Vec::new(),
            contour.samples,
        )
    } else {
        let order = settings.regular_degree + 1;
        let count = 2 * order;
        let abscissa = |x: f64| lo + (hi - lo) * (x + 1.0) * 0.5;
        let mut training = Vec::with_capacity(count);
        let mut regular = vec![[[Complex::ZERO; 2]; 2]; order];
        for sample in 0..count {
            let x = (PI * (sample as f64 + 0.5) / count as f64).cos();
            let k = abscissa(x);
            let (target, _) = complex_direct(k.into())?;
            let principal = principal_parts(&poles, k.into());
            for (d, (coefficient, t)) in regular.iter_mut().zip(chebyshev(x, order)).enumerate() {
                let weight = if d == 0 { 1.0 } else { 2.0 } / count as f64;
                for i in 0..2 {
                    for j in 0..2 {
                        coefficient[i][j] += (target[i][j] - principal[i][j]) * (weight * t);
                    }
                }
            }
            training.push((k, target));
        }
        if regular
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Fit);
        }
        (RegularContribution::Chebyshev(regular), training, count)
    };
    // Gather held-out direct values before transferring ownership of the poles.
    let mut validation = Vec::with_capacity(settings.validation_points);
    for i in 0..settings.validation_points {
        let k = lo + (hi - lo) * (i as f64 / (settings.validation_points - 1) as f64);
        if training.iter().all(|(sample, _)| *sample != k) {
            let (target, weights) = complex_direct(k.into())?;
            validation.push((k, target, weights));
        }
    }
    if validation.len() < 3 {
        return Err(Fit);
    }
    let mut result = QnmScatteringWindow {
        poles,
        settings,
        channels,
        beta_per_cm: beta.re,
        channel_field: match pol {
            Polarisation::TransverseElectric => QnmChannelField::ElectricY,
            Polarisation::TransverseMagnetic => QnmChannelField::ImpedanceScaledMagneticY,
        },
        regular,
        evidence: QnmWindowEvidence {
            fit_max_amplitude_error: contour.is_none().then_some(0.0),
            direct_construction_samples,
            held_out_samples: validation.len(),
            held_out_max_phase_error_rad: 0.0,
            held_out_max_reciprocity_defect: 0.0,
            held_out_max_unitarity_defect: 0.0,
            held_out_max_passivity_excess: 0.0,
            contour_max_refinement_change: contour.map(|_| 0.0),
            held_out_max_amplitude_error: 0.0,
            held_out_max_pole_only_error: 0.0,
            held_out_max_power_ratio_error: 0.0,
            held_out_max_flux_balance_defect: 0.0,
            meets_amplitude_tolerance: false,
        },
    };
    let error = |actual: ChannelMatrix, target: ChannelMatrix| {
        actual
            .iter()
            .flatten()
            .zip(target.iter().flatten())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0, f64::max)
    };
    for (k, target) in training {
        let actual = result.amplitudes(k).ok_or(Fit)?;
        result.evidence.fit_max_amplitude_error = result
            .evidence
            .fit_max_amplitude_error
            .map(|prior| prior.max(error(actual, target)));
    }
    for (k, target, weights) in validation {
        result.evidence.held_out_max_pole_only_error = result
            .evidence
            .held_out_max_pole_only_error
            .max(error(principal_parts(&result.poles, k.into()), target));
        let actual = result.amplitudes(k).ok_or(Fit)?;
        result.evidence.held_out_max_amplitude_error = result
            .evidence
            .held_out_max_amplitude_error
            .max(error(actual, target));
        let flux_normalized: ChannelMatrix = std::array::from_fn(|i| {
            std::array::from_fn(|j| actual[i][j] * (weights[i] / weights[j]).sqrt())
        });
        let [[a, b], [c, d]] = flux_normalized;
        let diag0 = a.norm_sqr() + c.norm_sqr();
        let diag1 = b.norm_sqr() + d.norm_sqr();
        let cross = a.conj() * b + c.conj() * d;
        let defect = (diag0 - 1.0)
            .hypot(diag1 - 1.0)
            .hypot(2.0_f64.sqrt() * cross.norm());
        let excess =
            (0.5 * (diag0 + diag1 + (diag0 - diag1).hypot(2.0 * cross.norm())) - 1.0).max(0.0);
        result.evidence.held_out_max_reciprocity_defect = result
            .evidence
            .held_out_max_reciprocity_defect
            .max((b - c).norm());
        result.evidence.held_out_max_unitarity_defect =
            result.evidence.held_out_max_unitarity_defect.max(defect);
        result.evidence.held_out_max_passivity_excess =
            result.evidence.held_out_max_passivity_excess.max(excess);
        for (&actual, &target) in actual.iter().flatten().zip(target.iter().flatten()) {
            if actual.norm() >= 1e-6 && target.norm() >= 1e-6 {
                result.evidence.held_out_max_phase_error_rad = result
                    .evidence
                    .held_out_max_phase_error_rad
                    .max((actual / target).arg().abs());
            }
        }
        if let RegularContribution::Cauchy(nodes) = &result.regular {
            let fine = cauchy(nodes, k, 1);
            let coarse = cauchy(nodes, k, 2);
            result.evidence.contour_max_refinement_change = result
                .evidence
                .contour_max_refinement_change
                .map(|prior| prior.max(error(fine, coarse)));
        }
        for j in 0..2 {
            let ratio = |s: ChannelMatrix| {
                (0..2)
                    .map(|i| weights[i] / weights[j] * s[i][j].norm_sqr())
                    .sum::<f64>()
            };
            let observed = ratio(actual);
            result.evidence.held_out_max_power_ratio_error = result
                .evidence
                .held_out_max_power_ratio_error
                .max((observed - ratio(target)).abs());
            result.evidence.held_out_max_flux_balance_defect = result
                .evidence
                .held_out_max_flux_balance_defect
                .max((observed - 1.0).abs());
        }
    }
    result.evidence.meets_amplitude_tolerance =
        result.evidence.held_out_max_amplitude_error <= settings.amplitude_atol;
    Ok(result)
}
