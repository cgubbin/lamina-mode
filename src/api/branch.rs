//! Local dispersion branches over real control coordinates in inverse centimetres.
use super::*;
use crate::problem as p;
use crate::problem::material::{self, Channel};
use lamina_core::{
    Backend, ComplexPlane, ModalDerivativeBackend, PlaneWaveDeterminant,
    ProjectPlaneWaveModeDeterminant, StackRegion,
};

#[derive(Debug, Clone, Copy)]
pub struct BranchSettings {
    pub local_assumption: LocalAnalyticityAssumption,
    pub domain_assumption: AnalyticAssumption,
    /// Half-width in the station's native chart units.
    pub radius: f64,
    /// Smallest subdivided interval in inverse centimetres.
    pub minimum_step: f64,
    /// Maximum total subdivisions per requested station.
    pub retries: usize,
    pub refinement: RefinementConfig,
    /// Threshold for the physical implicit derivative denominator.
    pub slope: ImplicitSlopeSettings,
}

#[derive(Debug, thiserror::Error)]
pub enum BranchFailure {
    #[error("invalid continuation schedule, settings, or source backend")]
    InvalidRequest,
    #[error("exterior cutoff or ambiguous sheet transport")]
    Exterior,
    #[error("physical derivative unavailable: {0}")]
    Derivative(String),
    #[error(transparent)]
    Slope(#[from] ImplicitSlopeFailure),
    #[error("local correction failed: {0}")]
    Correction(String),
    #[error("station domain failed: {0}")]
    Domain(String),
}
#[derive(Debug)]
pub struct BranchStop {
    pub requested: f64,
    pub attempted: f64,
    pub cause: BranchFailure,
}
#[derive(Debug)]
pub struct BranchTrace {
    /// Seed followed by every accepted point, including inserted stations.
    pub points: Vec<ModePoint>,
    /// Requested destination for each accepted step after the seed.
    pub destinations: Vec<f64>,
    /// Failed attempted coordinates and causes for intervals later subdivided.
    pub retries: Vec<(f64, BranchFailure)>,
    pub stop: Option<BranchStop>,
}

fn control(point: &ModePoint) -> f64 {
    match point.provenance().direction() {
        SearchCoordinate::InPlane { fixed_k0 } => fixed_k0,
        SearchCoordinate::Spectral { fixed_k_parallel } => fixed_k_parallel,
    }
}

impl ModePoint {
    /// Trace a local root over strictly monotone real targets, excluding the seed.
    pub fn continue_with<B, E>(
        &self,
        backend: B,
        targets: &[f64],
        settings: BranchSettings,
    ) -> Result<BranchTrace, BranchFailure>
    where
        B: Backend<p::J, ComplexPlane, Entries = E> + ModalDerivativeBackend + Clone,
        E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<p::J>>,
    {
        let config = crate::refinement::RefinementConfig::from(settings.refinement)
            .validate()
            .map_err(|_| BranchFailure::InvalidRequest)?;
        let start = control(self);
        let mut previous = start;
        if settings.slope.validate().is_err()
            || settings.refinement.max_updates == 0
            || settings.refinement.max_evaluations < 2
            || settings.refinement.max_backtracks == 0
            || self.backend() != std::any::type_name::<B>()
            || targets.is_empty()
            || !settings.radius.is_finite()
            || settings.radius <= 0.
            || !settings.minimum_step.is_finite()
            || settings.minimum_step <= 0.
            || targets.iter().any(|&target| {
                let delta = target - previous;
                previous = target;
                !delta.is_finite() || delta == 0. || delta.signum() != (targets[0] - start).signum()
            })
        {
            return Err(BranchFailure::InvalidRequest);
        }
        let mut trace = BranchTrace {
            points: vec![self.clone()],
            destinations: vec![],
            retries: vec![],
            stop: None,
        };
        for &requested in targets {
            let mut pending = vec![requested];
            let mut retries = settings.retries;
            while let Some(target) = pending.pop() {
                let last = trace.points.last().unwrap();
                match last.branch_step(backend.clone(), target, settings, config) {
                    Ok(point) => {
                        trace.destinations.push(requested);
                        trace.points.push(point)
                    }
                    Err(cause) => {
                        let here = control(last);
                        let mid = here + (target - here) * 0.5;
                        if retries == 0
                            || mid == target
                            || mid == here
                            || (mid - here).abs() < settings.minimum_step
                        {
                            trace.stop = Some(BranchStop {
                                requested,
                                attempted: target,
                                cause,
                            });
                            return Ok(trace);
                        }
                        retries -= 1;
                        trace.retries.push((target, cause));
                        pending.extend([target, mid]);
                    }
                }
            }
        }
        Ok(trace)
    }
    fn branch_step<B, E>(
        &self,
        backend: B,
        target: f64,
        s: BranchSettings,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<Self, BranchFailure>
    where
        B: Backend<p::J, ComplexPlane, Entries = E> + ModalDerivativeBackend + Clone,
        E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<p::J>>,
    {
        let provenance = self.provenance();
        let strategy = provenance.strategy();
        let spatial = strategy == Strategy::Propagation;
        let derivative = self
            .local_derivatives_with(
                backend.clone(),
                &[PhysicalParameter::K0, PhysicalParameter::Beta],
                s.local_assumption,
                Default::default(),
            )
            .map_err(|e| BranchFailure::Derivative(e.to_string()))?;
        let slope = derivative.implicit_slope(
            if spatial {
                ImplicitSlopeRequest::BetaPerK0
            } else {
                ImplicitSlopeRequest::K0PerBeta
            },
            s.slope,
        )?;
        let prediction = self.physical_coordinate() + slope.value * (target - control(self));
        let (k, b) = if spatial {
            (target.into(), prediction)
        } else {
            (prediction, target.into())
        };
        let old = self.signature().map_err(|_| BranchFailure::Exterior)?;
        let stack = provenance.stack();
        let pol = provenance
            .polarisation()
            .ok_or(BranchFailure::InvalidRequest)?;
        let [left, right] = old.open_exteriors().ok_or(BranchFailure::Exterior)?;
        // Continue each accepted root along the physical predictor segment.
        // The caller declares a regular neighbourhood covering these steps.
        let continue_root = |mut r: Complex, region, mat| {
            let mut channel = None;
            for part in 1..=8 {
                let t = f64::from(part) / 8.;
                let freq = old.k0 + (k - old.k0) * t;
                let beta = old.k_parallel + (b - old.k_parallel) * t;
                let tensors =
                    material::tensors(region, mat, freq).map_err(|_| BranchFailure::Exterior)?;
                let seen = Channel::of(&tensors, pol);
                let ratio = seen.kappa_squared(freq, beta) / (r * r);
                if !(ratio.re > 0. && (ratio - 1.).norm() < 0.75) {
                    return Err(BranchFailure::Exterior);
                }
                r *= ratio.sqrt();
                channel = Some(seen);
            }
            channel.map(|c| (r, c)).ok_or(BranchFailure::Exterior)
        };
        let sides = [
            continue_root(left, StackRegion::LeftExterior, stack.left_exterior())?,
            continue_root(right, StackRegion::RightExterior, stack.right_exterior())?,
        ];
        let roots = sides.map(|(r, _)| r);
        let chi = sides.map(|(_, c)| c.chi());
        // The chart's κ_χ: each root without its channel's √s.
        let unscaled = sides.map(|(r, c)| c.scale().map_or(r, |s| r / s.sqrt()));
        let fail = |e: String| BranchFailure::Domain(e);
        let (root, signature, enclosure) = if strategy == Strategy::DirectFrequency {
            let source = provenance.frequency().unwrap();
            let problem =
                p::frequency::FrequencyProblem::frequency(stack.clone(), target, pol, backend)
                    .map_err(|e| fail(e.to_string()))?;
            let domain = problem
                .frequency_domain(
                    FrequencyDomainInput {
                        bounds: FrequencyRegion::new(
                            k.re - s.radius,
                            k.re + s.radius,
                            k.im - s.radius,
                            k.im + s.radius,
                        )
                        .map_err(|e| fail(e.to_string()))?,
                        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
                            frequency: k,
                            kappa: roots,
                        }),
                        continuation: source.continuation,
                        exclusions: source.exclusions.clone(),
                        exterior_assumption: source.exterior_assumption,
                    },
                    s.domain_assumption,
                )
                .map_err(|e| fail(e.to_string()))?;
            problem.corrected_point(&domain, k, config)?
        } else {
            let exp_z = if spatial {
                (unscaled[0] + unscaled[1]) / (2. * Complex::i() * k)
            } else {
                (unscaled[0] / chi[0].sqrt() + unscaled[1] / chi[1].sqrt()) / (2. * Complex::i())
            };
            let mut z = exp_z.ln();
            z.im += std::f64::consts::TAU
                * ((self.native_coordinate().im - z.im) / std::f64::consts::TAU).round();
            let bounds = ConformalRegion::new(
                z.re - s.radius,
                z.re + s.radius,
                z.im - s.radius,
                z.im + s.radius,
            )
            .map_err(|e| fail(e.to_string()))?;
            let selection = ReferenceSelection::Coordinate {
                reference: z,
                coordinate: prediction,
            };
            if spatial {
                let problem = p::InPlaneProblem::in_plane(stack.clone(), target, pol, backend)
                    .map_err(|e| fail(e.to_string()))?;
                let domain = problem
                    .domain(bounds, selection, Default::default(), s.domain_assumption)
                    .map_err(|e| fail(e.to_string()))?;
                problem.corrected_point(&domain, z, config)?
            } else {
                let [l, r] = provenance
                    .inner
                    .problem()
                    .constants
                    .ok_or(BranchFailure::InvalidRequest)?;
                let problem = p::SpectralProblem::spectral(
                    l,
                    r,
                    stack.layers_left_to_right().iter().cloned(),
                    target,
                    pol,
                    backend,
                )
                .map_err(|e| fail(e.to_string()))?;
                let domain = problem
                    .domain(bounds, selection, Default::default(), s.domain_assumption)
                    .map_err(|e| fail(e.to_string()))?;
                problem.corrected_point(&domain, z, config)?
            }
        };
        Ok(Self {
            root,
            backend: self.backend(),
            chart: 0,
            candidate: 0,
            enclosure,
            signature: Ok(signature),
        })
    }
}
