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
    /// The corrector region does not hold exactly one root: none when the
    /// predictor has lost the branch, several when neighbouring roots lie
    /// within the chart radius (a near-degenerate group at this station).
    /// The step is refused rather than corrected onto an unknown root.
    #[error("the corrector region holds {zeros} roots, not one")]
    RootCount {
        zeros: usize,
        /// The corrector region in the station's native chart.
        bounds: Rect<f64>,
    },
    #[error("the roots in the corrector region could not be counted ({0:?})")]
    Uncounted(RootCountFailure),
}

/// Why the roots in a corrector region could not be counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootCountFailure {
    /// A root on or within round-off of the region's boundary.
    BoundaryRoot,
    /// The characteristic could not be evaluated on the boundary, or its
    /// winding was not a non-negative integer.
    NonIntegerWinding,
}

impl BranchTrace {
    /// Refused steps and the stop whose corrector region held several roots:
    /// near-degenerate neighbours of the branch, as (attempted control
    /// coordinate, root count, region in the native chart).
    pub fn near_degenerate(&self) -> Vec<(f64, usize, Rect<f64>)> {
        self.retries
            .iter()
            .map(|(at, cause)| (*at, cause))
            .chain(self.stop.iter().map(|stop| (stop.attempted, &stop.cause)))
            .filter_map(|(at, cause)| match cause {
                BranchFailure::RootCount { zeros, bounds } if *zeros > 1 => {
                    Some((at, *zeros, *bounds))
                }
                _ => None,
            })
            .collect()
    }
}

/// Require the corrector region to hold exactly one root.
fn one_root(
    counted: Result<(usize, Complex), crate::winding::WindingFailure>,
    bounds: Rect<f64>,
) -> Result<(), BranchFailure> {
    use crate::winding::WindingFailure;
    match counted {
        Ok((1, _)) => Ok(()),
        Ok((zeros, _)) => Err(BranchFailure::RootCount { zeros, bounds }),
        Err(WindingFailure::BoundaryZero) => {
            Err(BranchFailure::Uncounted(RootCountFailure::BoundaryRoot))
        }
        Err(WindingFailure::NonInteger) => Err(BranchFailure::Uncounted(
            RootCountFailure::NonIntegerWinding,
        )),
    }
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

/// The real coordinate held fixed at a station.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Vacuum wavenumber k0 fixed; β solved.
    K0,
    /// In-plane wavenumber β fixed; k0 solved.
    Beta,
}

/// The frequency-domain evidence of a station that solves k0 directly, for a
/// point found on another chart.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyStation {
    pub continuation: ContinuationConfig,
    pub exterior_assumption: ExteriorAnalyticAssumption,
    pub exclusions: Vec<FrequencyExclusion>,
}

pub(super) struct Station<'a> {
    pub control: Control,
    pub target: f64,
    pub prediction: Complex,
    pub radius: f64,
    pub frequency: Option<&'a FrequencyStation>,
    pub domain_assumption: AnalyticAssumption,
}

fn control_value(point: &ModePoint) -> f64 {
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
        let config = settings
            .refinement
            .resolve(self.root.provenance().wavenumber())
            .validate()
            .map_err(|_| BranchFailure::InvalidRequest)?;
        let start = control_value(self);
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
                        let here = control_value(last);
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
        let spatial = self.provenance().strategy() == Strategy::Propagation;
        let control = if spatial { Control::K0 } else { Control::Beta };
        let slope = self.slope(backend.clone(), control, &s)?;
        let prediction = self.physical_coordinate() + slope * (target - control_value(self));
        self.station(
            backend,
            Station {
                control,
                target,
                prediction,
                radius: s.radius,
                frequency: None,
                domain_assumption: s.domain_assumption,
            },
            config,
        )
    }

    /// The rate of the free coordinate per unit of `control` at this point:
    /// dβ/dk0 under k0 control, dk0/dβ under β control.
    pub(super) fn slope<B>(
        &self,
        backend: B,
        control: Control,
        s: &BranchSettings,
    ) -> Result<Complex, BranchFailure>
    where
        B: ModalDerivativeBackend + Clone,
    {
        let derivative = self
            .local_derivatives_with(
                backend,
                &[PhysicalParameter::K0, PhysicalParameter::Beta],
                s.local_assumption,
                Default::default(),
            )
            .map_err(|e| BranchFailure::Derivative(e.to_string()))?;
        Ok(derivative
            .implicit_slope(
                match control {
                    Control::K0 => ImplicitSlopeRequest::BetaPerK0,
                    Control::Beta => ImplicitSlopeRequest::K0PerBeta,
                },
                s.slope,
            )?
            .value)
    }

    /// Correct `prediction` of the free coordinate at the control value
    /// `target`. k0 control solves β on the in-plane chart; β control solves
    /// k0 on the spectral chart of a stack with constant exteriors seeded on
    /// it, else directly in k0, with the frequency evidence of the source
    /// point or `frequency`.
    pub(super) fn station<B, E>(
        &self,
        backend: B,
        station: Station<'_>,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<Self, BranchFailure>
    where
        B: Backend<p::J, ComplexPlane, Entries = E> + ModalDerivativeBackend + Clone,
        E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<p::J>>,
    {
        let Station {
            control,
            target,
            prediction,
            radius,
            frequency,
            domain_assumption,
        } = station;
        let provenance = self.provenance();
        let strategy = provenance.strategy();
        let spatial = control == Control::K0;
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
        let (root, signature, enclosure) = if !spatial && strategy != Strategy::ConformalSpectral {
            let source = provenance
                .frequency()
                .map(|f| FrequencyStation {
                    continuation: f.continuation,
                    exterior_assumption: f.exterior_assumption,
                    exclusions: f.exclusions.clone(),
                })
                .or(frequency.cloned())
                .ok_or(BranchFailure::InvalidRequest)?;
            let problem =
                p::frequency::FrequencyProblem::frequency(stack.clone(), target, pol, backend)
                    .map_err(|e| fail(e.to_string()))?;
            let domain = problem
                .frequency_domain(
                    FrequencyDomainInput {
                        bounds: FrequencyRegion::new(
                            k.re - radius,
                            k.re + radius,
                            k.im - radius,
                            k.im + radius,
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
                    domain_assumption,
                )
                .map_err(|e| fail(e.to_string()))?;
            let bounds = Rect {
                x_min: k.re - radius,
                x_max: k.re + radius,
                y_min: k.im - radius,
                y_max: k.im + radius,
            };
            one_root(problem.zeros_in(&domain), bounds)?;
            problem.corrected_point(&domain, k, config)?
        } else {
            let exp_z = if spatial {
                (unscaled[0] + unscaled[1]) / (2. * Complex::i() * k)
            } else {
                (unscaled[0] / chi[0].sqrt() + unscaled[1] / chi[1].sqrt()) / (2. * Complex::i())
            };
            let mut z = exp_z.ln();
            if strategy == Strategy::Propagation {
                // Stay on the source chart's sheet.
                z.im += std::f64::consts::TAU
                    * ((self.native_coordinate().im - z.im) / std::f64::consts::TAU).round();
            }
            let region = Rect {
                x_min: z.re - radius,
                x_max: z.re + radius,
                y_min: z.im - radius,
                y_max: z.im + radius,
            };
            let bounds =
                ConformalRegion::new(z.re - radius, z.re + radius, z.im - radius, z.im + radius)
                    .map_err(|e| fail(e.to_string()))?;
            let selection = ReferenceSelection::Coordinate {
                reference: z,
                coordinate: prediction,
            };
            if spatial {
                let problem = p::InPlaneProblem::in_plane(stack.clone(), target, pol, backend)
                    .map_err(|e| fail(e.to_string()))?;
                let domain = problem
                    .domain(bounds, selection, Default::default(), domain_assumption)
                    .map_err(|e| fail(e.to_string()))?;
                one_root(problem.zeros_in(&domain), region)?;
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
                    .domain(bounds, selection, Default::default(), domain_assumption)
                    .map_err(|e| fail(e.to_string()))?;
                one_root(problem.zeros_in(&domain), region)?;
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
