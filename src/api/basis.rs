//! Complete modal bases in a window of the complex β plane.
//!
//! A stack closed on both sides, by walls or wall-backed PMLs, is a regular
//! problem on a bounded interval: its spectrum is discrete. Its
//! characteristic is entire and even in β at fixed real k0, so the argument
//! principle over the boundary of a β rectangle counts the modes inside it,
//! guided, radiation-like, evanescent and complex alike. With a PML these are
//! the modes converging to the open stack's guided and exposed leaky modes as
//! the PML thickens, and the modes near the rotated continuum
//! arg κ ≈ −arg(g + τ), g the gap and τ the PML's complex thickness; all are
//! counted. A PML normalises the characteristic by exp(iκ_ref τ), leaving it
//! of order exp(Σ [−Im(κ_ref τ) + |Im(κ τ)|]); the search's near-zero
//! threshold is scaled by
//! the least of that, exp(Σ −Im(κ_ref τ)), and the residual limit of root
//! correction by its value at the seed, so that both are relative to the
//! characteristic's scale as between walls. The count is formed before, and
//! independently of, the search; each counted cell is refined on its own, and
//! every cell that does not end as accepted roots matching its count is
//! reported with its cause.
//!
//! Modes are normalised to unit unconjugated self-pairing
//! ⟨f, f⟩ = ¼ ∫ (E × H + E × H) · x̂ dz = 1 W/m from fields recovered at a
//! canonical unit seed, signed by a reference at the first interface, ordered
//! by decreasing Re β², then Im β², then Re β, and near-degenerate modes are
//! replaced by the unconjugated Löwdin basis of their span.
use super::integration::{
    Layout, Sample, converge, equivalent, field_scale, integral, samples, split_pmls, unconjugated,
};
use super::*;
use crate::problem as p;
use crate::search::ModalSearchResult;
use argloc::ArgumentConfig;
use lamina_core::{
    AnalyticalMaterialStack, FieldSampling, ModalFieldBackend, Polarisation, Scatter2, StackSide,
};
use lamina_units::Length;
use nalgebra::DMatrix;
use std::cmp::Ordering;
use std::f64::consts::TAU;

/// A window of the complex β plane.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModalWindow {
    /// A β rectangle in cm^-1 on the direct chart, for stacks closed on both
    /// sides by walls or PMLs: real parts `re_cm`, imaginary parts `im_cm`, each ascending.
    Wavenumber { re_cm: [f64; 2], im_cm: [f64; 2] },
}

impl ModalWindow {
    /// Re β ∈ [−margin, beta_max], Im β ∈ [−margin, decay_max]: it straddles
    /// both axes, so lossless real and imaginary roots lie strictly inside,
    /// and meets its own mirror only in the square |Re β|, |Im β| ≤ margin.
    pub fn forward(beta_max_cm: f64, decay_max_cm: f64, margin_cm: f64) -> Self {
        Self::Wavenumber {
            re_cm: [-margin_cm, beta_max_cm],
            im_cm: [-margin_cm, decay_max_cm],
        }
    }

    fn rectangle(&self) -> Option<Rect<f64>> {
        let Self::Wavenumber { re_cm, im_cm } = *self;
        let finite = re_cm.iter().chain(&im_cm).all(|v| v.is_finite());
        (finite && re_cm[0] < re_cm[1] && im_cm[0] < im_cm[1]).then_some(Rect {
            x_min: re_cm[0],
            x_max: re_cm[1],
            y_min: im_cm[0],
            y_max: im_cm[1],
        })
    }
}

/// The field family a basis spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModeFamily {
    /// Modes of one polarisation of a separable stack.
    Polarised(Polarisation),
}

#[derive(Clone, Copy, Debug)]
pub struct ModalBasisRequest {
    pub wavelength: Length<f64>,
    pub family: ModeFamily,
    pub window: ModalWindow,
    /// Largest |Δβ| in cm^-1 joining two modes into one near-degenerate
    /// group; `None` for 1e-6 k0.
    pub degeneracy_gap_cm: Option<f64>,
    /// A mode whose |⟨u, u⟩| is below this fraction of its bounded-region
    /// field scale is kept as ill-conditioned, unnormalised.
    pub self_pairing_rtol: f64,
    pub discovery: DiscoveryConfig,
    pub refinement: RefinementConfig,
    /// Field recovery limits. The amplitude is not used: fields are always
    /// recovered at the canonical unit seed.
    pub recovery: FieldRecoverySettings,
    /// Cell splits before a cell that holds several roots, or cannot be
    /// searched whole, stays unresolved.
    pub max_subdivisions: usize,
    pub assumption: AnalyticAssumption,
}

impl ModalBasisRequest {
    /// A request with the default gap, threshold and numerical controls.
    /// Absolute residual and correction limits are those of the defaults at
    /// a 1 cm wavelength, scaled with k0: the characteristic is linear in
    /// wavenumber, so the limits mean the same at every length scale.
    pub fn new(wavelength: Length<f64>, family: ModeFamily, window: ModalWindow) -> Self {
        let scale = TAU / wavelength.as_centimetres() / TAU;
        let refinement = RefinementConfig::default();
        let recovery = FieldRecoverySettings::default();
        Self {
            wavelength,
            family,
            window,
            degeneracy_gap_cm: None,
            self_pairing_rtol: 1e-8,
            discovery: DiscoveryConfig::default(),
            refinement: RefinementConfig {
                residual_atol: refinement.residual_atol * scale,
                native_atol: refinement.native_atol * scale,
                ..refinement
            },
            recovery: FieldRecoverySettings {
                characteristic_limit: recovery.characteristic_limit * scale,
                ..recovery
            },
            max_subdivisions: 1000,
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        }
    }

    pub fn k0_cm(&self) -> f64 {
        TAU / self.wavelength.as_centimetres()
    }
}

/// Which modes a basis claims to hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BasisScope {
    /// Every mode in the window of a discrete spectrum.
    Discrete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisStatus {
    /// Expected equals found, nothing is unresolved and every mode is normalised.
    Complete,
    Incomplete,
    /// The winding over the window boundary is not integer-close.
    CountFailed,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum BasisLimit {
    /// The window boundary passes within clearance of a zero.
    BoundaryRoot,
    /// A winding that is not integer-close or cannot be evaluated.
    NonIntegerCount,
    /// The subdivision or evaluation budget ran out.
    Budget,
    /// A counted cell's root failed correction or acceptance.
    Refinement(BranchFailure),
    /// Roots that stay together at the resolution limit, or a near-degenerate
    /// group whose Gram is singular: a possible exceptional point.
    Multiplicity { count: usize },
    /// A root where the window meets its own mirror image around β = 0.
    Cutoff,
    /// A cell on which the PML normalisation scale of the characteristic
    /// reaches exp(`log_magnitude`), outside the representable range.
    Scale { side: StackSide, log_magnitude: i64 },
}

#[derive(Debug)]
pub struct BasisGap {
    /// Bounds in the chart's native coordinate.
    pub bounds: Rect<f64>,
    /// The bounds' image in β, cm^-1.
    pub beta_image: Rect<f64>,
    /// Zeros counted inside, where a count exists.
    pub count: Option<usize>,
    pub cause: BasisLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalisation {
    /// ⟨f, f⟩ = 1 W/m; `ratio` is |⟨u, u⟩| over the field scale.
    Unit { ratio: f64 },
    /// |⟨u, u⟩| below the threshold of the field scale: no normalised fields.
    IllConditioned { ratio: f64 },
}

/// The sign reference at the first interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignReference {
    /// E_y + Z0 H_y.
    FirstInterface,
    /// E_y + Z0 H_y + i (Z0 H_x − E_x), when a wall lies on the first interface.
    WallOnFirstInterface,
}

/// A sign margin below this is flagged as ambiguous.
pub const SIGN_AMBIGUITY: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct BasisMode {
    pub point: ModePoint,
    /// ⟨u, u⟩ of the fields at the canonical unit seed, W/m.
    pub self_pairing_raw_w_per_m: Complex,
    pub normalisation: Normalisation,
    pub sign_reference: SignReference,
    /// |Re ℓ| / max |E_y + Z0 H_y| over the bounded samples; `None` when
    /// ill-conditioned.
    pub sign_margin: Option<f64>,
    /// Seed-field coefficients of the normalised mode: its own, or those of
    /// its near-degenerate group's members.
    terms: Vec<(ModePoint, Complex)>,
}

#[derive(Debug, thiserror::Error)]
pub enum BasisFieldError {
    #[error("mode is ill-conditioned and has no normalised fields")]
    IllConditioned,
    #[error(transparent)]
    Recovery(#[from] Box<FieldRecoveryFailure>),
}

impl BasisMode {
    pub fn sign_ambiguous(&self) -> bool {
        self.sign_margin.is_some_and(|m| m < SIGN_AMBIGUITY)
    }

    /// Normalised fields [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz] at `sampling`.
    /// Recovery uses the limits of `settings` at the canonical unit seed.
    pub fn fields_with<B: ModalFieldBackend + Clone>(
        &self,
        backend: B,
        sampling: &FieldSampling<f64>,
        settings: FieldRecoverySettings,
    ) -> Result<Vec<Sample>, BasisFieldError> {
        if self.terms.is_empty() {
            return Err(BasisFieldError::IllConditioned);
        }
        let seeds = self
            .terms
            .iter()
            .map(|(point, _)| seed_samples(point, backend.clone(), sampling, settings))
            .collect::<Result<Vec<_>, _>>()?;
        let weights: Vec<_> = self.terms.iter().map(|(_, c)| *c).collect();
        Ok(combine(&seeds, &weights))
    }

    pub(super) fn is_normalised(&self) -> bool {
        !self.terms.is_empty()
    }
}

fn seed_samples<B: ModalFieldBackend>(
    point: &ModePoint,
    backend: B,
    sampling: &FieldSampling<f64>,
    settings: FieldRecoverySettings,
) -> Result<Vec<Sample>, Box<FieldRecoveryFailure>> {
    let unit = FieldRecoverySettings {
        amplitude: Complex::ONE,
        ..settings
    };
    Ok(samples(
        &point.recover_fields_with(backend, sampling, unit)?.fields,
    ))
}

fn combine(seeds: &[Vec<Sample>], weights: &[Complex]) -> Vec<Sample> {
    (0..seeds[0].len())
        .map(|i| {
            std::array::from_fn(|c| {
                seeds
                    .iter()
                    .zip(weights)
                    .fold(Complex::ZERO, |sum, (seed, w)| sum + w * seed[i][c])
            })
        })
        .collect()
}

#[derive(Debug)]
pub struct ModalBasis {
    pub request: ModalBasisRequest,
    pub stack: AnalyticalMaterialStack<Complex>,
    pub chart: CoordinateKind,
    /// The window in the chart's native coordinate.
    pub window: Rect<f64>,
    /// The window's image in β, cm^-1.
    pub window_beta_image: Rect<f64>,
    /// The resolved near-degeneracy gap, cm^-1.
    pub degeneracy_gap_cm: f64,
    pub scope: BasisScope,
    pub status: BasisStatus,
    /// Zeros in the window from the boundary winding; `None` when it fails.
    pub expected: Option<usize>,
    /// In key order: decreasing Re β², then Im β², then Re β.
    pub modes: Vec<BasisMode>,
    pub unresolved: Vec<BasisGap>,
    /// Near-degenerate groups of two or more, as mode indices in key order.
    pub groups: Vec<Vec<usize>>,
}

impl ModalBasis {
    pub const ORDERING: &'static str =
        "decreasing Re β², then decreasing Im β², then decreasing Re β";

    /// Distinct accepted modes.
    pub fn found(&self) -> usize {
        self.modes.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModalBasisError {
    #[error("invalid window, wavelength or numerical controls")]
    InvalidRequest,
    #[error("a wavenumber window needs a stack closed by walls or PMLs on both sides")]
    OpenSide,
    #[error(
        "the {side:?} PML normalisation scale reaches exp({log_magnitude}) on the window, \
         outside the representable range"
    )]
    Scale { side: StackSide, log_magnitude: i64 },
    #[error(transparent)]
    Problem(#[from] ConstructionError),
    #[error("field recovery failed for mode {mode}")]
    Fields {
        mode: usize,
        #[source]
        source: Box<FieldRecoveryFailure>,
    },
    #[error("self-pairing quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
}

/// Every mode of a stack closed by walls or PMLs on both sides in a β window,
/// with counts and evidence. Searches use Scatter2; fields use `backend`.
pub fn modal_basis_in_window_with<B: ModalFieldBackend + Clone>(
    backend: &B,
    stack: AnalyticalMaterialStack<Complex>,
    request: ModalBasisRequest,
) -> Result<ModalBasis, ModalBasisError> {
    solve(backend, stack, request, false)
}

pub(super) fn solve<B: ModalFieldBackend + Clone>(
    backend: &B,
    stack: AnalyticalMaterialStack<Complex>,
    request: ModalBasisRequest,
    reverse: bool,
) -> Result<ModalBasis, ModalBasisError> {
    use ModalBasisError::*;
    let k0 = request.k0_cm();
    let window = request.window.rectangle().ok_or(InvalidRequest)?;
    let gap = request.degeneracy_gap_cm.unwrap_or(1e-6 * k0);
    let ModeFamily::Polarised(polarisation) = request.family;
    if !k0.is_finite()
        || k0 <= 0.0
        || !gap.is_finite()
        || gap < 0.0
        || !(request.self_pairing_rtol > 0.0 && request.self_pairing_rtol < 1.0)
    {
        return Err(InvalidRequest);
    }
    crate::refinement::RefinementConfig::from(request.refinement)
        .validate()
        .map_err(|_| InvalidRequest)?;
    if p::open_sides(&stack).contains(&true) {
        return Err(OpenSide);
    }
    let problem = p::InPlaneProblem::in_plane(stack.clone(), k0, polarisation, Scatter2::new())?;
    let bounds = ConformalRegion::new(window.x_min, window.x_max, window.y_min, window.y_max)
        .map_err(|_| InvalidRequest)?;
    if let Err(p::direct::DirectChartError::PmlScale {
        side,
        log_magnitude,
    }) = problem.direct_domain(bounds, request.assumption)
    {
        return Err(Scale {
            side,
            log_magnitude,
        });
    }
    // The near-zero threshold relative to the least scale of the
    // characteristic, which the PML normalisation sets.
    let least_scale = p::direct::pml_log_floor(&stack, k0).min(0.0).exp();
    let argument = DiscoveryConfig {
        zero_tolerance: request.discovery.zero_tolerance * least_scale,
        ..request.discovery
    }
    .argument_config()
    .map_err(|_| InvalidRequest)?;
    let search = Search {
        problem: &problem,
        assumption: request.assumption,
        argument,
        refinement: request.refinement,
        stack: &stack,
        k0,
        mirror: mirror_square(window),
        resolution: 1e-9 * k0,
        reverse,
    };
    let (counted, mut unresolved, interior) = search.count(window);
    let expected = counted.map(|(n, _)| n);
    let (points, more) = search.roots(interior, counted, request.max_subdivisions);
    unresolved.extend(more);
    let (modes, groups, singular) = normalise(backend, &stack, points, gap, &request)?;
    unresolved.extend(singular);
    unresolved.sort_by(|a, b| {
        let key = |r: &BasisGap| [r.bounds.x_min, r.bounds.y_min, r.bounds.x_max];
        key(a)
            .iter()
            .zip(key(b).iter())
            .map(|(x, y)| x.total_cmp(y))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    });
    let status = if expected.is_none() {
        BasisStatus::CountFailed
    } else if expected == Some(modes.len())
        && unresolved.is_empty()
        && modes.iter().all(BasisMode::is_normalised)
    {
        BasisStatus::Complete
    } else {
        BasisStatus::Incomplete
    };
    Ok(ModalBasis {
        request,
        stack,
        chart: CoordinateKind::Wavenumber,
        window,
        window_beta_image: window,
        degeneracy_gap_cm: gap,
        scope: BasisScope::Discrete,
        status,
        expected,
        modes,
        unresolved,
        groups,
    })
}

/// Where the window meets its mirror image −W, if anywhere.
fn mirror_square(w: Rect<f64>) -> Option<Rect<f64>> {
    let r = Rect {
        x_min: w.x_min.max(-w.x_max),
        x_max: w.x_max.min(-w.x_min),
        y_min: w.y_min.max(-w.y_max),
        y_max: w.y_max.min(-w.y_min),
    };
    (r.x_min <= r.x_max && r.y_min <= r.y_max).then_some(r)
}

fn contains(r: Rect<f64>, z: Complex) -> bool {
    r.x_min <= z.re && z.re <= r.x_max && r.y_min <= z.im && z.im <= r.y_max
}

fn region(bounds: Rect<f64>, count: Option<usize>, cause: BasisLimit) -> BasisGap {
    BasisGap {
        bounds,
        beta_image: bounds,
        count,
        cause,
    }
}

type Problem = p::InPlaneProblem<Scatter2>;

struct Search<'a> {
    problem: &'a Problem,
    assumption: AnalyticAssumption,
    argument: ArgumentConfig<f64>,
    refinement: RefinementConfig,
    stack: &'a AnalyticalMaterialStack<Complex>,
    k0: f64,
    mirror: Option<Rect<f64>>,
    /// Cells no larger than this (cm^-1) are not split further.
    resolution: f64,
    reverse: bool,
}

impl Search<'_> {
    /// The direct domain of `cell`; a cell whose PML scale leaves the
    /// representable range is `Scale`, any other failure `NonIntegerCount`.
    fn domain(
        &self,
        cell: Rect<f64>,
    ) -> Result<p::direct::DirectDomain<crate::problem::Compiled<Scatter2>>, BasisLimit> {
        let bounds = ConformalRegion::new(cell.x_min, cell.x_max, cell.y_min, cell.y_max)
            .map_err(|_| BasisLimit::NonIntegerCount)?;
        self.problem
            .direct_domain(bounds, self.assumption)
            .map_err(|e| match e {
                p::direct::DirectChartError::PmlScale {
                    side,
                    log_magnitude,
                } => BasisLimit::Scale {
                    side,
                    log_magnitude,
                },
                _ => BasisLimit::NonIntegerCount,
            })
    }

    fn search(
        &self,
        cell: Rect<f64>,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>> {
        let result = self.problem.search(&self.domain(cell).ok()?, config).ok()?;
        Some(result.into_parts().0)
    }

    /// Zeros inside `window` from the change of arg D around its boundary,
    /// tracked in steps over which the phase turns by less than π/4 and
    /// agrees with the integrated logarithmic derivative, and the first
    /// moment ∮ β d(log D) / 2πi, the sum of the zeros. On failure, the
    /// point where tracking stopped and why: a zero within `clearance` of the
    /// boundary, or a characteristic that could not be evaluated.
    fn winding(
        &self,
        window: Rect<f64>,
        clearance: f64,
    ) -> Result<(usize, Complex), (Complex, BasisLimit)> {
        let corner = |x: f64, y: f64| Complex::new(x, y);
        let (x, y) = ([window.x_min, window.x_max], [window.y_min, window.y_max]);
        let path = [
            corner(x[0], y[0]),
            corner(x[1], y[0]),
            corner(x[1], y[1]),
            corner(x[0], y[1]),
            corner(x[0], y[0]),
        ];
        let failed = |z| (z, BasisLimit::NonIntegerCount);
        let domain = self.domain(window).map_err(|cause| (path[0], cause))?;
        let evaluate = |z: Complex| {
            let bundle = self.problem.evaluate(&domain, z).map_err(|_| failed(z))?;
            let (value, derivative) = (*bundle.value(), *bundle.derivative());
            if value == Complex::ZERO {
                return Err((z, BasisLimit::BoundaryRoot));
            }
            let log = derivative / value;
            if !value.re.is_finite()
                || !value.im.is_finite()
                || !log.re.is_finite()
                || !log.im.is_finite()
            {
                return Err(failed(z));
            }
            Ok((value, log))
        };
        let mut turns = 0.0;
        let mut moment = Complex::ZERO;
        for edge in path.windows(2) {
            let (a, b) = (edge[0], edge[1]);
            let length = (b - a).norm();
            let (mut t, mut step) = (0.0_f64, 1.0_f64 / 16.0);
            let mut here = evaluate(a)?;
            while t < 1.0 {
                step = step.min(1.0 - t);
                let (z0, z1) = (a + (b - a) * t, a + (b - a) * (t + step));
                let there = evaluate(z1)?;
                let turn = (there.0 / here.0).arg();
                let predicted = ((here.1 + there.1) * 0.5 * (z1 - z0)).im;
                if turn.abs() > std::f64::consts::FRAC_PI_4 || (turn - predicted).abs() > 0.05 {
                    if step * length <= clearance {
                        return Err((z1, BasisLimit::BoundaryRoot));
                    }
                    step *= 0.5;
                    continue;
                }
                turns += turn;
                moment += (z0 + z1) * 0.5 * Complex::new((there.0 / here.0).norm().ln(), turn);
                t += step;
                here = there;
                step *= 2.0;
            }
        }
        let winding = turns / std::f64::consts::TAU;
        let count = winding.round();
        if (winding - count).abs() > 1e-6 || count < 0.0 {
            return Err(failed(path[0]));
        }
        Ok((
            count as usize,
            moment / Complex::new(0.0, std::f64::consts::TAU),
        ))
    }

    /// The expected count over the window boundary, the region where it
    /// failed, and the cell to search: the window, or when the count failed
    /// the window less a margin, so that a zero on the boundary is left to
    /// the unresolved region rather than invented or dropped.
    #[allow(clippy::type_complexity)]
    fn count(&self, window: Rect<f64>) -> (Option<(usize, Complex)>, Vec<BasisGap>, Rect<f64>) {
        let scale = (window.x_max - window.x_min).max(window.y_max - window.y_min);
        let (clearance, margin) = (1e-6 * scale, 1e-4 * scale);
        match self.winding(window, clearance) {
            Ok(counted) => (Some(counted), vec![], window),
            Err((z, cause)) => {
                let bounds = match cause {
                    BasisLimit::BoundaryRoot => Rect {
                        x_min: z.re - margin,
                        x_max: z.re + margin,
                        y_min: z.im - margin,
                        y_max: z.im + margin,
                    },
                    _ => window,
                };
                let interior = Rect {
                    x_min: window.x_min + margin,
                    x_max: window.x_max - margin,
                    y_min: window.y_min + margin,
                    y_max: window.y_max - margin,
                };
                (None, vec![region(bounds, None, cause)], interior)
            }
        }
    }

    /// Correct the one root counted in `cell` from its estimate. Newton steps
    /// may leave the cell by up to its size; the accepted root must lie in it.
    fn correct(&self, cell: Rect<f64>, estimate: Complex) -> Result<ModePoint, BasisLimit> {
        let reach = (cell.x_max - cell.x_min).max(cell.y_max - cell.y_min);
        let around = Rect {
            x_min: cell.x_min - reach,
            x_max: cell.x_max + reach,
            y_min: cell.y_min - reach,
            y_max: cell.y_max + reach,
        };
        let domain = self.domain(around)?;
        let seed = Complex::new(
            estimate.re.clamp(cell.x_min, cell.x_max),
            estimate.im.clamp(cell.y_min, cell.y_max),
        );
        // The residual limit relative to the PML scale of the characteristic
        // at the seed.
        let scale = p::direct::pml_log_scale(self.stack, self.k0, seed).exp();
        let refinement = crate::refinement::RefinementConfig::from(RefinementConfig {
            residual_atol: self.refinement.residual_atol * scale,
            ..self.refinement
        })
        .validate()
        .map_err(|_| {
            BasisLimit::Refinement(BranchFailure::Correction(
                "residual limit out of range at the PML scale".into(),
            ))
        })?;
        let (mut root, mut signature, _) = self
            .problem
            .corrected_point(&domain, seed, refinement)
            .map_err(BasisLimit::Refinement)?;
        // The accepted point's own Newton step is within the correction
        // limit; accepting again from the stepped point keeps the root to the
        // square of that step, which fields far from the seed side need.
        let evidence = root.evidence();
        let stepped = root.coordinate() - evidence.value / evidence.derivative;
        if contains(cell, stepped)
            && let Ok((polished, polished_signature, _)) =
                self.problem.corrected_point(&domain, stepped, refinement)
            && contains(cell, polished.coordinate())
        {
            (root, signature) = (polished, polished_signature);
        }
        if !contains(cell, root.coordinate()) {
            return Err(BasisLimit::Refinement(BranchFailure::Correction(
                "converged outside its counted cell".into(),
            )));
        }
        let enclosure = cell;
        Ok(ModePoint {
            root,
            signature: Ok(signature),
            enclosure,
            chart: 0,
            candidate: 0,
            backend: std::any::type_name::<Scatter2>(),
        })
    }

    /// Accepted roots in `interior`, and the regions that did not resolve.
    /// argloc counts and localises; a cell it cannot search, or a cluster it
    /// leaves, is split into quadrants counted by [`Self::winding`], and a
    /// cell holding one root is corrected from its estimate.
    fn roots(
        &self,
        interior: Rect<f64>,
        counted: Option<(usize, Complex)>,
        mut budget: usize,
    ) -> (Vec<ModePoint>, Vec<BasisGap>) {
        let mut roots = vec![];
        let mut unresolved = vec![];
        let scale = (interior.x_max - interior.x_min).max(interior.y_max - interior.y_min);
        let (count, sum) = match counted.map_or_else(|| self.winding(interior, 1e-6 * scale), Ok) {
            Ok(counted) => counted,
            Err((_, cause)) => {
                unresolved.push(region(interior, None, cause));
                return (roots, unresolved);
            }
        };
        let at_cutoff = |z: Complex| self.mirror.is_some_and(|m| contains(m, z));
        // (cell, count, sum of its zeros, whether that sum came from argloc
        // or a cluster's quadrants rather than a cell argloc has not seen)
        let mut cells = vec![(interior, count, sum, false)];
        while let Some((cell, count, sum, localised)) = if self.reverse {
            cells.pop()
        } else {
            (!cells.is_empty()).then(|| cells.remove(0))
        } {
            let centre = sum / count.max(1) as f64;
            if count == 0 {
                continue;
            }
            if at_cutoff(centre) {
                unresolved.push(region(cell, Some(count), BasisLimit::Cutoff));
                continue;
            }
            let search = (!localised)
                .then(|| self.search(cell, self.argument.clone()))
                .flatten();
            if count == 1 && search.is_none() {
                match self.correct(cell, sum) {
                    Ok(point) if at_cutoff(point.physical_coordinate()) => {
                        unresolved.push(region(cell, Some(1), BasisLimit::Cutoff))
                    }
                    Ok(point) => roots.push(point),
                    Err(cause) => unresolved.push(region(cell, Some(1), cause)),
                }
                continue;
            }
            if let Some(result) = search {
                let mut open = vec![];
                for leaf in result.leaves() {
                    let score = leaf.data.score;
                    if score.is_finite() && score <= self.argument.quad_tree.target_score {
                        continue;
                    }
                    let count = usize::try_from(leaf.data.root_count)
                        .ok()
                        .filter(|&n| n > 0);
                    let cause = if leaf.data.residual > self.argument.residual_tolerance {
                        BasisLimit::NonIntegerCount
                    } else {
                        BasisLimit::Budget
                    };
                    open.push(leaf.bounds);
                    unresolved.push(region(leaf.bounds, count, cause));
                }
                for estimate in result.estimates().iter().map(|e| *e.estimate()) {
                    let (bounds, n) = (estimate.enclosure, estimate.multiplicity);
                    if !open.contains(&bounds) {
                        cells.push((bounds, n, estimate.location * n as f64, true));
                    }
                }
                continue;
            }
            let size = (cell.x_max - cell.x_min).max(cell.y_max - cell.y_min);
            let cause = if size <= self.resolution {
                BasisLimit::Multiplicity { count }
            } else if budget == 0 {
                BasisLimit::Budget
            } else {
                budget -= 1;
                match self.quadrants(cell, count) {
                    Ok(quadrants) => {
                        cells.extend(
                            quadrants
                                .into_iter()
                                .map(|(q, n, sum)| (q, n, sum, localised)),
                        );
                        continue;
                    }
                    Err(cause) => cause,
                }
            };
            unresolved.push(region(cell, Some(count), cause));
        }
        roots.sort_by(|a, b| key_order(a.physical_coordinate(), b.physical_coordinate()));
        let mut distinct: Vec<ModePoint> = vec![];
        for point in roots {
            let beta = point.physical_coordinate();
            match distinct.last() {
                Some(last) if (last.physical_coordinate() - beta).norm() <= self.resolution => {
                    unresolved.push(region(
                        point.enclosure,
                        Some(1),
                        BasisLimit::Refinement(BranchFailure::Correction(
                            "converged to a root of another cell".into(),
                        )),
                    ))
                }
                _ => distinct.push(point),
            }
        }
        (distinct, unresolved)
    }

    /// The four quadrants of a cell holding `count` roots with their counts
    /// and the sums of their zeros, at a split whose quadrant boundaries all
    /// count, to `count` in total; otherwise `Scale` if a quadrant's PML scale
    /// failed, else `NonIntegerCount`.
    #[allow(clippy::type_complexity)]
    fn quadrants(
        &self,
        cell: Rect<f64>,
        count: usize,
    ) -> Result<Vec<(Rect<f64>, usize, Complex)>, BasisLimit> {
        let clearance = 1e-6 * (cell.x_max - cell.x_min).max(cell.y_max - cell.y_min);
        let mut cause = BasisLimit::NonIntegerCount;
        let split = [0.4637, 0.5381, 0.4129].into_iter().find_map(|f| {
            let x = cell.x_min + f * (cell.x_max - cell.x_min);
            let y = cell.y_min + f * (cell.y_max - cell.y_min);
            let quadrants = [
                [cell.x_min, x, cell.y_min, y],
                [x, cell.x_max, cell.y_min, y],
                [cell.x_min, x, y, cell.y_max],
                [x, cell.x_max, y, cell.y_max],
            ]
            .map(|[x_min, x_max, y_min, y_max]| Rect {
                x_min,
                x_max,
                y_min,
                y_max,
            });
            let counted = quadrants
                .iter()
                .map(|&q| match self.winding(q, clearance) {
                    Ok((n, sum)) => Some((q, n, sum)),
                    Err((_, scale @ BasisLimit::Scale { .. })) => {
                        cause = scale;
                        None
                    }
                    Err(_) => None,
                })
                .collect::<Option<Vec<_>>>()?;
            (counted.iter().map(|(_, n, _)| n).sum::<usize>() == count).then_some(counted)
        });
        split.ok_or(cause)
    }
}

/// Decreasing Re β², then decreasing Im β², then decreasing Re β.
fn key_order(a: Complex, b: Complex) -> Ordering {
    let (a2, b2) = (a * a, b * b);
    b2.re
        .total_cmp(&a2.re)
        .then(b2.im.total_cmp(&a2.im))
        .then(b.re.total_cmp(&a.re))
}

/// The sign reference ℓ of fields at the first interface.
fn reference(sample: &Sample, kind: SignReference) -> Complex {
    let primary = sample[1] + sample[4];
    match kind {
        SignReference::FirstInterface => primary,
        SignReference::WallOnFirstInterface => primary + Complex::I * (sample[3] - sample[0]),
    }
}

/// σ ∈ {±1} making Re ℓ positive, or Im ℓ when Re ℓ is zero to round-off.
fn sign(l: Complex) -> f64 {
    let decisive = if l.re.abs() <= 64.0 * f64::EPSILON * l.norm() {
        l.im
    } else {
        l.re
    };
    if decisive < 0.0 { -1.0 } else { 1.0 }
}

/// Normalise, sign and group accepted points already in key order.
#[allow(clippy::type_complexity)]
fn normalise<B: ModalFieldBackend + Clone>(
    backend: &B,
    stack: &AnalyticalMaterialStack<Complex>,
    points: Vec<ModePoint>,
    gap: f64,
    request: &ModalBasisRequest,
) -> Result<(Vec<BasisMode>, Vec<Vec<usize>>, Vec<BasisGap>), ModalBasisError> {
    use ModalBasisError::*;
    let n = points.len();
    let layout = Layout::new(stack, 0.0).ok_or(InvalidRequest)?;
    let beta: Vec<_> = points.iter().map(ModePoint::physical_coordinate).collect();
    let partition = split_pmls(
        Layout::partition(&[&layout]),
        &[(&layout, stack)],
        request.k0_cm(),
        &beta,
    );
    let kind = if layout.walls[0].is_some_and(|(_, z)| z == 0.0) {
        SignReference::WallOnFirstInterface
    } else {
        SignReference::FirstInterface
    };
    let first = partition
        .iter()
        .position(|&z| z == 0.0)
        .unwrap_or(0)
        .min(partition.len().saturating_sub(2));
    // Near-degenerate groups: connected components of |Δβ| ≤ gap.
    let mut component: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in 0..i {
            if (beta[i] - beta[j]).norm() <= gap {
                let (from, to) = (
                    component[i].max(component[j]),
                    component[i].min(component[j]),
                );
                component
                    .iter_mut()
                    .filter(|c| **c == from)
                    .for_each(|c| *c = to);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    for i in 0..n {
        if component[i] == i {
            let members: Vec<_> = (i..n).filter(|&j| component[j] == i).collect();
            if members.len() > 1 {
                groups.push(members);
            }
        }
    }
    let pairs: Vec<(usize, usize)> = (0..n)
        .map(|i| (i, i))
        .chain(groups.iter().flat_map(|g| {
            g.iter()
                .enumerate()
                .flat_map(move |(a, &i)| g[a + 1..].iter().map(move |&j| (i, j)))
        }))
        .collect();
    // Seeds at each level, scaled by their first level to about unit
    // self-pairing, or to the threshold's share of the field scale when the
    // self-pairing is smaller, so that the convergence test is relative for
    // every mode's self-pairing as for the pairings of normalised modes.
    let mut unit: Vec<f64> = vec![];
    let converged = converge(
        &partition,
        || Quadrature,
        |panels, _| {
            let sampling = layout.sampling(&partition, panels, false);
            let stretches = layout.stretches(&partition, panels);
            let mut seeds = points
                .iter()
                .enumerate()
                .map(|(mode, point)| {
                    let mut seed =
                        seed_samples(point, backend.clone(), &sampling, request.recovery)
                            .map_err(|source| Fields { mode, source })?;
                    equivalent(&mut seed, &stretches);
                    Ok::<_, ModalBasisError>(seed)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if unit.is_empty() {
                unit = seeds
                    .iter()
                    .map(|u| {
                        let scale = integral(&partition, panels, |s| field_scale(&u[s])).re;
                        let pairing = integral(&partition, panels, |s| unconjugated(&u[s], &u[s]));
                        pairing
                            .norm()
                            .max(request.self_pairing_rtol * scale)
                            .sqrt()
                            .recip()
                    })
                    .collect();
            }
            for (u, &w) in seeds.iter_mut().zip(&unit) {
                u.iter_mut().flatten().for_each(|v| *v *= w);
            }
            let integral = |f: &dyn Fn(usize) -> Complex| integral(&partition, panels, f);
            let mut entries: Vec<_> = pairs
                .iter()
                .map(|&(i, j)| integral(&|s| unconjugated(&seeds[i][s], &seeds[j][s])))
                .collect();
            entries.extend(seeds.iter().map(|u| integral(&|s| field_scale(&u[s]))));
            Ok((entries, (seeds, 1 + first * (panels + 1))))
        },
    )?;
    let (seeds, at) = converged.extra;
    let raw = |i: usize, j: usize| {
        let index = pairs.iter().position(|&p| p == (i, j)).unwrap_or(i);
        converged.entries[index]
    };
    let field = |i: usize| converged.entries[pairs.len() + i].re;
    let margin = |weights: &[(usize, Complex)], l: Complex| {
        let peak = (0..seeds[0].len())
            .map(|s| {
                weights
                    .iter()
                    .fold(Complex::ZERO, |t, &(j, w)| {
                        t + w * (seeds[j][s][1] + seeds[j][s][4])
                    })
                    .norm()
            })
            .fold(0.0, f64::max);
        l.re.abs() / peak
    };
    let mut modes = vec![];
    let mut scales = vec![None; n];
    for i in 0..n {
        let uu = raw(i, i);
        let ratio = uu.norm() / field(i);
        let mut mode = BasisMode {
            point: points[i].clone(),
            self_pairing_raw_w_per_m: uu / (unit[i] * unit[i]),
            normalisation: Normalisation::IllConditioned { ratio },
            sign_reference: kind,
            sign_margin: None,
            terms: vec![],
        };
        if ratio >= request.self_pairing_rtol && ratio.is_finite() {
            let s = uu.sqrt().inv();
            let l = s * reference(&seeds[i][at], kind);
            let scale = sign(l) * s;
            scales[i] = Some(scale);
            mode.normalisation = Normalisation::Unit { ratio };
            mode.sign_margin = Some(margin(&[(i, scale)], sign(l) * l));
            mode.terms = vec![(points[i].clone(), scale * unit[i])];
        }
        modes.push(mode);
    }
    let mut singular = vec![];
    for group in &groups {
        let members: Vec<_> = group
            .iter()
            .copied()
            .filter(|&i| scales[i].is_some())
            .collect();
        let k = members.len();
        if k < 2 {
            continue;
        }
        let scale = |i: usize| scales[i].unwrap_or(Complex::ZERO);
        let gram = DMatrix::from_fn(k, k, |a, b| {
            let (i, j) = (members[a].min(members[b]), members[a].max(members[b]));
            if i == j {
                Complex::ONE
            } else {
                scale(i) * scale(j) * raw(i, j)
            }
        });
        let Some(root) = inverse_sqrt(&gram, request.self_pairing_rtol) else {
            let bounds = members.iter().fold(
                Rect {
                    x_min: f64::INFINITY,
                    x_max: f64::NEG_INFINITY,
                    y_min: f64::INFINITY,
                    y_max: f64::NEG_INFINITY,
                },
                |r, &i| Rect {
                    x_min: r.x_min.min(beta[i].re),
                    x_max: r.x_max.max(beta[i].re),
                    y_min: r.y_min.min(beta[i].im),
                    y_max: r.y_max.max(beta[i].im),
                },
            );
            singular.push(region(
                bounds,
                Some(k),
                BasisLimit::Multiplicity { count: k },
            ));
            continue;
        };
        for (b, &i) in members.iter().enumerate() {
            let weights: Vec<_> = members
                .iter()
                .enumerate()
                .map(|(a, &j)| (j, scale(j) * root[(a, b)]))
                .collect();
            let l = weights.iter().fold(Complex::ZERO, |t, &(j, w)| {
                t + w * reference(&seeds[j][at], kind)
            });
            let sigma = sign(l);
            let weights: Vec<_> = weights.into_iter().map(|(j, w)| (j, sigma * w)).collect();
            modes[i].sign_margin = Some(margin(&weights, sigma * l));
            modes[i].terms = weights
                .into_iter()
                .map(|(j, w)| (points[j].clone(), w * unit[j]))
                .collect();
        }
    }
    Ok((modes, groups, singular))
}

/// The principal G^(−1/2) of a complex symmetric matrix by the Denman–Beavers
/// iteration; `None` when G is singular to `rtol` or the iteration stalls.
fn inverse_sqrt(gram: &DMatrix<Complex>, rtol: f64) -> Option<DMatrix<Complex>> {
    let values = gram.clone().singular_values();
    let largest = values.max();
    if values.min().partial_cmp(&(rtol * largest)) != Some(Ordering::Greater) {
        return None;
    }
    let mut y = gram.clone();
    let mut z = DMatrix::identity(gram.nrows(), gram.ncols());
    for _ in 0..100 {
        let y_inverse = y.clone().try_inverse()?;
        let z_inverse = z.clone().try_inverse()?;
        let next_y = (&y + z_inverse) * Complex::new(0.5, 0.0);
        let next_z = (&z + y_inverse) * Complex::new(0.5, 0.0);
        let change = (&next_z - &z).norm();
        y = next_y;
        z = next_z;
        if change <= 1e-15 * z.norm() {
            // Symmetrise: G^(−1/2) is a function of the symmetric G.
            return Some((&z + z.transpose()) * Complex::new(0.5, 0.0));
        }
    }
    None
}

#[cfg(test)]
mod tests;
