//! Complete modal bases in a window of the complex β plane, and bases of
//! guided and leaky modes of stacks with open sides in a window of the
//! conformal chart.
//!
//! A stack with an open side has a radiation continuum besides its discrete
//! modes; its basis, of scope [`BasisScope::DiscreteWithoutContinuum`], never
//! includes it, so its modes do not span the stack's fields. It is searched
//! on the conformal chart of its open exteriors (see [`ModalWindow::Chart`]
//! and [`ModalWindow::covering`]), on which the characteristic is entire, so
//! the winding counts its zeros there as it does between walls.
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
    Kappas, Layout, Sample, TailFailure, converge, equivalent, field_scale, integral, samples,
    split_pmls, tail_scale, tails, unconjugated,
};
use super::open::{BRANCH_POINT_CLEARANCE, OpenChart, check_period};
use super::*;
use crate::conformal::in_plane::ComplexInPlaneMap;
use crate::problem as p;
use crate::search::ModalSearchResult;
use argloc::ArgumentConfig;
use lamina_core::{FieldSampling, ModalFieldBackend, Polarisation, Scatter2, StackSide};
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
    /// A rectangle in the dimensionless conformal coordinate z of a stack
    /// with an open side: real parts `re`, imaginary parts `im`, each
    /// ascending, less than one period, 2π, tall. β is continued from its
    /// principal root at the window's centre.
    Chart { re: [f64; 2], im: [f64; 2] },
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
        let (Self::Wavenumber { re_cm, im_cm }
        | Self::Chart {
            re: re_cm,
            im: im_cm,
        }) = *self;
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
    /// Every mode: the coupled TE/TM channel, whose modes are hybrid on a
    /// stack that is not separable.
    All,
}

impl ModeFamily {
    /// The polarisation of a TE or TM family; `None` for every mode.
    pub fn polarisation(self) -> Option<Polarisation> {
        match self {
            Self::Polarised(polarisation) => Some(polarisation),
            Self::All => None,
        }
    }

    /// The channel this family solves.
    pub fn channel(self) -> lamina_core::Channel {
        self.polarisation()
            .map_or(lamina_core::Channel::Coupled, Into::into)
    }
}

impl From<Polarisation> for ModeFamily {
    fn from(polarisation: Polarisation) -> Self {
        Self::Polarised(polarisation)
    }
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
    /// On a chart window the correction limit is in the dimensionless z and
    /// is not scaled.
    pub fn new(wavelength: Length<f64>, family: ModeFamily, window: ModalWindow) -> Self {
        let scale = TAU / wavelength.as_centimetres() / TAU;
        let native = match window {
            ModalWindow::Wavenumber { .. } => scale,
            ModalWindow::Chart { .. } => 1.0,
        };
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
                native_atol: refinement.native_atol * native,
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
    /// The guided and leaky modes in a chart window of a stack with an open
    /// side. Such a stack also has a radiation continuum, which is not
    /// included: its discrete modes never span its fields.
    DiscreteWithoutContinuum,
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
    /// A root whose exterior wavevector on `side` lies within
    /// [`BRANCH_POINT_CLEARANCE`] k0 of zero, where its self-pairing tail
    /// i/(2κ) diverges.
    BranchPoint { side: StackSide },
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
    /// Each open side's sheet, `None` on closed sides.
    pub sheets: [Option<ExteriorSheet>; 2],
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
    pub fn fields_with<B: ModalFieldBackend<Region> + Clone>(
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

fn seed_samples<B: ModalFieldBackend<Region>>(
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
    pub stack: ModeStack,
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
    #[error("a stack with an open side needs a chart window, ModalWindow::Chart")]
    OpenSide,
    #[error(transparent)]
    Window(#[from] WindowError),
    #[error("the {side:?} open tail of a self-pairing is singular")]
    TailSingular { side: StackSide },
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
    /// Modal bases span one polarisation of a separable stack.
    #[error("modal bases need a TE or TM family")]
    Family,
}

/// Every mode in a window, with counts and evidence: a β window for a stack
/// closed by walls or PMLs on both sides, a chart window for one with an open
/// side. Searches use Scatter2; fields use `backend`.
pub fn modal_basis_in_window_with<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    stack: impl Into<ModeStack>,
    request: ModalBasisRequest,
) -> Result<ModalBasis, ModalBasisError> {
    solve(backend, stack.into(), request, false)
}

pub(super) fn solve<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    stack: ModeStack,
    request: ModalBasisRequest,
    reverse: bool,
) -> Result<ModalBasis, ModalBasisError> {
    use ModalBasisError::*;
    let k0 = request.k0_cm();
    let window = request.window.rectangle().ok_or(InvalidRequest)?;
    let gap = request.degeneracy_gap_cm.unwrap_or(1e-6 * k0);
    let polarisation = request.family.polarisation().ok_or(Family)?;
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
    let open = match request.window {
        ModalWindow::Wavenumber { .. } if p::open_sides(&stack).contains(&true) => {
            return Err(OpenSide);
        }
        ModalWindow::Wavenumber { .. } => None,
        ModalWindow::Chart { .. } => {
            check_period(window)?;
            let open = OpenChart::new(&stack, k0)?.ok_or(WindowError::ClosedStack)?;
            if open.holds_branch_point(window) {
                return Err(WindowError::CoordinateBranchPoint.into());
            }
            Some(open)
        }
    };
    let problem = p::InPlaneProblem::in_plane(stack.clone(), k0, polarisation, Scatter2::new())?;
    let bounds = ConformalRegion::new(window.x_min, window.x_max, window.y_min, window.y_max)
        .map_err(|_| InvalidRequest)?;
    let top = match open {
        None => {
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
            None
        }
        Some(_) => Some(
            problem
                .domain(
                    bounds,
                    ReferenceSelection::Sheet(CoordinateSheet::Principal),
                    ConformalChartConfig::default(),
                    request.assumption,
                )
                .map_err(|_| InvalidRequest)?,
        ),
    };
    // The near-zero threshold relative to the least scale of the
    // characteristic, which the PML normalisation sets.
    let pml_sides = p::direct::pml_sides(&stack, polarisation, k0)
        .map_err(|e| Problem(ConstructionError::Constitutive(e)))?;
    let least_scale = p::direct::pml_log_floor(&pml_sides).min(0.0).exp();
    let argument = DiscoveryConfig {
        zero_tolerance: request.discovery.zero_tolerance * least_scale,
        ..request.discovery
    }
    .argument_config()
    .map_err(|_| InvalidRequest)?;
    let mirror = top.is_none().then(|| mirror_square(window)).flatten();
    let resolution = if top.is_some() { 1e-9 } else { 1e-9 * k0 };
    let search = Search {
        cells: BasisCells {
            problem: &problem,
            assumption: request.assumption,
            window,
            top,
        },
        argument,
        refinement: request.refinement,
        pml_sides,
        k0,
        mirror,
        resolution,
        reverse,
        window,
        open,
    };
    let (counted, mut unresolved, interior) = search.count(window);
    let expected = counted.map(|(n, _)| n);
    let (points, more) = search.roots(interior, counted, request.max_subdivisions);
    unresolved.extend(more);
    let (modes, groups, singular) = normalise(backend, &stack, points, gap, &request)?;
    let window_beta_image = search.image(window);
    let open = search.open.is_some();
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
        chart: if open {
            CoordinateKind::Conformal
        } else {
            CoordinateKind::Wavenumber
        },
        window,
        window_beta_image,
        degeneracy_gap_cm: gap,
        scope: if open {
            BasisScope::DiscreteWithoutContinuum
        } else {
            BasisScope::Discrete
        },
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

type Problem = p::InPlaneProblem<Scatter2>;
type Compiled = p::Compiled<Scatter2>;
type ChartDomain = p::ProblemDomain<ComplexInPlaneMap<p::J>, Compiled>;

/// A cell's domain on the direct β chart or the conformal chart.
enum Domain {
    Direct(Box<p::direct::DirectDomain<Compiled>>),
    Chart(Box<ChartDomain>),
}

macro_rules! on_domain {
    ($domain:expr, $d:ident => $body:expr) => {
        match $domain {
            Domain::Direct($d) => $body,
            Domain::Chart($d) => $body,
        }
    };
}

impl Domain {
    /// The characteristic and its derivative in the native coordinate.
    fn evaluate(&self, problem: &Problem, z: Complex) -> Option<(Complex, Complex)> {
        on_domain!(self, d => problem.evaluate(d, z).ok().map(|b| (*b.value(), *b.derivative())))
    }

    fn search(
        &self,
        problem: &Problem,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>> {
        on_domain!(self, d => Some(problem.search(d, config).ok()?.into_parts().0))
    }

    fn corrected(
        &self,
        problem: &Problem,
        seed: Complex,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<(crate::refinement::AcceptedRoot, BranchSignature), BranchFailure> {
        on_domain!(self, d => problem.corrected_point(d, seed, config).map(|(r, s, _)| (r, s)))
    }
}

/// The cells of a counted search: the domain of a rectangle, evaluation,
/// localisation and correction on it, and β at a native point.
pub(super) trait Cells {
    type Domain;
    fn domain(&self, cell: Rect<f64>) -> Result<Self::Domain, BasisLimit>;
    /// The characteristic and its derivative in the native coordinate.
    fn evaluate(&self, domain: &Self::Domain, z: Complex) -> Option<(Complex, Complex)>;
    fn search(
        &self,
        domain: &Self::Domain,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>>;
    fn corrected(
        &self,
        domain: &Self::Domain,
        seed: Complex,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<(crate::refinement::AcceptedRoot, BranchSignature), BranchFailure>;
    /// β at a point of the window.
    fn beta(&self, z: Complex) -> Option<Complex>;
    /// The core backend named on accepted modes.
    fn backend(&self) -> &'static str;
    /// The scale of the characteristic at `seed` against which a root's
    /// residual is judged, when the cells set one.
    fn residual_scale(&self, _domain: &Self::Domain, _seed: Complex) -> Option<f64> {
        None
    }
}

/// The cells of a modal basis: the direct β chart, or the conformal chart
/// over the whole window for a stack with an open side.
struct BasisCells<'a> {
    problem: &'a Problem,
    assumption: AnalyticAssumption,
    window: Rect<f64>,
    /// The conformal chart over the whole window, for a stack with an open
    /// side; each cell's chart continues β from it.
    top: Option<ChartDomain>,
}

impl Cells for BasisCells<'_> {
    type Domain = Domain;

    /// The domain of `cell`: on the conformal chart with β continued from
    /// the window's chart at its centre, else the direct domain, where a
    /// cell whose PML scale leaves the representable range is `Scale`. Any
    /// other failure is `NonIntegerCount`.
    fn domain(&self, cell: Rect<f64>) -> Result<Domain, BasisLimit> {
        let bounds = ConformalRegion::new(cell.x_min, cell.x_max, cell.y_min, cell.y_max)
            .map_err(|_| BasisLimit::NonIntegerCount)?;
        if self.top.is_some() {
            let reference = Complex::new(
                0.5 * (cell.x_min + cell.x_max),
                0.5 * (cell.y_min + cell.y_max),
            );
            let coordinate = self.beta(reference).ok_or(BasisLimit::NonIntegerCount)?;
            return self
                .problem
                .domain(
                    bounds,
                    ReferenceSelection::Coordinate {
                        reference,
                        coordinate,
                    },
                    ConformalChartConfig::default(),
                    self.assumption,
                )
                .map(|d| Domain::Chart(Box::new(d)))
                .map_err(|_| BasisLimit::NonIntegerCount);
        }
        self.problem
            .direct_domain(bounds, self.assumption)
            .map(|d| Domain::Direct(Box::new(d)))
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

    fn evaluate(&self, domain: &Domain, z: Complex) -> Option<(Complex, Complex)> {
        domain.evaluate(self.problem, z)
    }

    fn search(
        &self,
        domain: &Domain,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>> {
        domain.search(self.problem, config)
    }

    fn corrected(
        &self,
        domain: &Domain,
        seed: Complex,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<(crate::refinement::AcceptedRoot, BranchSignature), BranchFailure> {
        domain.corrected(self.problem, seed, config)
    }

    /// β at a point of the window: itself on the direct chart.
    fn beta(&self, z: Complex) -> Option<Complex> {
        match &self.top {
            None => Some(z),
            Some(top) => {
                let w = self.window;
                let z = Complex::new(z.re.clamp(w.x_min, w.x_max), z.im.clamp(w.y_min, w.y_max));
                top.chart().coordinate(z).ok()
            }
        }
    }

    fn backend(&self) -> &'static str {
        std::any::type_name::<Scatter2>()
    }
}

pub(super) struct Search<X> {
    pub(super) cells: X,
    pub(super) argument: ArgumentConfig<f64>,
    pub(super) refinement: RefinementConfig,
    /// The PML sides, which set the characteristic's scale.
    pub(super) pml_sides: Vec<p::direct::PmlSide>,
    pub(super) k0: f64,
    pub(super) mirror: Option<Rect<f64>>,
    /// Cells no larger than this, in the native coordinate, are not split
    /// further.
    pub(super) resolution: f64,
    pub(super) reverse: bool,
    pub(super) window: Rect<f64>,
    pub(super) open: Option<OpenChart>,
}

impl<X: Cells> Search<X> {
    fn beta(&self, z: Complex) -> Option<Complex> {
        self.cells.beta(z)
    }

    /// The β image of `bounds`: themselves on the direct chart, else the
    /// bounding box of the image of their boundary.
    fn image(&self, bounds: Rect<f64>) -> Rect<f64> {
        match &self.open {
            None => bounds,
            Some(open) => {
                open.beta_image(bounds, self.beta(Complex::new(bounds.x_min, bounds.y_min)))
            }
        }
    }

    fn gap(&self, bounds: Rect<f64>, count: Option<usize>, cause: BasisLimit) -> BasisGap {
        BasisGap {
            bounds,
            beta_image: self.image(bounds),
            count,
            cause,
        }
    }

    /// The open side on which a root's exterior wavevector is within the
    /// branch-point clearance of zero, if any.
    fn branch_point(&self, point: &ModePoint) -> Option<StackSide> {
        let signature = point.signature().ok()?;
        [
            (StackSide::Left, signature.left),
            (StackSide::Right, signature.right),
        ]
        .into_iter()
        .find(|(_, kappa)| kappa.is_some_and(|k| k.norm() <= BRANCH_POINT_CLEARANCE * self.k0))
        .map(|(side, _)| side)
    }

    fn domain(&self, cell: Rect<f64>) -> Result<X::Domain, BasisLimit> {
        self.cells.domain(cell)
    }

    fn search(
        &self,
        cell: Rect<f64>,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>> {
        self.cells.search(&self.domain(cell).ok()?, config)
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
            let (value, derivative) = self.cells.evaluate(&domain, z).ok_or_else(|| failed(z))?;
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
    pub(super) fn count(
        &self,
        window: Rect<f64>,
    ) -> (Option<(usize, Complex)>, Vec<BasisGap>, Rect<f64>) {
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
                (None, vec![self.gap(bounds, None, cause)], interior)
            }
        }
    }

    /// Correct the one root counted in `cell` from its estimate. Newton steps
    /// may leave the cell by up to its size; the accepted root must lie in it.
    fn correct(&self, cell: Rect<f64>, estimate: Complex) -> Result<ModePoint, BasisLimit> {
        let reach = (cell.x_max - cell.x_min).max(cell.y_max - cell.y_min);
        let mut around = Rect {
            x_min: cell.x_min - reach,
            x_max: cell.x_max + reach,
            y_min: cell.y_min - reach,
            y_max: cell.y_max + reach,
        };
        if self.open.is_some() {
            // Within the window, a validated chart region.
            let w = self.window;
            around = Rect {
                x_min: around.x_min.max(w.x_min),
                x_max: around.x_max.min(w.x_max),
                y_min: around.y_min.max(w.y_min),
                y_max: around.y_max.min(w.y_max),
            };
        }
        let domain = self.domain(around)?;
        let seed = Complex::new(
            estimate.re.clamp(cell.x_min, cell.x_max),
            estimate.im.clamp(cell.y_min, cell.y_max),
        );
        // The residual limit relative to the PML scale of the characteristic
        // at the seed.
        let beta = self.beta(seed).unwrap_or(seed);
        let scale = match self.cells.residual_scale(&domain, seed) {
            Some(cells) => cells,
            None => p::direct::pml_log_scale(&self.pml_sides, beta).exp(),
        };
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
        let (mut root, mut signature) = self
            .cells
            .corrected(&domain, seed, refinement)
            .map_err(BasisLimit::Refinement)?;
        // The accepted point's own Newton step is within the correction
        // limit; accepting again from the stepped point keeps the root to the
        // square of that step, which fields far from the seed side need.
        let evidence = root.evidence();
        let stepped = evidence.z - evidence.value / evidence.derivative;
        if contains(cell, stepped)
            && let Ok((polished, polished_signature)) =
                self.cells.corrected(&domain, stepped, refinement)
            && contains(cell, polished.evidence().z)
        {
            (root, signature) = (polished, polished_signature);
        }
        if !contains(cell, root.evidence().z) {
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
            backend: self.cells.backend(),
        })
    }

    /// Accepted roots in `interior`, and the regions that did not resolve.
    ///
    /// The boundary winding of [`Self::winding`] counts a cell and gives the
    /// sum of its zeros. A cell holding one zero is corrected from that sum;
    /// a cell holding more is split into quadrants counted the same way.
    /// argloc localises a cell only where this fails: a correction that is
    /// not accepted, or a split whose quadrants do not count. Every accepted
    /// root lies in a cell whose boundary counted exactly one zero.
    pub(super) fn roots(
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
                unresolved.push(self.gap(interior, None, cause));
                return (roots, unresolved);
            }
        };
        let at_cutoff = |z: Complex| self.mirror.is_some_and(|m| contains(m, z));
        // (cell, count, sum of its zeros, whether argloc has localised it)
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
                unresolved.push(self.gap(cell, Some(count), BasisLimit::Cutoff));
                continue;
            }
            // argloc's leaves and estimates in place of this cell; `false`
            // when it cannot search the cell.
            let localise = |cells: &mut Vec<_>, unresolved: &mut Vec<BasisGap>| {
                let Some(result) = (!localised)
                    .then(|| self.search(cell, self.argument.clone()))
                    .flatten()
                else {
                    return false;
                };
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
                    unresolved.push(self.gap(leaf.bounds, count, cause));
                }
                for estimate in result.estimates().iter().map(|e| *e.estimate()) {
                    let (bounds, n) = (estimate.enclosure, estimate.multiplicity);
                    if !open.contains(&bounds) {
                        cells.push((bounds, n, estimate.location * n as f64, true));
                    }
                }
                true
            };
            if count == 1 {
                match self.correct(cell, sum) {
                    Ok(point) if at_cutoff(point.physical_coordinate()) => {
                        unresolved.push(self.gap(cell, Some(1), BasisLimit::Cutoff))
                    }
                    Ok(point) => match self.branch_point(&point) {
                        Some(side) => unresolved.push(self.gap(
                            cell,
                            Some(1),
                            BasisLimit::BranchPoint { side },
                        )),
                        None => roots.push(point),
                    },
                    Err(cause) => {
                        if !localise(&mut cells, &mut unresolved) {
                            unresolved.push(self.gap(cell, Some(1), cause));
                        }
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
            if matches!(cause, BasisLimit::Multiplicity { .. })
                || !localise(&mut cells, &mut unresolved)
            {
                unresolved.push(self.gap(cell, Some(count), cause));
            }
        }
        roots.sort_by(|a, b| key_order(a.physical_coordinate(), b.physical_coordinate()));
        let mut distinct: Vec<ModePoint> = vec![];
        for point in roots {
            let z = point.native_coordinate();
            match distinct.last() {
                Some(last) if (last.native_coordinate() - z).norm() <= self.resolution => {
                    unresolved.push(self.gap(
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
fn normalise<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    stack: &ModeStack,
    points: Vec<ModePoint>,
    gap: f64,
    request: &ModalBasisRequest,
) -> Result<(Vec<BasisMode>, Vec<Vec<usize>>, Vec<BasisGap>), ModalBasisError> {
    use ModalBasisError::*;
    let n = points.len();
    let k0 = request.k0_cm();
    let layout = Layout::new(stack, 0.0).ok_or(InvalidRequest)?;
    let beta: Vec<_> = points.iter().map(ModePoint::physical_coordinate).collect();
    let polarisation = request.family.polarisation().ok_or(Family)?;
    let kappas: Vec<Kappas> = points
        .iter()
        .map(|point| point.signature().map_or([None; 2], |s| [s.left, s.right]))
        .collect();
    // The unconjugated self-pairing and pairings of open modes include their
    // continued tails, and the field scale its open-side share.
    let singular = |failure: TailFailure| match failure {
        TailFailure::Singular(side) | TailFailure::Unbounded(side) => TailSingular { side },
    };
    let tail = |seeds: &[Vec<Sample>], i: usize, j: usize| {
        tails(&seeds[i], &seeds[j], [kappas[i], kappas[j]], k0, false).map_err(singular)
    };
    let partition = split_pmls(
        Layout::partition(&[&layout]),
        &[(&layout, stack, polarisation)],
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
            let scale = |u: &[Sample], i: usize| {
                integral(&partition, panels, |s| field_scale(&u[s])).re + tail_scale(u, kappas[i])
            };
            if unit.is_empty() {
                unit = (0..n)
                    .map(|i| {
                        let u = &seeds[i];
                        let pairing = integral(&partition, panels, |s| unconjugated(&u[s], &u[s]))
                            + tail(&seeds, i, i)?;
                        Ok(pairing
                            .norm()
                            .max(request.self_pairing_rtol * scale(u, i))
                            .sqrt()
                            .recip())
                    })
                    .collect::<Result<_, ModalBasisError>>()?;
            }
            for (u, &w) in seeds.iter_mut().zip(&unit) {
                u.iter_mut().flatten().for_each(|v| *v *= w);
            }
            let mut entries = pairs
                .iter()
                .map(|&(i, j)| {
                    Ok(integral(&partition, panels, |s| {
                        unconjugated(&seeds[i][s], &seeds[j][s])
                    }) + tail(&seeds, i, j)?)
                })
                .collect::<Result<Vec<_>, ModalBasisError>>()?;
            entries.extend((0..n).map(|i| Complex::new(scale(&seeds[i], i), 0.0)));
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
            sheets: kappas[i].map(|k| k.map(ExteriorSheet::of)),
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
    let mut singular_groups = vec![];
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
            let bounds = |at: &dyn Fn(usize) -> Complex| {
                members.iter().fold(
                    Rect {
                        x_min: f64::INFINITY,
                        x_max: f64::NEG_INFINITY,
                        y_min: f64::INFINITY,
                        y_max: f64::NEG_INFINITY,
                    },
                    |r, &i| Rect {
                        x_min: r.x_min.min(at(i).re),
                        x_max: r.x_max.max(at(i).re),
                        y_min: r.y_min.min(at(i).im),
                        y_max: r.y_max.max(at(i).im),
                    },
                )
            };
            singular_groups.push(BasisGap {
                bounds: bounds(&|i| points[i].native_coordinate()),
                beta_image: bounds(&|i| beta[i]),
                count: Some(k),
                cause: BasisLimit::Multiplicity { count: k },
            });
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
    Ok((modes, groups, singular_groups))
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
