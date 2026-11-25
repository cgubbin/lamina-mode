//! Projection of supplied fields onto modal bases and local coupling on them:
//! recovery of known combinations between walls, through PMLs and on open
//! stacks with leaky modes; the continuum residual of a field with radiation
//! content; the coupling diagonal against the modes' dβ/dp and the
//! off-diagonal against pairings of neighbouring stacks; groups; refusals.
use lamina_core::{
    AnalyticalMaterialStack, Constant, ExteriorSampling, FieldSampling, LayerSampling,
    Polarisation, Scatter2, Stack, StackSide, Termination, Wall, WallKind,
};
use lamina_mode::*;
use nalgebra::DMatrix;
use std::cell::RefCell;
use std::collections::HashMap;
mod pml_support;
use pml_support::{A, CLAD, CORE, K, PEC, PMC, PROFILE, TE, TM, cm, pml};

const GAP: f64 = 0.45;
const ORIGIN: GuidedOverlapRegistration = GuidedOverlapRegistration {
    origins_cm: [0., 0.],
};
/// A forward β window with real, imaginary and complex roots of the walled slab.
const FORWARD: ModalWindow = ModalWindow::Wavenumber {
    re_cm: [-0.05 * K, 2.1 * K],
    im_cm: [-0.05 * K, 3. * K],
};
/// A β window of the PML-closed slab below the rotated continuum's large modes.
const SMALL: ModalWindow = ModalWindow::Wavenumber {
    re_cm: [1.0 * K, 2.03 * K],
    im_cm: [-0.031 * K, 1.0 * K],
};
/// A chart window holding two guided modes and one leaky mode of the open slab.
const CHART: ModalWindow = ModalWindow::Chart {
    re: [-3., 0.35],
    im: [-2.9, 0.4],
};
/// The PML thickness, its start lying `PML_GAP` beyond each interface.
const PML_D: f64 = 0.75;
const PML_GAP: f64 = 0.1;

/// A core of permittivity `core` and thickness `t` in cladding of
/// permittivity `clad`, closed by `left` and `right` at absolute positions.
fn stack(
    core: f64,
    clad: f64,
    t: f64,
    left: impl Into<Termination<f64>>,
    right: impl Into<Termination<f64>>,
) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(clad), Constant::dielectric(clad))
        .analytical_layer(Constant::dielectric(core), cm(t))
        .left_termination(left)
        .right_termination(right)
        .finalise()
}

/// The core in cladding between walls at −GAP and 2A + GAP, whatever its thickness.
fn walled(core: f64, clad: f64, t: f64, kind: WallKind) -> AnalyticalMaterialStack<Complex> {
    stack(
        core,
        clad,
        t,
        Wall::new(kind, cm(-GAP)),
        Wall::new(kind, cm(2. * A + GAP)),
    )
}

/// The core between PEC-backed PMLs starting at −PML_GAP and 2A + PML_GAP.
fn pml_closed(core: f64, clad: f64, t: f64) -> AnalyticalMaterialStack<Complex> {
    stack(
        core,
        clad,
        t,
        pml(StackSide::Left, PML_GAP, PML_D, PROFILE, PEC),
        pml(StackSide::Right, PML_GAP, PML_D, PROFILE, PEC),
    )
}

fn open(core: f64, clad: f64, t: f64) -> AnalyticalMaterialStack<Complex> {
    stack(core, clad, t, Termination::Open, Termination::Open)
}

fn solve(
    stack: AnalyticalMaterialStack<Complex>,
    pol: Polarisation,
    window: ModalWindow,
) -> ModalBasis {
    let request = ModalBasisRequest::new(cm(1.), ModeFamily::Polarised(pol), window);
    let basis = modal_basis_in_window_with(&Scatter2::new(), stack, request).unwrap();
    assert_eq!(
        basis.status,
        BasisStatus::Complete,
        "{:?} {:?} {}",
        basis.unresolved,
        basis.expected,
        basis.found()
    );
    basis
}

/// One sample of a stack at z (cm, stack frame), from the side `side`
/// selects at an interface. Exterior samples continue into gaps and PMLs.
fn sampling_at(stack: &ModeStack, z: f64, side: ProjectionSide) -> FieldSampling<f64> {
    let widths: Vec<f64> = stack
        .layers_left_to_right()
        .iter()
        .map(|l| l.thickness().as_centimetres())
        .collect();
    let from_left = side == ProjectionSide::Left;
    if z < 0. || (z == 0. && from_left) {
        return FieldSampling::new().left_exterior(ExteriorSampling::point(cm(-z)));
    }
    let mut left = 0.;
    for (i, &w) in widths.iter().enumerate() {
        let right = left + w;
        if z < right || (z == right && from_left) {
            return FieldSampling::new()
                .layer(i, LayerSampling::offsets([cm((z - left).clamp(0., w))]));
        }
        left = right;
    }
    FieldSampling::new().right_exterior(ExteriorSampling::point(cm(z - left)))
}

/// Σ c_n f_n at one point, the fields continued into PMLs.
fn combination(
    basis: &ModalBasis,
    c: &[(usize, Complex)],
    z: f64,
    side: ProjectionSide,
) -> [Complex; 6] {
    let sampling = sampling_at(&basis.stack, z, side);
    let mut total = [Complex::ZERO; 6];
    for &(mode, weight) in c {
        let fields = basis.modes[mode]
            .fields_with(Scatter2::new(), &sampling, basis.request.recovery)
            .unwrap();
        for (t, v) in total.iter_mut().zip(fields[0]) {
            *t += weight * v;
        }
    }
    total
}

/// Known coefficients of magnitude about one, distinct for every mode.
fn known(n: usize) -> Vec<Complex> {
    (0..n)
        .map(|i| Complex::new(0.3 + 0.2 * i as f64, 0.5 - 0.15 * i as f64))
        .collect()
}

fn project(
    basis: &ModalBasis,
    support: [f64; 2],
    field: &dyn Fn(f64, ProjectionSide) -> [Complex; 6],
    exterior: [&[ExteriorWave]; 2],
) -> BasisFieldProjection {
    let memo: RefCell<HashMap<(u64, u8), [Complex; 6]>> = RefCell::default();
    let sample = |z: f64, side: ProjectionSide| {
        let key = (z.to_bits(), side as u8);
        if let Some(v) = memo.borrow().get(&key) {
            return *v;
        }
        let v = field(z, side);
        memo.borrow_mut().insert(key, v);
        v
    };
    basis_projection_with(
        &Scatter2::new(),
        basis,
        0.,
        BasisProjectionInput {
            field: GuidedProjectionInput {
                wavelength_cm: 1.,
                support_cm: support,
                interfaces_cm: &[],
                sample: &sample,
            },
            exterior,
        },
        1e-8,
    )
    .unwrap()
}

/// Recover known coefficients and report the worst error, the residual and
/// the Gram defect.
fn recover(
    basis: &ModalBasis,
    support: [f64; 2],
    exterior: impl Fn(&[(usize, Complex)]) -> [Vec<ExteriorWave>; 2],
) -> (f64, f64, f64) {
    let c: Vec<(usize, Complex)> = known(basis.modes.len()).into_iter().enumerate().collect();
    let waves = exterior(&c);
    let field = |z: f64, side| combination(basis, &c, z, side);
    let p = project(basis, support, &field, [&waves[0], &waves[1]]);
    let solution = p.solution.as_ref().unwrap();
    let error = p
        .modes
        .iter()
        .zip(solution.coefficients.iter())
        .map(|(&i, a)| (a - c[i].1).norm())
        .fold(0., f64::max);
    let residual = solution.relative_residual.unwrap();
    assert!(solution.equation_residual_w_per_m < 1e-9);
    (error, residual, p.gram_defect)
}

#[test]
fn known_combinations_of_walled_and_pml_closed_modes_are_recovered() {
    for pol in [TE, TM] {
        for kind in [PEC, PMC] {
            let basis = solve(walled(CORE, CLAD, 2. * A, kind), pol, FORWARD);
            let support = [-GAP, 2. * A + GAP];
            let (error, residual, defect) = recover(&basis, support, |_| [vec![], vec![]]);
            eprintln!(
                "{pol:?} {kind:?} walls, {} modes: max |a − c| {error:.1e}, relative residual \
                 {residual:.1e}, |G − I| {defect:.1e}",
                basis.found()
            );
            assert!(error < 1e-8 && residual < 1e-7 && defect < 1e-9);
        }
        let basis = solve(pml_closed(CORE, CLAD, 2. * A), pol, SMALL);
        let support = [-PML_GAP - PML_D, 2. * A + PML_GAP + PML_D];
        let (error, residual, defect) = recover(&basis, support, |_| [vec![], vec![]]);
        eprintln!(
            "{pol:?} PMLs, {} modes: max |a − c| {error:.1e}, relative residual {residual:.1e}, \
             |G − I| {defect:.1e}",
            basis.found()
        );
        assert!(error < 1e-8 && residual < 1e-7 && defect < 1e-9);
    }
}

/// The exterior waves of Σ c_n f_n beyond the support edges `support`.
fn continuation(
    basis: &ModalBasis,
    c: &[(usize, Complex)],
    support: [f64; 2],
) -> [Vec<ExteriorWave>; 2] {
    std::array::from_fn(|side| {
        let (z, from) = if side == 0 {
            (support[0], ProjectionSide::Left)
        } else {
            (support[1], ProjectionSide::Right)
        };
        c.iter()
            .map(|&(mode, weight)| {
                let s = basis.modes[mode].point.signature().unwrap();
                ExteriorWave {
                    kappa_cm: [s.left, s.right][side].unwrap(),
                    edge: combination(basis, &[(mode, weight)], z, from),
                }
            })
            .collect()
    })
}

#[test]
fn guided_and_leaky_combinations_are_recovered_with_their_continuation() {
    for pol in [TE, TM] {
        let basis = solve(open(CORE, CLAD, 2. * A), pol, CHART);
        assert_eq!(basis.scope, BasisScope::DiscreteWithoutContinuum);
        let leaky = basis
            .modes
            .iter()
            .filter(|m| m.sheets.contains(&Some(ExteriorSheet::Improper)))
            .count();
        assert!(leaky >= 1);
        let support = [-0.4, 2. * A + 0.4];
        let (error, residual, defect) =
            recover(&basis, support, |c| continuation(&basis, c, support));
        eprintln!(
            "{pol:?} open slab, {} modes ({leaky} leaky): max |a − c| {error:.1e}, relative \
             residual {residual:.1e}, |G − I| {defect:.1e}",
            basis.found()
        );
        assert!(error < 1e-8 && residual < 1e-7 && defect < 1e-9);
        // Without its continuation the field is cut at the support edges and
        // the leaky tails are missing: the coefficients are wrong.
        let (cut, _, _) = recover(&basis, support, |_| [vec![], vec![]]);
        assert!(cut > 1e-3, "{cut:e}");
    }
}

/// A TE Gaussian beam profile centred on the core, n_eff 1.8.
fn gaussian(z: f64) -> [Complex; 6] {
    let ey = Complex::new((-((z - A) / 0.3).powi(2)).exp(), 0.);
    [
        Complex::ZERO,
        ey,
        Complex::ZERO,
        Complex::ZERO,
        Complex::ZERO,
        1.8 * ey,
    ]
}

#[test]
fn radiation_content_of_an_open_stack_is_left_as_a_continuum_residual() {
    let basis = solve(open(CORE, CLAD, 2. * A), TE, CHART);
    let support = [-0.75, 2. * A + 0.75];
    let beam = project(&basis, support, &|z, _| gaussian(z), [&[], &[]]);
    assert!(beam.residual_includes_continuum);
    let solution = beam.solution.as_ref().unwrap();
    let residual = solution.relative_residual.unwrap();
    eprintln!(
        "TE Gaussian on the open slab: relative residual {residual:.3} outside the discrete \
         scope, equation residual {:.1e} W/m, coefficients {:?}",
        solution.equation_residual_w_per_m,
        solution.coefficients.as_slice()
    );
    assert!(residual > 1e-2);
    assert!(solution.equation_residual_w_per_m < 1e-9);
    // Linearity: the beam plus a modal combination returns both.
    let c: Vec<(usize, Complex)> = known(basis.modes.len()).into_iter().enumerate().collect();
    let waves = continuation(&basis, &c, support);
    let both = |z: f64, side| {
        let (g, m) = (gaussian(z), combination(&basis, &c, z, side));
        std::array::from_fn(|k| g[k] + m[k])
    };
    let sum = project(&basis, support, &both, [&waves[0], &waves[1]]);
    let worst = sum
        .solution
        .as_ref()
        .unwrap()
        .coefficients
        .iter()
        .zip(solution.coefficients.iter())
        .zip(&c)
        .map(|((s, g), (_, c))| (s - g - c).norm())
        .fold(0., f64::max);
    assert!(worst < 1e-8, "{worst:e}");
    // A closed stack's basis has no continuum to miss.
    let walled_basis = solve(walled(CORE, CLAD, 2. * A, PEC), TE, FORWARD);
    let p = project(
        &walled_basis,
        [-GAP, 2. * A + GAP],
        &|z, _| gaussian(z),
        [&[], &[]],
    );
    assert!(!p.residual_includes_continuum);
}

#[test]
fn projection_refuses_supports_beyond_walls_and_continuations_on_closed_sides() {
    let basis = solve(walled(CORE, CLAD, 2. * A, PEC), TE, FORWARD);
    let zero = |_: f64, _: ProjectionSide| [Complex::ZERO; 6];
    let run = |support: [f64; 2], exterior: [&[ExteriorWave]; 2]| {
        basis_projection_with(
            &Scatter2::new(),
            &basis,
            0.,
            BasisProjectionInput {
                field: GuidedProjectionInput {
                    wavelength_cm: 1.,
                    support_cm: support,
                    interfaces_cm: &[],
                    sample: &zero,
                },
                exterior,
            },
            1e-8,
        )
    };
    let wave = [ExteriorWave {
        kappa_cm: Complex::I,
        edge: [Complex::ONE; 6],
    }];
    assert!(matches!(
        run([-GAP - 0.1, 0.5], [&[], &[]]),
        Err(BasisProjectionError::InvalidInput)
    ));
    assert!(matches!(
        run([-GAP, 2. * A + GAP], [&wave, &[]]),
        Err(BasisProjectionError::InvalidInput)
    ));
    // A zero field has no relative residual but is a result.
    let p = run([-GAP, 2. * A + GAP], [&[], &[]]).unwrap();
    assert_eq!(p.field_scale_w_per_m, 0.);
    assert!(p.solution.unwrap().relative_residual.is_none());
    // On an open stack a continuation needs its edge in the exterior.
    let open_basis = solve(open(CORE, CLAD, 2. * A), TE, CHART);
    let inside = basis_projection_with(
        &Scatter2::new(),
        &open_basis,
        0.,
        BasisProjectionInput {
            field: GuidedProjectionInput {
                wavelength_cm: 1.,
                support_cm: [0.1, 0.5],
                interfaces_cm: &[],
                sample: &zero,
            },
            exterior: [&wave, &[]],
        },
        1e-8,
    );
    assert!(matches!(inside, Err(BasisProjectionError::InvalidInput)));
}

// ---------------------------------------------------------------- coupling

/// A stack of the given core permittivity, cladding permittivity and core thickness.
type Make = Box<dyn Fn(f64, f64, f64) -> AnalyticalMaterialStack<Complex>>;
/// A labelled parameter direction, the basis at its parameter displaced by
/// the argument, and the finite-difference step.
type Direction<'a> = (
    &'a str,
    GuidedPerturbation,
    Box<dyn Fn(f64) -> ModalBasis + 'a>,
    f64,
);

fn thickness() -> GuidedPerturbation {
    GuidedPerturbation::thickness(1, 0).unwrap()
}

/// The permittivity of `regions` (0 left exterior, 1 core, 2 right exterior).
fn permittivity(regions: &[usize]) -> GuidedPerturbation {
    let mut rates = vec![ConstitutiveRate::default(); 3];
    for &r in regions {
        rates[r].relative_permittivity = 1.;
    }
    GuidedPerturbation::new("permittivity", "1", vec![0., 0.], 0., rates).unwrap()
}

fn coupling(basis: &ModalBasis, direction: &GuidedPerturbation) -> BasisLocalCoupling {
    basis_local_coupling_with(&Scatter2::new(), basis, direction, None).unwrap()
}

/// dβ/dt of a mode from the characteristic's implicit slope.
fn slope(mode: &BasisMode) -> Complex {
    mode.point
        .local_derivatives_with(
            Scatter2::new(),
            &[
                PhysicalParameter::Beta,
                PhysicalParameter::LayerThickness(0.into()),
            ],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            Default::default(),
        )
        .unwrap()
        .implicit_slope(
            ImplicitSlopeRequest::BetaPerThickness(0.into()),
            Default::default(),
        )
        .unwrap()
        .value
}

/// The β of each mode of `basis` matched to its nearest in `other`.
fn matched(basis: &ModalBasis, other: &ModalBasis) -> Vec<Complex> {
    basis
        .modes
        .iter()
        .map(|m| {
            let b = m.point.physical_coordinate();
            other
                .modes
                .iter()
                .map(|o| o.point.physical_coordinate())
                .min_by(|x, y| (x - b).norm().total_cmp(&(y - b).norm()))
                .unwrap()
        })
        .collect()
}

/// Largest |K_nn − reference_n| relative to |reference_n|.
fn worst(diagonal: &[Complex], reference: &[Complex]) -> f64 {
    diagonal
        .iter()
        .zip(reference)
        .map(|(k, r)| (k - r).norm() / r.norm())
        .fold(0., f64::max)
}

#[test]
fn coupling_diagonals_are_the_modes_beta_rates() {
    let backend = Scatter2::new();
    let h = 1e-5;
    let cases: Vec<(&str, Polarisation, Make, ModalWindow)> = vec![
        (
            "walls",
            TE,
            Box::new(|core, clad, t| walled(core, clad, t, PEC)),
            FORWARD,
        ),
        (
            "walls",
            TM,
            Box::new(|core, clad, t| walled(core, clad, t, PMC)),
            FORWARD,
        ),
        ("PMLs", TE, Box::new(pml_closed), SMALL),
        ("PMLs", TM, Box::new(pml_closed), SMALL),
        ("open", TE, Box::new(open), CHART),
        ("open", TM, Box::new(open), CHART),
    ];
    for (name, pol, make, window) in &cases {
        let basis = solve(make(CORE, CLAD, 2. * A), *pol, *window);
        let diagonal = |c: &BasisLocalCoupling| {
            (0..c.modes.len())
                .map(|i| c.perturbation_matrix[(i, i)])
                .collect::<Vec<_>>()
        };
        // Thickness: the implicit slope of the characteristic, every mode.
        let k = diagonal(&coupling(&basis, &thickness()));
        let slopes: Vec<_> = basis.modes.iter().map(slope).collect();
        let thickness_error = worst(&k, &slopes);
        // Core permittivity: finite differences of re-solved β, and the
        // guided sensitivity for real modes.
        let core = diagonal(&coupling(&basis, &permittivity(&[1])));
        let up = solve(make(CORE + h, CLAD, 2. * A), *pol, *window);
        let down = solve(make(CORE - h, CLAD, 2. * A), *pol, *window);
        let fd: Vec<_> = matched(&basis, &up)
            .iter()
            .zip(matched(&basis, &down))
            .map(|(u, d)| (u - d) / (2. * h))
            .collect();
        let core_error = worst(&core, &fd);
        let mut guided_error: f64 = 0.;
        for (mode, k) in basis.modes.iter().zip(&core) {
            let beta = mode.point.physical_coordinate();
            if beta.im.abs() > 1e-9 * beta.norm()
                || beta.re <= 0.
                || mode
                    .sheets
                    .iter()
                    .any(|s| s == &Some(ExteriorSheet::Improper))
            {
                continue;
            }
            let rate = mode
                .point
                .guided_beta_sensitivity_with(
                    &backend,
                    &permittivity(&[1]),
                    Default::default(),
                    Default::default(),
                )
                .unwrap()
                .beta_rate;
            guided_error = guided_error.max((k - rate).norm() / rate.norm());
        }
        eprintln!(
            "{name} {pol:?}, {} modes: dβ/dt against the implicit slope {thickness_error:.1e}; \
             dβ/dε_core against finite differences {core_error:.1e}, against the guided \
             sensitivity {guided_error:.1e}",
            basis.found()
        );
        assert!(thickness_error < 1e-7, "{k:?} {slopes:?}");
        assert!(guided_error < 1e-7);
        assert!(core_error < 1e-5, "{core:?} {fd:?}");
        {
            // Cladding on both sides: gaps, PMLs or continued tails carry the rate.
            let clad = diagonal(&coupling(&basis, &permittivity(&[0, 2])));
            let up = solve(make(CORE, CLAD + h, 2. * A), *pol, *window);
            let down = solve(make(CORE, CLAD - h, 2. * A), *pol, *window);
            let fd: Vec<_> = matched(&basis, &up)
                .iter()
                .zip(matched(&basis, &down))
                .map(|(u, d)| (u - d) / (2. * h))
                .collect();
            let clad_error = worst(&clad, &fd);
            eprintln!("{name} {pol:?}: dβ/dε_clad against finite differences {clad_error:.1e}");
            assert!(clad_error < 1e-5, "{clad:?} {fd:?}");
        }
    }
}

/// Central differences of the pairing of `basis` with bases at p ± h,
/// against the coefficients; returns the worst disagreement relative to the
/// largest coefficient, and that coefficient.
fn against_pairings(
    basis: &ModalBasis,
    coupling: &BasisLocalCoupling,
    neighbour: &dyn Fn(f64) -> ModalBasis,
    h: f64,
) -> (f64, f64) {
    let backend = Scatter2::new();
    let margins: Vec<_> = basis.modes.iter().filter_map(|m| m.sign_margin).collect();
    assert!(margins.iter().all(|&m| m > 1e-2), "{margins:?}");
    let pair = |b: &ModalBasis| {
        assert_eq!(b.found(), basis.found());
        basis_pairing_with(&backend, [basis, b], ORIGIN)
            .unwrap()
            .unconjugated
    };
    let (up, down) = (pair(&neighbour(h)), pair(&neighbour(-h)));
    let fd: DMatrix<Complex> = (up - down) / Complex::new(2. * h, 0.);
    let n = basis.found();
    let largest = coupling
        .coefficients
        .iter()
        .flatten()
        .map(|c| c.norm())
        .fold(0., f64::max);
    let mut disagreement: f64 = 0.;
    for m in 0..n {
        for j in 0..n {
            let expected = coupling.coefficients[(m, j)].unwrap_or(Complex::ZERO);
            disagreement = disagreement.max((fd[(m, j)] - expected).norm());
        }
    }
    (disagreement / largest, largest)
}

#[test]
fn coupling_coefficients_are_derivatives_of_neighbouring_pairings() {
    let cases: Vec<(&str, Polarisation, Make, ModalWindow)> = vec![
        (
            "walls",
            TE,
            Box::new(|core, clad, t| walled(core, clad, t, PEC)),
            FORWARD,
        ),
        (
            "walls",
            TM,
            Box::new(|core, clad, t| walled(core, clad, t, PMC)),
            FORWARD,
        ),
        ("PMLs", TE, Box::new(pml_closed), SMALL),
        ("open", TE, Box::new(open), CHART),
        ("open", TM, Box::new(open), CHART),
    ];
    for (name, pol, make, window) in &cases {
        let basis = solve(make(CORE, CLAD, 2. * A), *pol, *window);
        let t = 2. * A;
        let directions: Vec<Direction> = vec![
            (
                "thickness",
                thickness(),
                Box::new(|d| solve(make(CORE, CLAD, t + d), *pol, *window)),
                1e-4 * t,
            ),
            (
                "core permittivity",
                permittivity(&[1]),
                Box::new(|d| solve(make(CORE + d, CLAD, t), *pol, *window)),
                1e-4,
            ),
            (
                "cladding permittivity",
                permittivity(&[0, 2]),
                Box::new(|d| solve(make(CORE, CLAD + d, t), *pol, *window)),
                1e-4,
            ),
        ];
        for (label, direction, neighbour, h) in &directions {
            let c = coupling(&basis, direction);
            assert!(c.groups.is_empty());
            let (error, largest) = against_pairings(&basis, &c, neighbour, *h);
            eprintln!(
                "{name} {pol:?} {label}: {} modes, central differences of the pairings agree \
                 with the coefficients to {error:.1e} of the largest, {largest:.3}",
                basis.found()
            );
            assert!(error < 1e-4, "{error:e}");
            if *label == "thickness" {
                // The disagreement is the differences' own error: it falls with
                // h, as h² where the fields are smooth across the moving
                // interface (TE) and as h where E_z jumps there (TM: the
                // pairing is C¹ with a curvature jump at the base thickness).
                let (finer, _) = against_pairings(&basis, &c, neighbour, h / 4.);
                eprintln!("    at h/4: {finer:.1e}");
                assert!(finer < error / 3. || finer < 1e-7, "{finer:e} {error:e}");
            }
        }
    }
}

/// Two identical cores far apart between PEC walls.
fn twin() -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(CLAD))
        .analytical_layer(Constant::dielectric(CORE), cm(0.3))
        .analytical_layer(Constant::dielectric(CLAD), cm(2.15))
        .analytical_layer(Constant::dielectric(CORE), cm(0.3))
        .left_termination(Wall::perfect_electric(cm(-0.3)))
        .right_termination(Wall::perfect_electric(cm(3.05)))
        .finalise()
}

#[test]
fn near_degenerate_modes_are_reported_as_groups() {
    let window = ModalWindow::forward(2.05 * K, 0.5 * K, 0.05 * K);
    let basis = solve(twin(), TE, window);
    assert_eq!(basis.groups.len(), 1);
    let mut rates = vec![ConstitutiveRate::default(); 5];
    rates[1].relative_permittivity = 1.;
    let first_core = GuidedPerturbation::new("first core", "1", vec![0.; 4], 0., rates).unwrap();
    let c = basis_local_coupling_with(&Scatter2::new(), &basis, &first_core, None).unwrap();
    assert_eq!(c.groups, basis.groups);
    let [i, j] = [c.groups[0][0], c.groups[0][1]];
    let n = c.modes.len();
    for m in 0..n {
        for k in 0..n {
            let within = [i, j].contains(&m) && [i, j].contains(&k);
            assert_eq!(c.coefficients[(m, k)].is_none(), m == k || within);
        }
    }
    let finite = c
        .coefficients
        .iter()
        .flatten()
        .all(|v| v.re.is_finite() && v.im.is_finite());
    assert!(finite);
    // The group's block holds the split: one core's perturbation mixes the
    // even and odd supermodes with a coupling as large as their own rates.
    let block = [c.perturbation_matrix[(i, i)], c.perturbation_matrix[(i, j)]];
    eprintln!(
        "twin cores: group {:?}, block diagonal {:.4} and coupling {:.4} per unit ε",
        c.groups[0], block[0], block[1]
    );
    assert!(block[1].norm() > 0.1 * block[0].norm());
    // A wider gap links more modes.
    let wide =
        basis_local_coupling_with(&Scatter2::new(), &basis, &first_core, Some(0.5 * K)).unwrap();
    assert!(wide.groups.iter().map(Vec::len).sum::<usize>() > 2);
}

#[test]
fn coupling_refuses_wavelength_magnetic_and_wall_moving_directions() {
    let basis = solve(walled(CORE, CLAD, 2. * A, PEC), TE, FORWARD);
    let run = |d: &GuidedPerturbation| basis_local_coupling_with(&Scatter2::new(), &basis, d, None);
    let rates = vec![ConstitutiveRate::default(); 3];
    let wavelength = GuidedPerturbation::new("λ", "cm", vec![0., 0.], 1., rates.clone()).unwrap();
    assert!(matches!(
        run(&wavelength),
        Err(BasisCouplingError::Wavelength)
    ));
    let mut magnetic = rates.clone();
    magnetic[1].relative_permeability = 1.;
    let magnetic = GuidedPerturbation::new("μ", "1", vec![0., 0.], 0., magnetic).unwrap();
    assert!(matches!(run(&magnetic), Err(BasisCouplingError::Magnetic)));
    let translation = GuidedPerturbation::new("x", "cm", vec![1., 1.], 0., rates).unwrap();
    assert!(matches!(
        run(&translation),
        Err(BasisCouplingError::Termination {
            side: StackSide::Left
        })
    ));
    assert!(matches!(
        basis_local_coupling_with(&Scatter2::new(), &basis, &thickness(), Some(-1.)),
        Err(BasisCouplingError::Threshold)
    ));
    // An interface on a wall may not move.
    let flush = solve(
        stack(
            CORE,
            CLAD,
            2. * A,
            Wall::perfect_electric(cm(-GAP)),
            Wall::perfect_electric(cm(2. * A)),
        ),
        TE,
        FORWARD,
    );
    assert!(matches!(
        basis_local_coupling_with(&Scatter2::new(), &flush, &thickness(), None),
        Err(BasisCouplingError::Termination {
            side: StackSide::Right
        })
    ));
}

/// A Drude core with Re ε ≈ 3.5 and Im ε ≈ 0.5 at the 1 cm wavelength, of
/// thickness `t`, between walls at −0.3 and 1.1: lossy and complex modes.
fn lossy(t: f64) -> AnalyticalMaterialStack<Complex> {
    use lamina_units::VacuumAngularWavenumber as W;
    let core = lamina_materials::Drude::new(
        4.,
        W::inverse_centimetres(40f64.sqrt()),
        W::inverse_centimetres(K),
    )
    .unwrap();
    Stack::from_analytical_materials(Constant::dielectric(CLAD), Constant::dielectric(1.))
        .analytical_layer(core, cm(t))
        .left_termination(Wall::perfect_electric(cm(-0.3)))
        .right_termination(Wall::perfect_electric(cm(1.1)))
        .finalise()
}

#[test]
fn coupling_of_lossy_complex_modes_matches_slopes_and_pairings() {
    let window = ModalWindow::forward(2.2 * K, 2. * K, 0.05 * K);
    for pol in [TE, TM] {
        let t = 0.6;
        let basis = solve(lossy(t), pol, window);
        let complex = basis
            .modes
            .iter()
            .filter(|m| {
                let b = m.point.physical_coordinate();
                b.re > 1e-3 * K && b.im > 1e-3 * K
            })
            .count();
        assert!(complex > 0);
        let c = coupling(&basis, &thickness());
        let diagonal: Vec<_> = (0..c.modes.len())
            .map(|i| c.perturbation_matrix[(i, i)])
            .collect();
        let slopes: Vec<_> = basis.modes.iter().map(slope).collect();
        let diagonal_error = worst(&diagonal, &slopes);
        let h = 1e-4 * t;
        let neighbour = |d: f64| solve(lossy(t + d), pol, window);
        let (error, largest) = against_pairings(&basis, &c, &neighbour, h);
        let (finer, _) = against_pairings(&basis, &c, &neighbour, h / 4.);
        eprintln!(
            "lossy {pol:?}, {} modes ({complex} complex): dβ/dt against the implicit slope \
             {diagonal_error:.1e}; pairings' central differences {error:.1e}, at h/4 {finer:.1e}, \
             of the largest coefficient {largest:.3}",
            basis.found()
        );
        assert!(diagonal_error < 1e-7);
        assert!(error < 1e-4 && (finer < error / 3. || finer < 1e-7));
    }
}
