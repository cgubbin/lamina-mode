//! Guided modes of stacks with full-tensor regions: counts by the argument
//! principle of the coupled characteristic, labels, separable consistency,
//! isotropic recovery and uniaxial benchmarks.
use lamina_core::{
    AnalyticalTensorMaterialHandle, Constant, Polarisation, Rotation, Scatter4, Stack, Transfer4,
};
use lamina_materials::PrincipalAxes;
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::TAU;

const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;
/// k0 at a 1 cm wavelength, in cm⁻¹.
const K0: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn window(family: ModeFamily, abs_beta: [f64; 2]) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.0),
        family,
        direction: GuidedDirection::Forward,
        abs_beta,
        cutoff_margin: 1e-3 * K0,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

/// A uniaxial region with its optic axis at polar angle `theta` from z and
/// azimuth `phi` from x.
fn uniaxial(ordinary: f64, extraordinary: f64, theta: f64, phi: f64) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        Rotation::from_optic_axis(theta, phi),
    )
}

fn isotropic(epsilon: f64, mu: f64) -> Region {
    lamina_core::AnalyticalMaterialHandle::new(Constant::magnetodielectric(epsilon, mu)).into()
}

fn betas(basis: &GuidedBasis) -> Vec<f64> {
    basis
        .modes
        .iter()
        .map(|m| m.physical_coordinate().re)
        .collect()
}

fn backends() -> [(
    &'static str,
    Box<dyn Fn(ModeStack, GuidedWindow) -> GuidedBasis>,
); 2] {
    [
        (
            "Scatter4",
            Box::new(|s, w| enumerate_guided_modes_with(Scatter4::new(), s, w).unwrap()),
        ),
        (
            "Transfer4",
            Box::new(|s, w| enumerate_guided_modes_with(Transfer4::new(), s, w).unwrap()),
        ),
    ]
}

/// Every mode of `basis` matches one of `reference` to `tol` relative, in order.
fn same_ladder(basis: &[f64], reference: &[f64], tol: f64) -> f64 {
    assert_eq!(basis.len(), reference.len(), "{basis:?} {reference:?}");
    basis
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .inspect(|d| assert!(*d < tol, "{basis:?} {reference:?}"))
        .fold(0.0, f64::max)
}

/// The TE and TM modes of a separable stack through the 2×2 scattering
/// backend, merged in decreasing β with their labels.
fn union(stack: &ModeStack, abs_beta: [f64; 2]) -> (Vec<f64>, Vec<ModeLabel>) {
    let mut modes = vec![];
    for pol in [TE, TM] {
        let basis = enumerate_guided_modes_with(
            lamina_core::Scatter2::new(),
            stack.clone(),
            window(pol.into(), abs_beta),
        )
        .unwrap();
        assert!(basis.is_complete());
        modes.extend(betas(&basis).into_iter().map(|b| (b, ModeLabel::from(pol))));
    }
    modes.sort_by(|a, b| b.0.total_cmp(&a.0));
    modes.into_iter().unzip()
}

#[test]
fn every_mode_of_an_isotropic_stack_is_the_union_of_te_and_tm() {
    let stack: ModeStack = Stack::builder(isotropic(1.0, 1.0), isotropic(2.25, 1.0))
        .layer(isotropic(4.0, 1.0), cm(0.9))
        .layer(isotropic(3.0, 1.3), cm(0.4))
        .finalise();
    for (name, enumerate) in backends() {
        let basis = enumerate(stack.clone(), window(ModeFamily::All, [0.0, 3.0 * K0]));
        let [lo, hi] = basis.eligible.unwrap();
        let (reference, labels) = union(&stack, [lo, hi]);
        assert!(basis.is_complete(), "{name}: {:?}", basis.unresolved);
        let worst = same_ladder(&betas(&basis), &reference, 1e-10);
        assert_eq!(basis.labels, labels);
        let evidence = basis.hybrid.as_ref().unwrap();
        let counts = evidence.separable_counts.unwrap();
        assert_eq!(
            counts.iter().flatten().sum::<usize>(),
            basis.expected.unwrap()
        );
        let off = evidence.off_family.iter().fold(0.0, |a: f64, b| a.max(*b));
        println!(
            "{name}: {} modes, counts {counts:?}, largest relative |Δβ| = {worst:e}, \
             largest off-family part = {off:e}, {} of {} cells counted by Transfer4",
            reference.len(),
            evidence
                .count_backends
                .iter()
                .filter(|b| b.contains("Transfer4"))
                .count(),
            evidence.count_backends.len()
        );
    }
}

#[test]
fn a_rotated_uniaxial_film_has_a_complete_hybrid_basis() {
    let stack = Stack::builder(isotropic(1.0, 1.0), isotropic(2.1, 1.0))
        .layer(uniaxial(4.0, 6.25, 0.6, 0.7), cm(0.8))
        .finalise();
    for backend in ["scatter4", "transfer4"] {
        let w = window(ModeFamily::All, [0.0, 3.0 * K0]);
        let basis = match backend {
            "scatter4" => enumerate_guided_modes_with(Scatter4::new(), stack.clone(), w),
            _ => enumerate_guided_modes_with(Transfer4::new(), stack.clone(), w),
        }
        .unwrap();
        println!(
            "{backend}: {:?} {:?} {:?}",
            basis.expected,
            betas(&basis),
            basis.unresolved
        );
        println!("{:?}", basis.hybrid);
        assert!(basis.is_complete());
        assert!(basis.labels.iter().all(|l| *l == ModeLabel::Hybrid));
    }
}
