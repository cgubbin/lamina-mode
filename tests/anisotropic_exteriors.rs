//! Open exteriors that couple TE and TM are refused before any guided search;
//! separable ones (isotropic, diagonal, an optic axis in the xz plane) are
//! counted as before.
//!
//! An exterior whose ε or μ has an `xy`, `yx`, `yz` or `zy` entry can make
//! the `(E_y, Z₀H_y)` reference basis of its incoming waves lose rank at a
//! real β above the light line. The characteristic then vanishes there for
//! every stack, and the zero was reported as a guided mode. The fixtures are
//! a uniaxial crystal (ε_o = 2.25, ε_e = 2.56) with its optic axis in the
//! interface plane at angle θ to the propagation direction, against an
//! isotropic half-space with ε = 2.40.
use lamina_core::{
    AnalyticalTensorMaterialHandle, Channel, Constant, ConstitutiveClass, ConstitutiveError,
    Polarisation, Rotation, Scatter2, Scatter4, Stack, StackRegion, Transfer4,
};
use lamina_materials::PrincipalAxes;
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::{FRAC_PI_2, TAU};

const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;
/// k0 at a 1 cm wavelength, in cm⁻¹.
const K0: f64 = TAU;
const ORDINARY: f64 = 2.25;
const EXTRAORDINARY: f64 = 2.56;
const PARTNER: f64 = 2.40;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn window(family: ModeFamily) -> GuidedWindow {
    GuidedWindow {
        wavelength: cm(1.0),
        family,
        direction: GuidedDirection::Forward,
        abs_beta: [0.0, 3.0 * K0],
        cutoff_margin: 1e-3 * K0,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}

/// The crystal with its optic axis at polar angle `theta` from z and
/// azimuth `phi` from x.
fn crystal(theta: f64, phi: f64) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ORDINARY),
            Constant::dielectric(EXTRAORDINARY),
        )),
        Rotation::from_optic_axis(theta, phi),
    )
}

/// The crystal with its optic axis in the interface plane at `degrees` to
/// the propagation direction x.
fn in_plane(degrees: f64) -> Region {
    crystal(FRAC_PI_2, degrees.to_radians())
}

fn isotropic(epsilon: f64) -> Region {
    lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(epsilon)).into()
}

fn betas(basis: &GuidedBasis) -> Vec<f64> {
    basis
        .modes
        .iter()
        .map(|m| m.physical_coordinate().re)
        .collect()
}

/// The refusal of an open exterior that couples TE and TM.
fn refused(region: StackRegion) -> impl Fn(&EnumerationError) -> bool {
    move |e| {
        matches!(
            e,
            EnumerationError::Constitutive(ConstitutiveError::Unsupported {
                region: r,
                class: ConstitutiveClass::Full,
            }) if *r == region
        )
    }
}

type Enumerate = fn(ModeStack, GuidedWindow) -> Result<GuidedBasis, EnumerationError>;

const ENUMERATIONS: [(&str, Enumerate); 3] = [
    ("default", |s, w| enumerate_guided_modes(s, w)),
    ("Scatter4", |s, w| {
        enumerate_guided_modes_with(Scatter4::new(), s, w)
    }),
    ("Transfer4", |s, w| {
        enumerate_guided_modes_with(Transfer4::new(), s, w)
    }),
];

#[test]
fn the_reported_spurious_dyakonov_cases_are_refused_on_every_backend() {
    // With a cutoff margin of 1e-6 k0, 0.2.0 reported as complete a mode at
    // ξ = 1.5732133 at 60°, where none exists, and a spurious mode beside the
    // true one at 45.9271° (ξ = 1.5493775) and at 45.85° (ξ = 1.5492434):
    // each a zero of the incoming waves' reference basis.
    let window = |family| GuidedWindow {
        cutoff_margin: 1e-6 * K0,
        ..window(family)
    };
    for degrees in [60.0, 45.9271, 45.85] {
        let left: ModeStack = Stack::builder(in_plane(degrees), isotropic(PARTNER)).finalise();
        let right: ModeStack = Stack::builder(isotropic(PARTNER), in_plane(degrees)).finalise();
        let layered: ModeStack = Stack::builder(in_plane(degrees), isotropic(PARTNER))
            .layer(isotropic(3.0), cm(0.6))
            .finalise();
        for (name, enumerate) in ENUMERATIONS {
            for (stack, region) in [
                (&left, StackRegion::LeftExterior),
                (&right, StackRegion::RightExterior),
                (&layered, StackRegion::LeftExterior),
            ] {
                let error = enumerate(stack.clone(), window(ModeFamily::All))
                    .expect_err(&format!("{degrees}° {name}: a non-separable exterior"));
                assert!(refused(region)(&error), "{degrees}° {name}: {error:?}");
                assert!(error.to_string().contains("not supported"), "{error}");
            }
        }
        // A single family is refused as before: TE and TM are not invariant.
        for pol in [TE, TM] {
            let error = enumerate_guided_modes(left.clone(), window(pol.into())).unwrap_err();
            assert!(
                matches!(
                    error,
                    EnumerationError::FamilyNotInvariant {
                        region: StackRegion::LeftExterior
                    }
                ),
                "{error:?}"
            );
        }
    }
}

#[test]
fn an_optic_axis_out_of_the_xz_plane_is_refused_with_a_finite_layer() {
    // A generic orientation and a tilt in the yz plane both couple TE and TM.
    for exterior in [crystal(0.6, 0.7), crystal(0.6, FRAC_PI_2)] {
        let stack: ModeStack = Stack::builder(isotropic(PARTNER), exterior)
            .layer(isotropic(4.0), cm(0.8))
            .finalise();
        for (name, enumerate) in ENUMERATIONS {
            let error = enumerate(stack.clone(), window(ModeFamily::All)).unwrap_err();
            assert!(
                refused(StackRegion::RightExterior)(&error),
                "{name}: {error:?}"
            );
        }
    }
}

#[test]
fn a_non_separable_finite_layer_between_isotropic_exteriors_is_still_counted() {
    // Only open exteriors are refused: a rotated film keeps its hybrid basis.
    let stack: ModeStack = Stack::builder(isotropic(1.0), isotropic(2.25))
        .layer(crystal(0.6, 0.7), cm(0.8))
        .finalise();
    let basis = enumerate_guided_modes(stack, window(ModeFamily::All)).unwrap();
    assert!(basis.is_complete(), "{:?}", basis.unresolved);
    assert!(!basis.modes.is_empty());
}

#[test]
fn in_plane_problems_refuse_a_non_separable_open_exterior_on_every_channel() {
    // The conformal in-plane chart needs isotropic exteriors on the coupled
    // channel and no full-class region on a scalar one.
    let stack: ModeStack = Stack::builder(in_plane(60.0), isotropic(PARTNER))
        .layer(isotropic(3.0), cm(0.6))
        .finalise();
    for channel in [Channel::Coupled, TE.into(), TM.into()] {
        for result in [
            PropagationProblem::with_backend(stack.clone(), K0, channel, Scatter4::new()).err(),
            PropagationProblem::with_backend(stack.clone(), K0, channel, Transfer4::new()).err(),
        ] {
            let diagnostic = result.expect("refused");
            assert!(
                matches!(
                    diagnostic.find_source::<ConstructionError>(),
                    Some(ConstructionError::Constitutive(
                        ConstitutiveError::Unsupported { .. }
                    ))
                ),
                "{channel:?}: {diagnostic:?}"
            );
        }
    }
}

#[test]
fn diagonal_exteriors_are_counted_and_equal_the_te_and_tm_union() {
    // An optic axis along x, y or z: the exterior is diagonal and TE/TM
    // separable, so its reference basis loses rank only on its light lines,
    // which the guided interval excludes. The coupled count must agree with
    // the scalar TE and TM counts of the 2×2 path.
    let along = [(FRAC_PI_2, 0.0), (FRAC_PI_2, FRAC_PI_2), (0.0, 0.0)];
    for (axis, (theta, phi)) in along.into_iter().enumerate() {
        let exterior = crystal(theta, phi);
        assert_eq!(exterior.class(), ConstitutiveClass::Diagonal);
        // Two half-spaces alone carry no surface wave.
        for stack in [
            Stack::builder(exterior.clone(), isotropic(PARTNER)).finalise(),
            Stack::builder(isotropic(PARTNER), exterior.clone()).finalise(),
        ] {
            for (name, enumerate) in ENUMERATIONS {
                let basis = enumerate(stack.clone(), window(ModeFamily::All)).unwrap();
                assert!(basis.is_complete(), "axis {axis} {name}: {basis:?}");
                assert!(basis.modes.is_empty(), "axis {axis} {name}: {basis:?}");
            }
        }
        let stack: ModeStack = Stack::builder(exterior, isotropic(PARTNER))
            .layer(isotropic(4.0), cm(0.8))
            .finalise();
        let mut union = vec![];
        for pol in [TE, TM] {
            let basis =
                enumerate_guided_modes_with(Scatter2::new(), stack.clone(), window(pol.into()))
                    .unwrap();
            assert!(
                basis.is_complete(),
                "axis {axis} {pol:?}: {:?}",
                basis.unresolved
            );
            union.extend(betas(&basis));
        }
        union.sort_by(|a, b| b.total_cmp(a));
        for (name, enumerate) in ENUMERATIONS {
            let all = enumerate(stack.clone(), window(ModeFamily::All)).unwrap();
            assert!(
                all.is_complete(),
                "axis {axis} {name}: {:?}",
                all.unresolved
            );
            let found = betas(&all);
            assert_eq!(
                found.len(),
                union.len(),
                "axis {axis} {name}: {found:?} {union:?}"
            );
            for (a, b) in found.iter().zip(&union) {
                assert!(
                    (a - b).abs() <= 1e-9 * b,
                    "axis {axis} {name}: {found:?} {union:?}"
                );
            }
        }
    }
}

#[test]
fn an_optic_axis_in_the_xz_plane_is_separable_and_still_counted() {
    // A full-class exterior without xy, yx, yz or zy entries.
    let exterior = crystal(0.6, 0.0);
    assert_eq!(exterior.class(), ConstitutiveClass::Full);
    assert!(exterior.is_separable());
    let stack: ModeStack = Stack::builder(exterior, isotropic(PARTNER))
        .layer(isotropic(4.0), cm(0.8))
        .finalise();
    let all = enumerate_guided_modes(stack.clone(), window(ModeFamily::All)).unwrap();
    assert!(all.is_complete(), "{:?}", all.unresolved);
    let counts = [TE, TM].map(|pol| {
        let basis = enumerate_guided_modes(stack.clone(), window(pol.into())).unwrap();
        assert!(basis.is_complete(), "{pol:?}: {:?}", basis.unresolved);
        basis.modes.len()
    });
    assert_eq!(all.modes.len(), counts.iter().sum::<usize>());
}
