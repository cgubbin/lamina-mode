#[path = "../examples/support/pole_material.rs"]
mod material;
#[path = "../examples/support/pole_green.rs"]
mod reference;
use lamina_core::{Constant, Polarisation, Stack};
use lamina_mode::*;
use lamina_units::Length;
use reference::{Green, c};
fn cavity() -> lamina_core::AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(4.), Length::centimetres(0.3))
        .finalise()
}
#[test]
fn independent_cavity_reference_and_positive_source_jump() {
    let stack = cavity();
    let k = reference::cavity_pole();
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let sig = BranchSignature {
            k0: k,
            k_parallel: c(0.),
            left: k,
            right: k,
        };
        let center = Green::new(&stack, sig, true, k, pol);
        assert!(center.wronskian().norm() < 1e-12);
        let z = 0.09;
        let zp = 0.21;
        let expected = center.right(z).field * center.right(zp).field / reference::cavity_norm(pol);
        let regular = Green::new(&stack, sig, true, k + c(0.05), pol);
        assert!((regular.source_flux_jump(z) - c(1.)).norm() < 1e-12);
        for off in [Complex::new(1e-5, 2e-5), Complex::new(-2e-6, 1e-6)] {
            let g = Green::new(&stack, sig, true, k + off, pol);
            assert!((off * g.value(z, zp) - expected).norm() < off.norm() * 10.);
        }
        let residue = reference::contour_residue(&stack, sig, true, pol, [z, zp], 0.01, 128);
        assert!((residue - expected).norm() < 1e-10);
        // residue transforms by c when changing from k0 to omega.
        let speed = 2.99792458e10;
        assert!((residue * speed / (expected * speed) - c(1.)).norm() < 1e-8);
    }
}

#[path = "../examples/support/pole_fixtures.rs"]
mod fixtures;
use lamina_core::{ExteriorSampling, FieldSampling, LayerSampling, Scatter2, Transfer2};
fn recovery() -> FieldRecoverySettings {
    FieldRecoverySettings {
        characteristic_limit: 1e-5,
        ..Default::default()
    }
}
fn variable(point: &ModePoint) -> PoleVariable {
    match point.provenance().direction() {
        SearchCoordinate::InPlane { .. } => PoleVariable::BetaAtFixedK0,
        _ => PoleVariable::K0AtFixedBeta,
    }
}
fn positions(point: &ModePoint, z: &[f64]) -> FieldSampling<f64> {
    let provenance = point.provenance();
    let stack = provenance.stack();
    let width: f64 = stack
        .layers_left_to_right()
        .iter()
        .map(|l| l.thickness().as_centimetres())
        .sum();
    let mut sampling = FieldSampling::new();
    for &x in z {
        if x <= 0. {
            sampling = sampling.left_exterior(ExteriorSampling::point(Length::centimetres(-x)));
        } else if x >= width {
            sampling =
                sampling.right_exterior(ExteriorSampling::point(Length::centimetres(x - width)));
        } else {
            let mut left = 0.;
            for (i, l) in stack.layers_left_to_right().iter().enumerate() {
                let d = l.thickness().as_centimetres();
                if x <= left + d {
                    sampling =
                        sampling.layer(i, LayerSampling::point(Length::centimetres(x - left)));
                    break;
                }
                left += d;
            }
        }
    }
    sampling
}
fn primary(fields: &ModeFields) -> &ndarray::Array1<Complex> {
    if fields.point.provenance().polarisation() == Polarisation::TransverseElectric {
        fields.fields.quantity().electric().y()
    } else {
        fields.fields.quantity().magnetic().y()
    }
}
#[test]
fn complete_norm_predicts_independent_residues_and_invariances() {
    for (name, point) in fixtures::cases() {
        let point = point.unwrap_or_else(|e| panic!("{name}: {e}"));
        let sig = *point.signature().unwrap();
        let pol = point.provenance().polarisation();
        let v = variable(&point);
        let spectral = v == PoleVariable::K0AtFixedBeta;
        let scale = 1. / sig.k0.norm().max(sig.k_parallel.norm());
        let width: f64 = point
            .provenance()
            .stack()
            .layers_left_to_right()
            .iter()
            .map(|l| l.thickness().as_centimetres())
            .sum();
        let z = [-0.17 * scale, 0.23 * width, width + 0.13 * scale];
        let sampling = positions(&point, &z);
        let mode = point
            .normalize_with(
                Transfer2::new(),
                v,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                Default::default(),
                recovery(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let fields = mode.fields(&sampling).unwrap();
        let psi = primary(&fields);
        let predicted = psi[0] * psi[2];
        let center = if spectral { sig.k0 } else { sig.k_parallel };
        let radius = center.norm() * 1e-4;
        let provenance = point.provenance();
        let stack = provenance.stack();
        let direct =
            reference::contour_residue(stack, sig, spectral, pol, [z[0], z[2]], radius, 128);
        let refined =
            reference::contour_residue(stack, sig, spectral, pol, [z[0], z[2]], radius, 256);
        let error = (predicted - direct).norm() / direct.norm();
        eprintln!(
            "{name}: residue error {error:e}, condition {}",
            mode.normalization.condition
        );
        assert!(error < 1e-7, "{name}: {predicted} != {direct}");
        assert!((direct - refined).norm() / direct.norm() < 1e-8);
        for direction in [Complex::new(1., 0.4), Complex::new(-0.3, 1.)] {
            let mut errors = Vec::new();
            for fraction in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8] {
                let offset = direction * center.norm() * fraction;
                let g = Green::new(stack, sig, spectral, center + offset, pol);
                errors.push((offset * g.value(z[0], z[2]) - direct).norm() / direct.norm());
            }
            let best = errors.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(best < 1e-5 && best < errors[0] * 0.01, "{name}: {errors:?}");
        }
        let scatter = point
            .normalize_with(
                Scatter2::new(),
                v,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                Default::default(),
                recovery(),
            )
            .unwrap();
        let sf = scatter.fields(&sampling).unwrap();
        let sp = primary(&sf);
        assert!(
            (sp[0] * sp[2] - predicted).norm() / predicted.norm() < 1e-7,
            "{name}: backend disagreement"
        );
        for amplitude in [Complex::new(0.7, -0.8), Complex::new(-0.4, 0.9)] {
            let n = point
                .normalize_with(
                    Transfer2::new(),
                    v,
                    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                    PoleNormalizationSettings {
                        amplitude,
                        ..Default::default()
                    },
                    recovery(),
                )
                .unwrap();
            let f = n.fields(&sampling).unwrap();
            let p = primary(&f);
            assert!((p[0] * p[2] - predicted).norm() / predicted.norm() < 1e-8);
            let sign = if (p[0] - psi[0]).norm() < (p[0] + psi[0]).norm() {
                1.
            } else {
                -1.
            };
            for (a, b) in p.iter().zip(psi) {
                assert!((*a - sign * b).norm() < 1e-7 * (1. + a.norm() + b.norm()));
            }
        }
        let shifted = point
            .normalize_with(
                Transfer2::new(),
                v,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                PoleNormalizationSettings {
                    exterior_offsets_cm: [0.11 * scale, 0.23 * scale],
                    ..Default::default()
                },
                recovery(),
            )
            .unwrap();
        assert!(
            (shifted.normalization.total - mode.normalization.total).norm()
                / mode.normalization.total.norm()
                < 1e-8,
            "{name}: partition"
        );
        let wrong = if spectral {
            PoleVariable::BetaAtFixedK0
        } else {
            PoleVariable::K0AtFixedBeta
        };
        assert!(matches!(
            point
                .normalize_with(
                    Transfer2::new(),
                    wrong,
                    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                    Default::default(),
                    recovery()
                )
                .unwrap_err()
                .cause,
            NormalizationError::IncompatibleVariable
        ));
        let failed = point
            .normalize_with(
                Transfer2::new(),
                v,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                PoleNormalizationSettings {
                    factor_atol: 1e99,
                    ..Default::default()
                },
                recovery(),
            )
            .unwrap_err();
        assert_eq!(failed.point.native_coordinate(), point.native_coordinate());
        assert!(matches!(
            failed.cause,
            NormalizationError::Core(lamina_core::PoleNormalizationError::Unavailable { .. })
        ));
    }
}

#[test]
fn normalization_failures_are_separate_and_source_preserving() {
    let point = fixtures::cavity(lamina_core::Polarisation::TransverseElectric, true).unwrap();
    for settings in [
        PoleNormalizationSettings {
            amplitude: Complex::new(f64::NAN, 0.),
            ..Default::default()
        },
        PoleNormalizationSettings {
            kappa_cutoff: 1e12,
            ..Default::default()
        },
        PoleNormalizationSettings {
            max_condition: 1.,
            ..Default::default()
        },
    ] {
        let failure = point
            .normalize_with(
                lamina_core::Transfer2::new(),
                PoleVariable::K0AtFixedBeta,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                settings,
                recovery(),
            )
            .unwrap_err();
        assert_eq!(failure.point.native_coordinate(), point.native_coordinate());
        assert!(matches!(failure.cause, NormalizationError::Core(_)));
    }
    let sampling = FieldSampling::new().left_exterior(ExteriorSampling::point(Length::zero()));
    assert!(
        point
            .recover_fields_with(lamina_core::Scatter2::new(), &sampling, recovery())
            .is_ok()
    );
}
