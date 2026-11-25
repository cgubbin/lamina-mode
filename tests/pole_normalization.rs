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
            left: Some(k),
            right: Some(k),
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
        let stack = &reference::scalar_stack(provenance.stack());
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

fn check_scattering_contours(residues: &QnmScatteringResidue) {
    use lamina_core::{
        CanonicalCoordinates, ComplexPlaneEvaluator, EvaluateMeromorphicMaterial,
        ExteriorWavevectors, IncidentSide, ModeJet1, Scalar, Scatter2, SeedJet,
    };
    use ndarray::{Ix0, arr0};
    use std::f64::consts::TAU;
    type J = ModeJet1<Complex, Ix0>;
    let seed = |value| <J as SeedJet>::constant(arr0(value));
    let sig = residues.signature;
    let provenance = residues.point.provenance();
    let pol = provenance.polarisation();
    let stack = &reference::scalar_stack(provenance.stack());
    let solver = ComplexPlaneEvaluator::<J, _, _>::compile(stack, Scatter2::new()).unwrap();
    let mut max_error = 0.0_f64;
    for (radius, count) in [(0.03, 64), (0.015, 128)] {
        let mut contour = [[Complex::ZERO; 2]; 2];
        for s in 0..count {
            let delta = radius * (Complex::I * (TAU * s as f64 / count as f64)).exp();
            let k = sig.k0 + delta;
            let [left, right] = [stack.left_exterior(), stack.right_exterior()].map(|m| {
                (m.evaluate_relative_permittivity_complex(Scalar(k))
                    * m.evaluate_relative_permeability_complex(Scalar(k))
                    * k
                    * k
                    - sig.k_parallel * sig.k_parallel)
                    .sqrt()
            });
            let root = |candidate: Complex, reference: Complex| {
                if (candidate - reference).norm() < (candidate + reference).norm() {
                    candidate
                } else {
                    -candidate
                }
            };
            let result = solver
                .solve(
                    &CanonicalCoordinates::new(seed(k), seed(sig.k_parallel)),
                    &ExteriorWavevectors::new(
                        seed(root(left, sig.left.unwrap())),
                        seed(root(right, sig.right.unwrap())),
                    ),
                    pol,
                )
                .unwrap();
            let l = result.amplitudes(IncidentSide::Left);
            let r = result.amplitudes(IncidentSide::Right);
            let entries = [
                [l.reflection(), r.transmission()],
                [l.transmission(), r.reflection()],
            ];
            for i in 0..2 {
                for j in 0..2 {
                    contour[i][j] += entries[i][j].value()[()] * delta / count as f64;
                }
            }
        }
        for (predicted, observed) in residues
            .residue_per_cm
            .iter()
            .flatten()
            .zip(contour.iter().flatten())
        {
            max_error = max_error.max((*predicted - *observed).norm());
            assert!(
                (*predicted - *observed).norm() < 1e-7,
                "{pol:?}: predicted={predicted}, contour={observed}"
            );
        }
    }
    let [[a, b], [c, d]] = residues.residue_per_cm;
    assert!((a * d - b * c).norm() < 1e-10);
    let [left, right] = residues.exterior_admittances_per_cm;
    assert!((b / right - c / left).norm() < 1e-10);
    println!("{pol:?} contour residue max error: {max_error:e}");
}

#[test]
fn normalized_cavity_scattering_residue_matches_direct_contours() {
    use lamina_core::Scatter2;
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let point = fixtures::cavity(pol, true).unwrap();
        let mode = point
            .normalize_with(
                Scatter2::new(),
                PoleVariable::K0AtFixedBeta,
                fixtures::fields::ASSUME,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let residues = mode.scattering_residue().unwrap();
        check_scattering_contours(&residues);
    }
}

#[test]
fn qnm_windows_reconstruct_lossless_symmetric_and_oblique_asymmetric_slabs() {
    use lamina_core::Scatter2;
    for (right_epsilon, beta, pol) in [
        (1.0, 0.0, Polarisation::TransverseElectric),
        (2.25, 1.3, Polarisation::TransverseMagnetic),
    ] {
        let stack = Stack::from_analytical_materials(
            Constant::vacuum(),
            Constant::dielectric(right_epsilon),
        )
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise();
        let determinant = |k: Complex| {
            let [left, middle, right] =
                [1.0, 4.0, right_epsilon].map(|e| (e * k * k - beta * beta).sqrt());
            let factor = |e| {
                if pol == Polarisation::TransverseMagnetic {
                    e
                } else {
                    1.0
                }
            };
            let [a, b, c] = [
                left / factor(1.0),
                middle / factor(4.0),
                right / factor(right_epsilon),
            ];
            (b + a) * (b + c) - (b - a) * (b - c) * (2.0 * Complex::I * middle * 0.3).exp()
        };
        let mut target = Complex::new(5.25, -2.0);
        for _ in 0..12 {
            let derivative = (determinant(target + 1e-5) - determinant(target - 1e-5)) / 2e-5;
            target -= determinant(target) / derivative;
        }
        assert!(determinant(target).norm() < 1e-9);
        let point = fixtures::frequency_at_beta(stack, target, beta, pol).unwrap();
        assert!((point.physical_coordinate() - target).norm() < 1e-8);
        let mode = point
            .normalize_with(
                Scatter2::new(),
                PoleVariable::K0AtFixedBeta,
                fixtures::fields::ASSUME,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let residue = mode.scattering_residue().unwrap();
        check_scattering_contours(&residue);
        let channels = QnmConstantChannels {
            epsilon: [1.0, right_epsilon],
            mu: [1.0; 2],
        };
        let settings = QnmWindowSettings {
            k0_per_cm: [4.2, 6.3],
            regular_degree: 10,
            validation_points: 103,
            amplitude_atol: 1e-7,
        };
        let result =
            reconstruct_qnm_scattering_window(vec![residue.clone()], channels, settings).unwrap();
        println!("{pol:?}: pole={target}, {:?}", result.evidence);
        assert!(result.evidence.meets_amplitude_tolerance);
        assert!(result.evidence.held_out_max_flux_balance_defect < 5e-7);
        assert!(result.amplitudes(4.1).is_none());
        let power = result.power_ratios(5.25).unwrap();
        assert!((power[0][0] + power[1][0] - 1.0).abs() < 5e-7);
        assert!((power[0][1] + power[1][1] - 1.0).abs() < 5e-7);
        println!("R/T at k0=5.25: {power:?}");
        let coarse = reconstruct_qnm_scattering_window(
            vec![residue.clone()],
            channels,
            QnmWindowSettings {
                regular_degree: 0,
                ..settings
            },
        )
        .unwrap();
        assert!(!coarse.evidence.meets_amplitude_tolerance);
        if beta > 0.0 {
            let cutoff = reconstruct_qnm_scattering_window(
                vec![residue],
                channels,
                QnmWindowSettings {
                    k0_per_cm: [beta, 6.3],
                    ..settings
                },
            );
            assert!(matches!(cutoff, Err(QnmWindowError::ExteriorThreshold)));
        }
    }
}

#[test]
fn compare_equal_budget_poles_and_contour_reconstructions() {
    use lamina_core::Scatter2;
    for (right_epsilon, beta, pol) in [
        (1.0, 0.0, Polarisation::TransverseElectric),
        (2.25, 1.3, Polarisation::TransverseMagnetic),
    ] {
        let stack = Stack::from_analytical_materials(
            Constant::vacuum(),
            Constant::dielectric(right_epsilon),
        )
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
        .finalise();
        let determinant = |k: Complex| {
            let [a, b, c] = [1.0, 4.0, right_epsilon].map(|e| {
                (e * k * k - beta * beta).sqrt()
                    / if pol == Polarisation::TransverseMagnetic {
                        e
                    } else {
                        1.0
                    }
            });
            (b + a) * (b + c)
                - (b - a) * (b - c) * (0.6 * Complex::I * (4.0 * k * k - beta * beta).sqrt()).exp()
        };
        let pole = |order: f64| {
            let mut target = Complex::new(order * std::f64::consts::PI / 0.6, -2.0);
            for _ in 0..12 {
                let derivative = (determinant(target + 1e-5) - determinant(target - 1e-5)) / 2e-5;
                target -= determinant(target) / derivative;
            }
            let point = fixtures::frequency_at_beta(stack.clone(), target, beta, pol).unwrap();
            point
                .normalize_with(
                    Scatter2::new(),
                    PoleVariable::K0AtFixedBeta,
                    fixtures::fields::ASSUME,
                    Default::default(),
                    Default::default(),
                )
                .unwrap()
                .scattering_residue()
                .unwrap()
        };
        let poles = vec![pole(1.0), pole(2.0)];
        let channels = QnmConstantChannels {
            epsilon: [1.0, right_epsilon],
            mu: [1.0; 2],
        };
        let settings = QnmWindowSettings {
            k0_per_cm: [3.2, 12.0],
            regular_degree: 10,
            validation_points: 129,
            amplitude_atol: 1e-7,
        };
        let reference = &poles[0].point;
        let direct =
            reconstruct_direct_scattering_window(&stack, beta, pol, channels, settings).unwrap();
        let one =
            reconstruct_qnm_scattering_window(poles[..1].to_vec(), channels, settings).unwrap();
        let two = reconstruct_qnm_scattering_window(poles.clone(), channels, settings).unwrap();
        assert!(
            two.evidence.held_out_max_amplitude_error
                < 0.01 * direct.evidence.held_out_max_amplitude_error
        );
        assert!(
            two.evidence.held_out_max_amplitude_error < one.evidence.held_out_max_amplitude_error
        );
        for (count, model) in [direct, one, two].iter().enumerate() {
            assert_eq!(model.evidence.direct_construction_samples, 22);
            println!("BUDGET {pol:?} {count} {:?}", model.evidence);
        }
        let contour = |samples| {
            reconstruct_qnm_scattering_contour(
                reference,
                poles.clone(),
                channels,
                settings,
                QnmContourSettings {
                    radius_per_cm: 5.1,
                    samples,
                },
            )
            .unwrap()
        };
        let coarse = contour(64);
        let fine = contour(256);
        for model in [&coarse, &fine] {
            println!("CAUCHY {pol:?} {:?}", model.evidence);
        }
        assert!(fine.evidence.meets_amplitude_tolerance);
        assert!(
            fine.evidence.held_out_max_amplitude_error
                < coarse.evidence.held_out_max_amplitude_error
        );
        assert!(fine.evidence.contour_max_refinement_change.unwrap() < 1e-7);
        assert!(fine.evidence.held_out_max_unitarity_defect < 1e-7);
        assert!(fine.evidence.held_out_max_reciprocity_defect < 1e-7);
        assert!(fine.evidence.held_out_max_passivity_excess < 1e-7);
        assert!(fine.evidence.held_out_max_phase_error_rad < 1e-5);
        let missing = reconstruct_qnm_scattering_contour(
            reference,
            poles[..1].to_vec(),
            channels,
            settings,
            QnmContourSettings {
                radius_per_cm: 5.1,
                samples: 256,
            },
        )
        .unwrap();
        assert!(!missing.evidence.meets_amplitude_tolerance);
    }
}

#[test]
fn dispersive_exterior_window_uses_one_outgoing_sheet() {
    use lamina_core::Scatter2;
    let beta = 1.2;
    let pol = Polarisation::TransverseElectric;
    let epsilon_right = |k: Complex| 2.1 + 0.025 * k;
    let stack = Stack::from_analytical_materials(
        Constant::vacuum(),
        material::linear(2.1, 0.025, 1.0, 0.0),
    )
    .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.3))
    .finalise();
    let factors = |k: Complex| {
        [1.0.into(), 4.0.into(), epsilon_right(k)]
            .map(|e: Complex| (e * k * k - beta * beta).sqrt())
    };
    let determinant = |k: Complex| {
        let [a, b, c] = factors(k);
        (b + a) * (b + c) - (b - a) * (b - c) * (0.6 * Complex::I * b).exp()
    };
    let mut target = Complex::new(5.25, -2.5);
    for _ in 0..12 {
        let derivative = (determinant(target + 1e-5) - determinant(target - 1e-5)) / 2e-5;
        target -= determinant(target) / derivative;
    }
    let point = fixtures::frequency_at_beta(stack, target, beta, pol).unwrap();
    let pole = point
        .normalize_with(
            Scatter2::new(),
            PoleVariable::K0AtFixedBeta,
            fixtures::fields::ASSUME,
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .scattering_residue()
        .unwrap();
    check_scattering_contours(&pole);
    let domain = FrequencyDomainInput {
        bounds: FrequencyRegion::new(2.1, 8.4, -3.1, 3.1).unwrap(),
        reference: FrequencyReference::RealRadiation {
            frequency: 5.25,
            choices: [RadiationChoice::OutgoingOrDecaying; 2],
            tolerance: Default::default(),
        },
        continuation: Default::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    };
    let settings = QnmWindowSettings {
        k0_per_cm: [4.2, 6.3],
        regular_degree: 10,
        validation_points: 103,
        amplitude_atol: 1e-7,
    };
    let contour = QnmContourSettings {
        radius_per_cm: 2.9,
        samples: 128,
    };
    let solve = |pole: QnmScatteringResidue, domain, contour| {
        reconstruct_dispersive_qnm_scattering(
            &point,
            vec![pole],
            domain,
            fixtures::fields::ASSUME,
            settings,
            contour,
        )
    };
    for method in [None, Some(contour)] {
        let reconstruction = solve(pole.clone(), domain.clone(), method).unwrap();
        assert!(reconstruction.constant_channels().is_none());
        assert_eq!(
            reconstruction
                .continuation_evidence()
                .unwrap()
                .reference
                .frequency,
            5.25.into()
        );
        assert!(reconstruction.evidence.meets_amplitude_tolerance);
        assert!(reconstruction.evidence.held_out_max_unitarity_defect < 1e-7);
        let mut analytic_error = 0.0_f64;
        for i in 0..=40 {
            let k = if i == 40 {
                6.3
            } else {
                4.2 + (6.3 - 4.2) * i as f64 / 40.0
            };
            let [a, b, c] = factors(k.into());
            let phase = (Complex::I * 0.3 * b).exp();
            let r01 = (a - b) / (a + b);
            let r12 = (b - c) / (b + c);
            let denominator = 1.0 + r01 * r12 * phase * phase;
            let expected_r = (r01 + r12 * phase * phase) / denominator;
            let expected_t = (2.0 * a / (a + b)) * (2.0 * b / (b + c)) * phase / denominator;
            let result = reconstruction.amplitudes(k).unwrap();
            analytic_error = analytic_error
                .max((result[0][0] - expected_r).norm())
                .max((result[1][0] - expected_t).norm());
            let flux = reconstruction.power_ratios(k).unwrap();
            assert!((flux[0][0] + flux[1][0] - 1.0).abs() < 1e-7);
        }
        println!(
            "DISPERSIVE contour={} pole={target} analytic_error={analytic_error:e} {:?}",
            method.is_some(),
            reconstruction.evidence
        );
        assert!(analytic_error < 1e-7);
    }
    let mut opposite = pole.clone();
    opposite.signature.right = opposite.signature.right.map(|k| -k);
    assert!(matches!(
        solve(opposite, domain.clone(), None),
        Err(QnmWindowError::ChannelMismatch)
    ));
    let mut excluded = domain.clone();
    excluded.exclusions.push(FrequencyExclusion {
        region: FrequencyRegion::new(4.8, 5.0, -0.1, 0.1).unwrap(),
        reason: ExclusionReason::ExteriorThreshold { side: 1 },
    });
    assert!(matches!(
        solve(pole.clone(), excluded, None),
        Err(QnmWindowError::Continuation(FrequencyError::Excluded(_)))
    ));
    let mut narrow = domain;
    narrow.bounds = FrequencyRegion::new(3.5, 7.0, -3.1, 3.1).unwrap();
    assert!(matches!(
        solve(pole, narrow, Some(contour)),
        Err(QnmWindowError::ContinuationDomain)
    ));
}

#[test]
fn absorbing_slab_scattering_is_passive_without_unitarity() {
    use lamina_core::Scatter2;
    let e = Complex::new(4.0, 0.12);
    let n = e.sqrt();
    let reflection = (n - 1.0) / (n + 1.0);
    // This branch of the analytic slab pole lies near the first positive resonance.
    let target = (std::f64::consts::PI + Complex::I * reflection.ln()) / (0.3 * n);
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(
            material::Frozen {
                epsilon: e,
                mu: Complex::ONE,
            },
            Length::centimetres(0.3),
        )
        .finalise();
    let point = fixtures::frequency(stack, target, Polarisation::TransverseElectric).unwrap();
    let pole = point
        .normalize_with(
            Scatter2::new(),
            PoleVariable::K0AtFixedBeta,
            fixtures::fields::ASSUME,
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .scattering_residue()
        .unwrap();
    let model = reconstruct_qnm_scattering_window(
        vec![pole],
        QnmConstantChannels {
            epsilon: [1.0; 2],
            mu: [1.0; 2],
        },
        QnmWindowSettings {
            k0_per_cm: [4.2, 6.3],
            regular_degree: 10,
            validation_points: 103,
            amplitude_atol: 1e-7,
        },
    )
    .unwrap();
    let flux = model.power_ratios(5.25).unwrap();
    let absorbed = 1.0 - flux[0][0] - flux[1][0];
    println!("ABSORBING loss={absorbed:e} {:?}", model.evidence);
    assert!(model.evidence.meets_amplitude_tolerance);
    assert!(model.evidence.held_out_max_passivity_excess < 1e-9);
    assert!(model.evidence.held_out_max_unitarity_defect > 1e-2);
    assert!(absorbed > 1e-2);
}

fn lossy_sweep_stack(
    core_width: f64,
    core_epsilon: f64,
) -> lamina_core::AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.44))
        .analytical_layer(
            material::Frozen {
                epsilon: Complex::new(core_epsilon, 0.12),
                mu: Complex::ONE,
            },
            Length::centimetres(core_width),
        )
        .analytical_layer(Constant::dielectric(2.3), Length::centimetres(0.07))
        .finalise()
}

fn lossy_sweep_request() -> QnmPoleRequest {
    QnmPoleRequest {
        beta_per_cm: 0.0,
        polarisation: Polarisation::TransverseElectric,
        domain: FrequencyDomainInput {
            bounds: FrequencyRegion::new(3.0, 11.6, -4.2, 0.35).unwrap(),
            reference: FrequencyReference::RealRadiation {
                frequency: 5.0,
                choices: [RadiationChoice::OutgoingOrDecaying; 2],
                tolerance: Default::default(),
            },
            continuation: Default::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        assumption: fixtures::fields::ASSUME,
        discovery: Default::default(),
        refinement: Default::default(),
        matching: Default::default(),
        normalization: Default::default(),
        recovery: Default::default(),
    }
}

#[test]
fn lossy_multilayer_poles_are_selected_tracked_and_audited() {
    let request = lossy_sweep_request();
    let source = select_qnm_poles(lossy_sweep_stack(0.30, 4.0), request.clone()).unwrap();
    println!(
        "QNM selection count={:?}, coverage={:?}, poles={:?}",
        source.winding_count,
        source.coverage,
        source
            .poles
            .iter()
            .map(|p| p.signature.k0)
            .collect::<Vec<_>>()
    );
    assert_eq!(source.coverage, QnmPoleCoverage::Reconciled);
    assert_eq!(source.poles.len(), 2);
    verify_lossy_design_sensitivities(&source);
    let step = |radius, audit| {
        track_qnm_poles(
            &source,
            lossy_sweep_stack(0.302, 4.0),
            request.clone(),
            QnmPoleTrackingSettings {
                match_radius_per_cm: radius,
                audit_coverage: audit,
            },
        )
        .unwrap()
    };
    let tracked = step(0.2, false);
    let audit = step(0.2, true);
    assert_eq!(tracked.destination.coverage, QnmPoleCoverage::TrackedOnly);
    assert_eq!(audit.destination.coverage, QnmPoleCoverage::Reconciled);
    assert_eq!(tracked.matches, [(0, 0), (1, 1)]);
    assert_eq!(tracked.matches, audit.matches);
    for (a, b) in tracked
        .destination
        .poles
        .iter()
        .zip(&audit.destination.poles)
    {
        println!(
            "source tracked={} independent={}",
            a.signature.k0, b.signature.k0
        );
        assert!((a.signature.k0 - b.signature.k0).norm() < 1e-8);
    }
    let refreshed = step(1e-4, false);
    assert_eq!(refreshed.failed_local_sources, [0, 1]);
    assert_eq!(refreshed.destination.coverage, QnmPoleCoverage::Reconciled);
    assert_eq!(refreshed.unmatched_source, [0, 1]);
    let mut limited = request;
    limited.discovery.max_iterations = 1;
    let partial = select_qnm_poles(lossy_sweep_stack(0.30, 4.0), limited).unwrap();
    assert_eq!(partial.coverage, QnmPoleCoverage::Unresolved);
}

fn direct_lossy_cavity(k: Complex, width: f64, index: f64) -> (Complex, Complex, Complex) {
    let n = [
        Complex::ONE,
        Complex::new(index * index, 0.12).sqrt(),
        Complex::from(2.3_f64.sqrt()),
        Complex::from(1.2),
    ];
    let r = |i: usize| (n[i] - n[i + 1]) / (n[i] + n[i + 1]);
    let t = |i: usize| 2.0 * n[i] / (n[i] + n[i + 1]);
    let p = (Complex::I * k * n[1] * width).exp();
    let q = (Complex::I * k * n[2] * 0.07).exp();
    let tail_denom = 1.0 + r(1) * r(2) * q * q;
    let tail_numer = r(1) + r(2) * q * q;
    let denominator = tail_denom + r(0) * tail_numer * p * p;
    (
        (r(0) * tail_denom + tail_numer * p * p) / denominator,
        t(0) * t(1) * t(2) * p * q / denominator,
        denominator,
    )
}

fn verify_lossy_design_sensitivities(source: &QnmPoleSet) {
    for (label, width_rate, index_rate) in [("width", 1.0, 0.0), ("index", 0.0, 1.0)] {
        let width = |offset| 0.30 + width_rate * offset;
        let index = |offset: f64| 2.0 + index_rate * offset;
        let h = 2e-4;
        let model = differentiate_qnm_design(
            source,
            |offset| lossy_sweep_stack(width(offset), index(offset).powi(2)),
            QnmDesignSettings {
                step: h,
                match_radius_per_cm: 0.20,
                channels: QnmConstantChannels {
                    epsilon: [1.0, 1.44],
                    mu: [1.0; 2],
                },
                window: QnmWindowSettings {
                    k0_per_cm: [3.2, 11.3],
                    regular_degree: 18,
                    validation_points: 65,
                    amplitude_atol: 2e-8,
                },
            },
        )
        .unwrap();
        assert!(
            model
                .stations
                .iter()
                .all(|s| s.coverage == QnmPoleCoverage::TrackedOnly)
        );
        let d = |k, offset| direct_lossy_cavity(k, width(offset), index(offset));
        let mut amplitude_error = 0.0_f64;
        let mut step_error = 0.0_f64;
        for i in 0..=20 {
            let k = 3.3 + 7.9 * i as f64 / 20.0;
            let rate = model.amplitudes_rate(k).unwrap();
            let (r0, t0, _) = d(k.into(), -h / 2.0);
            let (r1, t1, _) = d(k.into(), h / 2.0);
            amplitude_error = amplitude_error
                .max((rate.total[0][0] - (r1 - r0) / h).norm())
                .max((rate.total[1][0] - (t1 - t0) / h).norm());
            step_error = step_error.max(rate.max_step_difference);
        }
        assert!(amplitude_error < 2e-4);
        assert!(amplitude_error < 2.0 * step_error);
        let direct_pole = |offset| {
            let mut k = source.poles[0].signature.k0;
            for _ in 0..10 {
                let derivative = (d(k + 1e-5, offset).2 - d(k - 1e-5, offset).2) / 2e-5;
                k -= d(k, offset).2 / derivative;
            }
            k
        };
        let (a, b) = (direct_pole(-h / 2.0), direct_pole(h / 2.0));
        let expected_wavelength = (std::f64::consts::TAU / b.re - std::f64::consts::TAU / a.re) / h;
        let expected_linewidth = -2.0 * (b.im - a.im) / h;
        assert!((model.pole_rates[0].wavelength_cm - expected_wavelength).abs() < 2e-6);
        assert!((model.pole_rates[0].linewidth_per_cm - expected_linewidth).abs() < 2e-6);
        let peak = model.transmission_peak_rate([3.4, 6.2]).unwrap();
        let direct_peak = |offset| {
            let sample = |k: f64| 1.2 * d(k.into(), offset).1.norm_sqr();
            let (mut lo, mut hi) = (3.4, 6.2);
            for _ in 0..50 {
                let a = lo + (hi - lo) / 3.0;
                let b = hi - (hi - lo) / 3.0;
                if sample(a) < sample(b) {
                    lo = a;
                } else {
                    hi = b;
                }
            }
            sample((lo + hi) / 2.0)
        };
        let expected_peak = (direct_peak(h / 2.0) - direct_peak(-h / 2.0)) / h;
        assert!((peak.rate - expected_peak).abs() < 2e-5);
        println!(
            "DESIGN {label} amplitude_error={amplitude_error:e}, step_difference={step_error:e}, pole={:?}, peak={peak:?}, direct_peak_rate={expected_peak:e}",
            model.pole_rates[0]
        );
    }
}
