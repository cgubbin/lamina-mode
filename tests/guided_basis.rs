#[path = "../examples/support/pole_material.rs"]
mod material;

use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Stack};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::{PI, TAU};
fn request(pol: Polarisation) -> GuidedWindow {
    GuidedWindow {
        wavelength: Length::centimetres(1.),
        polarisation: pol,
        direction: GuidedDirection::Forward,
        abs_beta: [0., 15.],
        cutoff_margin: 1e-8,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    }
}
fn slab(d: f64, r: f64) -> AnalyticalMaterialStack<Complex> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(r))
        .analytical_layer(Constant::dielectric(4.), Length::centimetres(d))
        .finalise()
}
fn ladder(d: f64, r: f64, pol: Polarisation) -> Vec<f64> {
    // Analytic round-trip reflection phases of an asymmetric three-layer slab.
    let phase = |b: f64| {
        let q = (4. * TAU * TAU - b * b).max(0.).sqrt();
        let shift = |e: f64| {
            ((b * b - e * TAU * TAU).max(0.).sqrt()
                * if pol == Polarisation::TransverseElectric {
                    1.
                } else {
                    4. / e
                })
            .atan2(q)
        };
        q * d - shift(1.) - shift(r)
    };
    let mut roots = vec![];
    for m in 0..100 {
        let (mut lo, mut hi) = (TAU * r.sqrt() + 1e-8, 2. * TAU);
        if phase(lo) <= m as f64 * PI {
            break;
        }
        for _ in 0..60 {
            let b = (lo + hi) / 2.;
            if phase(b) > m as f64 * PI {
                lo = b
            } else {
                hi = b
            }
        }
        roots.push((lo + hi) / 2.);
    }
    roots
}
#[test]
fn slab_ladders_and_cutoff() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for (r, d) in [
            (1., 2.),
            (2.25, 2.),
            (1., (1. - 0.01) / (2. * 3_f64.sqrt())),
            (1., (1. + 0.01) / (2. * 3_f64.sqrt())),
        ] {
            let mut req = request(pol);
            req.direction = GuidedDirection::Backward;
            let result = enumerate_guided_modes(slab(d, r), req).unwrap();
            let reference = ladder(d, r, pol);
            assert!(result.is_complete(), "{:?}", result.unresolved);
            assert_eq!(result.expected, Some(reference.len()));
            for (p, b) in result.modes.iter().zip(reference) {
                assert!((p.physical_coordinate().re + b).abs() < 1e-8);
            }
        }
    }
}
#[test]
fn inputs_endpoints_and_empty_interval() {
    let mut req = request(Polarisation::TransverseElectric);
    req.abs_beta = [4., 1.];
    assert!(enumerate_guided_modes(slab(1., 1.), req).is_err());
    req.abs_beta = [0., 2.];
    let result = enumerate_guided_modes(slab(1., 1.), req).unwrap();
    assert!(result.is_complete() && result.eligible.is_none());
    req.abs_beta = [11., 11.00001];
    let empty = enumerate_guided_modes(slab(1., 1.), req).unwrap();
    assert!(empty.is_complete() && empty.eligible.is_some() && empty.expected == Some(0));
    let root = ladder(1., 1., req.polarisation)[0];
    req.abs_beta = [root, 13.];
    let endpoint = enumerate_guided_modes(slab(1., 1.), req).unwrap();
    assert_eq!(endpoint.expected, None);
    req.abs_beta = [0., 15.];
    assert!(matches!(
        enumerate_guided_modes(slab(1., -1.), req),
        Err(EnumerationError::Material {
            region: 2,
            entry: TensorEntry::EpsilonYy
        })
    ));
    req.refinement.max_updates = 0;
    let failed = enumerate_guided_modes(slab(1., 1.), req).unwrap();
    assert!(failed.expected.unwrap() > 0 && !failed.is_complete());
    assert!(
        failed
            .unresolved
            .iter()
            .all(|g| matches!(g.cause, GuidedLimit::Correction(_)))
    );
}
#[test]
fn symmetric_coupled_cores_and_grazing_layer() {
    let make = || {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
            .analytical_layer(Constant::dielectric(4.), Length::centimetres(0.20))
            .analytical_layer(Constant::dielectric(1.21), Length::centimetres(0.55))
            .analytical_layer(Constant::dielectric(4.), Length::centimetres(0.20))
            .finalise()
    };
    let mut req = request(Polarisation::TransverseElectric);
    // Lower endpoint is the gap medium's internal grazing condition.
    req.abs_beta = [TAU * 1.1, 13.];
    let result = enumerate_guided_modes(make(), req).unwrap();
    assert!(result.is_complete(), "{:?}", result.unresolved);
    assert_eq!(result.expected, Some(2));
    assert!(result.modes[0].physical_coordinate().re > result.modes[1].physical_coordinate().re);
    req.max_subdivisions = 0;
    let partial = enumerate_guided_modes(make(), req).unwrap();
    assert_eq!(partial.unresolved[0].expected, Some(2));
    assert!(!partial.is_complete());
}

#[test]
fn partial_coverage_retains_power_ready_modes_and_counts() {
    let mut req = request(Polarisation::TransverseElectric);
    req.max_subdivisions = 32;
    let basis = enumerate_guided_modes(slab(2., 2.25), req).unwrap();
    assert!(!basis.is_complete() && !basis.modes.is_empty());
    let missing: usize = basis.unresolved.iter().map(|g| g.expected.unwrap()).sum();
    assert_eq!(basis.expected, Some(basis.modes.len() + missing));
    let p = &basis.modes[0];
    assert!(p.describe(Default::default()).is_ok());
    p.guided_power_profile_with(
        lamina_core::Scatter2::new(),
        &lamina_core::FieldSampling::new().layer_centres(),
        Default::default(),
    )
    .unwrap();
}

#[test]
fn lossy_finite_region_is_reported() {
    use lamina_materials::Drude;
    use lamina_units::VacuumAngularWavenumber as K;
    let film = Drude::new(4., K::inverse_centimetres(1.), K::inverse_centimetres(0.01)).unwrap();
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(film, Length::centimetres(0.3))
        .finalise();
    assert!(matches!(
        enumerate_guided_modes(stack, request(Polarisation::TransverseMagnetic)),
        Err(EnumerationError::Material {
            region: 1,
            entry: TensorEntry::EpsilonXx
        })
    ));
}

#[test]
fn guided_beta_sensitivity_matches_corrected_slabs_in_both_directions() {
    let backend = lamina_core::Scatter2::new();
    let thickness = 0.25;
    let direction = GuidedPerturbation::thickness(1, 0).unwrap();
    for propagation in [GuidedDirection::Forward, GuidedDirection::Backward] {
        let mut req = request(Polarisation::TransverseElectric);
        req.direction = propagation;
        let solve = |d| {
            enumerate_guided_modes(slab(d, 1.44), req)
                .unwrap()
                .modes
                .remove(0)
        };
        let point = solve(thickness);
        let result = point
            .guided_beta_sensitivity_with(
                &backend,
                &direction,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let h = 1e-5;
        let difference = (solve(thickness + h).physical_coordinate()
            - solve(thickness - h).physical_coordinate())
            / (2.0 * h);
        assert!((result.beta_rate - difference).norm() < 1e-6);
        let layer = lamina_core::FiniteLayerIndex::new(0);
        let local = point.local_derivatives_with(
            backend,
            &[PhysicalParameter::Beta, PhysicalParameter::LayerThickness(layer)],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            Default::default(),
        ).unwrap();
        let slope = local
            .implicit_slope(
                ImplicitSlopeRequest::BetaPerThickness(layer),
                Default::default(),
            )
            .unwrap();
        assert!((result.beta_rate - slope.value).norm() < 1e-10);
        let guarded = point.guided_beta_sensitivity_with(
            &backend,
            &direction,
            Default::default(),
            ImplicitSlopeSettings {
                denominator_atol: 1e6,
                ..Default::default()
            },
        );
        assert!(matches!(
            guarded,
            Err(GuidedBetaError::NonSimpleRoot { .. })
        ));
        let cutoff = point.guided_beta_sensitivity_with(
            &backend,
            &direction,
            LocalDerivativeSettings {
                exterior_kappa_cutoff: 1e6,
                ..Default::default()
            },
            Default::default(),
        );
        assert!(matches!(
            cutoff,
            Err(GuidedBetaError::ExteriorCutoff { .. })
        ));
        let mismatch = GuidedPerturbation::thickness(2, 0).unwrap();
        assert!(matches!(
            point.guided_beta_sensitivity_with(
                &backend,
                &mismatch,
                Default::default(),
                Default::default()
            ),
            Err(GuidedBetaError::Direction(DirectionError::InvalidGeometry))
        ));
    }
}

#[test]
fn guided_beta_rejects_a_loss_inducing_direction_at_a_lossless_point() {
    let mut core = material::linear(4.0, 0.0, 1.0, 0.0);
    core.epsilon_loss_slope = 0.1;
    core.lossless_at = TAU;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(core, Length::centimetres(0.25))
        .finalise();
    let req = request(Polarisation::TransverseElectric);
    let point = enumerate_guided_modes(stack, req).unwrap().modes.remove(0);
    let backend = lamina_core::Scatter2::new();
    let direction = GuidedPerturbation::new(
        "vacuum wavelength",
        "cm",
        vec![0.0; 2],
        1.0,
        vec![ConstitutiveRate::default(); 3],
    )
    .unwrap();
    let beta = point.physical_coordinate();
    assert!(matches!(
        point.guided_beta_sensitivity_with(
            &backend,
            &direction,
            Default::default(),
            Default::default()
        ),
        Err(GuidedBetaError::Direction(DirectionError::LossInducingRate))
    ));

    let thickness = GuidedPerturbation::thickness(1, 0).unwrap();
    let supported = point
        .guided_beta_sensitivity_with(&backend, &thickness, Default::default(), Default::default())
        .unwrap();
    assert_eq!(supported.point().physical_coordinate(), beta);
}

#[test]
fn co_moving_field_tangent_vanishes_under_rigid_translation() {
    use lamina_core::{ExteriorSampling, FieldSampling, LayerSampling, Scatter2};
    let backend = Scatter2::new();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform_with_boundaries(3, true, true))
        .right_exterior(ExteriorSampling::point(Length::zero()));
    let direction = GuidedPerturbation::new(
        "translation",
        "cm",
        vec![1.0; 2],
        0.0,
        vec![ConstitutiveRate::default(); 3],
    )
    .unwrap();
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let point = enumerate_guided_modes(slab(0.25, 1.44), request(pol))
            .unwrap()
            .modes
            .remove(0);
        let result = point
            .guided_beta_sensitivity_with(
                &backend,
                &direction,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let recovered = result
            .attached_field_tangent_with(&backend, &sampling, Complex::ONE)
            .unwrap();
        let e = recovered.tangent.quantity().electric();
        let h = recovered.tangent.quantity().magnetic();
        assert!(result.beta_rate.norm() < 1e-12);
        let lab = result
            .field_tangent_with(
                &backend,
                &sampling,
                GuidedObservationFrame::FixedLab,
                Complex::ONE,
            )
            .unwrap();
        for sample in [0, 1, 3, 4] {
            assert!(matches!(
                lab[sample].tangent,
                Err(GuidedObservationError::MovingInterface { .. })
            ));
        }
        let step = 1e-5;
        let displaced = |shift| {
            let sampling = FieldSampling::new()
                .layer(0, LayerSampling::point(Length::centimetres(0.125 - shift)));
            result
                .field_tangent_with(
                    &backend,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                    Complex::ONE,
                )
                .unwrap()[0]
                .field
        };
        let plus = displaced(step);
        let minus = displaced(-step);
        for (component, ((p, m), tangent)) in plus
            .into_iter()
            .zip(minus)
            .zip(lab[2].tangent.unwrap())
            .enumerate()
        {
            let fd = (p - m) / (2.0 * step);
            assert!(
                (fd - tangent).norm() < 1e-8 * fd.norm().max(1.0),
                "{pol:?} {component} {fd} {tangent}"
            );
        }
        assert_eq!(recovered.fields.sampling().len(), 5);
        assert!(
            [e.x(), e.y(), e.z(), h.x(), h.y(), h.z()]
                .into_iter()
                .flatten()
                .all(|v| v.norm() < 1e-12)
        );
    }
}

#[test]
fn guided_power_tangent_matches_independently_normalized_slabs() {
    use lamina_core::{ExteriorSampling, FieldSampling, LayerSampling, Scatter2};
    let backend = Scatter2::new();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform_with_boundaries(3, true, true))
        .right_exterior(ExteriorSampling::point(Length::zero()));
    let direction = GuidedPerturbation::thickness(1, 0).unwrap();
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let solve = |d| {
            enumerate_guided_modes(slab(d, 1.44), request(pol))
                .unwrap()
                .modes
                .remove(0)
        };
        let point = solve(0.25);
        let beta = point
            .guided_beta_sensitivity_with(
                &backend,
                &direction,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let evaluate = |amplitude| {
            beta.power_field_tangent_with(
                &backend,
                &sampling,
                GuidedObservationFrame::MaterialAttached,
                amplitude,
            )
            .unwrap()
        };
        let result = evaluate(Complex::ONE);
        let rescaled = evaluate(Complex::new(-2.0, 3.0));
        let primary = if pol == Polarisation::TransverseElectric {
            1
        } else {
            4
        };
        assert!(result.samples[0].field[primary].re > 0.0);
        assert!(result.samples[0].tangent.unwrap()[primary].im.abs() < 1e-9);
        assert!(
            (result
                .regional_power
                .iter()
                .map(|r| r.power_w_per_m)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            result
                .regional_power
                .iter()
                .map(|r| r.power_rate_w_per_m)
                .sum::<f64>()
                .abs()
                < 1e-12
        );
        let step = 1e-5;
        let profile = |d| {
            solve(d)
                .guided_power_profile_with(backend, &sampling, Default::default())
                .unwrap()
        };
        let plus = profile(0.25 + step);
        let minus = profile(0.25 - step);
        for (i, (base, rescaled)) in result.samples.iter().zip(&rescaled.samples).enumerate() {
            let derivative = base.tangent.unwrap();
            for (a, b) in derivative.into_iter().zip(rescaled.tangent.unwrap()) {
                assert!((a - b).norm() < 1e-8 * a.norm().max(1.0));
            }
            let sample = |profile: &ModalProfile| {
                let fields = profile.fields.quantity();
                if primary == 1 {
                    fields.electric().y()[i]
                } else {
                    fields.magnetic().y()[i]
                }
            };
            let fd = (sample(&plus) - sample(&minus)) / (2.0 * step);
            assert!((fd - derivative[primary]).norm() < 1e-5 * fd.norm().max(1.0));
        }
    }
}

#[test]
fn slab_sensitivities_converge_for_wavelength_index_and_interface_motion() {
    use lamina_core::{FieldSampling, Scatter2};
    let backend = Scatter2::new();
    let sampling = FieldSampling::new().layer_centres().layer_interfaces();
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for (label, wavelength, index, interface) in [
            ("dispersive wavelength", 1.0, 0.0, 0.0),
            ("material index", 0.0, 1.0, 0.0),
            ("internal interface", 0.0, 0.0, 1.0),
        ] {
            let solve = |p: f64| {
                let n = 2.0 + p * index;
                let stack = Stack::from_analytical_materials(
                    Constant::vacuum(),
                    Constant::dielectric(1.44),
                )
                .analytical_layer(
                    material::linear(n * n - 0.08 * TAU, 0.08, 1.0, 0.0),
                    Length::centimetres(0.15 + p * interface),
                )
                .analytical_layer(
                    Constant::dielectric(2.89),
                    Length::centimetres(0.13 - p * interface),
                )
                .finalise();
                let mut req = request(pol);
                req.wavelength = Length::centimetres(1.0 + p * wavelength);
                enumerate_guided_modes(stack, req).unwrap().modes.remove(0)
            };
            let mut rates = vec![ConstitutiveRate::default(); 4];
            rates[1] = ConstitutiveRate::nonmagnetic_index(2.0, index).unwrap();
            let direction = GuidedPerturbation::new(
                label,
                if index == 0.0 { "cm" } else { "1" },
                vec![0.0, interface, 0.0],
                wavelength,
                rates,
            )
            .unwrap();
            let tangent = solve(0.0)
                .guided_beta_sensitivity_with(
                    &backend,
                    &direction,
                    Default::default(),
                    Default::default(),
                )
                .unwrap()
                .power_field_tangent_with(
                    &backend,
                    &sampling,
                    GuidedObservationFrame::MaterialAttached,
                    Complex::ONE,
                )
                .unwrap();
            let mut errors = Vec::new();
            for step in [1e-3, 1e-4, 1e-5] {
                let plus = solve(step);
                let minus = solve(-step);
                let beta_fd =
                    (plus.physical_coordinate() - minus.physical_coordinate()) / (2.0 * step);
                let beta_error =
                    (beta_fd - tangent.eigenvalue.beta_rate).norm() / beta_fd.norm().max(1.0);
                let profile = |point: &ModePoint| {
                    point
                        .guided_power_profile_with(backend, &sampling, Default::default())
                        .unwrap()
                };
                let plus = profile(&plus);
                let minus = profile(&minus);
                let components = |p: &ModalProfile, i| {
                    let e = p.fields.quantity().electric();
                    let h = p.fields.quantity().magnetic();
                    [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]]
                };
                let mut field_error = 0.0f64;
                for (i, sample) in tangent.samples.iter().enumerate() {
                    for ((p, m), derivative) in components(&plus, i)
                        .into_iter()
                        .zip(components(&minus, i))
                        .zip(sample.tangent.unwrap())
                    {
                        let fd = (p - m) / (2.0 * step);
                        field_error =
                            field_error.max((fd - derivative).norm() / fd.norm().max(1.0));
                    }
                }
                errors.push((beta_error, field_error));
            }
            println!("{pol:?} {label}: {errors:?}");
            assert!(errors[2].0 < 2e-7 && errors[2].1 < 2e-5);
            assert!(errors[1].0 < errors[0].0 && errors[1].1 < errors[0].1);
        }
    }
}

#[test]
fn guided_power_overlaps_check_signed_norm_orthogonality_and_exchange() {
    let backend = lamina_core::Scatter2::new();
    let te = request(Polarisation::TransverseElectric);
    let modes = enumerate_guided_modes(slab(0.7, 1.44), te).unwrap().modes;
    let mut backward = te;
    backward.direction = GuidedDirection::Backward;
    let reverse = enumerate_guided_modes(slab(0.7, 1.44), backward)
        .unwrap()
        .modes;
    let tm = enumerate_guided_modes(slab(0.7, 1.44), request(Polarisation::TransverseMagnetic))
        .unwrap()
        .modes;
    let other = enumerate_guided_modes(slab(0.55, 1.21), te).unwrap().modes;
    let overlap = |a, b, origins_cm| {
        guided_power_overlap_with(
            &backend,
            [a, b],
            GuidedOverlapRegistration { origins_cm },
            Default::default(),
        )
        .unwrap()
        .power_w_per_m
    };
    assert!((overlap(&modes[0], &modes[0], [0.0; 2]) - 1.0).norm() < 1e-8);
    assert!((overlap(&reverse[0], &reverse[0], [0.0; 2]) + 1.0).norm() < 1e-8);
    assert!(overlap(&modes[0], &modes[1], [0.0; 2]).norm() < 1e-8);
    assert!(overlap(&modes[0], &tm[0], [0.0; 2]).norm() < 1e-12);
    let ab = overlap(&modes[0], &other[0], [0.0, 0.13]);
    let ba = overlap(&other[0], &modes[0], [0.13, 0.0]);
    let translated = overlap(&modes[0], &other[0], [2.0, 2.13]);
    assert!(ab.norm() > 0.1);
    assert!((ab - ba.conj()).norm() < 1e-10);
    assert!((ab - translated).norm() < 1e-10);
}

#[test]
fn guided_overlap_tangent_tracks_registration_and_moving_limits() {
    let backend = lamina_core::Scatter2::new();
    let solve = |d: f64| {
        enumerate_guided_modes(slab(d, 1.44), request(Polarisation::TransverseElectric))
            .unwrap()
            .modes
            .remove(0)
    };
    let a = solve(0.25);
    let b = solve(0.30);
    let shift = |rate| {
        GuidedPerturbation::new(
            "translation",
            "cm",
            vec![rate; 2],
            0.0,
            vec![ConstitutiveRate::default(); 3],
        )
        .unwrap()
    };
    let registration = GuidedOverlapRegistration {
        origins_cm: [0.0, 0.07],
    };
    let evaluate = |modes, dirs, registration| {
        guided_power_overlap_sensitivity_with(
            &backend,
            modes,
            dirs,
            registration,
            Default::default(),
        )
        .unwrap()
    };
    let translating = shift(1.0);
    let stationary = shift(0.0);
    let common = evaluate([&a, &b], [&translating, &translating], registration);
    assert!(common.tangent.unwrap().power_rate_w_per_m.norm() < 1e-10);
    let change = GuidedPerturbation::thickness(1, 0).unwrap();
    let result = evaluate([&a, &b], [&stationary, &change], registration);
    let rate = result.tangent.unwrap().power_rate_w_per_m;
    let overlap = |width| {
        guided_power_overlap_with(
            &backend,
            [&a, &solve(width)],
            registration,
            Default::default(),
        )
        .unwrap()
        .power_w_per_m
    };
    let h = 1e-4;
    let fd = (overlap(0.30 + h) - overlap(0.30 - h)) / (2.0 * h);
    assert!((rate - fd).norm() < 2e-6, "jet={rate}, difference={fd}");
    let displacement = evaluate([&a, &b], [&stationary, &translating], registration)
        .tangent
        .unwrap()
        .power_rate_w_per_m;
    let shifted = |delta| {
        guided_power_overlap_with(
            &backend,
            [&a, &b],
            GuidedOverlapRegistration {
                origins_cm: [0.0, 0.07 + delta],
            },
            Default::default(),
        )
        .unwrap()
        .power_w_per_m
    };
    assert!((displacement - (shifted(h) - shifted(-h)) / (2.0 * h)).norm() < 2e-6);
    let wavelength = GuidedPerturbation::new(
        "wavelength",
        "cm",
        vec![0.0; 2],
        1.0,
        vec![ConstitutiveRate::default(); 3],
    )
    .unwrap();
    let spectral = evaluate([&a, &b], [&wavelength, &wavelength], registration)
        .tangent
        .unwrap()
        .power_rate_w_per_m;
    let at_wavelength = |delta| {
        let mut req = request(Polarisation::TransverseElectric);
        req.wavelength = Length::centimetres(1.0 + delta);
        let a = enumerate_guided_modes(slab(0.25, 1.44), req)
            .unwrap()
            .modes
            .remove(0);
        let b = enumerate_guided_modes(slab(0.30, 1.44), req)
            .unwrap()
            .modes
            .remove(0);
        guided_power_overlap_with(&backend, [&a, &b], registration, Default::default())
            .unwrap()
            .power_w_per_m
    };
    let fd = (at_wavelength(h) - at_wavelength(-h)) / (2.0 * h);
    assert!((spectral - fd).norm() < 2e-6);
    let aligned = GuidedOverlapRegistration {
        origins_cm: [0.0; 2],
    };
    let split = evaluate([&a, &a], [&stationary, &translating], aligned);
    assert!(matches!(
        split.tangent,
        Err(GuidedOverlapDerivativeError::SeparatingInterfaces { .. })
    ));
    assert!((split.base.power_w_per_m - 1.0).norm() < 1e-8);
    let together = evaluate([&a, &a], [&change, &change], aligned);
    assert!(together.tangent.unwrap().power_rate_w_per_m.norm() < 1e-8);
}

#[test]
fn tracked_coupled_slabs_match_independent_enumeration() {
    let stack = |gap| {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
            .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.20))
            .analytical_layer(Constant::dielectric(1.21), Length::centimetres(gap))
            .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.20))
            .finalise()
    };
    let mut window = request(Polarisation::TransverseElectric);
    window.abs_beta = [TAU * 1.1, 13.0];
    let previous = enumerate_guided_modes(stack(0.55), window).unwrap();
    let step = GuidedTrackingStep {
        stack: stack(0.551).into(),
        window,
        perturbation: GuidedPerturbation::thickness(3, 1).unwrap(),
        increment: 0.001,
        registration: GuidedOverlapRegistration {
            origins_cm: [0.0; 2],
        },
        minimum_overlap: 0.2,
        ambiguity_margin: 0.01,
    };
    let tracked = track_guided_basis(&previous, step).unwrap();
    let reference = enumerate_guided_modes(stack(0.551), window).unwrap();
    assert!(tracked.used_local_correction, "{:?}", tracked.predictions);
    assert!(tracked.basis.is_complete());
    assert_eq!(tracked.matches.len(), 2);
    for (predicted, independent) in tracked.basis.modes.iter().zip(reference.modes) {
        assert!(
            (predicted.physical_coordinate() - independent.physical_coordinate()).norm() < 1e-8
        );
    }
}

#[test]
fn guided_tracking_refreshes_when_an_upper_mode_reaches_cutoff() {
    let window = request(Polarisation::TransverseElectric);
    let previous = enumerate_guided_modes(slab(0.35, 1.0), window).unwrap();
    let tracked = track_guided_basis(
        &previous,
        GuidedTrackingStep {
            stack: slab(0.25, 1.0).into(),
            window,
            perturbation: GuidedPerturbation::thickness(1, 0).unwrap(),
            increment: -0.1,
            registration: GuidedOverlapRegistration {
                origins_cm: [0.0; 2],
            },
            minimum_overlap: 0.2,
            ambiguity_margin: 0.01,
        },
    )
    .unwrap();
    assert!(!tracked.used_local_correction);
    assert!(tracked.basis.is_complete());
    assert_eq!(tracked.basis.modes.len(), 1);
    assert_eq!(tracked.unmatched_source, [1]);
}

/// Project an analytic superposition of a TE slab's two modes, with every
/// length scaled by `scale` (cm), and return the projection.
fn project_slab_superposition(
    scale: f64,
    settings: FieldRecoverySettings,
    duplicate_first_mode: bool,
) -> GuidedFieldProjection {
    use lamina_core::{ExteriorSampling, FieldSampling, Scatter2};
    let width = 0.4 * scale;
    let k0 = TAU / scale;
    let mut window = request(Polarisation::TransverseElectric);
    window.wavelength = Length::centimetres(scale);
    window.abs_beta = [0., 15. / scale];
    window.cutoff_margin = 1e-8 / scale;
    if scale != 1.0 {
        // Residual limit relative to k0, as for the field-recovery limits.
        window.refinement.residual_atol = 0.0;
        window.refinement.residual_rtol = 1e-9;
        window.refinement.residual_scale = k0;
    }
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(4.), Length::centimetres(width))
        .finalise();
    let mut basis = enumerate_guided_modes(stack, window).unwrap();
    assert_eq!(basis.modes.len(), 2);
    let expected = [Complex::new(0.8, 0.2), Complex::new(-0.3, 0.4)];
    let modes: Vec<_> = basis
        .modes
        .iter()
        .map(|mode| {
            let profile = mode
                .guided_power_profile_with(
                    Scatter2::new(),
                    &FieldSampling::new().left_exterior(ExteriorSampling::point(Length::zero())),
                    settings,
                )
                .unwrap();
            let beta = mode.physical_coordinate().re;
            let amplitude = profile.fields.quantity().electric().y()[0];
            (beta, amplitude)
        })
        .collect();
    let sample = |z: f64, _side| {
        let mut result = [Complex::ZERO; 6];
        for ((beta, amplitude), coefficient) in modes.iter().copied().zip(expected) {
            let alpha = (beta * beta - k0 * k0).sqrt();
            let q = (4.0 * k0 * k0 - beta * beta).sqrt();
            let core = |z: f64| (q * z).cos() + alpha / q * (q * z).sin();
            let (ey, derivative) = if z <= 0.0 {
                let ey = (alpha * z).exp();
                (ey, alpha * ey)
            } else if z >= width {
                let ey = core(width) * (-alpha * (z - width)).exp();
                (ey, -alpha * ey)
            } else {
                (core(z), -q * (q * z).sin() + alpha * (q * z).cos())
            };
            let scale = coefficient * amplitude;
            result[1] += scale * ey;
            result[3] += scale * Complex::I * derivative / k0;
            result[5] += scale * beta * ey / k0;
        }
        result
    };
    if duplicate_first_mode {
        basis.modes[1] = basis.modes[0].clone();
    }
    let input = GuidedProjectionInput {
        wavelength_cm: scale,
        support_cm: [-12.0 * scale, 12.0 * scale],
        interfaces_cm: &[0.0, width],
        sample: &sample,
    };
    let projection = project_guided_field(&basis, 0.0, input, 1e-8, settings).unwrap();
    if !duplicate_first_mode {
        assert_eq!(projection.rank, 2);
        let solution = projection.solution.as_ref().unwrap();
        assert!(solution.condition_number < 1.000001);
        assert!(
            solution.residual_w_per_m < 1e-9,
            "{solution:?} · gram {:?} · rhs {:?}",
            projection.gram,
            projection.right_hand_side
        );
        for (actual, expected) in solution.coefficients.iter().zip(expected) {
            assert!((*actual - expected).norm() < 1e-7, "{actual} != {expected}");
        }
    }
    projection
}

#[test]
fn guided_projection_recovers_complex_coefficients_and_reports_rank() {
    project_slab_superposition(1.0, Default::default(), false);
    let rank_deficient = project_slab_superposition(1.0, Default::default(), true);
    assert_eq!(rank_deficient.rank, 1);
    assert!(rank_deficient.solution.is_none());
}

#[test]
fn guided_projection_at_micron_scale_uses_k0_relative_recovery_limits() {
    // λ = 1 µm: lamina-mode's absolute default characteristic limit (1e-6 at
    // k0 = 2π cm^-1) is scaled with k0.
    let scale = 1e-4;
    let settings = FieldRecoverySettings {
        characteristic_limit: 1e-6 / scale,
        ..Default::default()
    };
    project_slab_superposition(scale, settings, false);
}

#[test]
fn tracking_keeps_indistinguishable_sources_in_an_ambiguous_cluster() {
    let window = request(Polarisation::TransverseElectric);
    let mut previous = enumerate_guided_modes(slab(0.4, 1.0), window).unwrap();
    previous.modes[1] = previous.modes[0].clone();
    let result = track_guided_basis(
        &previous,
        GuidedTrackingStep {
            stack: slab(0.4, 1.0).into(),
            window,
            perturbation: GuidedPerturbation::thickness(1, 0).unwrap(),
            increment: 0.0,
            registration: GuidedOverlapRegistration {
                origins_cm: [0.0; 2],
            },
            minimum_overlap: 0.2,
            ambiguity_margin: 0.01,
        },
    )
    .unwrap();
    assert!(!result.used_local_correction);
    assert!(result.matches.is_empty());
    assert_eq!(result.ambiguous.len(), 1);
    assert_eq!(result.ambiguous[0].source, [0, 1]);
    assert_eq!(result.ambiguous[0].destination, [0]);
}

#[test]
fn guided_reciprocity_is_antisymmetric_and_pairs_counterpropagating_modes() {
    let backend = lamina_core::Scatter2::new();
    let mut window = request(Polarisation::TransverseElectric);
    let a = enumerate_guided_modes(slab(0.25, 1.44), window)
        .unwrap()
        .modes
        .remove(0);
    window.direction = GuidedDirection::Backward;
    let b = enumerate_guided_modes(slab(0.25, 1.44), window)
        .unwrap()
        .modes
        .remove(0);
    let flux = |a, b| {
        guided_reciprocity_with(
            &backend,
            [a, b],
            GuidedOverlapRegistration {
                origins_cm: [0.0; 2],
            },
            Default::default(),
        )
        .unwrap()
        .flux_w_per_m
    };
    assert!(flux(&a, &a).norm() < 1e-12);
    assert!((flux(&a, &b) + flux(&b, &a)).norm() < 1e-12);
    assert!((flux(&a, &b) + 4.0).norm() < 1e-8);
}

#[test]
fn local_coupling_matches_neighbouring_overlaps_and_groups_small_gaps() {
    let backend = lamina_core::Scatter2::new();
    let stack = |interface: f64, epsilon_shift: f64| {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.44))
            .analytical_layer(
                Constant::dielectric(4.0 + epsilon_shift),
                Length::centimetres(0.42 + interface),
            )
            .analytical_layer(
                Constant::dielectric(2.25),
                Length::centimetres(0.10 - interface),
            )
            .finalise()
    };
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let window = request(pol);
        let base = enumerate_guided_modes(stack(0.0, 0.0), window).unwrap();
        assert!(base.modes.len() >= 2);
        for (motion, material) in [(1.0, 0.0), (0.0, 1.0)] {
            let mut rates = vec![ConstitutiveRate::default(); 4];
            rates[1].relative_permittivity = material;
            let direction =
                GuidedPerturbation::new("slab direction", "1", vec![0.0, motion, 0.0], 0.0, rates)
                    .unwrap();
            let coupling =
                guided_local_coupling_with(&backend, &base, &direction, 1e-7, Default::default())
                    .unwrap();
            let beta_rate = base.modes[0]
                .guided_beta_sensitivity_with(
                    &backend,
                    &direction,
                    Default::default(),
                    Default::default(),
                )
                .unwrap()
                .beta_rate;
            assert!((coupling.perturbation_matrix[(0, 0)] - beta_rate).norm() < 1e-6);
            let evaluate = |step| {
                let displaced =
                    enumerate_guided_modes(stack(step * motion, step * material), window).unwrap();
                guided_power_overlap_with(
                    &backend,
                    [&displaced.modes[1], &base.modes[0]],
                    GuidedOverlapRegistration {
                        origins_cm: [0.0; 2],
                    },
                    Default::default(),
                )
                .unwrap()
                .power_w_per_m
            };
            let step = 1e-5;
            let fd = (evaluate(step) - evaluate(-step)) / (2.0 * step);
            let coefficient = coupling.coefficients[(0, 1)].unwrap();
            println!(
                "{pol:?} interface={motion}, coefficient={coefficient}, finite difference={fd}, error={:e}",
                (fd - coefficient).norm()
            );
            assert!(
                (fd - coefficient).norm() < 1e-5,
                "{pol:?} motion={motion}, fd={fd}, coefficient={coefficient}"
            );
            assert!((coefficient + coupling.coefficients[(1, 0)].unwrap().conj()).norm() < 1e-10);
            let unresolved =
                guided_local_coupling_with(&backend, &base, &direction, 100.0, Default::default())
                    .unwrap();
            assert_eq!(unresolved.unresolved_groups.len(), 1);
            assert!(unresolved.coefficients[(0, 1)].is_none());
            assert_eq!(unresolved.perturbation_matrix, coupling.perturbation_matrix);
        }
    }
}

#[test]
fn coupled_slab_gap_has_the_predicted_local_mode_coupling() {
    let backend = lamina_core::Scatter2::new();
    let window = request(Polarisation::TransverseElectric);
    let solve = |gap| {
        let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
            .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.2))
            .analytical_layer(Constant::dielectric(1.21), Length::centimetres(gap))
            .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.2))
            .finalise();
        enumerate_guided_modes(stack, window).unwrap()
    };
    let gap = 0.55;
    let base = solve(gap);
    let direction = GuidedPerturbation::thickness(3, 1).unwrap();
    let local =
        guided_local_coupling_with(&backend, &base, &direction, 1e-7, Default::default()).unwrap();
    let overlap = |offset| {
        let displaced = solve(gap + offset);
        guided_power_overlap_with(
            &backend,
            [&displaced.modes[1], &base.modes[0]],
            GuidedOverlapRegistration {
                origins_cm: [0.0; 2],
            },
            Default::default(),
        )
        .unwrap()
        .power_w_per_m
    };
    let step = 1e-5;
    let fd = (overlap(step) - overlap(-step)) / (2.0 * step);
    let local = local.coefficients[(0, 1)].unwrap();
    println!("coupled slab gap: local={local}, finite difference={fd}");
    assert!((local - fd).norm() < 1e-5);
}

#[test]
fn power_tangent_converges_for_interface_motion_in_a_micron_multilayer() {
    use lamina_core::{FieldSampling, LayerSampling, Scatter2};
    // λ = 1 µm: vacuum | ε 4, 0.42 µm | ε 2.25, 0.10 µm | ε 1.44, with the
    // interface between the two layers moving and its neighbours fixed.
    let scale = 1e-4;
    let k0 = TAU / scale;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.44))
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(0.42 * scale))
        .analytical_layer(
            Constant::dielectric(2.25),
            Length::centimetres(0.10 * scale),
        )
        .finalise();
    let mut window = request(Polarisation::TransverseElectric);
    window.wavelength = Length::centimetres(scale);
    window.abs_beta = [0.0, 2.02 * k0];
    window.cutoff_margin = 1e-7;
    window.phase_tolerance = 1e-10;
    window.refinement.residual_atol = 0.0;
    window.refinement.residual_rtol = 1e-9;
    window.refinement.residual_scale = k0;
    let basis = enumerate_guided_modes(stack, window).unwrap();
    assert!(basis.is_complete());
    assert_eq!(basis.modes.len(), 2);
    let direction = GuidedPerturbation::new(
        "interface position",
        "cm",
        vec![0.0, 1.0, 0.0],
        0.0,
        vec![ConstitutiveRate::default(); 4],
    )
    .unwrap();
    let backend = Scatter2::new();
    for (index, mode) in basis.modes.iter().enumerate() {
        let power = mode
            .guided_beta_sensitivity_with(
                &backend,
                &direction,
                Default::default(),
                Default::default(),
            )
            .unwrap()
            .power_field_tangent_with(
                &backend,
                &FieldSampling::new().layer(0, LayerSampling::offsets([Length::zero()])),
                GuidedObservationFrame::MaterialAttached,
                Complex::ONE,
            )
            .unwrap_or_else(|e| panic!("mode {index}: {e:?}"));
        let total: f64 = power.regional_power.iter().map(|r| r.power_w_per_m).sum();
        let rate: f64 = power
            .regional_power
            .iter()
            .map(|r| r.power_rate_w_per_m)
            .sum();
        assert!((total - 1.0).abs() < 1e-9);
        let largest = power
            .regional_power
            .iter()
            .map(|r| r.power_rate_w_per_m.abs())
            .fold(0.0, f64::max);
        assert!(
            rate.abs() <= 1e-7 * largest,
            "mode {index}: {rate} vs {largest}"
        );
    }
}
