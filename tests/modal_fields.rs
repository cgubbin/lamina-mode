#[path = "../examples/support/mod.rs"]
mod existing;
#[path = "../examples/support/field_modes.rs"]
mod fixtures;
use fixtures::{KP, c};
use lamina_core::{
    Constant, ExteriorSampling, FieldSampling, LayerSampling, Polarisation, Scatter2, Stack,
    Transfer2,
};
use lamina_mode::*;
use lamina_units::Length;
fn close(a: Complex, b: Complex) {
    assert!(
        (a - b).norm() < 1e-7 * (1. + a.norm() + b.norm()),
        "{a} != {b}"
    );
}
fn settings() -> FieldRecoverySettings {
    FieldRecoverySettings {
        characteristic_limit: 1e-5,
        ..Default::default()
    }
}
fn compare(point: &ModePoint, sampling: &FieldSampling<f64>) -> ModeFields {
    let a = point
        .recover_fields_with(Scatter2::new(), sampling, settings())
        .unwrap();
    let b = point
        .recover_fields_with(Transfer2::new(), sampling, settings())
        .unwrap();
    let af = a.fields.quantity();
    let bf = b.fields.quantity();
    let av = [
        af.electric().x(),
        af.electric().y(),
        af.electric().z(),
        af.magnetic().x(),
        af.magnetic().y(),
        af.magnetic().z(),
    ];
    let bv = [
        bf.electric().x(),
        bf.electric().y(),
        bf.electric().z(),
        bf.magnetic().x(),
        bf.magnetic().y(),
        bf.magnetic().z(),
    ];
    let gauge = av
        .iter()
        .zip(bv)
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .find(|(a, b)| a.norm() > 1e-8 && b.norm() > 1e-8)
        .map(|(a, b)| *a / *b)
        .unwrap();
    let norm = av
        .iter()
        .flat_map(|a| a.iter())
        .map(|v| v.norm())
        .fold(0., f64::max);
    for (a, b) in av.into_iter().zip(bv) {
        for (x, y) in a.iter().zip(b.iter()) {
            close(*x / norm, gauge * y / norm);
        }
    }
    assert_eq!(a.fields.sampling(), b.fields.sampling());
    a
}
#[test]
fn interface_profiles_both_complex_cases_and_owned_gauge() {
    for temporal in [false, true] {
        let point = fixtures::interface(temporal).unwrap();
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::distances([
                Length::zero(),
                Length::centimetres(0.2 / KP),
            ]))
            .right_exterior(ExteriorSampling::distances([
                Length::zero(),
                Length::centimetres(0.2 / KP),
            ]));
        let a = compare(&point, &sampling);
        let f = a.fields.quantity();
        let h = f.magnetic().y();
        let signature = point.signature().unwrap();
        let epsilon = fixtures::metal(signature.k0 / KP);
        close(h[0], h[2]);
        close(f.electric().x()[0], f.electric().x()[2]);
        close(f.electric().z()[0], epsilon * f.electric().z()[2]);
        close(
            h[1] / h[0],
            (Complex::i() * signature.left.unwrap() * 0.2 / KP).exp(),
        );
        close(
            h[3] / h[2],
            (Complex::i() * signature.right.unwrap() * 0.2 / KP).exp(),
        );
        for (index, e) in [(0, c(1.)), (2, epsilon)] {
            close(
                f.electric().z()[index] / h[index],
                -signature.k_parallel / (signature.k0 * e),
            );
        }
        assert!(matches!(
            point
                .describe(InterpretationConfig::default())
                .unwrap()
                .interpretation
                .classification,
            Classification::Unsupported(_)
        ));
        let multiplier = Complex::new(0.7, -0.3);
        let b = point
            .recover_fields_with(
                Scatter2::new(),
                &sampling,
                FieldRecoverySettings {
                    amplitude: multiplier,
                    ..settings()
                },
            )
            .unwrap();
        for (x, y) in h.iter().zip(b.fields.quantity().magnetic().y()) {
            close(*y, *x * multiplier);
        }
        assert_eq!(a.relative_boundary_defect, b.relative_boundary_defect);
        assert!(
            point
                .recover_fields_with(
                    Scatter2::new(),
                    &sampling,
                    FieldRecoverySettings {
                        amplitude: c(0.),
                        ..settings()
                    }
                )
                .is_err()
        );
        let failed = point
            .recover_fields_with(
                Scatter2::new(),
                &sampling,
                FieldRecoverySettings {
                    characteristic_limit: 0.,
                    relative_boundary_limit: 0.,
                    ..settings()
                },
            )
            .unwrap_err();
        assert_eq!(
            failed.point.physical_coordinate(),
            point.physical_coordinate()
        );
        let invalid = FieldSampling::new().layer(0, LayerSampling::uniform(3));
        assert!(matches!(
            point
                .recover_fields_with(Scatter2::new(), &invalid, settings())
                .unwrap_err()
                .cause,
            FieldRecoveryError::Core(_)
        ));
        drop(point);
        assert_eq!(a.fields.sampling().len(), 4);
    }
}
#[test]
fn both_film_modes_match_boundary_conditions_and_magnetic_parity() {
    for even in [true, false] {
        let point = fixtures::film(even).unwrap();
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(Length::zero()))
            .layer(0, LayerSampling::uniform_with_boundaries(9, true, true))
            .right_exterior(ExteriorSampling::point(Length::zero()));
        let a = compare(&point, &sampling);
        let f = a.fields.quantity();
        let h = f.magnetic().y();
        let ex = f.electric().x();
        let ez = f.electric().z();
        let e = fixtures::metal(c(fixtures::U));
        close(h[0], h[1]);
        close(h[9], h[10]);
        close(ex[0], ex[1]);
        close(ex[9], ex[10]);
        close(e * ez[1], ez[0]);
        close(e * ez[9], ez[10]);
        for i in 1..=9 {
            close(h[i] / h[0], h[10 - i] / h[0] * if even { 1. } else { -1. });
        }
        assert!(fixtures::film_equation(point.physical_coordinate() / KP, even).norm() < 1e-8);
        assert_eq!(a.fields.sampling().len(), 11);
        let exterior = FieldSampling::new()
            .left_exterior(ExteriorSampling::distances([
                Length::zero(),
                Length::centimetres(0.2 / KP),
            ]))
            .right_exterior(ExteriorSampling::distances([
                Length::zero(),
                Length::centimetres(0.2 / KP),
            ]));
        let profile = point
            .recover_fields_with(Transfer2::new(), &exterior, settings())
            .unwrap();
        let h = profile.fields.quantity().magnetic().y();
        assert!(h[1].norm() < h[0].norm() && h[3].norm() < h[2].norm());
        let sig = point.signature().unwrap();
        close(
            h[1] / h[0],
            (Complex::i() * sig.left.unwrap() * 0.2 / KP).exp(),
        );
        close(
            h[3] / h[2],
            (Complex::i() * sig.right.unwrap() * 0.2 / KP).exp(),
        );
    }
}
#[test]
fn guided_te_tm_components_and_interfaces() {
    for tm in [false, true] {
        let point = if !tm {
            existing::guided().unwrap().observations()[0].clone()
        } else {
            let (u, v) = (0.5_f64, 0.8_f64);
            let q = (4. * u * u - v * v).sqrt();
            let al = (v * v - u * u).sqrt();
            let ar = (v * v - 2.25 * u * u).sqrt();
            let d = ((4. * al / q).atan() + (4. * ar / (2.25 * q)).atan()) / q;
            let stack =
                Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
                    .analytical_layer(Constant::dielectric(4.), Length::centimetres(d / KP))
                    .finalise();
            fixtures::propagation(
                stack,
                u,
                c(v),
                [c(1.), c(2.25)],
                Polarisation::TransverseMagnetic,
            )
            .unwrap()
        };
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(Length::zero()))
            .layer(0, LayerSampling::boundaries())
            .right_exterior(ExteriorSampling::point(Length::zero()));
        let a = compare(&point, &sampling);
        let f = a.fields.quantity();
        for (l, r) in [(0, 1), (2, 3)] {
            close(f.electric().x()[l], f.electric().x()[r]);
            close(f.electric().y()[l], f.electric().y()[r]);
            close(f.magnetic().x()[l], f.magnetic().x()[r]);
            close(f.magnetic().y()[l], f.magnetic().y()[r]);
            close(f.magnetic().z()[l], f.magnetic().z()[r]);
        }
        let eps = [c(1.), c(4.), c(4.), c(2.25)];
        for (l, r) in [(0, 1), (2, 3)] {
            close(eps[l] * f.electric().z()[l], eps[r] * f.electric().z()[r]);
        }
        let zero = if tm {
            [f.electric().y(), f.magnetic().x(), f.magnetic().z()]
        } else {
            [f.electric().x(), f.electric().z(), f.magnetic().y()]
        };
        assert!(
            zero.into_iter()
                .flat_map(|a| a.iter())
                .all(|v| v.norm() < 1e-12)
        );
    }
}
#[test]
fn leaky_cavity_finite_growth_and_overflow_are_separate() {
    let point = existing::constant_cavity().unwrap().observations()[0].clone();
    let sampling = FieldSampling::new().left_exterior(ExteriorSampling::distances([
        Length::zero(),
        Length::centimetres(0.2),
    ]));
    let a = compare(&point, &sampling);
    let e = a.fields.quantity().electric().y();
    assert!(e[1].norm() > e[0].norm());
    let invalid =
        FieldSampling::new().left_exterior(ExteriorSampling::point(Length::centimetres(-1.)));
    assert!(
        point
            .recover_fields_with(Scatter2::new(), &invalid, settings())
            .is_err()
    );
    let overflow =
        FieldSampling::new().left_exterior(ExteriorSampling::point(Length::centimetres(1e6)));
    let failure = point
        .recover_fields_with(Transfer2::new(), &overflow, settings())
        .unwrap_err();
    assert!(matches!(failure.cause, FieldRecoveryError::Core(_)));
    assert_eq!(failure.point.native_coordinate(), point.native_coordinate());
}

#[test]
fn physical_profile_keeps_depths_sides_and_normalization() {
    let point = fixtures::film(true).unwrap();
    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::nanometres(10.)))
        .layer(0, LayerSampling::uniform_with_boundaries(3, true, true))
        .right_exterior(ExteriorSampling::point(Length::centimetres(0.)));
    let raw = point
        .sample_profile_with(Scatter2::new(), &sampling, settings())
        .unwrap();
    let width = fixtures::D / KP;
    for (value, want) in raw
        .depths_cm
        .iter()
        .zip([-1e-6, 0., width / 2., width, width])
    {
        assert!((value - want).abs() < 1e-15);
    }
    assert_eq!(raw.depths_cm.len(), 5);
    assert_ne!(
        raw.fields.sampling().positions()[3],
        raw.fields.sampling().positions()[4]
    );
    assert!(matches!(
        raw.normalization,
        ProfileNormalization::BackendSeed { .. }
    ));
    let pole = point
        .normalize_with(
            Scatter2::new(),
            PoleVariable::BetaAtFixedK0,
            fixtures::ASSUME,
            Default::default(),
            settings(),
        )
        .unwrap();
    let profile = pole.sample_profile(&sampling).unwrap();
    assert_eq!(profile.depths_cm, raw.depths_cm);
    assert!(
        matches!(profile.normalization,ProfileNormalization::ScalarPole(ref factor) if factor.scale==pole.normalization.scale)
    );
    assert_eq!(profile.wavevectors, *point.signature().unwrap());
}
