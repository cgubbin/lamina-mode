use super::super::integration::{Sample, unconjugated};
use super::*;
use lamina_core::{Constant, Stack, Wall, WallKind};

const K: f64 = TAU;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn walled(kind: WallKind) -> ModeStack {
    Stack::from_analytical_materials(Constant::dielectric(2.25), Constant::dielectric(1.0))
        .analytical_layer(Constant::dielectric(4.0), cm(0.6))
        .left_termination(Wall::new(kind, cm(-0.4)))
        .right_termination(Wall::new(kind, cm(1.1)))
        .finalise()
        .into()
}

fn request(polarisation: Polarisation) -> ModalBasisRequest {
    ModalBasisRequest::new(
        cm(1.0),
        ModeFamily::Polarised(polarisation),
        ModalWindow::forward(2.05 * K, 2.0 * K, 0.03 * K),
    )
}

/// Everything a basis returns, as bits.
fn fingerprint(basis: &ModalBasis) -> Vec<u64> {
    let sampling = FieldSampling::new()
        .layer(0, lamina_core::LayerSampling::uniform(7))
        .left_exterior(lamina_core::ExteriorSampling::uniform(3, cm(0.4)));
    let mut bits = vec![basis.expected.unwrap_or(usize::MAX) as u64];
    for mode in &basis.modes {
        let beta = mode.point.physical_coordinate();
        bits.extend([beta.re.to_bits(), beta.im.to_bits()]);
        let fields = mode
            .fields_with(Scatter2::new(), &sampling, basis.request.recovery)
            .unwrap();
        bits.extend(
            fields
                .iter()
                .flatten()
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()]),
        );
    }
    bits.extend(basis.groups.iter().flatten().map(|&i| i as u64));
    bits
}

#[test]
fn reversed_traversal_and_rescaled_seeds_give_identical_bases() {
    for (kind, polarisation) in [
        (WallKind::PerfectElectric, Polarisation::TransverseElectric),
        (WallKind::PerfectMagnetic, Polarisation::TransverseMagnetic),
    ] {
        let backend = Scatter2::new();
        let forward = solve(&backend, walled(kind), request(polarisation), false).unwrap();
        assert_eq!(forward.status, BasisStatus::Complete);
        let reversed = solve(&backend, walled(kind), request(polarisation), true).unwrap();
        let mut seeded = request(polarisation);
        seeded.recovery.amplitude = Complex::new(-3.7e4, 2.2e-3);
        let seeded = solve(&backend, walled(kind), seeded, false).unwrap();
        let base = fingerprint(&forward);
        assert_eq!(base, fingerprint(&reversed));
        assert_eq!(base, fingerprint(&seeded));
        eprintln!(
            "{polarisation:?} {kind:?}: {} modes identical under reversed traversal and seed rescaling",
            forward.found()
        );
    }
}

fn mirror(a: &Sample) -> Sample {
    [-a[0], a[1], a[2], a[3], -a[4], -a[5]]
}

fn paired_with_mirror(a: &Sample, b: &Sample) -> Complex {
    unconjugated(a, &mirror(b))
}

/// R(a, b) = −4 ⟨a, M b⟩ for the merged antisymmetric flux, so the backward
/// partner b = σ M f of a unit-power guided mode gives R = −4σ W/m.
#[test]
fn reciprocity_flux_is_minus_four_unconjugated_pairings_with_the_mirror() {
    use super::super::overlap::integrate_guided_flux;
    let stack = Stack::from_analytical_materials(Constant::dielectric(2.25), Constant::vacuum())
        .analytical_layer(Constant::dielectric(4.0), cm(0.5))
        .finalise();
    let backend = Scatter2::new();
    let registration = GuidedOverlapRegistration {
        origins_cm: [0.0, 0.0],
    };
    let settings = FieldRecoverySettings::default();
    for (polarisation, merged) in [
        (Polarisation::TransverseElectric, -4.0),
        (Polarisation::TransverseMagnetic, 4.0),
    ] {
        let modes = |direction| {
            let window = GuidedWindow {
                wavelength: cm(1.0),
                family: polarisation.into(),
                direction,
                abs_beta: [1.5 * K, 1.999 * K],
                cutoff_margin: 1e-8,
                phase_tolerance: 1e-11,
                max_subdivisions: 1000,
                refinement: Default::default(),
                assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
            };
            enumerate_guided_modes(stack.clone(), window).unwrap().modes
        };
        let (forward, backward) = (
            modes(GuidedDirection::Forward),
            modes(GuidedDirection::Backward),
        );
        assert!(!forward.is_empty() && forward.len() == backward.len());
        for (f, b) in forward.iter().zip(&backward) {
            let flux = guided_reciprocity_with(&backend, [f, b], registration, settings)
                .unwrap()
                .flux_w_per_m;
            let integrate = |modes, density| {
                integrate_guided_flux(&backend, modes, registration, settings, density)
                    .unwrap()
                    .power_w_per_m
            };
            let with_mirror = integrate([f, b], paired_with_mirror);
            let own = integrate([f, f], unconjugated);
            let self_mirror = integrate([f, f], paired_with_mirror);
            assert!((flux - merged).norm() < 1e-8, "{flux}");
            assert!(
                (flux + 4.0 * with_mirror).norm() < 1e-8,
                "{flux} {with_mirror}"
            );
            // ⟨f, f⟩ is the signed power; ⟨f, M f⟩ = −R(f, f)/4 = 0.
            assert!((own - 1.0).norm() < 1e-8, "{own}");
            assert!(self_mirror.norm() < 1e-8, "{self_mirror}");
            eprintln!(
                "{polarisation:?} β/k0 {:.6}: R(f, b) = {flux:.10}, −4⟨f, Mb⟩ = {:.10}, ⟨f, f⟩ = {own:.10}",
                f.physical_coordinate().re / K,
                -4.0 * with_mirror
            );
        }
    }
}

#[test]
fn inverse_square_root_orthonormalises_and_refuses_singular_grams() {
    let c = Complex::new;
    let gram = DMatrix::from_row_slice(
        3,
        3,
        &[
            c(1.0, 0.0),
            c(0.02, -0.01),
            c(-0.03, 0.005),
            c(0.02, -0.01),
            c(1.0, 0.0),
            c(0.01, 0.04),
            c(-0.03, 0.005),
            c(0.01, 0.04),
            c(1.0, 0.0),
        ],
    );
    let root = inverse_sqrt(&gram, 1e-8).unwrap();
    let identity = root.transpose() * &gram * &root;
    let defect = (identity - DMatrix::identity(3, 3)).norm();
    assert!(defect < 1e-14, "{defect:e}");
    assert_eq!(
        inverse_sqrt(&DMatrix::identity(2, 2), 1e-8).unwrap(),
        DMatrix::identity(2, 2)
    );
    let singular = DMatrix::from_element(2, 2, Complex::ONE);
    assert!(inverse_sqrt(&singular, 1e-8).is_none());
}
