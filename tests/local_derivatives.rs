#[path = "../examples/support/pole_fixtures.rs"]
mod fixtures;
#[path = "../examples/support/pole_material.rs"]
mod material;
#[path = "../examples/support/pole_green.rs"]
mod reference;

use lamina_core::{
    AnalyticalMaterialStack, CanonicalCoordinates, Constant, EvaluateMeromorphicMaterial,
    ExteriorWavevectors, FiniteLayerIndex, ModalDerivativeBackend, ModeJet1, Polarisation, Scalar,
    Scatter2, SeedJet, Stack, ValidationConfig, compile_canonical_constant_stack,
};
use lamina_mode::*;
use lamina_units::{Length, LengthUnit};
use ndarray::{Ix0, arr0};

type J = ModeJet1<Complex, Ix0>;

fn characteristic<B: ModalDerivativeBackend>(
    backend: &B,
    stack: &AnalyticalMaterialStack<Complex>,
    k0: Complex,
    beta: Complex,
    branches: [Complex; 2],
    polarisation: Polarisation,
) -> Complex {
    let exteriors = [stack.left_exterior(), stack.right_exterior()];
    let kappa = exteriors
        .into_iter()
        .zip(branches)
        .map(|(material, branch)| {
            let epsilon = material.evaluate_relative_permittivity_complex(Scalar(k0));
            let mu = material.evaluate_relative_permeability_complex(Scalar(k0));
            let root = (epsilon * mu * k0 * k0 - beta * beta).sqrt();
            if (root - branch).norm() < (-root - branch).norm() {
                root
            } else {
                -root
            }
        })
        .collect::<Vec<_>>();
    let canonical =
        compile_canonical_constant_stack::<_, J>(stack, Ix0(), &ValidationConfig::permissive())
            .unwrap();
    let coordinates = CanonicalCoordinates::new(
        <J as SeedJet>::constant(arr0(k0)),
        <J as SeedJet>::constant(arr0(beta)),
    );
    let exterior = ExteriorWavevectors::new(
        <J as SeedJet>::constant(arr0(kappa[0])),
        <J as SeedJet>::constant(arr0(kappa[1])),
    );
    *backend
        .checked_characteristic(canonical, &coordinates, &exterior, polarisation)
        .unwrap()
        .value()
        .first()
        .unwrap()
}

fn close(actual: Complex, expected: Complex, tolerance: f64) {
    assert!(
        (actual - expected).norm() <= tolerance * (1.0 + expected.norm()),
        "{actual:?} != {expected:?}"
    );
}

fn partial(report: &LocalDerivativeReport, parameter: PhysicalParameter) -> Complex {
    report
        .outcomes
        .iter()
        .find(|outcome| outcome.parameter == parameter)
        .unwrap()
        .result
        .as_ref()
        .unwrap()
        .value
}

fn local(point: &ModePoint, parameters: &[PhysicalParameter]) -> LocalDerivativeReport {
    point
        .local_derivatives_with(
            Scatter2::new(),
            parameters,
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            LocalDerivativeSettings::default(),
        )
        .unwrap()
}

fn propagation_root_at(
    source: &ModePoint,
    k0: f64,
    beta_guess: Complex,
) -> Result<ModePoint, Box<dyn std::error::Error>> {
    let stack = source.provenance().stack().clone();
    let old = *source.signature().map_err(|error| format!("{error:?}"))?;
    let exteriors = [stack.left_exterior(), stack.right_exterior()];
    let branches = exteriors.map(|material| {
        let epsilon = material.evaluate_relative_permittivity_complex(Scalar(Complex::new(k0, 0.)));
        let mu = material.evaluate_relative_permeability_complex(Scalar(Complex::new(k0, 0.)));
        let root = (epsilon * mu * k0 * k0 - beta_guess * beta_guess).sqrt();
        let old_root = if std::ptr::eq(material, stack.left_exterior()) {
            old.left
        } else {
            old.right
        };
        if (root - old_root).norm() <= (-root - old_root).norm() {
            root
        } else {
            -root
        }
    });
    let z = ((branches[0] + branches[1]) / (Complex::i() * 2. * k0)).ln();
    let problem = PropagationProblem::new(stack, k0, source.provenance().polarisation())?;
    let domain = problem.domain(
        ConformalRegion::new(z.re - 0.04, z.re + 0.04, z.im - 0.04, z.im + 0.04)?,
        ReferenceSelection::Coordinate {
            reference: z,
            coordinate: beta_guess,
        },
        ConformalChartConfig::default(),
        AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    )?;
    let report = problem.plan(vec![domain], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig::default(),
        MatchConfig::default(),
    )?;
    report
        .observations()
        .iter()
        .min_by(|a, b| {
            (a.physical_coordinate() - beta_guess)
                .norm()
                .total_cmp(&(b.physical_coordinate() - beta_guess).norm())
        })
        .cloned()
        .ok_or_else(|| "no perturbed propagation roots were found".into())
}

fn temporal_root_at_beta(
    source: &ModePoint,
    beta: Complex,
    k0_guess: Complex,
) -> Result<Complex, Box<dyn std::error::Error>> {
    let stack = source.provenance().stack().clone();
    let old = *source.signature().map_err(|error| format!("{error:?}"))?;
    let mut root = k0_guess;
    for _ in 0..30 {
        let residual = |k0: Complex| {
            let exteriors = [stack.left_exterior(), stack.right_exterior()];
            let branches = exteriors.map(|material| {
                let epsilon = material.evaluate_relative_permittivity_complex(Scalar(k0));
                let mu = material.evaluate_relative_permeability_complex(Scalar(k0));
                let root = (epsilon * mu * k0 * k0 - beta * beta).sqrt();
                let index = if std::ptr::eq(material, stack.left_exterior()) {
                    0
                } else {
                    1
                };
                if (root - [old.left, old.right][index]).norm()
                    <= (-root - [old.left, old.right][index]).norm()
                {
                    root
                } else {
                    -root
                }
            });
            let signature = BranchSignature {
                k0,
                k_parallel: beta,
                left: branches[0],
                right: branches[1],
            };
            reference::Green::new(
                &stack,
                signature,
                true,
                k0,
                source.provenance().polarisation(),
            )
            .wronskian()
        };
        let h = Complex::new(1e-3, 0.);
        let f = residual(root);
        let df = (residual(root + h) - residual(root - h)) / (2. * h);
        let step = f / df;
        root -= step;
        if step.norm() < 1e-14 {
            break;
        }
    }
    let anchor = root + Complex::new(100., 20.);
    let exteriors = [stack.left_exterior(), stack.right_exterior()];
    let branches = exteriors.map(|material| {
        let epsilon = material.evaluate_relative_permittivity_complex(Scalar(anchor));
        let mu = material.evaluate_relative_permeability_complex(Scalar(anchor));
        let root = (epsilon * mu * anchor * anchor - beta * beta).sqrt();
        let index = if std::ptr::eq(material, stack.left_exterior()) {
            0
        } else {
            1
        };
        if (root - [old.left, old.right][index]).norm()
            <= (-root - [old.left, old.right][index]).norm()
        {
            root
        } else {
            -root
        }
    });
    let problem = FrequencyProblem::new(stack, beta.re, source.provenance().polarisation())?;
    let input = FrequencyDomainInput {
        bounds: FrequencyRegion::new(
            root.re - 1000.,
            root.re + 1000.,
            root.im - 1000.,
            root.im + 1000.,
        )?,
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: anchor,
            kappa: branches,
        }),
        continuation: ContinuationConfig::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    };
    let domain = problem.domain(input, AnalyticAssumption::CallerAssumedHolomorphicPoleFree)?;
    let evaluation = problem.evaluate(&domain, root)?;
    assert!(evaluation.value.norm() < 1e-8);
    Ok(root)
}

#[test]
fn physical_coordinate_partials_match_branch_anchored_finite_differences() {
    for point in [
        fixtures::guided(Polarisation::TransverseElectric).unwrap(),
        fixtures::guided(Polarisation::TransverseMagnetic).unwrap(),
        fixtures::dispersive(Polarisation::TransverseElectric, false).unwrap(),
        fixtures::dispersive(Polarisation::TransverseMagnetic, true).unwrap(),
    ] {
        let signature = *point.signature().unwrap();
        let provenance = point.provenance();
        let stack = provenance.stack();
        let pol = provenance.polarisation();
        let report = local(&point, &[PhysicalParameter::K0, PhysicalParameter::Beta]);
        let accepted_value = report.characteristic.as_ref().unwrap();
        assert!(accepted_value.re.is_finite() && accepted_value.im.is_finite());
        let mut errors_k0 = Vec::new();
        let mut errors_beta = Vec::new();
        for h in [0.1, 0.03, 0.01] {
            let fk = (characteristic(
                &Scatter2::new(),
                stack,
                signature.k0 + h,
                signature.k_parallel,
                [signature.left, signature.right],
                pol,
            ) - characteristic(
                &Scatter2::new(),
                stack,
                signature.k0 - h,
                signature.k_parallel,
                [signature.left, signature.right],
                pol,
            )) / (2. * h);
            let fb = (characteristic(
                &Scatter2::new(),
                stack,
                signature.k0,
                signature.k_parallel + h,
                [signature.left, signature.right],
                pol,
            ) - characteristic(
                &Scatter2::new(),
                stack,
                signature.k0,
                signature.k_parallel - h,
                [signature.left, signature.right],
                pol,
            )) / (2. * h);
            errors_k0.push((fk - partial(&report, PhysicalParameter::K0)).norm());
            errors_beta.push((fb - partial(&report, PhysicalParameter::Beta)).norm());
        }
        // Includes double-precision cancellation in the independently
        // reassembled backend characteristic at this small physical step.
        let final_tolerance = 1e-4;
        assert!(errors_k0[2] <= errors_k0[0] + final_tolerance);
        assert!(errors_beta[2] <= errors_beta[0] + final_tolerance);
        let scale_k0 = 1. + partial(&report, PhysicalParameter::K0).norm();
        let scale_beta = 1. + partial(&report, PhysicalParameter::Beta).norm();
        assert!(
            errors_k0[2] < final_tolerance * scale_k0,
            "k0 errors={errors_k0:?}"
        );
        assert!(
            errors_beta[2] < final_tolerance * scale_beta,
            "beta errors={errors_beta:?}, partial={}, point={:?}",
            partial(&report, PhysicalParameter::Beta),
            signature
        );
    }
}

#[test]
fn accepted_native_derivatives_follow_frequency_conformal_and_propagation_chain_rules() {
    // Direct frequency chart: native coordinate is physical k0.
    let direct = fixtures::cavity(Polarisation::TransverseElectric, true).unwrap();
    let direct_report = local(&direct, &[PhysicalParameter::K0]);
    close(
        direct.evidence().derivative,
        partial(&direct_report, PhysicalParameter::K0),
        2e-8,
    );

    // This conformal fixture uses z = log(-i k0), hence dk0/dz = k0.
    let conformal = fixtures::cavity(Polarisation::TransverseElectric, false).unwrap();
    let conformal_report = local(&conformal, &[PhysicalParameter::K0]);
    close(
        conformal.evidence().derivative / conformal.physical_coordinate(),
        partial(&conformal_report, PhysicalParameter::K0),
        2e-7,
    );

    // Propagation chart: z = log((kappa_L + kappa_R)/(2 i k0));
    // with fixed real k0 and constant exteriors, d kappa_j / d beta = -beta/kappa_j.
    let propagation = fixtures::guided(Polarisation::TransverseElectric).unwrap();
    let signature = *propagation.signature().unwrap();
    let report = local(&propagation, &[PhysicalParameter::Beta]);
    let dz_dbeta = -signature.k_parallel
        * (Complex::new(1., 0.) / signature.left + Complex::new(1., 0.) / signature.right)
        / (signature.left + signature.right);
    close(
        propagation.evidence().derivative * dz_dbeta,
        partial(&report, PhysicalParameter::Beta),
        2e-6,
    );
}

#[test]
fn plasmon_and_film_physical_partials_match_independent_finite_differences() {
    for point in [
        fixtures::fields::interface(false).unwrap(),
        fixtures::fields::interface(true).unwrap(),
        fixtures::fields::film(true).unwrap(),
        fixtures::fields::film(false).unwrap(),
    ] {
        let signature = *point.signature().unwrap();
        let provenance = point.provenance();
        let pol = provenance.polarisation();
        let stack = provenance.stack();
        let report = local(&point, &[PhysicalParameter::K0, PhysicalParameter::Beta]);
        for (parameter, error) in [(PhysicalParameter::K0, 0), (PhysicalParameter::Beta, 1)] {
            let h = 0.01;
            let (k_plus, b_plus, k_minus, b_minus) = if error == 0 {
                (
                    signature.k0 + h,
                    signature.k_parallel,
                    signature.k0 - h,
                    signature.k_parallel,
                )
            } else {
                (
                    signature.k0,
                    signature.k_parallel + h,
                    signature.k0,
                    signature.k_parallel - h,
                )
            };
            let fd = (characteristic(
                &Scatter2::new(),
                stack,
                k_plus,
                b_plus,
                [signature.left, signature.right],
                pol,
            ) - characteristic(
                &Scatter2::new(),
                stack,
                k_minus,
                b_minus,
                [signature.left, signature.right],
                pol,
            )) / (2. * h);
            close(partial(&report, parameter), fd, 2e-4);
        }
        let slope = report
            .implicit_slope(
                ImplicitSlopeRequest::BetaPerK0,
                ImplicitSlopeSettings::default(),
            )
            .unwrap();
        assert!(slope.value.re.is_finite() && slope.value.im.is_finite());
    }
}

#[test]
fn propagation_implicit_slopes_match_independent_perturbed_roots() {
    for point in [
        fixtures::guided(Polarisation::TransverseElectric).unwrap(),
        fixtures::guided(Polarisation::TransverseMagnetic).unwrap(),
        fixtures::fields::interface(false).unwrap(),
        fixtures::fields::film(true).unwrap(),
        fixtures::fields::film(false).unwrap(),
    ] {
        let signature = *point.signature().unwrap();
        let report = local(&point, &[PhysicalParameter::K0, PhysicalParameter::Beta]);
        let slope = report
            .implicit_slope(
                ImplicitSlopeRequest::BetaPerK0,
                ImplicitSlopeSettings::default(),
            )
            .unwrap();
        let step = 25.0;
        let plus = propagation_root_at(
            &point,
            signature.k0.re + step,
            signature.k_parallel + slope.value * step,
        )
        .unwrap();
        let minus = propagation_root_at(
            &point,
            signature.k0.re - step,
            signature.k_parallel - slope.value * step,
        )
        .unwrap();
        let measured = (plus.physical_coordinate() - minus.physical_coordinate()) / (2. * step);
        close(measured, slope.value, 2e-2);
    }
}

#[test]
fn temporal_interface_slope_matches_independent_frequency_roots() {
    let point = fixtures::fields::interface(true).unwrap();
    let signature = *point.signature().unwrap();
    let report = local(&point, &[PhysicalParameter::K0, PhysicalParameter::Beta]);
    let slope = report
        .implicit_slope(
            ImplicitSlopeRequest::K0PerBeta,
            ImplicitSlopeSettings::default(),
        )
        .unwrap();
    let step = 25.0;
    let plus = temporal_root_at_beta(
        &point,
        signature.k_parallel + step,
        signature.k0 + slope.value * step,
    )
    .unwrap();
    let minus = temporal_root_at_beta(
        &point,
        signature.k_parallel - step,
        signature.k0 - slope.value * step,
    )
    .unwrap();
    let measured = (plus - minus) / (2. * step);
    close(measured, slope.value, 2e-2);
}

#[test]
fn thickness_partial_matches_fixed_coordinate_finite_difference_and_slope_is_guarded() {
    let point = fixtures::cavity(Polarisation::TransverseMagnetic, true).unwrap();
    let signature = *point.signature().unwrap();
    let provenance = point.provenance();
    let stack = provenance.stack();
    let layer = stack.layers_left_to_right().first().unwrap();
    let thickness = layer.thickness().as_centimetres();
    let h = 1e-6;
    let displaced = |delta: f64| {
        Stack::builder(
            stack.left_exterior().clone(),
            stack.right_exterior().clone(),
        )
        .layer(
            layer.material().clone(),
            Length::centimetres(thickness + delta),
        )
        .finalise()
    };
    let report = local(
        &point,
        &[
            PhysicalParameter::K0,
            PhysicalParameter::Beta,
            PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0)),
        ],
    );
    let fd = (characteristic(
        &Scatter2::new(),
        &displaced(h),
        signature.k0,
        signature.k_parallel,
        [signature.left, signature.right],
        provenance.polarisation(),
    ) - characteristic(
        &Scatter2::new(),
        &displaced(-h),
        signature.k0,
        signature.k_parallel,
        [signature.left, signature.right],
        provenance.polarisation(),
    )) / (2. * h);
    close(
        partial(
            &report,
            PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0)),
        ),
        fd,
        2e-5,
    );

    let slope = report
        .implicit_slope(
            ImplicitSlopeRequest::K0PerBeta,
            ImplicitSlopeSettings::default(),
        )
        .unwrap();
    close(
        slope.value,
        -partial(&report, PhysicalParameter::Beta) / partial(&report, PhysicalParameter::K0),
        1e-12,
    );
    let thickness_slope = report
        .implicit_slope(
            ImplicitSlopeRequest::K0PerThickness(FiniteLayerIndex::new(0)),
            ImplicitSlopeSettings::default(),
        )
        .unwrap();
    assert_eq!(
        thickness_slope.units,
        SlopeUnits::InverseCentimetrePerLength(LengthUnit::Centimetre)
    );
    close(
        thickness_slope.value,
        -partial(
            &report,
            PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0)),
        ) / partial(&report, PhysicalParameter::K0),
        1e-12,
    );
}

#[test]
fn equivalent_centimetre_and_nanometre_sources_label_and_scale_thickness_partials() {
    let target = Complex::new(std::f64::consts::PI / 0.6, -21_f64.ln() / 1.2);
    let make_stack = |thickness| {
        Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
            .analytical_layer(Constant::dielectric(4.0), thickness)
            .finalise()
    };
    let cm_point = fixtures::frequency(
        make_stack(Length::centimetres(0.3)),
        target,
        Polarisation::TransverseMagnetic,
    )
    .unwrap();
    let nm_point = fixtures::frequency(
        make_stack(Length::nanometres(3.0e6)),
        target,
        Polarisation::TransverseMagnetic,
    )
    .unwrap();
    let parameter = PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0));
    let cm_partial = partial(&local(&cm_point, &[parameter]), parameter);
    let nm_partial = partial(&local(&nm_point, &[parameter]), parameter);
    close(nm_partial, cm_partial * 1e-7, 2e-8);
    assert_eq!(
        local(&cm_point, &[parameter]).outcomes[0]
            .result
            .as_ref()
            .unwrap()
            .units,
        PartialUnits::PerLength(LengthUnit::Centimetre)
    );
    assert_eq!(
        local(&nm_point, &[parameter]).outcomes[0]
            .result
            .as_ref()
            .unwrap()
            .units,
        PartialUnits::PerLength(LengthUnit::Nanometre)
    );
}

#[test]
fn near_zero_slope_denominator_is_reported_without_losing_partials() {
    let point = fixtures::guided(Polarisation::TransverseElectric).unwrap();
    let report = local(&point, &[PhysicalParameter::K0, PhysicalParameter::Beta]);
    let denominator = partial(&report, PhysicalParameter::K0);
    let result = report.implicit_slope(
        ImplicitSlopeRequest::K0PerBeta,
        ImplicitSlopeSettings {
            denominator_atol: denominator.norm() * 2.0,
            denominator_rtol: 0.0,
            denominator_reference_scale: 1.0,
        },
    );
    assert!(matches!(
        result,
        Err(ImplicitSlopeFailure::DegenerateDenominator { .. })
    ));
    assert!(partial(&report, PhysicalParameter::K0).is_finite());
    assert!(partial(&report, PhysicalParameter::Beta).is_finite());
}

#[test]
fn exterior_cutoff_is_a_typed_partial_failure_and_keeps_other_partials() {
    let point = fixtures::guided(Polarisation::TransverseElectric).unwrap();
    let left = point.signature().unwrap().left.norm();
    let report = point
        .local_derivatives_with(
            Scatter2::new(),
            &[
                PhysicalParameter::K0,
                PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0)),
            ],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            LocalDerivativeSettings {
                exterior_kappa_cutoff: left * 2.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(matches!(
        report.outcomes[0].result,
        Err(LocalDerivativeFailure::ExteriorCutoff { side: "left", .. })
    ));
    assert!(report.outcomes[1].result.is_ok());
    assert_eq!(
        report.point.physical_coordinate(),
        point.physical_coordinate()
    );
}

#[test]
fn request_and_selector_failures_are_separate_and_preserve_the_point() {
    let point = fixtures::guided(Polarisation::TransverseElectric).unwrap();
    let invalid = point
        .local_derivatives_with(
            Scatter2::new(),
            &[PhysicalParameter::Beta],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            LocalDerivativeSettings {
                branch_relation_rtol: f64::NAN,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(
        invalid.cause,
        LocalDerivativeRequestError::Settings(_)
    ));
    assert_eq!(
        invalid.point.physical_coordinate(),
        point.physical_coordinate()
    );

    let report = local(
        &point,
        &[
            PhysicalParameter::LayerThickness(FiniteLayerIndex::new(99)),
            PhysicalParameter::Beta,
        ],
    );
    assert!(matches!(
        report.outcomes[0].result,
        Err(LocalDerivativeFailure::LayerOutOfBounds(_))
    ));
    assert!(report.outcomes[1].result.is_ok());
    assert_eq!(
        report.point.physical_coordinate(),
        point.physical_coordinate()
    );
    assert_eq!(report.source_backend, point.backend());
    assert_eq!(report.evaluation_backend, std::any::type_name::<Scatter2>());
}

#[test]
fn guided_power_is_phase_fixed_and_independent_of_seed_scale() {
    use lamina_core::{ExteriorSampling, FieldSampling};
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let p = fixtures::guided(pol).unwrap();
        let sampling =
            FieldSampling::new().left_exterior(ExteriorSampling::point(Length::centimetres(0.)));
        let profile = |amplitude| {
            p.guided_power_profile_with(
                Scatter2::new(),
                &sampling,
                FieldRecoverySettings {
                    amplitude,
                    characteristic_limit: 1e-5,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let a = profile(Complex::new(1., 0.));
        let b = profile(Complex::new(-2., 3.));
        let primary = |p: &ModalProfile| match pol {
            Polarisation::TransverseElectric => p.fields.quantity().electric().y()[0],
            Polarisation::TransverseMagnetic => p.fields.quantity().magnetic().y()[0],
        };
        let x = primary(&a);
        assert!(x.re > 0. && x.im.abs() < 1e-10 * x.re);
        assert!((x - primary(&b)).norm() < 1e-9 * x.norm());
        let ProfileNormalization::GuidedPower(info) = a.normalization else {
            panic!()
        };
        assert_eq!(info.signed_power_w_per_m, 1.);
        assert!(info.seed_exterior_power_w_per_m.iter().all(|v| *v > 0.));
        // Independent slab integral with primary field equal to one at the left interface.
        let sig = p.signature().unwrap();
        let q = (4. * sig.k0.re.powi(2) - sig.k_parallel.re.powi(2)).sqrt();
        let tm = pol == Polarisation::TransverseMagnetic;
        let weight = |epsilon| if tm { epsilon } else { 1. };
        let a = sig.left.im * weight(4.) / q;
        let d = p.provenance().stack().layers_left_to_right()[0]
            .thickness()
            .as_centimetres();
        let v = q * d;
        let core = (1. + a * a) * d / 2.
            + (1. - a * a) * (2. * v).sin() / (4. * q)
            + a * v.sin().powi(2) / q;
        let right = v.cos() + a * v.sin();
        let integral = 1. / (2. * sig.left.im)
            + core / weight(4.)
            + right * right / (2. * sig.right.im * weight(2.25));
        let watts =
            x.norm_sqr() * 0.01 / 376.730313412 * sig.k_parallel.re / (2. * sig.k0.re) * integral;
        assert!((watts - 1.).abs() < 1e-7, "{pol:?} power {watts}");
    }
}

#[test]
fn guided_power_preserves_backward_flux_and_rejects_leaky_modes() {
    use fixtures::fields::{KP, propagation};
    use lamina_core::{ExteriorSampling, FieldSampling};
    let sampling =
        FieldSampling::new().right_exterior(ExteriorSampling::point(Length::centimetres(0.)));
    let settings = FieldRecoverySettings {
        characteristic_limit: 1e-5,
        ..Default::default()
    };
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let q = (1.0_f64 - 0.8 * 0.8).sqrt();
        let alpha = (0.8_f64 * 0.8 - 0.25).sqrt();
        let weight = if pol == Polarisation::TransverseElectric {
            1.
        } else {
            4.
        };
        let d = 2. * (weight * alpha / q).atan() / q;
        let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
            .analytical_layer(Constant::dielectric(4.), Length::centimetres(d / KP))
            .finalise();
        let p = propagation(stack, 0.5, (-0.8).into(), [1.0.into(); 2], pol).unwrap();
        let profile = p
            .guided_power_profile_with(Scatter2::new(), &sampling, settings)
            .unwrap();
        let ProfileNormalization::GuidedPower(info) = profile.normalization else {
            panic!()
        };
        assert_eq!(info.signed_power_w_per_m, -1.);
        let [l, r] = info.seed_exterior_power_w_per_m;
        assert!((l - r).abs() < 1e-7 * l.abs() && l < 0.);
    }
    let cavity = fixtures::cavity(Polarisation::TransverseElectric, true).unwrap();
    assert!(matches!(
        cavity.guided_power_profile_with(Scatter2::new(), &sampling, settings),
        Err(GuidedPowerError::Ineligible)
    ));
}

#[test]
fn guided_rates_combine_dispersion_and_explicit_index_variation() {
    use std::f64::consts::TAU;

    let stack = Stack::from_analytical_materials(
        material::linear(1.0, 0.1, 1.0, -0.02),
        material::linear(2.0, -0.03, 1.0, 0.04),
    )
    .finalise();
    let wavelength_rate = 0.25;
    let explicit = ConstitutiveRate::nonmagnetic_index(1.5, 0.2).unwrap();
    let direction = GuidedPerturbation::new(
        "combined variation",
        "dimensionless",
        vec![0.0],
        wavelength_rate,
        vec![explicit, ConstitutiveRate::default()],
    )
    .unwrap();
    let rates = direction.total_regional_rates(&stack, TAU).unwrap();
    assert!((explicit.relative_permittivity - 0.6).abs() < 1e-14);
    assert!((direction.k0_rate(TAU).unwrap() + TAU * wavelength_rate).abs() < 1e-14);
    let h = 1e-5;
    for (region, handle) in [stack.left_exterior(), stack.right_exterior()]
        .into_iter()
        .enumerate()
    {
        let evaluate = |step| {
            let k0 = Scalar(Complex::new(TAU / (1.0 + step * wavelength_rate), 0.0));
            let epsilon_rate = direction.regional_rates()[region].relative_permittivity;
            (
                handle.evaluate_relative_permittivity_complex(k0).re + epsilon_rate * step,
                handle.evaluate_relative_permeability_complex(k0).re,
            )
        };
        let plus = evaluate(h);
        let minus = evaluate(-h);
        assert!(
            (rates[region].relative_permittivity - (plus.0 - minus.0) / (2.0 * h)).abs() < 1e-10
        );
        assert!(
            (rates[region].relative_permeability - (plus.1 - minus.1) / (2.0 * h)).abs() < 1e-10
        );
    }
}
