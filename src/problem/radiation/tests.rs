use super::*;
use crate::{
    conformal::chart::CoordinateSheet,
    problem::orchestration::{ChartPlan, DiscoveryConfig, MatchConfig},
    refinement::RefinementConfig,
};
use lamina_core::{Constant, ScalarAlgebra, Scatter2};
use std::f64::consts::{FRAC_PI_2, PI};
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn c(x: f64) -> C {
    C::new(x, 0.0)
}
fn ext(e: f64) -> ConstantExterior {
    ConstantExterior::new(c(e), c(1.0)).unwrap()
}
fn region(z: C, h: f64) -> ConformalRegion<f64> {
    ConformalRegion::new(z.re - h, z.re + h, z.im - h, z.im + h).unwrap()
}
fn reference(z: C, coordinate: f64, choices: [RadiationChoice; 2]) -> RadiationReference {
    RadiationReference {
        z,
        coordinate,
        choices: choices.map(Some),
        tolerance: RadiationTolerance::default(),
    }
}
fn wave(e: f64, k: f64, b: f64, choice: RadiationChoice) -> C {
    let q = e * k * k - b * b;
    let v = if q > 0.0 {
        c(q.sqrt())
    } else {
        C::new(0.0, (-q).sqrt())
    };
    if choice == RadiationChoice::OutgoingOrDecaying {
        v
    } else {
        -v
    }
}
#[test]
fn real_references_verify_open_closed_and_reversed_channels_for_both_maps() {
    use RadiationChoice::*;
    for choices in [
        [OutgoingOrDecaying; 2],
        [IncomingOrGrowing; 2],
        [OutgoingOrDecaying, IncomingOrGrowing],
        [IncomingOrGrowing, OutgoingOrDecaying],
    ] {
        let p = InPlaneProblem::in_plane(
            Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
                .finalise(),
            2.0,
            Polarisation::TransverseElectric,
            Scatter2::new(),
        )
        .unwrap();
        for beta in [1.0, 2.5, 3.5] {
            let z = ((wave(1.0, 2.0, beta, choices[0]) + wave(2.25, 2.0, beta, choices[1]))
                / C::new(0.0, 4.0))
            .ln();
            let r = region(z, 0.001);
            let d = p
                .domain_with_radiation_reference(
                    r,
                    reference(z, beta, choices),
                    ConformalChartConfig::default(),
                    ASSUME,
                )
                .unwrap();
            assert_eq!(d.provenance().requested(), r);
            let RadiationProvenance::Verified(v) = d.provenance().radiation() else {
                panic!()
            };
            assert_eq!(v.reference().choices, choices.map(Some));
            assert_eq!(v.physical_coordinates(), (2.0, beta));
        }
        let p = SpectralProblem::spectral(
            ext(1.0),
            ext(2.25),
            [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
            1.5,
            Polarisation::TransverseMagnetic,
            Scatter2::new(),
        )
        .unwrap();
        for k0 in [0.5, 1.2, 3.0] {
            let z = ((wave(1.0, k0, 1.5, choices[0]) + wave(2.25, k0, 1.5, choices[1]) / 1.5)
                / C::new(0.0, 2.0))
            .ln();
            let d = p
                .domain_with_radiation_reference(
                    region(z, 0.001),
                    reference(z, k0, choices),
                    ConformalChartConfig::default(),
                    ASSUME,
                )
                .unwrap();
            assert!(matches!(
                d.provenance().radiation(),
                RadiationProvenance::Verified(_)
            ));
        }
    }
}
#[test]
fn invalid_references_fail_without_reselecting_a_sheet() {
    use RadiationChoice::*;
    let p = SpectralProblem::spectral(
        ext(1.0),
        ext(1.0),
        [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let z = C::new(2.0_f64.ln(), -FRAC_PI_2);
    let r = region(z, 0.02);
    let cfg = ConformalChartConfig::default();
    let a = reference(z, 2.0, [OutgoingOrDecaying; 2]);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = a;
        invalid.tolerance.relative = bad;
        assert!(matches!(
            p.domain_with_radiation_reference(r, invalid, cfg, ASSUME),
            Err(RadiationReferenceError::InvalidTolerance)
        ));
    }
    for invalid in [
        RadiationReference { z: z + c(1.0), ..a },
        RadiationReference {
            coordinate: -2.0,
            ..a
        },
        RadiationReference {
            z: C::new(f64::NAN, 0.0),
            ..a
        },
    ] {
        assert!(matches!(
            p.domain_with_radiation_reference(r, invalid, cfg, ASSUME),
            Err(RadiationReferenceError::InvalidReference)
        ));
    }
    assert!(
        p.domain_with_radiation_reference(
            r,
            RadiationReference {
                coordinate: 3.0,
                ..a
            },
            cfg,
            ASSUME
        )
        .is_err()
    );
    assert!(matches!(
        p.domain_with_radiation_reference(
            r,
            RadiationReference {
                choices: [Some(OutgoingOrDecaying), Some(IncomingOrGrowing)],
                ..a
            },
            cfg,
            ASSUME
        ),
        Err(RadiationReferenceError::BranchMismatch { side: 1, .. })
    ));
    let shifted = z + C::new(0.0, PI);
    assert!(matches!(
        p.domain_with_radiation_reference(
            region(shifted, 0.02),
            RadiationReference { z: shifted, ..a },
            cfg,
            ASSUME
        ),
        Err(RadiationReferenceError::BranchMismatch { .. })
    ));
    let backwards = p
        .domain(
            r,
            ReferenceSelection::Coordinate {
                reference: z,
                coordinate: c(-2.0),
            },
            cfg,
            ASSUME,
        )
        .unwrap();
    let mapped = backwards
        .search
        .dispersion()
        .chart()
        .map(J::lift_constant(arr0(z)))
        .unwrap();
    assert!(
        (mapped
            .exterior_wavevectors()
            .left()
            .unwrap()
            .scalar()
            .unwrap()
            .value()[()]
            - c(2.0))
        .norm()
            < 1e-12
    );
    assert_eq!(
        backwards.provenance().radiation(),
        &RadiationProvenance::Unspecified
    );
    for (e, m) in [(C::new(1.0, 0.1), c(1.0)), (c(1.0), c(2.0))] {
        let p = SpectralProblem::spectral(
            ConstantExterior::new(e, m).unwrap(),
            ext(1.0),
            [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
            0.0,
            Polarisation::TransverseElectric,
            Scatter2::new(),
        )
        .unwrap();
        assert!(matches!(
            p.domain_with_radiation_reference(r, a, cfg, ASSUME),
            Err(RadiationReferenceError::UnsupportedMaterial { side: 0 })
        ));
    }
    let p = SpectralProblem::spectral(
        ext(1.0),
        ext(2.25),
        [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
        1.5,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    assert!(matches!(
        p.domain_with_radiation_reference(
            r,
            RadiationReference {
                coordinate: 1.5,
                ..a
            },
            cfg,
            ASSUME
        ),
        Err(RadiationReferenceError::Cutoff { side: 0 })
    ));
    let unspecified = p
        .domain(
            region(c(0.0), 0.01),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            cfg,
            ASSUME,
        )
        .unwrap();
    assert_eq!(
        unspecified.provenance().radiation(),
        &RadiationProvenance::Unspecified
    );
}
#[test]
fn anchored_cavity_preserves_reference_through_search_refinement_and_batch() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let d = 0.3;
        let pole = C::new(PI, -3.0_f64.ln()) / (2.0 * d);
        let z = (C::new(0.0, -1.0) * pole).ln();
        let anchor = C::new(pole.norm().ln(), -FRAC_PI_2);
        let r =
            ConformalRegion::new(z.re - 0.04, z.re + 0.04, z.im - 0.04, anchor.im + 0.04).unwrap();
        let p = SpectralProblem::spectral(
            ext(1.0),
            ext(1.0),
            [Layer::new(
                lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                lamina_core::units::Length::centimetres(d),
            )],
            0.0,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        let domain = p
            .domain_with_radiation_reference(
                r,
                reference(
                    anchor,
                    pole.norm(),
                    [RadiationChoice::OutgoingOrDecaying; 2],
                ),
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        let clone = domain.clone();
        assert!(domain.provenance().same_domain(clone.provenance()));
        assert_eq!(domain.provenance().reference(), anchor);
        assert_eq!(domain.provenance().requested(), r);
        assert_eq!(
            clone.provenance().radiation(),
            domain.provenance().radiation()
        );
        let plan = ChartPlan::explicit(p.clone(), vec![domain.clone()], vec![]).unwrap();
        let batch = plan.discover(DiscoveryConfig::default()).unwrap();
        let report = plan
            .refine(&batch, RefinementConfig::default(), MatchConfig::default())
            .unwrap();
        assert_eq!(report.observations().len(), 1, "{report:?}");
        let root = report.observations()[0].root();
        assert!((root.coordinate() - pole).norm() < 1e-7);
        let point = p.describe_root(
            &domain,
            root,
            super::super::physical::InterpretationConfig::default(),
        );
        let detail = point.result.as_ref().unwrap();
        assert_eq!(
            detail.interpretation.radiation,
            super::super::physical::RadiationQualification::OutgoingContinuation
        );
        assert_eq!(
            detail.interpretation.evolution,
            super::super::physical::Evolution::TemporalDecay
        );
        let descriptions =
            report.describe_points(super::super::physical::InterpretationConfig::default());
        assert_eq!(descriptions[0].result.as_ref().unwrap(), detail);
        let existing = p
            .domain(
                r,
                ReferenceSelection::Coordinate {
                    reference: anchor,
                    coordinate: c(pole.norm()),
                },
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        assert_eq!(
            existing.provenance().radiation(),
            &RadiationProvenance::Unspecified
        );

        assert!(root.provenance().same_domain(domain.provenance()));
        for provenance in [
            root.provenance(),
            batch.entries()[0].as_ref().unwrap().provenance(),
            report.charts()[0].provenance(),
        ] {
            assert_eq!(provenance.radiation(), domain.provenance().radiation());
        }
        let other = SpectralProblem::spectral(
            ext(1.0),
            ext(1.0),
            [] as [lamina_core::Layer<crate::problem::Region, f64>; 0],
            0.0,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        assert!(matches!(
            other.evaluate(&domain, z),
            Err(ExecutionError::ForeignDomain)
        ));
        assert!(
            p.refine(
                &clone,
                batch.entries()[0].as_ref().unwrap(),
                RefinementConfig::default()
            )
            .is_ok()
        );
    }
}

#[test]
fn incoming_frequency_continuation_is_verified_other_not_outgoing() {
    use super::super::physical::{Evolution, InterpretationConfig, RadiationQualification};
    let d = 0.3;
    let pole = C::new(PI, 3.0_f64.ln()) / (2.0 * d);
    let z = (C::new(0.0, 1.0) * pole).ln();
    let anchor = C::new(pole.norm().ln(), FRAC_PI_2);
    let r = ConformalRegion::new(z.re - 0.04, z.re + 0.04, anchor.im - 0.04, z.im + 0.04).unwrap();
    let p = SpectralProblem::spectral(
        ext(1.0),
        ext(1.0),
        [Layer::new(
            lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
            lamina_core::units::Length::centimetres(d),
        )],
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let domain = p
        .domain_with_radiation_reference(
            r,
            reference(anchor, pole.norm(), [RadiationChoice::IncomingOrGrowing; 2]),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap();
    let plan = ChartPlan::explicit(p, vec![domain], vec![]).unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    let descriptions = report.describe_points(InterpretationConfig::default());
    assert_eq!(descriptions.len(), 1);
    assert!((descriptions[0].root.coordinate() - pole).norm() < 1e-7);
    let interpretation = &descriptions[0].result.as_ref().unwrap().interpretation;
    assert_eq!(
        interpretation.radiation,
        RadiationQualification::VerifiedOther
    );
    assert_eq!(interpretation.evolution, Evolution::TemporalGrowth);
}
