use super::*;
use crate::problem::orchestration::{
    ChartPlan, CoverageExclusion, DiscoveryConfig, MatchConfig, PlanGeometry,
};
use crate::problem::physical::{
    Classification, InterpretationConfig, RadiationQualification, Unsupported,
};
use crate::refinement::{CandidateStatus, RefinementConfig};
use lamina_core::{
    ComplexScalar, Constant, DerivativeOrder, DifferentiableMaterial,
    DifferentiableMeromorphicMaterial, Material, MeromorphicMaterial, Sampled, Scatter2,
};
use std::sync::atomic::{AtomicBool, Ordering};
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn c(x: f64) -> C {
    C::new(x, 0.0)
}
#[derive(Clone, Debug)]
enum Model {
    Exp { magnetic: bool },
    Lorentz { strength: f64, damping: f64 },
    BadRegion,
}
#[derive(Clone, Debug)]
struct TestMaterial {
    model: Model,
    fault: Option<Arc<AtomicBool>>,
    derivative_fault: bool,
}
impl TestMaterial {
    fn exponential(magnetic: bool) -> Self {
        Self {
            model: Model::Exp { magnetic },
            fault: None,
            derivative_fault: false,
        }
    }
    fn lorentz(strength: f64, damping: f64) -> Self {
        Self {
            model: Model::Lorentz { strength, damping },
            fault: None,
            derivative_fault: false,
        }
    }
    fn values(&self, k: C) -> [C; 6] {
        let mut v = match self.model {
            Model::Exp { magnetic } => {
                let a = if magnetic { 0.5 } else { 1.0 };
                let e = (k * a).exp() / (k * k);
                let mu = if magnetic { (k * 0.5).exp() } else { c(1.0) };
                [
                    e,
                    mu,
                    e * (c(a) - c(2.0) / k),
                    if magnetic { mu * 0.5 } else { c(0.0) },
                    e * ((c(a) - c(2.0) / k).powi(2) + c(2.0) / (k * k)),
                    if magnetic { mu * 0.25 } else { c(0.0) },
                ]
            }
            Model::Lorentz { strength, damping } => {
                let d = c(400.0) - k * k - C::new(0.0, damping) * k;
                let dp = -2.0 * k - C::new(0.0, damping);
                [
                    c(2.25) + strength / d,
                    c(1.0),
                    -strength * dp / (d * d),
                    c(0.0),
                    c(2.0 * strength) / (d * d) + 2.0 * strength * dp * dp / (d * d * d),
                    c(0.0),
                ]
            }
            Model::BadRegion => [
                if k.re > 6.0 && k.im < -0.2 {
                    C::new(f64::NAN, 0.0)
                } else {
                    c(1.0)
                },
                c(1.0),
                c(0.0),
                c(0.0),
                c(0.0),
                c(0.0),
            ],
        };
        if self
            .fault
            .as_ref()
            .is_some_and(|f| f.load(Ordering::SeqCst))
        {
            v[if self.derivative_fault { 2 } else { 0 }] = C::new(f64::NAN, 0.0);
        }
        v
    }
}
fn lift<T: ComplexScalar<RealField = f64>>(v: C) -> T {
    T::from_real(v.re) + T::i() * T::from_real(v.im)
}
fn lower<T: ComplexScalar<RealField = f64>>(v: T) -> C {
    C::new(v.real(), v.imaginary())
}
impl Material for TestMaterial {
    type Real = f64;
    fn relative_permittivity<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|x| lift(self.values(c(x))[0]))
    }
    fn relative_permeability<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|x| lift(self.values(c(x))[1]))
    }
}
impl DifferentiableMaterial for TestMaterial {
    fn relative_permittivity_derivative<I, T>(&self, k: I, order: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|x| {
            lift(
                self.values(c(x))[match order {
                    DerivativeOrder::First => 2,
                    DerivativeOrder::Second => 4,
                    DerivativeOrder::Third => panic!("fixture supports up to second derivative"),
                }],
            )
        })
    }
    fn relative_permeability_derivative<I, T>(&self, k: I, order: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|x| {
            lift(
                self.values(c(x))[match order {
                    DerivativeOrder::First => 3,
                    DerivativeOrder::Second => 5,
                    DerivativeOrder::Third => panic!("fixture supports up to second derivative"),
                }],
            )
        })
    }
}
impl MeromorphicMaterial for TestMaterial {
    fn relative_permittivity_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|x| lift(self.values(lower(x))[0]))
    }
    fn relative_permeability_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|x| lift(self.values(lower(x))[1]))
    }
}
impl DifferentiableMeromorphicMaterial for TestMaterial {
    fn relative_permittivity_complex_derivative<I, T>(
        &self,
        k: I,
        order: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|x| {
            lift(
                self.values(lower(x))[match order {
                    DerivativeOrder::First => 2,
                    DerivativeOrder::Second => 4,
                    DerivativeOrder::Third => panic!("fixture supports up to second derivative"),
                }],
            )
        })
    }
    fn relative_permeability_complex_derivative<I, T>(
        &self,
        k: I,
        order: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|x| {
            lift(
                self.values(lower(x))[match order {
                    DerivativeOrder::First => 3,
                    DerivativeOrder::Second => 5,
                    DerivativeOrder::Third => panic!("fixture supports up to second derivative"),
                }],
            )
        })
    }
}
fn input(bounds: FrequencyRegion, reference: FrequencyReference) -> FrequencyDomainInput {
    FrequencyDomainInput {
        bounds,
        reference,
        continuation: ContinuationConfig::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    }
}
fn explicit(k: C, kappa: [C; 2]) -> FrequencyReference {
    FrequencyReference::Explicit(ExteriorReferenceValues {
        frequency: k,
        kappa,
    })
}
fn radiation(k: f64) -> FrequencyReference {
    FrequencyReference::RealRadiation {
        frequency: k,
        choices: [RadiationChoice::OutgoingOrDecaying; 2],
        tolerance: RadiationTolerance::default(),
    }
}
fn exp_problem(magnetic: bool) -> FrequencyProblem<Scatter2> {
    let m = TestMaterial::exponential(magnetic);
    FrequencyProblem::frequency(
        Stack::from_analytical_materials(m.clone(), m).finalise(),
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap()
}
#[test]
fn dispersive_branches_cross_principal_cut_and_keep_both_material_derivatives() {
    for magnetic in [false, true] {
        let p = exp_problem(magnetic);
        let anchor = C::new(1.5, 2.9);
        let kappa = (anchor / 2.0).exp();
        let bounds = FrequencyRegion::new(1.0, 2.0, 2.7, 3.6).unwrap();
        let d = p
            .frequency_domain(input(bounds, explicit(anchor, [kappa; 2])), ASSUME)
            .unwrap();
        let opposite = p
            .frequency_domain(input(bounds, explicit(anchor, [-kappa, kappa])), ASSUME)
            .unwrap();
        assert_eq!(d.provenance().coordinate_kind(), CoordinateKind::Frequency);
        assert_eq!(d.provenance().coordinate_kind().units(), "cm^-1");
        assert_eq!(d.provenance().requested(), bounds.bounds());
        assert_eq!(d.provenance().reference(), anchor);
        assert_eq!(
            d.provenance().radiation(),
            &RadiationProvenance::Unspecified
        );
        let points = [
            C::new(1.8, 3.5),
            C::new(1.2, 2.8),
            C::new(1.5, 3.2),
            C::new(1.8, 3.5),
        ];
        for k in points {
            let chart = d.search.dispersion().chart();
            let point = chart.map(J::variable(arr0(k), 0).unwrap()).unwrap();
            let other = opposite
                .search
                .dispersion()
                .chart()
                .map(J::variable(arr0(k), 0).unwrap())
                .unwrap();
            let expected = (k / 2.0).exp();
            for wave in [
                point
                    .exterior_wavevectors()
                    .left()
                    .unwrap()
                    .scalar()
                    .unwrap(),
                point
                    .exterior_wavevectors()
                    .right()
                    .unwrap()
                    .scalar()
                    .unwrap(),
            ] {
                assert!((wave.value()[()] - expected).norm() < 1e-9);
                assert!((wave.first()[()] - expected / 2.0).norm() < 1e-9);
            }
            assert!(
                (other
                    .exterior_wavevectors()
                    .left()
                    .unwrap()
                    .scalar()
                    .unwrap()
                    .value()[()]
                    + expected)
                    .norm()
                    < 1e-9
            );
            assert!(
                (other
                    .exterior_wavevectors()
                    .right()
                    .unwrap()
                    .scalar()
                    .unwrap()
                    .value()[()]
                    - expected)
                    .norm()
                    < 1e-9
            );
            let h = 1e-5;
            let numerical = (chart.continuation_at(0, k + c(h)).unwrap().value
                - chart.continuation_at(0, k - c(h)).unwrap().value)
                / (2.0 * h);
            assert!((numerical - expected / 2.0).norm() < 1e-8);
            assert!(chart.continuation_at(0, k).unwrap().evaluations <= 7);
        }
    }
}
#[test]
fn frequency_domain_exclusions_cover_halo_and_never_claim_certification() {
    let p = exp_problem(false);
    let bounds = FrequencyRegion::new(1.0, 2.0, -0.2, 0.2).unwrap();
    let anchor = c(1.5);
    let base = input(bounds, explicit(anchor, [(anchor / 2.0).exp(); 2]));
    let d = p.frequency_domain(base.clone(), ASSUME).unwrap();
    let outer = d.provenance().evaluation().real_bounds().1;
    assert!(outer > 2.0);
    let closed_exclusion = FrequencyExclusion {
        region: FrequencyRegion::new(2.0, 2.1, -0.1, 0.1).unwrap(),
        reason: ExclusionReason::ExteriorThreshold { side: 1 },
    };
    for exclusion in [
        closed_exclusion,
        FrequencyExclusion {
            region: FrequencyRegion::new(outer, outer + 0.1, -0.1, 0.1).unwrap(),
            reason: ExclusionReason::MaterialSingularity,
        },
    ] {
        let mut bad = base.clone();
        bad.exclusions.push(exclusion.clone());
        assert!(
            matches!(p.frequency_domain(bad,ASSUME),Err(FrequencyError::Excluded(e)) if e==exclusion)
        );
    }
    let point = C::new(outer, 0.0);
    assert_eq!(
        d.search.dispersion().chart().coordinate(point).unwrap(),
        point
    );
    assert!(matches!(
        d.search.dispersion().chart().coordinate(c(2.1)),
        Err(FrequencyError::OutsideRegion)
    ));
    assert_eq!(
        d.provenance().frequency().unwrap().exterior_assumption,
        ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero
    );
    let mut bad = base.clone();
    bad.reference = explicit(anchor, [c(7.0); 2]);
    assert!(matches!(
        p.frequency_domain(bad, ASSUME),
        Err(FrequencyError::Exterior {
            source: ExteriorFailure::InconsistentValue,
            ..
        })
    ));
    let mut bad = base.clone();
    bad.continuation.integral_tolerance = f64::NAN;
    assert!(matches!(
        p.frequency_domain(bad, ASSUME),
        Err(FrequencyError::InvalidConfig)
    ));
    let mut bad = base;
    bad.reference = explicit(c(2.1), [c(1.0); 2]);
    assert!(matches!(
        p.frequency_domain(bad, ASSUME),
        Err(FrequencyError::InvalidReference)
    ));
    assert!(FrequencyRegion::new(f64::NAN, 2.0, 0.0, 1.0).is_err());
}
#[test]
fn exterior_budgets_depth_and_nonfinite_derivatives_remain_typed() {
    let p = exp_problem(false);
    let k = c(1.5);
    let mut i = input(
        FrequencyRegion::new(1.0, 2.0, -0.1, 0.1).unwrap(),
        explicit(k, [(k / 2.0).exp(); 2]),
    );
    i.continuation.max_evaluations = 1;
    let d = p.frequency_domain(i, ASSUME).unwrap();
    assert!(
        matches!(p.evaluate(&d,c(1.8)),Err(ExecutionError::Optical(e)) if matches!(e.as_ref(),ChartDispersionError::Chart(FrequencyError::Exterior{side:0,source:ExteriorFailure::EvaluationBudget,evaluations:1})))
    );
    let flag = Arc::new(AtomicBool::new(false));
    let m = TestMaterial {
        fault: Some(flag.clone()),
        derivative_fault: true,
        ..TestMaterial::exponential(true)
    };
    let p = FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), m).finalise(),
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let d = p
        .frequency_domain(
            input(
                FrequencyRegion::new(1.0, 2.0, -0.1, 0.1).unwrap(),
                explicit(k, [k, (k / 2.0).exp()]),
            ),
            ASSUME,
        )
        .unwrap();
    flag.store(true, Ordering::SeqCst);
    assert!(
        matches!(p.evaluate(&d,k),Err(ExecutionError::Optical(e)) if matches!(e.as_ref(),ChartDispersionError::Chart(FrequencyError::Exterior{side:1,source:ExteriorFailure::NonFiniteDerivative,..})))
    );
    // A smooth but non-polynomial logarithmic derivative requires refinement.
    let p = FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum()).finalise(),
        1.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let mut i = input(
        FrequencyRegion::new(1.1, 3.0, -0.2, 0.2).unwrap(),
        radiation(2.0),
    );
    i.continuation.max_depth = 0;
    i.continuation.integral_tolerance = 1e-15;
    let d = p.frequency_domain(i, ASSUME).unwrap();
    assert!(
        matches!(p.evaluate(&d,c(1.2)),Err(ExecutionError::Optical(e)) if matches!(e.as_ref(),ChartDispersionError::Chart(FrequencyError::Exterior{source:ExteriorFailure::IntegrationDepth,..})))
    );
}
fn slab(strength: f64, pol: Polarisation) -> FrequencyProblem<Scatter2> {
    FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), TestMaterial::lorentz(strength, 0.0))
            .analytical_layer(
                Constant::dielectric(4.0),
                lamina_units::Length::centimetres(0.3),
            )
            .finalise(),
        0.0,
        pol,
        Scatter2::new(),
    )
    .unwrap()
}
fn slab_input() -> FrequencyDomainInput {
    let mut i = input(
        FrequencyRegion::new(4.3, 6.3, -3.4, 0.2).unwrap(),
        radiation(5.5),
    );
    i.exclusions.push(FrequencyExclusion {
        region: FrequencyRegion::new(19.9, 20.1, -0.1, 0.1).unwrap(),
        reason: ExclusionReason::MaterialSingularity,
    });
    i
}
// Independent outgoing Fabry-Perot denominator, without core optical evaluation.
fn characteristic(k: C, strength: f64, pol: Polarisation) -> C {
    let epsilon = c(2.25) + strength / (c(400.0) - k * k);
    let (yc, yl, yr) = if pol == Polarisation::TransverseElectric {
        (k * 2.0, k, k * epsilon.sqrt())
    } else {
        (k / 2.0, k, k / epsilon.sqrt())
    };
    -C::i() * (yl + yc) * (yc + yr) / (2.0 * yc)
        * (-C::i() * k * 0.6).exp()
        * (c(1.0) - (yc - yl) / (yc + yl) * (yc - yr) / (yc + yr) * (C::i() * k * 1.2).exp())
}
fn independent_root(strength: f64, pol: Polarisation) -> C {
    let mut k = C::new(std::f64::consts::PI / 0.6, -21.0_f64.ln() / 1.2);
    for _ in 0..12 {
        let h = c(1e-5);
        k -= characteristic(k, strength, pol)
            / ((characteristic(k + h, strength, pol) - characteristic(k - h, strength, pol))
                / (2e-5));
    }
    assert!(characteristic(k, strength, pol).norm() < 1e-12);
    k
}
#[test]
fn dispersive_te_tm_slab_roots_and_derivatives_are_not_frozen() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let p = slab(100.0, pol);
        let d = p.frequency_domain(slab_input(), ASSUME).unwrap();
        assert!(matches!(
            d.provenance().radiation(),
            RadiationProvenance::Verified(_)
        ));
        let k = independent_root(100.0, pol);
        let h = c(1e-5);
        let evaluated = p.evaluate(&d, k).unwrap();
        assert!(evaluated.value().norm() < 1e-10, "{evaluated:?}");
        let derivative =
            (characteristic(k + h, 100.0, pol) - characteristic(k - h, 100.0, pol)) / (2e-5);
        assert!(
            (*evaluated.derivative() - derivative).norm() < 1e-7,
            "{evaluated:?} {derivative}"
        );
        let probe = k + C::new(0.05, 0.03);
        let sampled = p.evaluate(&d, probe).unwrap();
        assert!((*sampled.value() - characteristic(probe, 100.0, pol)).norm() < 1e-9);
        let derivative = (characteristic(probe + h, 100.0, pol)
            - characteristic(probe - h, 100.0, pol))
            / (2e-5);
        assert!((*sampled.derivative() - derivative).norm() < 1e-7);
        let frozen_e: f64 = 2.25 + 100.0 / (400.0 - 5.5 * 5.5);
        let reflection =
            (2.0 - 1.0) / (2.0 + 1.0) * (2.0 - frozen_e.sqrt()) / (2.0 + frozen_e.sqrt());
        let frozen = C::new(std::f64::consts::PI / 0.6, reflection.ln() / 1.2);
        assert!((k - frozen).norm() > 1e-3);
        let result = p.search(&d, ArgumentConfig::new(1e-6)).unwrap();
        assert_eq!(result.result().len(), 1);
        let refined = p.refine(&d, &result, RefinementConfig::default()).unwrap();
        let root = refined
            .outcomes()
            .iter()
            .find_map(|o| {
                if let CandidateStatus::Accepted(r) = o.status() {
                    Some(r.as_ref())
                } else {
                    None
                }
            })
            .expect("accepted slab root");
        assert!((root.coordinate() - k).norm() < 1e-8);
        assert_eq!(root.coordinate(), root.evidence().z);
        assert!(root.provenance().same_domain(d.provenance()));
        let physical = p
            .describe_root(&d, root, InterpretationConfig::default())
            .result
            .unwrap();
        assert_eq!(physical.data.k0, root.coordinate());
        assert_eq!(
            physical.interpretation.classification,
            Classification::Unsupported(Unsupported::SolveDirection)
        );
        assert!((physical.data.epsilon[1] - c(frozen_e)).norm() > 1e-3);
        assert!(matches!(
            slab(100.0, pol).evaluate(&d, k),
            Err(ExecutionError::ForeignDomain)
        ));
    }
}
#[test]
fn constant_te_tm_limit_matches_conformal_chain_rule_and_roots() {
    for pol in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let direct = slab(0.0, pol);
        let k = independent_root(0.0, pol);
        let anchor = k + c(0.1);
        let d = direct
            .frequency_domain(
                input(
                    FrequencyRegion::new(k.re - 0.2, k.re + 0.2, k.im - 0.2, k.im + 0.2).unwrap(),
                    explicit(anchor, [anchor, anchor * 1.5]),
                ),
                ASSUME,
            )
            .unwrap();
        let p = SpectralProblem::spectral(
            ConstantExterior::new(c(1.0), c(1.0)).unwrap(),
            ConstantExterior::new(c(2.25), c(1.0)).unwrap(),
            [Layer::new(
                AnalyticalMaterialHandle::new(Constant::dielectric(4.0)),
                lamina_units::Length::centimetres(0.3),
            )],
            0.0,
            pol,
            Scatter2::new(),
        )
        .unwrap();
        let z = (-C::i() * k).ln();
        let cd = p
            .domain(
                ConformalRegion::new(z.re - 0.08, z.re + 0.08, z.im - 0.08, z.im + 0.08).unwrap(),
                ReferenceSelection::Coordinate {
                    reference: z,
                    coordinate: k,
                },
                ConformalChartConfig::default(),
                ASSUME,
            )
            .unwrap();
        assert_eq!(cd.provenance().coordinate_kind(), CoordinateKind::Conformal);
        assert_eq!(cd.provenance().coordinate_kind().units(), "dimensionless");
        let a = direct.evaluate(&d, k).unwrap();
        let b = p.evaluate(&cd, z).unwrap();
        assert!((*a.value() - *b.value()).norm() < 1e-10);
        assert!(
            (*a.derivative() - *b.derivative() / k).norm() < 1e-9,
            "{a:?} {b:?}"
        );
        let direct_discovery = direct.search(&d, ArgumentConfig::new(1e-6)).unwrap();
        let direct_refined = direct
            .refine(&d, &direct_discovery, RefinementConfig::default())
            .unwrap();
        let direct_root = direct_refined
            .outcomes()
            .iter()
            .find_map(|o| {
                if let CandidateStatus::Accepted(r) = o.status() {
                    Some(r)
                } else {
                    None
                }
            })
            .unwrap();
        assert!((direct_root.coordinate() - k).norm() < 1e-8);
        assert_eq!(
            direct_root.provenance().coordinate_kind(),
            CoordinateKind::Frequency
        );
        let discovered = p.search(&cd, ArgumentConfig::new(1e-6)).unwrap();
        let refined = p
            .refine(&cd, &discovered, RefinementConfig::default())
            .unwrap();
        assert!(
            (refined
                .outcomes()
                .iter()
                .find_map(|o| if let CandidateStatus::Accepted(r) = o.status() {
                    Some(r.as_ref())
                } else {
                    None
                })
                .unwrap()
                .coordinate()
                - k)
                .norm()
                < 1e-8
        );
    }
}
#[test]
fn actual_anchor_radiation_is_separate_from_explicit_complex_branches() {
    for (m, unsupported) in [
        (TestMaterial::lorentz(100.0, 0.0), false),
        (TestMaterial::lorentz(100.0, 0.2), true),
        (TestMaterial::exponential(true), true),
    ] {
        let p = FrequencyProblem::frequency(
            Stack::from_analytical_materials(Constant::vacuum(), m.clone()).finalise(),
            0.0,
            Polarisation::TransverseElectric,
            Scatter2::new(),
        )
        .unwrap();
        let r = FrequencyRegion::new(1.0, 2.0, -0.2, 0.2).unwrap();
        let d = p.frequency_domain(input(r, radiation(1.5)), ASSUME);
        if unsupported {
            assert!(matches!(
                d,
                Err(FrequencyError::Radiation(
                    super::super::radiation::RadiationReferenceError::UnsupportedMaterial {
                        side: 1
                    }
                ))
            ));
        } else {
            let d = d.unwrap();
            let RadiationProvenance::Verified(ref v) = d.provenance().radiation else {
                panic!()
            };
            assert_eq!(v.epsilon()[1], m.values(c(1.5))[0].re);
            assert!(v.all_outgoing());
            let incoming = p
                .frequency_domain(
                    input(
                        r,
                        FrequencyReference::RealRadiation {
                            frequency: 1.5,
                            choices: [RadiationChoice::IncomingOrGrowing; 2],
                            tolerance: RadiationTolerance::default(),
                        },
                    ),
                    ASSUME,
                )
                .unwrap();
            assert!(
                incoming
                    .provenance()
                    .frequency()
                    .unwrap()
                    .reference
                    .kappa
                    .iter()
                    .all(|k| k.re < 0.0)
            );
        }
        let k = C::new(1.5, -0.1);
        let v = m.values(k);
        let d = p
            .frequency_domain(
                input(r, explicit(k, [k, (v[0] * v[1] * k * k).sqrt()])),
                ASSUME,
            )
            .unwrap();
        assert_eq!(
            d.provenance().radiation(),
            &RadiationProvenance::Unspecified
        );
    }
    let p = FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum()).finalise(),
        1.5,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    assert!(matches!(
        p.frequency_domain(
            input(
                FrequencyRegion::new(1.0, 2.0, -0.1, 0.1).unwrap(),
                radiation(1.5)
            ),
            ASSUME
        ),
        Err(FrequencyError::Radiation(
            super::super::radiation::RadiationReferenceError::Cutoff { side: 0 }
        ))
    ));
    // k0=0 is not a coordinate branch point: q=-beta² remains nonzero here.
    let d = p
        .frequency_domain(
            input(
                FrequencyRegion::new(-0.2, 0.2, -0.2, 0.2).unwrap(),
                explicit(c(0.0), [C::new(0.0, 1.5); 2]),
            ),
            ASSUME,
        )
        .unwrap();
    assert_eq!(
        d.search.dispersion().chart().coordinate(c(0.0)).unwrap(),
        c(0.0)
    );
    let mapped = d
        .search
        .dispersion()
        .chart()
        .map(J::variable(arr0(c(0.0)), 0).unwrap())
        .unwrap();
    assert_eq!(
        mapped.coordinates().vacuum_angular_wavenumber().first()[()],
        c(1.0)
    );
}
#[test]
fn frequency_overlap_keeps_native_evidence_and_finite_coverage() {
    let p = slab(100.0, Polarisation::TransverseElectric);
    let d = p.frequency_domain(slab_input(), ASSUME).unwrap();
    let mut i = slab_input();
    i.bounds = FrequencyRegion::new(4.4, 6.2, -3.3, 0.15).unwrap();
    let other = p.frequency_domain(i, ASSUME).unwrap();
    let plan = ChartPlan::explicit(p.clone(), vec![d.clone(), other], vec![]).unwrap();
    assert_eq!(*plan.geometry(), PlanGeometry::ExplicitFrequency);
    assert!(
        plan.exclusions()
            .iter()
            .any(|e| matches!(e, CoverageExclusion::Frequency(_)))
    );
    assert_eq!(
        plan.exclusions()
            .iter()
            .filter(|e| matches!(e, CoverageExclusion::OutsideFrequencyRegion { .. }))
            .count(),
        2
    );
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert_eq!(report.observations().len(), 2);
    assert_eq!(report.groups().len(), 1);
    assert_eq!(report.groups()[0].members().len(), 2);
    for observation in report.observations() {
        let root = observation.root();
        let k = root.evidence().z;
        assert_eq!(k, root.coordinate());
        assert!(contains(root.provenance().requested(), k));
        assert!(
            root.provenance()
                .same_domain(plan.domains()[observation.chart()].provenance())
        );
        let e = observation.enclosure();
        assert!(k.re >= e.x_min && k.re <= e.x_max && k.im >= e.y_min && k.im <= e.y_max);
    }
    for (n, description) in report
        .describe_points(InterpretationConfig::default())
        .into_iter()
        .enumerate()
    {
        let detail = description.result.unwrap();
        assert_eq!(
            detail.interpretation.radiation,
            RadiationQualification::OutgoingContinuation
        );
        let observation = &report.observations()[n];
        let single = p
            .describe_root(
                &plan.domains()[observation.chart()],
                observation.root(),
                InterpretationConfig::default(),
            )
            .result
            .unwrap();
        assert_eq!(detail, single);
    }
}
// Synthetic optical characteristic isolates chart plumbing/coverage from optics.
#[derive(Clone, Debug)]
struct Polynomial(Vec<C>);
impl ModeDeterminantSolver<J> for Polynomial {
    type Error = std::convert::Infallible;
    fn determinant(
        &self,
        coordinates: &lamina_core::CanonicalCoordinates<J>,
        _: &lamina_core::ExteriorWavevectors<J>,
    ) -> Result<J, Self::Error> {
        let k = coordinates.vacuum_angular_wavenumber();
        Ok(self.0.iter().fold(k.constant(c(1.0)), |value, root| {
            value.multiply(&k.subtract(&k.constant(*root)))
        }))
    }
}
fn synthetic(material: TestMaterial, roots: Vec<C>) -> ModalProblem<(), Polynomial> {
    let original = FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), material).finalise(),
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    ModalProblem {
        context: original.context,
        map: (),
        solver: Polynomial(roots),
    }
}
fn source_is<E: std::error::Error + 'static>(mut e: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if e.is::<E>() {
            return true;
        }
        match e.source() {
            Some(next) => e = next,
            None => return false,
        }
    }
}
#[test]
fn frequency_plans_keep_opposite_sheets_failures_and_clusters() {
    let p = synthetic(TestMaterial::lorentz(0.0, 0.0), vec![C::new(5.0, -0.5)]);
    let bounds = FrequencyRegion::new(4.0, 5.8, -0.8, 0.1).unwrap();
    let d = p
        .frequency_domain(input(bounds, radiation(5.0)), ASSUME)
        .unwrap();
    let mut branches = d.provenance().frequency().unwrap().reference;
    branches.kappa[1] = -branches.kappa[1];
    let other = p
        .frequency_domain(
            input(bounds, FrequencyReference::Explicit(branches)),
            ASSUME,
        )
        .unwrap();
    let plan = ChartPlan::explicit(p, vec![d, other], vec![]).unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert_eq!(report.observations().len(), 2);
    assert_eq!(report.groups().len(), 2);
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|d| d.reason == super::super::orchestration::Ambiguity::IncompatibleBranch)
    );

    let p = synthetic(
        TestMaterial {
            model: Model::BadRegion,
            fault: None,
            derivative_fault: false,
        },
        vec![],
    );
    let good = p
        .frequency_domain(
            input(
                FrequencyRegion::new(4.0, 5.0, -0.4, 0.1).unwrap(),
                radiation(4.5),
            ),
            ASSUME,
        )
        .unwrap();
    let bad = p
        .frequency_domain(
            input(
                FrequencyRegion::new(6.1, 6.8, -0.4, 0.1).unwrap(),
                radiation(6.5),
            ),
            ASSUME,
        )
        .unwrap();
    let exclusion = FrequencyExclusion {
        region: FrequencyRegion::new(7.0, 8.0, -0.4, 0.1).unwrap(),
        reason: ExclusionReason::CallerDeclared("unsearched model region".into()),
    };
    let plan = ChartPlan::explicit(
        p,
        vec![good, bad],
        vec![CoverageExclusion::Frequency(exclusion.clone())],
    )
    .unwrap();
    let batch = plan.discover(DiscoveryConfig::default()).unwrap();
    assert!(batch.entries()[0].is_ok());
    let error = batch.entries()[1].as_ref().unwrap_err();
    assert!(source_is::<FrequencyError>(error));
    assert!(format!("{error:?}").contains("NonFiniteMaterial"));
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    assert!(report.charts()[0].refinement().is_some());
    assert!(report.charts()[1].refinement().is_none());
    assert!(
        report
            .exclusions()
            .contains(&CoverageExclusion::Frequency(exclusion))
    );
    assert!(std::ptr::eq(
        report.charts()[1].discovery(),
        &batch.entries()[1]
    ));

    let p = synthetic(
        TestMaterial::lorentz(0.0, 0.0),
        vec![C::new(4.5, -0.6), C::new(4.55, -0.61), C::new(5.5, -0.3)],
    );
    let d = p
        .frequency_domain(input(bounds, radiation(5.0)), ASSUME)
        .unwrap();
    let plan = ChartPlan::explicit(p, vec![d], vec![]).unwrap();
    let batch = plan
        .discover(DiscoveryConfig {
            max_iterations: 0,
            // Unsplit, so that the clusters stay visible.
            max_subdivisions: 0,
            ..DiscoveryConfig::default()
        })
        .unwrap();
    let report = plan
        .refine(&batch, RefinementConfig::default(), MatchConfig::default())
        .unwrap();
    let raw = report.charts()[0].discovery().as_ref().unwrap();
    assert_eq!(
        *raw.result().termination(),
        argloc::Termination::ExceededMaxIterations
    );
    assert!(!raw.result().leaves().is_empty());
    assert!(
        report.charts()[0]
            .refinement()
            .unwrap()
            .outcomes()
            .iter()
            .any(|o| matches!(o.status(), CandidateStatus::UnresolvedCluster))
    );
    assert!(
        raw.result()
            .estimates()
            .iter()
            .any(|e| e.estimate().multiplicity > 1)
    );
}
#[test]
fn frequency_refinement_and_description_retain_original_material_failures() {
    let flag = Arc::new(AtomicBool::new(false));
    let p = synthetic(
        TestMaterial {
            fault: Some(flag.clone()),
            derivative_fault: false,
            ..TestMaterial::lorentz(0.0, 0.0)
        },
        vec![C::new(5.0, -0.5)],
    );
    let d = p
        .frequency_domain(
            input(
                FrequencyRegion::new(4.0, 6.0, -0.8, 0.1).unwrap(),
                radiation(5.0),
            ),
            ASSUME,
        )
        .unwrap();
    let discovery = p.search(&d, ArgumentConfig::new(1e-6)).unwrap();
    // Deliberately mutate the test model to inject failures after discovery.
    flag.store(true, Ordering::SeqCst);
    let report = p
        .refine(&d, &discovery, RefinementConfig::default())
        .unwrap();
    assert!(std::ptr::eq(report.discovery(), &discovery));
    assert!(report.outcomes().iter().any(|o|matches!(o.status(),CandidateStatus::EvaluationFailed(e) if source_is::<FrequencyError>(e))));
    flag.store(false, Ordering::SeqCst);
    let accepted = p
        .refine(&d, &discovery, RefinementConfig::default())
        .unwrap();
    let root = accepted
        .outcomes()
        .iter()
        .find_map(|o| {
            if let CandidateStatus::Accepted(r) = o.status() {
                Some(r.as_ref())
            } else {
                None
            }
        })
        .unwrap();
    flag.store(true, Ordering::SeqCst);
    let description = p.describe_root(&d, root, InterpretationConfig::default());
    assert!(std::ptr::eq(description.root, root));
    let error = description.result.unwrap_err();
    assert!(source_is::<FrequencyError>(&error));
    assert!(format!("{error:?}").contains("NonFiniteMaterial"));
}
#[test]
fn sampled_cutoff_and_right_side_budget_preserve_identity_and_work() {
    let p = FrequencyProblem::frequency(
        Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum()).finalise(),
        1.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let d = p
        .frequency_domain(
            input(
                FrequencyRegion::new(0.5, 2.0, -0.2, 0.2).unwrap(),
                radiation(1.5),
            ),
            ASSUME,
        )
        .unwrap();
    assert!(
        matches!(p.evaluate(&d,c(1.0)),Err(ExecutionError::Optical(e)) if matches!(e.as_ref(),ChartDispersionError::Chart(FrequencyError::Exterior{side:0,source:ExteriorFailure::Cutoff,evaluations:1})))
    );
    let p = FrequencyProblem::frequency(
        Stack::from_analytical_materials(TestMaterial::exponential(false), Constant::vacuum())
            .finalise(),
        0.0,
        Polarisation::TransverseElectric,
        Scatter2::new(),
    )
    .unwrap();
    let anchor = c(1.5);
    let mut i = input(
        FrequencyRegion::new(1.0, 2.0, -0.2, 0.2).unwrap(),
        explicit(anchor, [(anchor / 2.0).exp(), anchor]),
    );
    i.continuation.max_evaluations = 7;
    let d = p.frequency_domain(i, ASSUME).unwrap();
    assert!(
        matches!(p.evaluate(&d,c(1.8)),Err(ExecutionError::Optical(e)) if matches!(e.as_ref(),ChartDispersionError::Chart(FrequencyError::Exterior{side:1,source:ExteriorFailure::EvaluationBudget,evaluations:7})))
    );
}
#[test]
fn tiles_cover_the_box_less_its_holes() {
    let r = |a, b, c, d| ConformalRegion::new(a, b, c, d).unwrap();
    let area = |t: &ConformalRegion<f64>| {
        let ((a, b), (c, d)) = (t.real_bounds(), t.imaginary_bounds());
        (b - a) * (d - c)
    };
    let bounds = r(0.0, 4.0, 0.0, 4.0);
    // One hole inside: bottom, left, right and top tiles.
    let tiles = super::tiles(bounds, [r(1.0, 2.0, 1.0, 3.0)].into_iter());
    assert_eq!(
        tiles,
        vec![
            r(0.0, 4.0, 0.0, 1.0),
            r(0.0, 1.0, 1.0, 3.0),
            r(2.0, 4.0, 1.0, 3.0),
            r(0.0, 4.0, 3.0, 4.0)
        ]
    );
    // A hole across a corner and one past an edge, clipped to the box.
    let holes = [r(-1.0, 1.0, -1.0, 1.0), r(3.0, 5.0, 2.0, 3.0)];
    let tiles = super::tiles(bounds, holes.into_iter());
    let total: f64 = tiles.iter().map(area).sum();
    assert!((total - (16.0 - 1.0 - 1.0)).abs() < 1e-12, "{tiles:?}");
    for (i, a) in tiles.iter().enumerate() {
        assert!(holes.iter().all(|h| !interiors_overlap(*a, *h)));
        assert!(tiles[i + 1..].iter().all(|b| !interiors_overlap(*a, *b)));
    }
}
