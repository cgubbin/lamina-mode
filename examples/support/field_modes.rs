#![allow(dead_code)]
use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Stack};
use lamina_materials::Drude;
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber as K};
pub const KP: f64 = 1e5;
pub const U: f64 = 0.5;
pub const D: f64 = 0.4;
pub const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
pub type Error = Box<dyn std::error::Error>;
pub fn c(v: f64) -> Complex {
    Complex::new(v, 0.)
}
pub fn metal(u: Complex) -> Complex {
    c(1.) - c(1.) / (u * (u + Complex::new(0., 0.02)))
}
pub fn film_equation(v: Complex, even: bool) -> Complex {
    let u = c(U);
    let e = metal(u);
    let ad = (v * v - u * u).sqrt();
    let am = (v * v - e * u * u).sqrt();
    let t = (am * (D / 2.)).tanh();
    ad + am / e * if even { t } else { c(1.) / t }
}
pub fn film_reference(even: bool) -> Complex {
    let mut v = if even {
        Complex::new(0.505, 0.001)
    } else {
        Complex::new(1.8, 0.08)
    };
    for _ in 0..60 {
        let h = c(1e-6);
        let step = film_equation(v, even)
            / ((film_equation(v + h, even) - film_equation(v - h, even)) / (2. * h));
        v -= step;
        if step.norm() < 1e-13 {
            break;
        }
    }
    assert!(film_equation(v, even).norm() < 1e-10);
    assert!(v.im > 0.);
    v
}
pub fn film(even: bool) -> Result<ModePoint, Error> {
    let material = Drude::new(
        1.,
        K::inverse_centimetres(KP),
        K::inverse_centimetres(0.02 * KP),
    )?;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(material, Length::centimetres(D / KP))
        .finalise();
    propagation(
        stack,
        U,
        film_reference(even),
        [c(1.); 2],
        Polarisation::TransverseMagnetic,
    )
}
pub fn propagation(
    stack: AnalyticalMaterialStack<Complex>,
    u: f64,
    target: Complex,
    epsilon: [Complex; 2],
    pol: Polarisation,
) -> Result<ModePoint, Error> {
    let kappa = |v: Complex| {
        epsilon.map(|e| {
            let k = (e * u * u - v * v).sqrt();
            if k.im < 0. { -k } else { k }
        })
    };
    let z_at = |v: Complex| {
        let k = kappa(v);
        ((k[0] + k[1]) / (Complex::i() * 2. * u)).ln()
    };
    let z = z_at(target);
    let anchor = target + Complex::new(1e-6, 1e-6);
    let zr = z_at(anchor);
    let p = PropagationProblem::new(stack, u * KP, pol)?;
    let d = p.domain(
        ConformalRegion::new(z.re - 0.025, z.re + 0.025, z.im - 0.025, z.im + 0.025)?,
        ReferenceSelection::Coordinate {
            reference: zr,
            coordinate: anchor * KP,
        },
        ConformalChartConfig::default(),
        ASSUME,
    )?;
    let report = p.plan(vec![d], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig {
            residual_atol: Some(1e-8),
            residual_rtol: 1e-11,
            residual_scale: KP,
            ..Default::default()
        },
        MatchConfig::default(),
    )?;
    if report.observations().len() != 1 {
        return Err(format!("expected one root: {report:?}").into());
    }
    let point = report.observations()[0].clone();
    assert!((point.physical_coordinate() / KP - target).norm() < 1e-8);
    Ok(point)
}
pub fn interface(temporal: bool) -> Result<ModePoint, Error> {
    let m = Drude::new(
        1.,
        K::inverse_centimetres(KP),
        K::inverse_centimetres(0.02 * KP),
    )?;
    let stack = Stack::from_analytical_materials(Constant::vacuum(), m).finalise();
    if !temporal {
        let e = metal(c(U));
        return propagation(
            stack,
            U,
            c(U) * (e / (c(1.) + e)).sqrt(),
            [c(1.), e],
            Polarisation::TransverseMagnetic,
        );
    }
    let v = 1.;
    let mut u = Complex::new(0.618, -0.007);
    for _ in 0..30 {
        let f = u.powi(4) + Complex::i() * 0.02 * u.powi(3) - 3. * u * u - Complex::i() * 0.04 * u
            + c(v);
        let df = 4. * u.powi(3) + Complex::i() * 0.06 * u * u - 6. * u - Complex::i() * 0.04;
        u -= f / df;
    }
    let anchor = u + Complex::new(0.001, 0.0002);
    let roots = [c(1.), metal(anchor)].map(|e| {
        let k = (e * anchor * anchor - c(v * v)).sqrt();
        if k.im < 0. { -k * KP } else { k * KP }
    });
    let p = FrequencyProblem::new(stack, v * KP, Polarisation::TransverseMagnetic)?;
    let d = p.domain(
        FrequencyDomainInput {
            bounds: FrequencyRegion::new(
                (u.re - 0.01) * KP,
                (u.re + 0.01) * KP,
                (u.im - 0.003) * KP,
                (u.im + 0.003) * KP,
            )?,
            reference: FrequencyReference::Explicit(ExteriorReferenceValues {
                frequency: anchor * KP,
                kappa: roots,
            }),
            continuation: ContinuationConfig::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        ASSUME,
    )?;
    let r = p.plan(vec![d], vec![])?.solve(
        DiscoveryConfig::default(),
        RefinementConfig {
            residual_atol: Some(1e-8),
            residual_rtol: 1e-11,
            residual_scale: KP,
            native_atol: 1e-5,
            ..Default::default()
        },
        MatchConfig::default(),
    )?;
    if r.observations().len() != 1 {
        return Err(format!("expected interface pole: {r:?}").into());
    }
    Ok(r.observations()[0].clone())
}
