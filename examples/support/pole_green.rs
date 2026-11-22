//! Independent value-only fundamental-solution Green reference. No production
//! normalization, determinant derivative or recovered mode fields are used.
#![allow(dead_code)]
use lamina_core::{AnalyticalMaterialStack, EvaluateMeromorphicMaterial, Polarisation, Scalar};
use lamina_mode::{BranchSignature, Complex};
pub fn c(v: f64) -> Complex {
    Complex::new(v, 0.)
}
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub field: Complex,
    pub flux: Complex,
}
#[derive(Clone, Copy)]
struct Medium {
    f: Complex,
    q: Complex,
}
fn advance(s: State, m: Medium, d: f64) -> State {
    let k = m.q.sqrt();
    let x = k * d;
    let sinc = if x.norm() < 1e-5 {
        c(1.) - x * x / 6. + x.powi(4) / 120.
    } else {
        x.sin() / x
    };
    State {
        field: x.cos() * s.field + m.f * d * sinc * s.flux,
        flux: -m.q / m.f * d * sinc * s.field + x.cos() * s.flux,
    }
}
pub struct Green {
    exterior: [Medium; 2],
    layers: Vec<(Medium, f64)>,
    kappa: [Complex; 2],
    pub width: f64,
}
impl Green {
    pub fn new(
        stack: &AnalyticalMaterialStack<Complex>,
        signature: BranchSignature,
        spectral: bool,
        lambda: Complex,
        pol: Polarisation,
    ) -> Self {
        let k0 = if spectral { lambda } else { signature.k0 };
        let beta = if spectral {
            signature.k_parallel
        } else {
            lambda
        };
        let medium = |m: &lamina_core::AnalyticalMaterialHandle<Complex>| {
            let e = m.evaluate_relative_permittivity_complex(Scalar(k0));
            let u = m.evaluate_relative_permeability_complex(Scalar(k0));
            Medium {
                f: if pol == Polarisation::TransverseElectric {
                    u
                } else {
                    e
                },
                q: e * u * k0 * k0 - beta * beta,
            }
        };
        let exterior = [
            medium(stack.left_exterior()),
            medium(stack.right_exterior()),
        ];
        let roots = [signature.left, signature.right];
        let kappa =
            std::array::from_fn(|i| roots[i] * (exterior[i].q / (roots[i] * roots[i])).sqrt());
        let layers: Vec<_> = stack
            .layers_left_to_right()
            .iter()
            .map(|l| (medium(l.material()), l.thickness().as_centimetres()))
            .collect();
        let width = layers.iter().map(|(_, d)| d).sum();
        Self {
            exterior,
            layers,
            kappa,
            width,
        }
    }
    pub fn left(&self, z: f64) -> State {
        let mut s = State {
            field: c(1.),
            flux: -Complex::i() * self.kappa[0] / self.exterior[0].f,
        };
        if z < 0. {
            return advance(s, self.exterior[0], z);
        }
        let mut x = 0.;
        for &(m, d) in &self.layers {
            let h = (z - x).min(d);
            if h > 0. {
                s = advance(s, m, h);
            }
            x += d;
            if z <= x {
                return s;
            }
        }
        advance(s, self.exterior[1], z - self.width)
    }
    pub fn right(&self, z: f64) -> State {
        let mut s = State {
            field: c(1.),
            flux: Complex::i() * self.kappa[1] / self.exterior[1].f,
        };
        if z > self.width {
            return advance(s, self.exterior[1], z - self.width);
        }
        let mut x = self.width;
        for &(m, d) in self.layers.iter().rev() {
            let h = (x - z).min(d);
            if h > 0. {
                s = advance(s, m, -h);
            }
            x -= d;
            if z >= x {
                return s;
            }
        }
        advance(s, self.exterior[0], z)
    }
    pub fn wronskian(&self) -> Complex {
        let a = self.left(0.);
        let b = self.right(0.);
        a.field * b.flux - a.flux * b.field
    }
    pub fn value(&self, z: f64, source: f64) -> Complex {
        self.left(z.min(source)).field * self.right(z.max(source)).field / self.wronskian()
    }
    pub fn source_flux_jump(&self, z: f64) -> Complex {
        let l = self.left(z);
        let r = self.right(z);
        (l.field * r.flux - l.flux * r.field) / self.wronskian()
    }
}
/// Known first normal-incidence pole for vacuum | n=2, d=.3 cm | vacuum.
pub fn cavity_pole() -> Complex {
    Complex::new(std::f64::consts::PI / 0.6, -3_f64.ln() / 0.6)
}
/// Unit right-outgoing primary field; scalar operator norm in the k0 plane.
pub fn cavity_norm(pol: Polarisation) -> Complex {
    cavity_pole()
        * 0.9
        * if pol == Polarisation::TransverseElectric {
            1.
        } else {
            -1.
        }
}
/// Contour quadrature of the independent Green function, not a norm derivative.
pub fn contour_residue(
    stack: &AnalyticalMaterialStack<Complex>,
    sig: BranchSignature,
    spectral: bool,
    pol: Polarisation,
    positions: [f64; 2],
    radius: f64,
    points: usize,
) -> Complex {
    let [z, zp] = positions;
    let center = if spectral { sig.k0 } else { sig.k_parallel };
    (0..points)
        .map(|j| {
            let offset = Complex::from_polar(
                radius,
                2. * std::f64::consts::PI * (j as f64 + 0.5) / points as f64,
            );
            Green::new(stack, sig, spectral, center + offset, pol).value(z, zp) * offset
        })
        .sum::<Complex>()
        / points as f64
}
