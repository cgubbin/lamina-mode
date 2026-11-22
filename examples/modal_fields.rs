#[path = "support/mod.rs"]
mod existing;
#[path = "support/field_modes.rs"]
mod fixtures;
use lamina_core::{ExteriorSampling, FieldSampling, LayerSampling, Scatter2};
use lamina_mode::*;
use lamina_units::Length;
use std::{fs::File, io::Write, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .ok_or("usage: modal_fields NEW_DIRECTORY")?;
    let out = Path::new(&out);
    std::fs::create_dir(out)?;
    let mut csv = File::create(out.join("fields.csv"))?;
    let mut evidence = File::create(out.join("evidence.txt"))?;
    writeln!(
        csv,
        "mode,z,ex_re,ex_im,ey_re,ey_im,ez_re,ez_im,hx_re,hx_im,hy_re,hy_im,hz_re,hz_im"
    )?;
    let modes = [
        ("interface_spatial", fixtures::interface(false)?),
        ("interface_temporal", fixtures::interface(true)?),
        ("film_even_H", fixtures::film(true)?),
        ("film_odd_H", fixtures::film(false)?),
        (
            "leaky_cavity",
            existing::constant_cavity()?.observations()[0].clone(),
        ),
    ];
    for (name, point) in modes {
        let film = name.starts_with("film");
        let leaky = name == "leaky_cavity";
        let scale = if leaky { 1. } else { fixtures::KP };
        let width = if film {
            fixtures::D
        } else if leaky {
            0.3
        } else {
            0.
        };
        let extent = if leaky { 0.4 } else { 4. };
        let mut sampling = FieldSampling::new().left_exterior(ExteriorSampling::uniform(
            81,
            Length::centimetres(extent / scale),
        ));
        let mut x: Vec<f64> = (0..81).map(|i| -extent + extent * i as f64 / 80.).collect();
        if film || leaky {
            sampling = sampling.layer(0, LayerSampling::uniform_with_boundaries(81, true, true));
            x.extend((0..81).map(|i| width * i as f64 / 80.));
        }
        sampling = sampling.right_exterior(ExteriorSampling::uniform(
            81,
            Length::centimetres(extent / scale),
        ));
        x.extend((0..81).map(|i| width + extent * i as f64 / 80.));
        let settings = FieldRecoverySettings {
            characteristic_limit: 1e-5,
            ..Default::default()
        };
        let raw = point.recover_fields_with(Scatter2::new(), &sampling, settings)?;
        let primary = if leaky {
            raw.fields.quantity().electric().y()[80]
        } else {
            raw.fields.quantity().magnetic().y()[80]
        };
        let fields = point.recover_fields_with(
            Scatter2::new(),
            &sampling,
            FieldRecoverySettings {
                amplitude: Complex::new(1., 0.) / primary,
                ..settings
            },
        )?;
        writeln!(
            evidence,
            "{name}: units z*{} cm^-1; interface primary=1 complex gauge; {:#?}",
            scale, fields
        )?;
        let f = fields.fields.quantity();
        let arrays = [
            f.electric().x(),
            f.electric().y(),
            f.electric().z(),
            f.magnetic().x(),
            f.magnetic().y(),
            f.magnetic().z(),
        ];
        for (i, z) in x.iter().enumerate() {
            write!(csv, "{name},{z}")?;
            for a in arrays {
                write!(csv, ",{},{}", a[i].re, a[i].im)?;
            }
            writeln!(csv)?;
        }
    }
    Ok(())
}
