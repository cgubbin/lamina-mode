//! Centered local design derivatives of a tracked QNM reconstruction.

use super::*;

type Matrix = [[Complex; 2]; 2];

#[derive(Clone, Copy, Debug)]
pub struct QnmDesignSettings {
    pub step: f64,
    pub match_radius_per_cm: f64,
    pub channels: QnmConstantChannels,
    pub window: QnmWindowSettings,
}

#[derive(Clone, Copy, Debug)]
pub struct QnmDesignStation {
    pub offset: f64,
    pub coverage: QnmPoleCoverage,
    pub construction_samples: usize,
    pub held_out_samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct QnmPoleDesignRate {
    pub pole_per_cm: Complex,
    pub residue_per_cm: Matrix,
    /// Rate of 2π/Re(k₀) in centimetres per parameter unit.
    pub wavelength_cm: f64,
    /// Rate of −2 Im(k₀) in inverse centimetres per parameter unit.
    pub linewidth_per_cm: f64,
    pub pole_step_difference_per_cm: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QnmAmplitudeDesignRate {
    pub total: Matrix,
    pub principal: Matrix,
    pub regular: Matrix,
    pub max_step_difference: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QnmPeakTransmissionRate {
    pub wavenumber_per_cm: f64,
    pub transmission: f64,
    pub rate: f64,
    pub displaced_peak_rate: f64,
    pub max_step_difference: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum QnmDesignError {
    #[error("expected a positive finite step and reconciled nonempty base")]
    InvalidInput,
    #[error(transparent)]
    Tracking(#[from] QnmPoleSearchError),
    #[error("pole identity is unresolved at parameter offset {offset}")]
    Unresolved { offset: f64 },
    #[error(
        "real-window approximation failed its amplitude tolerance at parameter offset {offset}"
    )]
    Accuracy { offset: f64 },
    #[error(transparent)]
    Reconstruction(#[from] QnmWindowError),
}

#[derive(Debug)]
pub struct QnmDesignSensitivity {
    base: QnmScatteringWindow,
    step: f64,
    coarse: [QnmScatteringWindow; 2],
    fine: [QnmScatteringWindow; 2],
    pub stations: [QnmDesignStation; 4],
    pub pole_rates: Vec<QnmPoleDesignRate>,
}

/// Build a centered design derivative from a reconciled base and a parameterized
/// stack. The parameter offset is relative to the base geometry. The two
/// exterior materials and the transverse channel reference convention are fixed.
pub fn differentiate_qnm_design<S: Into<ModeStack>>(
    source: &QnmPoleSet,
    stack_at_offset: impl Fn(f64) -> S,
    settings: QnmDesignSettings,
) -> Result<QnmDesignSensitivity, QnmDesignError> {
    use QnmDesignError::*;
    let step = settings.step;
    if !step.is_finite()
        || step <= 0.0
        || 0.5 * step == 0.0
        || source.coverage != QnmPoleCoverage::Reconciled
        || source.poles.is_empty()
    {
        return Err(InvalidInput);
    }
    let reconstruct = |poles, offset| {
        let model = reconstruct_qnm_scattering_window(poles, settings.channels, settings.window)?;
        if !model.evidence.meets_amplitude_tolerance {
            return Err(Accuracy { offset });
        }
        Ok(model)
    };
    let base = reconstruct(source.poles.clone(), 0.0)?;
    let evaluate = |offset| {
        let tracked = track_qnm_poles(
            source,
            stack_at_offset(offset),
            source.request.clone(),
            QnmPoleTrackingSettings {
                match_radius_per_cm: settings.match_radius_per_cm,
                audit_coverage: false,
            },
        )?;
        if tracked.matches.len() != source.poles.len()
            || tracked.destination.poles.len() != source.poles.len()
            || tracked.destination.coverage == QnmPoleCoverage::Unresolved
        {
            return Err(Unresolved { offset });
        }
        let poles = tracked
            .matches
            .iter()
            .map(|&(_, j)| tracked.destination.poles[j].clone())
            .collect();
        let model = reconstruct(poles, offset)?;
        let station = QnmDesignStation {
            offset,
            coverage: tracked.destination.coverage,
            construction_samples: model.evidence.direct_construction_samples,
            held_out_samples: model.evidence.held_out_samples,
        };
        Ok((model, station))
    };
    let (cm, cm_station) = evaluate(-step)?;
    let (cp, cp_station) = evaluate(step)?;
    let (fm, fm_station) = evaluate(-step / 2.0)?;
    let (fp, fp_station) = evaluate(step / 2.0)?;
    let pole_rates = source
        .poles
        .iter()
        .enumerate()
        .map(|(i, pole)| {
            let rate = (fp.poles()[i].signature.k0 - fm.poles()[i].signature.k0) / step;
            let coarse = (cp.poles()[i].signature.k0 - cm.poles()[i].signature.k0) / (2.0 * step);
            QnmPoleDesignRate {
                pole_per_cm: rate,
                residue_per_cm: std::array::from_fn(|r| {
                    std::array::from_fn(|c| {
                        (fp.poles()[i].residue_per_cm[r][c] - fm.poles()[i].residue_per_cm[r][c])
                            / step
                    })
                }),
                wavelength_cm: -std::f64::consts::TAU * rate.re / pole.signature.k0.re.powi(2),
                linewidth_per_cm: -2.0 * rate.im,
                pole_step_difference_per_cm: (rate - coarse).norm(),
            }
        })
        .collect();
    Ok(QnmDesignSensitivity {
        base,
        step,
        coarse: [cm, cp],
        fine: [fm, fp],
        stations: [cm_station, cp_station, fm_station, fp_station],
        pole_rates,
    })
}

impl QnmDesignSensitivity {
    pub fn base(&self) -> &QnmScatteringWindow {
        &self.base
    }

    pub fn amplitudes_rate(&self, k: f64) -> Option<QnmAmplitudeDesignRate> {
        self.base.amplitudes(k)?;
        let (coarse, _) = self.evaluate(&self.coarse, k, 2.0 * self.step)?;
        let (principal, regular) = self.evaluate(&self.fine, k, self.step)?;
        let coarse_regular = regular_rate(&self.coarse, k, 2.0 * self.step)?;
        let total =
            std::array::from_fn(|i| std::array::from_fn(|j| principal[i][j] + regular[i][j]));
        let max_step_difference = (0..2)
            .flat_map(|i| {
                (0..2).map(move |j| (total[i][j] - coarse[i][j] - coarse_regular[i][j]).norm())
            })
            .fold(0.0, f64::max);
        Some(QnmAmplitudeDesignRate {
            total,
            principal,
            regular,
            max_step_difference,
        })
    }

    fn evaluate(
        &self,
        pair: &[QnmScatteringWindow; 2],
        k: f64,
        denominator: f64,
    ) -> Option<(Matrix, Matrix)> {
        let [minus, plus] = pair;
        let principal = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                self.base
                    .poles()
                    .iter()
                    .enumerate()
                    .map(|(p, pole)| {
                        let delta = k - pole.signature.k0;
                        let dk = (plus.poles()[p].signature.k0 - minus.poles()[p].signature.k0)
                            / denominator;
                        let dr = (plus.poles()[p].residue_per_cm[i][j]
                            - minus.poles()[p].residue_per_cm[i][j])
                            / denominator;
                        dr / delta + pole.residue_per_cm[i][j] * dk / (delta * delta)
                    })
                    .sum()
            })
        });
        Some((principal, regular_rate(pair, k, denominator)?))
    }

    /// Evaluate the power sensitivity at an isolated interior maximum of
    /// left-incident transmission. The caller supplies a bracket containing
    /// one resolved peak.
    pub fn transmission_peak_rate(&self, interval: [f64; 2]) -> Option<QnmPeakTransmissionRate> {
        let (k, transmission) = peak(&self.base, interval)?;
        let t = self.base.amplitudes(k)?[1][0];
        if t.norm_sqr() == 0.0 {
            return None;
        }
        let dt = self.amplitudes_rate(k)?.total[1][0];
        let rate = 2.0 * (t.conj() * dt).re * transmission / t.norm_sqr();
        let displaced = |pair: &[QnmScatteringWindow; 2], denominator| {
            Some((peak(&pair[1], interval)?.1 - peak(&pair[0], interval)?.1) / denominator)
        };
        let fine = displaced(&self.fine, self.step)?;
        let coarse = displaced(&self.coarse, 2.0 * self.step)?;
        Some(QnmPeakTransmissionRate {
            wavenumber_per_cm: k,
            transmission,
            rate,
            displaced_peak_rate: fine,
            max_step_difference: (fine - coarse).abs(),
        })
    }
}

fn regular_rate(pair: &[QnmScatteringWindow; 2], k: f64, denominator: f64) -> Option<Matrix> {
    let regular = |model: &QnmScatteringWindow| {
        let value = model.amplitudes(k)?;
        let pole = super::qnm_scattering::principal_parts(model.poles(), k.into());
        Some(std::array::from_fn::<_, 2, _>(|i| {
            std::array::from_fn::<_, 2, _>(|j| value[i][j] - pole[i][j])
        }))
    };
    let a = regular(&pair[0])?;
    let b = regular(&pair[1])?;
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| (b[i][j] - a[i][j]) / denominator)
    }))
}

fn peak(model: &QnmScatteringWindow, [lo, hi]: [f64; 2]) -> Option<(f64, f64)> {
    if lo >= hi {
        return None;
    }
    let sample = |k| model.power_ratios(k).map(|s| s[1][0]);
    let at = |i| {
        if i == 64 {
            hi
        } else {
            lo + (hi - lo) * i as f64 / 64.0
        }
    };
    let grid = (0..=64)
        .map(|i| sample(at(i)))
        .collect::<Option<Vec<_>>>()?;
    let mut indices = (1..64).filter(|&i| grid[i] > grid[i - 1] && grid[i] > grid[i + 1]);
    let i = indices.next()?;
    if indices.next().is_some() {
        return None;
    }
    let (mut lo, mut hi) = (at(i - 1), at(i + 1));
    for _ in 0..40 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if sample(a)? < sample(b)? {
            lo = a;
        } else {
            hi = b;
        }
    }
    let k = lo + (hi - lo) / 2.0;
    Some((k, sample(k)?))
}
