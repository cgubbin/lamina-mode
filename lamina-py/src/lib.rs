//! Native module for the `lamina` Python package.
//!
//! Inputs and results cross the boundary in their serialised form (see
//! [`records`]). The Python package builds typed results with numpy arrays
//! from that form, so the canonical serialisation, its schema checks and its
//! hashes live in one place.
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

mod pair;
mod records;
mod sens;
mod solve;

use lamina_mode::Complex;
use records as rec;
use solve::Failure;

create_exception!(
    _native,
    LaminaError,
    PyException,
    "Base class for lamina failures."
);
create_exception!(
    _native,
    SchemaError,
    PyValueError,
    "A serialised record failed its schema checks."
);
create_exception!(
    _native,
    InvalidInput,
    PyValueError,
    "An input failed validation."
);
create_exception!(
    _native,
    EnumerationFailed,
    LaminaError,
    "Guided-mode enumeration failed."
);
create_exception!(
    _native,
    FieldsFailed,
    LaminaError,
    "Field recovery or power normalisation failed."
);
create_exception!(
    _native,
    DerivativesFailed,
    LaminaError,
    "Local derivative evaluation failed."
);
create_exception!(
    _native,
    IntegrationFailed,
    LaminaError,
    "A pairing integral exceeded its accuracy or sampling budget."
);

fn raise(failure: Failure) -> PyErr {
    match failure {
        Failure::Invalid(m) => InvalidInput::new_err(m),
        Failure::Enumeration(m) => EnumerationFailed::new_err(m),
        Failure::Fields(m) => FieldsFailed::new_err(m),
        Failure::Derivatives(m) => DerivativesFailed::new_err(m),
        Failure::Integration(m) => IntegrationFailed::new_err(m),
    }
}

fn json<R: rec::Record>(record: &R) -> PyResult<String> {
    rec::to_canonical(record).map_err(SchemaError::new_err)
}

fn parameter(json: &str) -> PyResult<rec::Parameter> {
    serde_json::from_str(json).map_err(|e| InvalidInput::new_err(format!("parameter: {e}")))
}

/// An enumerated basis whose accepted points stay available for sampling.
#[pyclass(frozen, module = "lamina._native")]
struct NativeBasis {
    solved: solve::Solved,
}

#[pymethods]
impl NativeBasis {
    /// Canonical serialised basis.
    fn record_json(&self) -> PyResult<String> {
        json(&self.solved.record)
    }

    /// Unit-power fields of one mode at the given z positions (m), serialised.
    fn sample_fields(&self, py: Python<'_>, mode_index: usize, z_m: Vec<f64>) -> PyResult<String> {
        let record = py
            .detach(|| solve::profile(&self.solved, mode_index, &z_m))
            .map_err(raise)?;
        json(&record)
    }

    /// Total sensitivity of one mode to a serialised parameter: dβ/dp, unit-power
    /// fields and their derivatives at z (m), and regional power fractions.
    #[pyo3(signature = (mode_index, parameter_json, z_m, frame))]
    fn sensitivity(
        &self,
        py: Python<'_>,
        mode_index: usize,
        parameter_json: &str,
        z_m: Vec<f64>,
        frame: &str,
    ) -> PyResult<String> {
        let parameter = parameter(parameter_json)?;
        let frame: rec::Frame = serde_json::from_value(serde_json::Value::String(frame.to_owned()))
            .map_err(|e| InvalidInput::new_err(format!("frame: {e}")))?;
        let record = py
            .detach(|| sens::sensitivity(&self.solved, mode_index, parameter, &z_m, frame))
            .map_err(raise)?;
        json(&record)
    }

    /// The certified group index of one mode, serialised.
    fn group_index(&self, py: Python<'_>, mode_index: usize, tolerance: f64) -> PyResult<String> {
        let record = py
            .detach(|| sens::group_index(&self.solved, mode_index, tolerance))
            .map_err(raise)?;
        json(&record)
    }

    /// Perturbation matrix and coupling coefficients of the basis along a
    /// serialised fixed-wavelength parameter, serialised.
    #[pyo3(signature = (parameter_json, gap_rad_per_m=None))]
    fn local_coupling(
        &self,
        py: Python<'_>,
        parameter_json: &str,
        gap_rad_per_m: Option<f64>,
    ) -> PyResult<String> {
        let parameter = parameter(parameter_json)?;
        let record = py
            .detach(|| pair::local_coupling(&self.solved, parameter, gap_rad_per_m))
            .map_err(raise)?;
        json(&record)
    }

    /// Project a supplied field onto the basis, serialised. `field(z_m, side)`
    /// returns twelve floats: Re E, Im E (V/m), Re H, Im H (A/m), each (x, y, z).
    /// It runs with the GIL held.
    #[pyo3(signature = (field, support_m, interfaces_m, origin_m, rank_rtol, field_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn project_field(
        &self,
        field: Bound<'_, PyAny>,
        support_m: [f64; 2],
        interfaces_m: Vec<f64>,
        origin_m: f64,
        rank_rtol: f64,
        field_id: Option<String>,
    ) -> PyResult<String> {
        let sample = |z_m: f64, side: &'static str| -> Result<[Complex; 6], pair::SampleError> {
            let values: Vec<f64> = field
                .call1((z_m, side))
                .and_then(|v| v.extract())
                .map_err(pair::SampleError::Raised)?;
            if values.len() != 12 {
                return Err(pair::SampleError::Invalid(format!(
                    "the field sampler must return 12 values, got {}",
                    values.len()
                )));
            }
            Ok(std::array::from_fn(|k| {
                let (part, component) = (k / 3, k % 3);
                let (re, im) = if part == 0 {
                    (values[component], values[3 + component])
                } else {
                    (values[6 + component], values[9 + component])
                };
                Complex::new(re, im)
            }))
        };
        let request = pair::ProjectionRequest {
            support_m,
            interfaces_m: &interfaces_m,
            origin_m,
            rank_rtol,
            field_id,
        };
        match pair::project(&self.solved, request, &sample) {
            Ok(record) => json(&record),
            Err(pair::ProjectionFailure::Failure(f)) => Err(raise(f)),
            Err(pair::ProjectionFailure::Raised(e)) => Err(e),
        }
    }

    /// Local characteristic partials and implicit slopes of one mode, serialised.
    fn local_derivatives(
        &self,
        py: Python<'_>,
        mode_index: usize,
        thickness_layers: Vec<usize>,
    ) -> PyResult<String> {
        let record = py
            .detach(|| solve::derivatives(&self.solved, mode_index, &thickness_layers))
            .map_err(raise)?;
        json(&record)
    }
}

/// Power overlap of two modes, serialised.
#[pyfunction]
fn power_overlap(
    py: Python<'_>,
    first: PyRef<'_, NativeBasis>,
    first_mode: usize,
    second: PyRef<'_, NativeBasis>,
    second_mode: usize,
    origins_m: [f64; 2],
) -> PyResult<String> {
    let (a, b) = (&first.solved, &second.solved);
    let record = py
        .detach(|| pair::power_overlap(a, first_mode, b, second_mode, origins_m))
        .map_err(raise)?;
    json(&record)
}

/// Derivative of the power overlap of two modes along one serialised
/// parameter per side (None holds that side fixed), serialised.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn power_overlap_sensitivity(
    py: Python<'_>,
    first: PyRef<'_, NativeBasis>,
    first_mode: usize,
    second: PyRef<'_, NativeBasis>,
    second_mode: usize,
    first_parameter_json: Option<&str>,
    second_parameter_json: Option<&str>,
    origins_m: [f64; 2],
) -> PyResult<String> {
    let first_parameter = first_parameter_json.map(parameter).transpose()?;
    let second_parameter = second_parameter_json.map(parameter).transpose()?;
    let (a, b) = (&first.solved, &second.solved);
    let record = py
        .detach(|| {
            pair::overlap_sensitivity(
                a,
                first_mode,
                b,
                second_mode,
                first_parameter,
                second_parameter,
                origins_m,
            )
        })
        .map_err(raise)?;
    json(&record)
}

/// Reciprocity flux of two modes, serialised.
#[pyfunction]
fn reciprocity(
    py: Python<'_>,
    first: PyRef<'_, NativeBasis>,
    first_mode: usize,
    second: PyRef<'_, NativeBasis>,
    second_mode: usize,
    origins_m: [f64; 2],
) -> PyResult<String> {
    let (a, b) = (&first.solved, &second.solved);
    let record = py
        .detach(|| pair::reciprocity(a, first_mode, b, second_mode, origins_m))
        .map_err(raise)?;
    json(&record)
}

/// Enumerate guided modes for a serialised input. Returns the native basis.
#[pyfunction]
fn enumerate_guided_modes(py: Python<'_>, input_json: &str) -> PyResult<NativeBasis> {
    let (input, _) =
        rec::canonicalise::<rec::GuidedInput>(input_json).map_err(SchemaError::new_err)?;
    let solved = py.detach(|| solve::enumerate(input)).map_err(raise)?;
    Ok(NativeBasis { solved })
}

/// Canonical form and digest of a serialised record of the named kind.
fn canonical(kind: &str, json: &str) -> PyResult<(String, String)> {
    fn go<R: rec::Record>(json: &str) -> Result<(String, String), String> {
        let (record, canonical) = rec::canonicalise::<R>(json)?;
        Ok((canonical, rec::digest(&record)?))
    }
    match kind {
        "guided_input" => go::<rec::GuidedInput>(json),
        "guided_basis" => go::<rec::GuidedBasis>(json),
        "guided_profile" => go::<rec::GuidedProfile>(json),
        "local_derivatives" => go::<rec::LocalDerivatives>(json),
        "guided_sensitivity" => go::<rec::GuidedSensitivity>(json),
        "group_index" => go::<rec::GroupIndex>(json),
        "guided_power_overlap" => go::<rec::PowerOverlap>(json),
        "guided_overlap_sensitivity" => go::<rec::OverlapSensitivity>(json),
        "guided_reciprocity" => go::<rec::Reciprocity>(json),
        "guided_local_coupling" => go::<rec::LocalCoupling>(json),
        "guided_projection" => go::<rec::FieldProjection>(json),
        other => Err(format!("unknown record kind {other:?}")),
    }
    .map_err(SchemaError::new_err)
}

/// Check a serialised record of the given kind and return its canonical form.
#[pyfunction]
fn canonical_json(kind: &str, json: &str) -> PyResult<String> {
    Ok(canonical(kind, json)?.0)
}

/// SHA-256 (hex) of a record's canonical form.
#[pyfunction]
fn canonical_hash(kind: &str, json: &str) -> PyResult<String> {
    Ok(canonical(kind, json)?.1)
}

/// Validate and normalise a serialised input without solving it.
#[pyfunction]
fn normalise_input(input_json: &str) -> PyResult<String> {
    let (input, _) =
        rec::canonicalise::<rec::GuidedInput>(input_json).map_err(SchemaError::new_err)?;
    json(&solve::normalise(input).map_err(raise)?)
}

/// lamina-mode's default refinement controls, serialised.
#[pyfunction]
fn default_refinement() -> PyResult<String> {
    serde_json::to_string(&solve::default_refinement())
        .map_err(|e| SchemaError::new_err(e.to_string()))
}

/// The conventions every result of this schema version carries, serialised.
#[pyfunction]
fn conventions() -> PyResult<String> {
    serde_json::to_string(&rec::Conventions::current())
        .map_err(|e| SchemaError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("SCHEMA_VERSION", rec::SCHEMA_VERSION)?;
    m.add("ASSUMPTION", solve::ASSUMPTION)?;
    m.add("LaminaError", py.get_type::<LaminaError>())?;
    m.add("SchemaError", py.get_type::<SchemaError>())?;
    m.add("InvalidInput", py.get_type::<InvalidInput>())?;
    m.add("EnumerationFailed", py.get_type::<EnumerationFailed>())?;
    m.add("FieldsFailed", py.get_type::<FieldsFailed>())?;
    m.add("DerivativesFailed", py.get_type::<DerivativesFailed>())?;
    m.add("IntegrationFailed", py.get_type::<IntegrationFailed>())?;
    m.add_class::<NativeBasis>()?;
    m.add_function(wrap_pyfunction!(enumerate_guided_modes, m)?)?;
    m.add_function(wrap_pyfunction!(power_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(power_overlap_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocity, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_hash, m)?)?;
    m.add_function(wrap_pyfunction!(normalise_input, m)?)?;
    m.add_function(wrap_pyfunction!(default_refinement, m)?)?;
    m.add_function(wrap_pyfunction!(conventions, m)?)?;
    Ok(())
}
