//! Native module for the `lamina` Python package.
//!
//! Inputs and results cross the boundary in their serialised form (see
//! [`records`]). The Python package builds typed results with numpy arrays
//! from that form, so the canonical serialisation, its schema checks and its
//! hashes live in one place.
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

mod records;
mod solve;

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

fn raise(failure: Failure) -> PyErr {
    match failure {
        Failure::Invalid(m) => InvalidInput::new_err(m),
        Failure::Enumeration(m) => EnumerationFailed::new_err(m),
        Failure::Fields(m) => FieldsFailed::new_err(m),
        Failure::Derivatives(m) => DerivativesFailed::new_err(m),
    }
}

fn json<R: rec::Record>(record: &R) -> PyResult<String> {
    rec::to_canonical(record).map_err(SchemaError::new_err)
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
    m.add_class::<NativeBasis>()?;
    m.add_function(wrap_pyfunction!(enumerate_guided_modes, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_hash, m)?)?;
    m.add_function(wrap_pyfunction!(normalise_input, m)?)?;
    m.add_function(wrap_pyfunction!(default_refinement, m)?)?;
    m.add_function(wrap_pyfunction!(conventions, m)?)?;
    Ok(())
}
