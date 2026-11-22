#![doc = include_str!("../docs/point-solver.md")]
#![doc = include_str!("../docs/modal-transform.md")]
#![allow(dead_code)]

mod analytic;
mod apm;
pub(crate) mod conformal;
mod dispersion;
mod mode;
mod problem;
mod refinement;
mod search;

#[cfg(test)]
mod checked_tests;

/// Physical-input point solvers and owned numerical evidence.
pub mod api;
pub use api::*;
