//! Conformal coordinate mappings for modal analysis.
//!
//! A conformal map parameterises a planar optical problem by a complex
//! coordinate `z`. Evaluating the map produces the canonical plane-wave
//! coordinates required by `lamina-core`, together with the analytically
//! continued exterior transverse wavevectors.
//!
//! The exterior wavevectors are supplied explicitly because their branch
//! choice forms part of the modal radiation condition. Interior transverse
//! wavevectors remain the responsibility of `lamina-core`.

mod branch;
pub(crate) mod chart;
mod landmarks;
mod region;

pub mod in_plane;
pub mod wavenumber;

pub use region::{ConformalRegion, ConformalRegionError};

use lamina_core::{CanonicalCoordinates, ExteriorWavevectors};

/// A planar optical problem evaluated at one point in the conformal plane.
///
/// All fields use the same algebra type `A`. Consequently `A` may be a
/// complex scalar for value evaluation or a jet algebra seeded in the
/// conformal coordinate for derivative evaluation.
#[derive(Clone, Debug)]
pub struct ConformalPoint<A> {
    coordinates: CanonicalCoordinates<A>,
    exterior_wavevectors: ExteriorWavevectors<A>,
}

impl<A> ConformalPoint<A> {
    pub fn new(
        coordinates: CanonicalCoordinates<A>,
        exterior_wavevectors: ExteriorWavevectors<A>,
    ) -> Self {
        Self {
            coordinates,
            exterior_wavevectors,
        }
    }

    pub fn coordinates(&self) -> &CanonicalCoordinates<A> {
        &self.coordinates
    }

    pub fn exterior_wavevectors(&self) -> &ExteriorWavevectors<A> {
        &self.exterior_wavevectors
    }

    pub fn into_parts(self) -> (CanonicalCoordinates<A>, ExteriorWavevectors<A>) {
        (self.coordinates, self.exterior_wavevectors)
    }
}

/// Maps a conformal coordinate onto a canonical planar-wave problem.
///
/// Implementations must analytically continue the exterior transverse
/// wavevectors onto the modal sheet represented by the map. In particular,
/// implementations must not reconstruct the exterior wavevectors using a
/// pointwise principal square root.
///
/// Construction and search-domain validation are responsible for ensuring
/// that the map is only used where its analytic assumptions hold.
pub trait RawConformalMap<A> {
    /// Evaluate the physical problem corresponding to conformal coordinate `z`.
    fn map_raw(&self, z: A) -> ConformalPoint<A>;
}

pub trait ValidateConformalRegion<R> {
    type Error: Send + Sync + std::fmt::Display + 'static + Clone + std::error::Error;

    fn validate_region(&self, region: &ConformalRegion<R>) -> Result<(), Self::Error>;
}
