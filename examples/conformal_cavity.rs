//! Constant-exterior outgoing cavity pole on an explicit conformal chart.
mod support;
fn main() -> Result<(), lamina_mode::Diagnostic> {
    support::print(&support::constant_cavity()?)
}
