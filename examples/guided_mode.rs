//! TE guided propagation mode from an existing core analytical stack.
mod support;
fn main() -> Result<(), lamina_mode::Diagnostic> {
    support::print(&support::guided()?)
}
