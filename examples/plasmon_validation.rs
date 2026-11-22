#[path = "support/plasmon.rs"]
mod plasmon;
fn main() -> Result<(), plasmon::Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4
        || args[0] != "--suite"
        || args[2] != "--output"
        || !matches!(args[1].as_str(), "quick" | "full")
    {
        return Err("usage: plasmon_validation --suite quick|full --output NEW_DIRECTORY".into());
    }
    plasmon::run(std::path::Path::new(&args[3]), args[1] == "full")
}
