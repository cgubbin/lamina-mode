#[path = "support/pole_validation.rs"]
mod validation;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4
        || args[0] != "--suite"
        || args[2] != "--output"
        || !matches!(args[1].as_str(), "quick" | "full")
    {
        return Err("usage: pole_normalization --suite quick|full --output NEW_DIRECTORY".into());
    }
    validation::run(std::path::Path::new(&args[3]), args[1] == "full")
}
