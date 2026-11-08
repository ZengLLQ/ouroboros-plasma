//! OuroborosPlasma executable entry point.

use clap::Parser;
use ouroboros_plasma::cli::Cli;

fn main() {
    let cli = Cli::parse();
    cli.run();
}
