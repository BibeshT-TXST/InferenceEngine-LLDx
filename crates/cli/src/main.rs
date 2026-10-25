//! `lldx-engine`: development and benchmark CLI for the `lldx-engine` library.
//!
//! Logs go to stderr so stdout stays free for generated text.

use std::io::IsTerminal;

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "lldx-engine", version, about)]
struct Cli {
    /// Format of the log lines written to stderr.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    log_format: LogFormat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LogFormat {
    /// One JSON object per line, for machines.
    Json,
    /// Multi-line, human-readable output.
    Pretty,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.log_format)?;
    tracing::info!(engine_version = lldx_engine::VERSION, "starting");
    Ok(())
}

/// Installs the global tracing subscriber. Only binaries do this; the library never does.
fn init_tracing(format: LogFormat) -> anyhow::Result<()> {
    let builder = tracing_subscriber::fmt().with_writer(std::io::stderr);
    match format {
        LogFormat::Json => builder.json().try_init(),
        LogFormat::Pretty => builder
            .pretty()
            .with_ansi(std::io::stderr().is_terminal())
            .try_init(),
    }
    .map_err(|err| anyhow::anyhow!(err))
}
