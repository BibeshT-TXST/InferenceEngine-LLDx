//! Development tasks for the workspace. Run with `cargo xtask <task>`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Args, Parser, Subcommand};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(name = "xtask", about = "Development tasks for InferenceEngine-LLDx")]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Debug, Subcommand)]
enum Task {
    /// Regenerate reference fixtures and check them against fixtures/manifest.json.
    Fixtures(FixturesArgs),
}

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("target").required(true).args(["tiny", "model"])))]
struct FixturesArgs {
    /// Tiny random-weight model, written to fixtures/tiny/.
    #[arg(long)]
    tiny: bool,
    /// SmolLM3-3B directory at the pinned revision, written to fixtures/out/smollm3-3b/.
    #[arg(long, value_name = "DIR")]
    model: Option<PathBuf>,
    /// Accept hashes that differ from the manifest (after an intended generator or version change).
    #[arg(long)]
    update: bool,
}

/// sha256 per fixture file, keyed by path relative to `fixtures/`.
type Hashes = BTreeMap<String, String>;

fn main() -> Result<()> {
    match Cli::parse().task {
        Task::Fixtures(args) => fixtures(&args),
    }
}

fn fixtures(args: &FixturesArgs) -> Result<()> {
    let root = workspace_root()?;
    let fixtures_dir = root.join("fixtures");
    let manifest_path = fixtures_dir.join("manifest.json");
    let (key, flag) = match &args.model {
        Some(dir) => (
            "smollm3-3b",
            vec!["--model".into(), dir.clone().into_os_string()],
        ),
        None => ("tiny", vec!["--tiny".into()]),
    };

    let previous_manifest = fs::read(&manifest_path).ok();
    let previous = match &previous_manifest {
        Some(bytes) => manifest_hashes(bytes, key)?,
        None => Hashes::new(),
    };

    let status = Command::new("uv")
        .args([
            "run",
            "--project",
            "fixtures",
            "--locked",
            "python",
            "-I",
            "fixtures/generate.py",
        ])
        .args(&flag)
        .current_dir(&root)
        .status()
        .context("failed to run `uv`; install it from https://docs.astral.sh/uv/")?;
    if !status.success() {
        bail!("fixtures/generate.py failed: {status}");
    }

    let recorded = manifest_hashes(&fs::read(&manifest_path)?, key)?;
    if recorded.is_empty() {
        bail!(
            "generate.py recorded no files for `{key}` in {}",
            manifest_path.display()
        );
    }
    for (file, want) in &recorded {
        let got = sha256(&fixtures_dir.join(file))?;
        if &got != want {
            bail!("{file}: sha256 {got} does not match the {want} generate.py recorded");
        }
    }

    if previous.is_empty() {
        println!(
            "recorded {} fixture hashes for `{key}`; run again to check reproducibility",
            recorded.len()
        );
        return Ok(());
    }
    let mismatches = diff(&previous, &recorded);
    if mismatches.is_empty() {
        println!(
            "all {} `{key}` fixtures match fixtures/manifest.json",
            recorded.len()
        );
        return Ok(());
    }
    if args.update {
        println!(
            "updated `{key}` hashes in fixtures/manifest.json:\n{}",
            mismatches.join("\n")
        );
        return Ok(());
    }
    // Keep the manifest as it was, so a failed check can't overwrite the recorded hashes.
    if let Some(bytes) = previous_manifest {
        fs::write(&manifest_path, bytes)?;
    }
    bail!(
        "`{key}` fixtures differ from fixtures/manifest.json:\n{}\n\
         If the change is intended, rerun with --update.",
        mismatches.join("\n")
    )
}

/// The workspace root, two levels above this crate's manifest.
fn workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .context("xtask is not inside the workspace")
}

fn manifest_hashes(bytes: &[u8], key: &str) -> Result<Hashes> {
    let manifest: Value =
        serde_json::from_slice(bytes).context("fixtures/manifest.json is not valid JSON")?;
    let Some(files) = manifest.get(key).and_then(|entry| entry.get("files")) else {
        return Ok(Hashes::new());
    };
    serde_json::from_value(files.clone())
        .with_context(|| format!("`{key}.files` is not a map of path to sha256"))
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            // Writing to a String cannot fail.
            let _ = write!(hex, "{byte:02x}");
            hex
        }))
}

fn diff(previous: &Hashes, current: &Hashes) -> Vec<String> {
    let mut lines = Vec::new();
    for (file, old) in previous {
        match current.get(file) {
            None => lines.push(format!("  missing  {file}")),
            Some(new) if new != old => lines.push(format!("  changed  {file}: {old} -> {new}")),
            Some(_) => {}
        }
    }
    for file in current.keys().filter(|file| !previous.contains_key(*file)) {
        lines.push(format!("  new      {file}"));
    }
    lines
}
