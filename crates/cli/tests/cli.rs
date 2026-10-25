//! Runs the `lldx-engine` binary and checks its log output.

use std::process::Command;

use serde_json::Value;

#[test]
fn json_log_format_writes_one_json_object_per_stderr_line() {
    let output = Command::new(env!("CARGO_BIN_EXE_lldx-engine"))
        .args(["--log-format", "json"])
        .output()
        .expect("binary should run");

    assert!(output.status.success(), "exit status: {}", output.status);
    assert!(output.stdout.is_empty(), "logs must not go to stdout");

    let stderr = String::from_utf8(output.stderr).expect("stderr should be UTF-8");
    let lines: Vec<Value> = stderr
        .lines()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("line is not JSON ({err}): {line}"))
        })
        .collect();

    assert!(
        lines
            .iter()
            .any(|event| event["fields"]["engine_version"] == lldx_engine::VERSION),
        "no startup event with engine_version in: {stderr}"
    );
}
