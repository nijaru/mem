//! Integration coverage for repo-local default storage.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use uuid::Uuid;

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("mem-repo-local-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    // Canonicalize so expectations match the git-reported repo root: on macOS
    // the OS temp dir is /var/... while git resolves the same directory as
    // /private/var/...
    std::fs::canonicalize(&path).expect("canonicalize temp dir")
}

fn command(cwd: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mem"));
    command
        .current_dir(cwd)
        .env_remove("MEM_DB")
        .env_remove("MEM_HOME");
    command
}

fn run(cwd: &Path, args: &[&str]) -> Output {
    command(cwd).args(args).output().expect("spawn mem")
}

fn run_json(cwd: &Path, args: &[&str]) -> serde_json::Value {
    let output = command(cwd)
        .arg("--json")
        .args(args)
        .output()
        .expect("spawn mem");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("parse JSON")
}

#[test]
fn init_creates_repo_local_store_and_ignore_file() {
    let cwd = temp_dir();
    let value = run_json(&cwd, &["init"]);
    let db = cwd.join(".mem/mem.db");
    assert!(db.is_file());
    assert_eq!(value["database"], db.display().to_string());
    assert_eq!(
        std::fs::read_to_string(cwd.join(".mem/.gitignore")).unwrap(),
        "*\n"
    );
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn absent_store_reads_do_not_initialize_storage() {
    let cwd = temp_dir();
    let status = run_json(&cwd, &["status"]);
    assert_eq!(status["schema_version"], 0);
    assert_eq!(status["total"], 0);
    assert_eq!(
        run_json(&cwd, &["search", "nothing"])
            .as_array()
            .map(Vec::len),
        Some(0)
    );
    assert_eq!(
        run_json(&cwd, &["context", "nothing"])["memories"]
            .as_array()
            .map(Vec::len),
        Some(0)
    );
    assert!(!cwd.join(".mem").exists());
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn write_initializes_repo_local_storage_without_init() {
    let cwd = temp_dir();
    let value = run_json(
        &cwd,
        &["remember", "repo-local memory", "--source-type", "test"],
    );
    assert_eq!(value["text"], "repo-local memory");
    assert!(cwd.join(".mem/mem.db").is_file());
    assert!(cwd.join(".mem/.gitignore").is_file());
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn explicit_database_bypasses_repo_local_storage() {
    let cwd = temp_dir();
    let db = cwd.join("exact.db");
    let output = command(&cwd)
        .args([
            "--db",
            db.to_str().unwrap(),
            "remember",
            "exact memory",
            "--source-type",
            "test",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(db.is_file());
    assert!(!cwd.join(".mem").exists());
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn nested_commands_reuse_nearest_existing_mem_directory() {
    let root = temp_dir();
    let nested = root.join("a/b/c");
    std::fs::create_dir_all(&nested).unwrap();
    assert!(run(&root, &["init"]).status.success());
    let value = run_json(&nested, &["status"]);
    assert_eq!(
        value["database"],
        root.join(".mem/mem.db").display().to_string()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn git_repo_root_is_default_project_boundary() {
    let root = temp_dir();
    let git = Command::new("git")
        .args(["init", "-q", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(git.status.success());
    let nested = root.join("src/deep");
    std::fs::create_dir_all(&nested).unwrap();
    let value = run_json(&nested, &["status"]);
    assert_eq!(
        value["database"],
        root.join(".mem/mem.db").display().to_string()
    );
    assert!(
        !root.join(".mem").exists(),
        "read must not initialize the repo store"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn nested_mem_directory_does_not_shadow_git_root_store() {
    let root = temp_dir();
    let git = Command::new("git")
        .args(["init", "-q", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(git.status.success());
    assert!(run(&root, &["init"]).status.success());

    let nested = root.join("src/deep");
    std::fs::create_dir_all(nested.join(".mem")).unwrap();
    let value = run_json(&nested, &["status"]);
    assert_eq!(
        value["database"],
        root.join(".mem/mem.db").display().to_string()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn context_max_bytes_is_a_hard_bound() {
    let cwd = temp_dir();
    run_json(
        &cwd,
        &[
            "remember",
            "oversized memory payload",
            "--source-type",
            "test",
        ],
    );
    let value = run_json(&cwd, &["context", "oversized", "--max-bytes", "4"]);
    assert_eq!(value["memories"].as_array().map(Vec::len), Some(0));
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn version_and_status_report_build_identity() {
    let cwd = temp_dir();
    let status = run_json(&cwd, &["status"]);
    let build = status["build_commit"]
        .as_str()
        .expect("status reports build_commit");
    assert!(!build.trim().is_empty(), "build identity must not be empty");
    let version = String::from_utf8_lossy(
        &command(&cwd)
            .arg("--version")
            .output()
            .expect("spawn mem")
            .stdout,
    )
    .to_string();
    assert!(
        version.contains(build),
        "--version must report the same build identity: {version}"
    );
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn export_emits_ndjson_active_by_default_and_preserves_lineage() {
    let cwd = temp_dir();
    let first = run_json(&cwd, &["remember", "first memory", "--source-type", "test"]);
    let second = run_json(
        &cwd,
        &["remember", "second memory", "--source-type", "test"],
    );
    run_json(
        &cwd,
        &[
            "correct",
            second["id"].as_str().unwrap(),
            "second memory, corrected",
        ],
    );
    run(&cwd, &["forget", first["id"].as_str().unwrap()]);

    let output = command(&cwd).args(["export"]).output().expect("spawn mem");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let records: Vec<serde_json::Value> = stdout
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .expect("export output is valid NDJSON");
    // Deleted memories never export; the corrected replacement is active.
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["text"], "second memory, corrected");
    assert_eq!(records[0]["status"], "active");
    assert!(records[0]["superseded_by"].is_null());
    // Full field set on every record.
    for field in [
        "id",
        "kind",
        "text",
        "actor",
        "source_type",
        "source_ref",
        "status",
        "superseded_by",
        "created_at",
        "updated_at",
        "deleted_at",
    ] {
        assert!(records[0].get(field).is_some(), "missing field: {field}");
    }

    let lineage = command(&cwd)
        .args(["export", "--include-superseded"])
        .output()
        .expect("spawn mem");
    assert!(lineage.status.success());
    let lineage_records: Vec<serde_json::Value> = String::from_utf8_lossy(&lineage.stdout)
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .expect("lineage export is valid NDJSON");
    assert_eq!(lineage_records.len(), 2);
    assert_eq!(lineage_records[0]["status"], "superseded");
    assert_eq!(
        lineage_records[0]["superseded_by"],
        lineage_records[1]["id"]
    );
    assert_eq!(lineage_records[1]["status"], "active");

    // Absent store: export succeeds with zero lines and creates nothing.
    let empty = temp_dir();
    let output = command(&empty)
        .args(["export"])
        .output()
        .expect("spawn mem");
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!empty.join(".mem").exists());
    std::fs::remove_dir_all(cwd).unwrap();
    std::fs::remove_dir_all(empty).unwrap();
}

#[test]
fn export_records_are_stable_across_runs_and_composable_with_jq() {
    let cwd = temp_dir();
    run_json(
        &cwd,
        &["remember", "stable memory", "--source-type", "test"],
    );
    let first = command(&cwd).args(["export"]).output().expect("spawn mem");
    let second = command(&cwd).args(["export"]).output().expect("spawn mem");
    assert_eq!(
        first.stdout, second.stdout,
        "repeated exports must be diffable"
    );
    std::fs::remove_dir_all(cwd).unwrap();
}

#[test]
fn export_to_early_exiting_consumer_exits_quietly() {
    // A consumer closing the pipe after reading enough records is a normal
    // exit, not an export failure: mem must not print a broken-pipe error or
    // fail the pipeline. Output must exceed the pipe buffer to force EPIPE.
    let cwd = temp_dir();
    for index in 0..400 {
        run(
            &cwd,
            &[
                "remember",
                &format!("padding record to exceed the pipe buffer {index}"),
                "--source-type",
                "test",
            ],
        );
    }
    let mut mem = command(&cwd)
        .args(["export"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn mem");
    let mut stdout = mem.stdout.take().expect("capture mem stdout");
    // Read a little, then drop the pipe so the remaining export hits EPIPE.
    let mut buffer = [0u8; 256];
    stdout.read_exact(&mut buffer).expect("read first records");
    drop(stdout);
    let output = mem.wait_with_output().expect("wait for mem");
    assert_eq!(
        output.status.code(),
        Some(0),
        "early-exiting consumer must not fail mem; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "broken pipe must be silent; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(cwd).unwrap();
}
