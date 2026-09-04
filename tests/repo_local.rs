//! Integration coverage for repo-local default storage.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use uuid::Uuid;

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("mem-repo-local-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
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
