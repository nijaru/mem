//! Managed-layout default activation (storage slice E) integration tests.
//!
//! Each test spawns the real `mem` binary with an isolated `MEM_HOME`, so
//! these exercise the exact router construction the shipped CLI performs.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(home: &Path, args: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mem"));
    command.env("MEM_HOME", home).env_remove("MEM_DB");
    command.args(args).output().expect("spawn mem")
}

fn test_home(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mem-slice-e-{name}-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&path).expect("create test home");
    path
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("utf-8 stdout")
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("utf-8 stderr")
}

fn json_field(output: &str, field: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(output).expect("valid json");
    value[field].as_str().unwrap_or_default().to_owned()
}

fn context_texts(output: &std::process::Output) -> Vec<String> {
    let value: serde_json::Value = serde_json::from_str(&stdout(output)).expect("json");
    value["memories"]
        .as_array()
        .expect("memories array")
        .iter()
        .map(|memory| {
            memory["memory"]["text"]
                .as_str()
                .expect("memory text")
                .to_owned()
        })
        .collect()
}

#[test]
fn managed_layout_is_the_default_without_overrides() {
    let home = test_home("default");
    let output = run(&home, &["--json", "remember", "default layout fact"]);
    assert!(output.status.success(), "{}", stderr(&output));

    let output = run(&home, &["--json", "status"]);
    assert!(output.status.success());
    let database = json_field(&stdout(&output), "database");
    assert!(
        database.ends_with("layout-v1/user.db"),
        "expected the managed user store, got {database}"
    );

    let output = run(&home, &["--json", "context", "default layout fact"]);
    assert!(output.status.success());
    assert!(
        context_texts(&output).contains(&"default layout fact".to_owned()),
        "global memory must be recallable through the managed default"
    );

    std::fs::remove_dir_all(&home).expect("cleanup");
}

#[test]
fn db_override_keeps_exact_file_operation() {
    let home = test_home("exact");
    let db = home.join("exact.db");
    let output = run(
        &home,
        &[
            "--json",
            "--db",
            db.to_str().unwrap(),
            "remember",
            "exact fact",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));

    let output = run(&home, &["--json", "--db", db.to_str().unwrap(), "status"]);
    assert!(output.status.success());
    let database = json_field(&stdout(&output), "database");
    assert!(database.ends_with("exact.db"), "got {database}");
    // Exact operation must not create the managed layout.
    assert!(!home.join("layout-v1").is_dir());

    std::fs::remove_dir_all(&home).expect("cleanup");
}

#[test]
fn legacy_store_surfaces_migration_guidance() {
    let home = test_home("legacy");
    std::fs::write(home.join("memory.db"), b"legacy marker").expect("write legacy");

    // Storage-touching commands refuse with migration guidance.
    let output = run(&home, &["--json", "remember", "should not work"]);
    assert!(
        !output.status.success(),
        "must refuse while migration is pending"
    );
    let error = stderr(&output);
    assert!(
        error.contains("mem storage migrate"),
        "expected migration guidance, got: {error}"
    );

    // Inventory stays usable.
    let output = run(&home, &["--json", "storage", "status"]);
    assert!(output.status.success(), "{}", stderr(&output));

    std::fs::remove_dir_all(&home).expect("cleanup");
}

#[test]
fn migrated_layout_becomes_the_default_surface() {
    let home = test_home("migrated");
    let legacy = home.join("memory.db");
    let output = run(
        &home,
        &[
            "--json",
            "--db",
            legacy.to_str().unwrap(),
            "remember",
            "migrated legacy fact",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));

    let output = run(&home, &["--json", "storage", "migrate"]);
    assert!(output.status.success(), "{}", stderr(&output));

    let output = run(&home, &["--json", "context", "migrated legacy fact"]);
    assert!(output.status.success());
    assert!(
        context_texts(&output).contains(&"migrated legacy fact".to_owned()),
        "migrated rows must be visible through the managed default"
    );

    std::fs::remove_dir_all(&home).expect("cleanup");
}

#[test]
fn recall_respects_byte_budget() {
    let home = test_home("budget");
    // Global memories; from /tmp (no project) context reads them all.
    for text in [
        "budget fact one with a reasonably sized body",
        "budget fact two with a reasonably sized body",
        "budget fact three with a reasonably sized body",
    ] {
        let output = run(&home, &["--json", "remember", text]);
        assert!(output.status.success(), "{}", stderr(&output));
    }

    // A tight budget must cut recall below the item limit.
    let output = run(
        &home,
        &["--json", "context", "budget fact", "--max-bytes", "64"],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let texts = context_texts(&output);
    assert!(
        !texts.is_empty(),
        "the first hit must be included even when the budget is tight"
    );
    let total: usize = texts.iter().map(|text| text.len()).sum();
    // First hit always included; anything after must fit the budget.
    assert!(
        total <= 64 + texts[0].len(),
        "total {total} far exceeds the 64-byte budget"
    );

    // A generous budget returns everything up to the item limit.
    let output = run(
        &home,
        &["--json", "context", "budget fact", "--max-bytes", "100000"],
    );
    assert_eq!(context_texts(&output).len(), 3);

    // Zero disables the budget.
    let output = run(
        &home,
        &["--json", "context", "budget fact", "--max-bytes", "0"],
    );
    assert_eq!(context_texts(&output).len(), 3);

    std::fs::remove_dir_all(&home).expect("cleanup");
}
