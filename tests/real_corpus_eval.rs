use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use uuid::Uuid;

const LIMIT: usize = 10;

struct Seed<'a> {
    key: &'a str,
    text: &'a str,
    kind: &'a str,
}

struct Case<'a> {
    key: &'a str,
    query: &'a str,
}

#[test]
#[ignore = "downloads/loads the embedding model; intended for the retrieval-eval workflow"]
fn current_product_knowledge_survives_paraphrased_retrieval() {
    let db = test_path();
    let seeds = corpus();
    let cases = cases();
    let mut ids = HashMap::new();
    for seed in &seeds {
        let output = remember(&db, seed);
        ids.insert(seed.key, output["id"].as_str().expect("id").to_owned());
    }
    let index = run_json(&db, &["index", "-n", "64"]);
    assert_eq!(index["indexed"].as_u64(), Some(seeds.len() as u64));

    let mut hits = 0usize;
    for case in &cases {
        let expected = ids.get(case.key).expect("target seed");
        let ranking = context_ids(&db, case.query);
        if ranking.iter().take(3).any(|id| id == expected) {
            hits += 1;
        }
    }
    assert!(
        hits * 3 >= cases.len() * 2,
        "current-product context hit@3 below two thirds: {hits}/{}",
        cases.len()
    );
    cleanup(&db);
}

fn corpus() -> Vec<Seed<'static>> {
    vec![
        Seed {
            key: "package",
            text: "The crates.io package is mem-cli while the installed executable and repository are named mem.",
            kind: "fact",
        },
        Seed {
            key: "toolchain",
            text: "Rust 1.98.0 is pinned and CI gates changes with rustfmt, all-target tests, strict Clippy, and a release build.",
            kind: "constraint",
        },
        Seed {
            key: "core",
            text: "The durable core contains semantic memories and compact workspace continuation state; episodic session projection is outside the current product.",
            kind: "decision",
        },
        Seed {
            key: "provenance",
            text: "Memory provenance is stored directly as actor, source type, and optional source reference rather than through a generic relation graph.",
            kind: "decision",
        },
        Seed {
            key: "fts",
            text: "FTS5 lexical retrieval is synchronous and always available; embeddings are derived local data and never a correctness dependency.",
            kind: "decision",
        },
        Seed {
            key: "fencing",
            text: "Embedding commits are fenced by the memory updated_at that produced them, so a stale vector is discarded if the source changes before commit.",
            kind: "constraint",
        },
        Seed {
            key: "queue",
            text: "There is no durable embedding job queue; index discovers active memories missing a current-model vector each time it runs.",
            kind: "decision",
        },
        Seed {
            key: "oneshot",
            text: "mem remains a one-shot CLI because current latency does not justify a daemon or resident service.",
            kind: "decision",
        },
        Seed {
            key: "writes",
            text: "Durable semantic memories are written explicitly; automatic conversation extraction is not part of the core.",
            kind: "constraint",
        },
        Seed {
            key: "backend",
            text: "SQLite-specific storage is kept direct instead of introducing a generic backend abstraction without a real second backend.",
            kind: "constraint",
        },
        Seed {
            key: "correction",
            text: "Correcting a memory creates a replacement and marks the predecessor superseded in the same transaction.",
            kind: "procedure",
        },
        Seed {
            key: "workspace",
            text: "Workspace continuation state is keyed inside the repo-local database so separate worktrees can keep independent resume points.",
            kind: "decision",
        },
        Seed {
            key: "tasks",
            text: "Task state stays outside mem; workspace state can carry an external task reference without owning task management.",
            kind: "constraint",
        },
        Seed {
            key: "local",
            text: "The normal storage model is one repo-local SQLite database, with --db and MEM_DB reserved as explicit overrides.",
            kind: "decision",
        },
        Seed {
            key: "compact",
            text: "Durable memories should be concise semantic statements rather than copies of entire source documents.",
            kind: "preference",
        },
    ]
}

fn cases() -> Vec<Case<'static>> {
    vec![
        Case {
            key: "package",
            query: "Why is the Cargo package name different from the command name?",
        },
        Case {
            key: "toolchain",
            query: "What exact checks qualify a change before it lands?",
        },
        Case {
            key: "core",
            query: "Does the current tool still ingest and project whole past sessions?",
        },
        Case {
            key: "provenance",
            query: "Where does a remembered statement keep its source information?",
        },
        Case {
            key: "fts",
            query: "Can search still work when embeddings are absent?",
        },
        Case {
            key: "fencing",
            query: "How do we prevent an old vector from replacing one for newer text?",
        },
        Case {
            key: "queue",
            query: "What persistent worker queue must be repaired after an indexing crash?",
        },
        Case {
            key: "oneshot",
            query: "Why isn't there a background memory service?",
        },
        Case {
            key: "writes",
            query: "Does every conversation turn automatically become durable memory?",
        },
        Case {
            key: "backend",
            query: "Should we add a storage trait for hypothetical databases now?",
        },
        Case {
            key: "correction",
            query: "How is a wrong memory fixed without erasing history?",
        },
        Case {
            key: "workspace",
            query: "How do different worktrees keep separate continuation checkpoints?",
        },
        Case {
            key: "tasks",
            query: "Is this also the task tracker?",
        },
        Case {
            key: "local",
            query: "Where does a project normally store its memory database?",
        },
        Case {
            key: "compact",
            query: "Should we save entire research documents as individual memories?",
        },
    ]
}

fn remember(db: &Path, seed: &Seed<'_>) -> Value {
    run_json(
        db,
        &[
            "remember",
            seed.text,
            "--kind",
            seed.kind,
            "--source-type",
            "real-corpus-eval",
            "--source-ref",
            seed.key,
        ],
    )
}

fn context_ids(db: &Path, query: &str) -> Vec<String> {
    run_json(db, &["context", query, "-n", &LIMIT.to_string()])["memories"]
        .as_array()
        .expect("context memories")
        .iter()
        .map(|memory| memory["id"].as_str().expect("id").to_owned())
        .collect()
}

fn run_json(db: &Path, args: &[&str]) -> Value {
    let output = mem_command(db, args).output().expect("run mem subprocess");
    assert!(
        output.status.success(),
        "mem {args:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("parse mem JSON")
}

fn mem_command(db: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mem"));
    command.arg("--json").arg("--db").arg(db).args(args);
    command
}

fn test_path() -> PathBuf {
    std::env::temp_dir().join(format!("mem-real-corpus-eval-{}.db", Uuid::now_v7()))
}

fn cleanup(path: &Path) {
    let paths = HashSet::from([
        path.to_owned(),
        PathBuf::from(format!("{}-shm", path.display())),
        PathBuf::from(format!("{}-wal", path.display())),
    ]);
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}
