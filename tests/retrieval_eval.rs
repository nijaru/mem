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

#[derive(Default)]
struct Metrics {
    cases: usize,
    hit_at_1: usize,
    hit_at_3: usize,
}

impl Metrics {
    fn observe(&mut self, rank: Option<usize>) {
        self.cases += 1;
        if rank == Some(1) {
            self.hit_at_1 += 1;
        }
        if rank.is_some_and(|rank| rank <= 3) {
            self.hit_at_3 += 1;
        }
    }

    fn print(&self, label: &str) {
        println!(
            "{label}: hit@1={:.3} hit@3={:.3}",
            self.hit_at_1 as f64 / self.cases as f64,
            self.hit_at_3 as f64 / self.cases as f64,
        );
    }
}

#[test]
#[ignore = "downloads/loads the embedding model; intended for the retrieval-eval workflow"]
fn semantic_and_context_retrieve_paraphrased_product_knowledge() {
    let db = test_path();
    let seeds = corpus();
    let cases = cases();
    let mut ids = HashMap::new();

    for seed in &seeds {
        let output = remember(&db, seed);
        ids.insert(seed.key, output["id"].as_str().expect("id").to_owned());
    }
    let indexed = run_json(&db, &["index", "-n", "64"]);
    assert_eq!(indexed["indexed"].as_u64(), Some(seeds.len() as u64));

    let mut lexical = Metrics::default();
    let mut semantic = Metrics::default();
    let mut context = Metrics::default();
    for case in &cases {
        let expected = ids.get(case.key).expect("target seed");
        lexical.observe(rank_of(&lexical_ids(&db, case.query), expected));
        semantic.observe(rank_of(&semantic_ids(&db, case.query), expected));
        context.observe(rank_of(&context_ids(&db, case.query), expected));
    }

    lexical.print("lexical");
    semantic.print("semantic");
    context.print("context");
    assert!(
        semantic.hit_at_3 * 3 >= semantic.cases * 2,
        "semantic hit@3 below two thirds: {}/{}",
        semantic.hit_at_3,
        semantic.cases
    );
    assert!(
        context.hit_at_3 * 3 >= context.cases * 2,
        "context hit@3 below two thirds: {}/{}",
        context.hit_at_3,
        context.cases
    );

    cleanup(&db);
}

#[test]
#[ignore = "loads the cached embedding model; intended for the retrieval-eval workflow"]
fn incomplete_embedding_coverage_falls_back_without_hiding_memories() {
    let db = test_path();
    let embedded = remember_text(
        &db,
        "Deployments require glibc 2.35 on Ubuntu 22.04 for the runtime.",
    );
    run_json(&db, &["index", "-n", "1"]);

    let fresh = remember_text(
        &db,
        "Paraphrase queries with no shared vocabulary still rank semantically in retrieval.",
    );
    let ids = context_ids(&db, "shared vocabulary rank semantically");
    assert!(
        ids.contains(&fresh),
        "fresh memory hidden before indexing: {ids:?}"
    );

    run_json(&db, &["index", "-n", "64"]);
    let ids = context_ids(&db, "what system library does the linux machine need");
    assert!(ids.contains(&embedded));

    let corrected = run_json(
        &db,
        &[
            "correct",
            &fresh,
            "Paraphrase retrieval keeps working after a correction supersedes the earlier phrasing.",
            "--source-type",
            "test",
        ],
    );
    let replacement = corrected["replacement"]["id"]
        .as_str()
        .expect("replacement id")
        .to_owned();
    let ids = context_ids(&db, "correction supersedes earlier phrasing");
    assert!(ids.contains(&replacement));
    assert!(!ids.contains(&fresh));

    cleanup(&db);
}

fn corpus() -> Vec<Seed<'static>> {
    vec![
        Seed {
            key: "workspace",
            text: "Continuation state is keyed by workspace inside the repo-local database so branches and worktrees keep independent resume points.",
            kind: "decision",
        },
        Seed {
            key: "stale",
            text: "An embedding is committed only when the memory is still active at the exact updated_at value that was embedded; stale source versions are discarded.",
            kind: "constraint",
        },
        Seed {
            key: "storage",
            text: "SQLite is the canonical project store, with synchronous FTS5 and rebuildable local embeddings.",
            kind: "decision",
        },
        Seed {
            key: "ann",
            text: "Use exact cosine scanning for the small local corpus and add ANN only if profiling shows a real need.",
            kind: "decision",
        },
        Seed {
            key: "corrections",
            text: "Corrections preserve the predecessor as superseded and point it directly at the replacement instead of overwriting history.",
            kind: "procedure",
        },
        Seed {
            key: "provenance",
            text: "Each memory stores the actor, source type, and optional source reference directly so recalled knowledge retains lightweight provenance.",
            kind: "decision",
        },
        Seed {
            key: "indexing",
            text: "Indexing scans active memories missing a current-model vector; failed work remains missing and a later index run retries it naturally.",
            kind: "procedure",
        },
        Seed {
            key: "tasks",
            text: "Task tracking stays outside mem; continuation state may reference an external task without depending on a task system.",
            kind: "constraint",
        },
        Seed {
            key: "oneshot",
            text: "Keep mem as a one-shot CLI unless measured latency demonstrates that a persistent service is necessary.",
            kind: "decision",
        },
        Seed {
            key: "authority",
            text: "Provenance is descriptive audit context, not authenticated authority and not a retrieval-ranking weight.",
            kind: "constraint",
        },
        Seed {
            key: "compactness",
            text: "Prefer concise durable memories over large ambient context dumps that duplicate source material.",
            kind: "preference",
        },
        Seed {
            key: "hosting",
            text: "Hosted project systems and synchronization stay outside the local memory storage and retrieval core.",
            kind: "decision",
        },
    ]
}

fn cases() -> Vec<Case<'static>> {
    vec![
        Case {
            key: "workspace",
            query: "How do two worktrees avoid clobbering the current resume point?",
        },
        Case {
            key: "stale",
            query: "Can a vector computed from old text overwrite a newer memory?",
        },
        Case {
            key: "storage",
            query: "What database is the durable source of truth for a project?",
        },
        Case {
            key: "ann",
            query: "Do we need HNSW for the local memory corpus yet?",
        },
        Case {
            key: "corrections",
            query: "What happens to the old record when a remembered decision is fixed?",
        },
        Case {
            key: "provenance",
            query: "What source information travels with recalled knowledge?",
        },
        Case {
            key: "indexing",
            query: "If embedding generation fails, what state has to be repaired?",
        },
        Case {
            key: "tasks",
            query: "Is task management part of this memory tool?",
        },
        Case {
            key: "oneshot",
            query: "Do queries require a resident daemon?",
        },
        Case {
            key: "authority",
            query: "Does a caller label make a stored statement trusted?",
        },
        Case {
            key: "compactness",
            query: "Should full documents be stuffed into durable memory by default?",
        },
        Case {
            key: "hosting",
            query: "Should Linear or GitHub synchronization live inside the storage core?",
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
            "retrieval-eval",
            "--source-ref",
            seed.key,
        ],
    )
}

fn remember_text(db: &Path, text: &str) -> String {
    run_json(db, &["remember", text])["id"]
        .as_str()
        .expect("id")
        .to_owned()
}

fn lexical_ids(db: &Path, query: &str) -> Vec<String> {
    ids(run_json(db, &["search", query, "-n", &LIMIT.to_string()]))
}

fn semantic_ids(db: &Path, query: &str) -> Vec<String> {
    ids(run_json(
        db,
        &["search", query, "--semantic", "-n", &LIMIT.to_string()],
    ))
}

fn context_ids(db: &Path, query: &str) -> Vec<String> {
    run_json(db, &["context", query, "-n", &LIMIT.to_string()])["memories"]
        .as_array()
        .expect("context memories")
        .iter()
        .map(|memory| memory["id"].as_str().expect("id").to_owned())
        .collect()
}

fn ids(value: Value) -> Vec<String> {
    value
        .as_array()
        .expect("search results")
        .iter()
        .map(|item| item["memory"]["id"].as_str().expect("id").to_owned())
        .collect()
}

fn rank_of(ranking: &[String], expected: &str) -> Option<usize> {
    ranking
        .iter()
        .position(|id| id == expected)
        .map(|index| index + 1)
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
    std::env::temp_dir().join(format!("mem-retrieval-eval-{}.db", Uuid::now_v7()))
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
