use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use uuid::Uuid;

const PROJECT: &str = "retrieval-eval";
const WORKSPACE: &str = "main";
const LIMIT: usize = 10;
const RRF_K: f64 = 60.0;

struct Seed<'a> {
    key: &'a str,
    text: &'a str,
    kind: &'a str,
    project: Option<&'a str>,
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
    reciprocal_rank: f64,
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
        if let Some(rank) = rank {
            self.reciprocal_rank += 1.0 / rank as f64;
        }
    }

    fn print(&self, label: &str) {
        println!(
            "{label}: hit@1={:.3} hit@3={:.3} mrr={:.3}",
            self.hit_at_1 as f64 / self.cases as f64,
            self.hit_at_3 as f64 / self.cases as f64,
            self.reciprocal_rank / self.cases as f64
        );
    }
}

#[test]
#[ignore = "downloads/loads the embedding model and is intended for explicit retrieval evaluation"]
fn compare_lexical_semantic_and_rrf_retrieval() {
    let db = test_path();
    let seeds = corpus();
    let cases = cases();
    let mut ids = HashMap::new();

    for seed in &seeds {
        let output = remember(&db, seed);
        let id = output["id"]
            .as_str()
            .expect("remember JSON should contain memory ID")
            .to_owned();
        assert!(ids.insert(seed.key, id).is_none(), "duplicate seed key");
    }

    let index = run_json(&db, &["index", "run", "-n", "64"]);
    assert_eq!(
        index["committed"].as_u64(),
        Some(seeds.len() as u64),
        "all seeded memories should be embedded: {index}"
    );

    let other_project_id = ids
        .get("other-project-distractor")
        .expect("other-project distractor ID");
    let mut lexical_metrics = Metrics::default();
    let mut semantic_metrics = Metrics::default();
    let mut equal_rrf_metrics = Metrics::default();
    let mut semantic_weighted_metrics = Metrics::default();
    let mut context_default_metrics = Metrics::default();

    println!("case\tlexical\tsemantic\trrf\trrf-s2\tcontext-default\tquery");
    for case in &cases {
        let expected = ids
            .get(case.key)
            .expect("case should reference seeded memory");
        let lexical = context_ids(&db, case.query, true);
        let semantic = semantic_ids(&db, case.query);
        let context_default = context_ids(&db, case.query, false);

        assert!(
            !lexical.contains(other_project_id),
            "lexical retrieval leaked other-project memory"
        );
        assert!(
            !semantic.contains(other_project_id),
            "semantic retrieval leaked other-project memory"
        );
        assert!(
            !context_default.contains(other_project_id),
            "default context ranking leaked other-project memory"
        );

        let equal_rrf = rrf(&lexical, &semantic, 1.0, 1.0, LIMIT);
        let semantic_weighted = rrf(&lexical, &semantic, 1.0, 2.0, LIMIT);
        let lexical_rank = rank_of(&lexical, expected);
        let semantic_rank = rank_of(&semantic, expected);
        let equal_rrf_rank = rank_of(&equal_rrf, expected);
        let semantic_weighted_rank = rank_of(&semantic_weighted, expected);
        let context_default_rank = rank_of(&context_default, expected);

        lexical_metrics.observe(lexical_rank);
        semantic_metrics.observe(semantic_rank);
        equal_rrf_metrics.observe(equal_rrf_rank);
        semantic_weighted_metrics.observe(semantic_weighted_rank);
        context_default_metrics.observe(context_default_rank);

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.key,
            display_rank(lexical_rank),
            display_rank(semantic_rank),
            display_rank(equal_rrf_rank),
            display_rank(semantic_weighted_rank),
            display_rank(context_default_rank),
            case.query
        );
    }

    println!();
    lexical_metrics.print("lexical-or (context --lexical)");
    semantic_metrics.print("semantic (search --semantic)");
    equal_rrf_metrics.print("rrf");
    semantic_weighted_metrics.print("rrf-semantic-2x");
    context_default_metrics.print("context default (semantic-first)");

    cleanup(&db);
}

#[test]
#[ignore = "loads the cached embedding model; intended for the retrieval-eval workflow"]
fn incomplete_embedding_coverage_never_hides_active_memories() {
    // Reproduces the production failure: the model is cached and older
    // memories are embedded, so a naive "any embedding exists" gate would
    // activate semantic ranking, which joins from the embeddings table and
    // silently hides any active memory whose indexing job is still queued.
    let db = test_path();
    let embedded = remember_text(
        &db,
        "Deployments require glibc 2.35 on Ubuntu 22.04 for the ort runtime.",
        Some(PROJECT),
    );
    run_json(&db, &["index", "run", "-n", "1"]);

    // A fresh memory with a pending embedding job must stay visible.
    let fresh = remember_text(
        &db,
        "Paraphrase queries with no shared vocabulary still rank semantically in the retrieval evaluation.",
        Some(PROJECT),
    );
    let ids = context_ids(&db, "shared vocabulary rank semantically", false);
    assert!(
        ids.contains(&fresh),
        "fresh memory must stay visible while its embedding job is queued: {ids:?}"
    );

    // After indexing completes, full coverage lets semantic ranking activate.
    // Semantic search returns every visible active memory regardless of query
    // overlap, while lexical OR over this query matches nothing, so finding
    // both memories proves semantic mode is active.
    run_json(&db, &["index", "run", "-n", "64"]);
    let ids = context_ids(
        &db,
        "what system library does the linux machine need",
        false,
    );
    assert!(
        ids.contains(&embedded) && ids.contains(&fresh),
        "fully indexed memories must be retrievable under semantic ranking: {ids:?}"
    );

    // Supersession queues a replacement: the replacement is fresh (no
    // vector) and the predecessor is superseded (invisible), so recall must
    // fall back to lexical and keep the replacement visible.
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
    let replacement = corrected["replacement"]["memory"]["id"]
        .as_str()
        .expect("correct JSON should contain replacement memory ID")
        .to_owned();
    let ids = context_ids(&db, "correction supersedes earlier phrasing", false);
    assert!(
        ids.contains(&replacement),
        "replacement memory must stay visible while its embedding job is queued: {ids:?}"
    );
    assert!(
        !ids.contains(&fresh),
        "superseded predecessor must not appear in recall: {ids:?}"
    );

    // Re-indexing restores complete coverage and semantic ranking, now over
    // the replacement instead of its superseded predecessor.
    run_json(&db, &["index", "run", "-n", "64"]);
    let ids = context_ids(
        &db,
        "what system library does the linux machine need",
        false,
    );
    assert!(
        ids.contains(&embedded) && ids.contains(&replacement),
        "replacement must be retrievable under semantic ranking after re-indexing: {ids:?}"
    );
    assert!(
        !ids.contains(&fresh),
        "superseded predecessor must stay out of semantic recall: {ids:?}"
    );

    cleanup(&db);
}

fn remember_text(db: &Path, text: &str, project: Option<&str>) -> String {
    let mut args = vec!["remember", text];
    if let Some(project) = project {
        args.extend(["--project", project]);
    } else {
        args.push("--global");
    }
    run_json(db, &args)["id"]
        .as_str()
        .expect("remember JSON should contain memory ID")
        .to_owned()
}

fn corpus() -> Vec<Seed<'static>> {
    vec![
        Seed {
            key: "workspace",
            text: "Continuation state is scoped to a project plus workspace so separate branches and worktrees do not overwrite each other's active state.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "stale",
            text: "When canonical source text changes, stale embeddings must be invalidated immediately and rebuilt for the new source generation.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "github",
            text: "The core memory CLI is agent-neutral and must not depend on GitHub; backup or push automation belongs in an external wrapper.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "storage",
            text: "SQLite is the canonical v1 storage backend, with synchronous FTS5 and derived embeddings.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "ann",
            text: "Start vector retrieval with exact cosine scanning and add ANN or HNSW only after profiling shows it is necessary.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "corrections",
            text: "Corrections are non-destructive: preserve the old memory as superseded and link it to the replacement with provenance.",
            kind: "procedure",
            project: Some(PROJECT),
        },
        Seed {
            key: "episodes",
            text: "Episodic history is a searchable projection that keeps exact references back to the original session entries and tool evidence.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "indexing",
            text: "Canonical writes and FTS5 updates are synchronous; embedding generation is derived work that may run later without blocking the write.",
            kind: "procedure",
            project: Some(PROJECT),
        },
        Seed {
            key: "tasks",
            text: "The task tracker remains separate from memory; memory may reference task IDs but must not depend on the task system.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "stdio",
            text: "Use one-shot CLI operation first; add a warm stdio service only if profiling shows startup or model latency justifies it.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "authority",
            text: "User-authored memories and verified agent conclusions have different authority and evidence requirements.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "freshness",
            text: "Coding memories backed by Git paths should eventually be checked for evidence freshness when those paths change.",
            kind: "procedure",
            project: Some(PROJECT),
        },
        Seed {
            key: "portability",
            text: "The memory system should remain usable by multiple agents; Pi is only the first adapter, not the architecture boundary.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "confidence",
            text: "Do not reduce epistemic confidence merely because a memory has not been used recently; decay retrieval utility separately.",
            kind: "constraint",
            project: None,
        },
        Seed {
            key: "compactness",
            text: "Prefer concise durable memory over large ambient context dumps that duplicate source material.",
            kind: "preference",
            project: None,
        },
        Seed {
            key: "other-project-distractor",
            text: "A different project stores branch state in GitHub issues and uses HNSW for every memory query.",
            kind: "fact",
            project: Some("other-project"),
        },
    ]
}

fn cases() -> Vec<Case<'static>> {
    vec![
        Case {
            key: "workspace",
            query: "How do we stop two checkouts of the same repository from clobbering the current resume point?",
        },
        Case {
            key: "stale",
            query: "What happens to a vector after the text it represents is edited?",
        },
        Case {
            key: "github",
            query: "Should syncing to the hosting service be built into the memory engine itself?",
        },
        Case {
            key: "storage",
            query: "What database are we trusting for the first usable release?",
        },
        Case {
            key: "ann",
            query: "Do we need an approximate nearest-neighbor index yet?",
        },
        Case {
            key: "corrections",
            query: "When a remembered decision is corrected, do we overwrite the original record?",
        },
        Case {
            key: "episodes",
            query: "How can an agent drill from a past debugging hit back into the exact conversation or tool output?",
        },
        Case {
            key: "indexing",
            query: "Which indexing work has to finish before a memory write can return?",
        },
        Case {
            key: "confidence",
            query: "Should an old but still valid preference become less true just because nobody retrieved it lately?",
        },
        Case {
            key: "tasks",
            query: "Is the task tracker part of the memory tool?",
        },
        Case {
            key: "stdio",
            query: "Do we need a persistent daemon for every query?",
        },
        Case {
            key: "authority",
            query: "Can agent guesses be stored with the same weight as something the user explicitly said?",
        },
        Case {
            key: "freshness",
            query: "How should implementation memories react when the files supporting them have changed?",
        },
        Case {
            key: "portability",
            query: "Is Pi the architecture boundary or just the first client?",
        },
    ]
}

fn remember(db: &Path, seed: &Seed<'_>) -> Value {
    let mut args = vec![
        "remember".to_owned(),
        seed.text.to_owned(),
        "--kind".to_owned(),
        seed.kind.to_owned(),
        "--source-type".to_owned(),
        "retrieval-eval".to_owned(),
        "--source-ref".to_owned(),
        seed.key.to_owned(),
    ];
    if let Some(project) = seed.project {
        args.extend(["--project".to_owned(), project.to_owned()]);
    } else {
        args.push("--global".to_owned());
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_json(db, &refs)
}

fn context_ids(db: &Path, query: &str, lexical: bool) -> Vec<String> {
    context_ids_env(db, query, lexical, &[])
}

fn context_ids_env(db: &Path, query: &str, lexical: bool, env: &[(&str, &str)]) -> Vec<String> {
    let limit = LIMIT.to_string();
    let mut args = vec![
        "context",
        query,
        "--project",
        PROJECT,
        "--workspace",
        WORKSPACE,
        "-n",
        limit.as_str(),
    ];
    if lexical {
        args.push("--lexical");
    }
    run_json_env(db, &args, env)["memories"]
        .as_array()
        .expect("context memories array")
        .iter()
        .map(|item| {
            item["memory"]["id"]
                .as_str()
                .expect("context memory ID")
                .to_owned()
        })
        .collect()
}

fn semantic_ids(db: &Path, query: &str) -> Vec<String> {
    let limit = LIMIT.to_string();
    let output = run_json(
        db,
        &[
            "search",
            query,
            "--project",
            PROJECT,
            "--semantic",
            "-n",
            &limit,
        ],
    );
    output
        .as_array()
        .expect("semantic search array")
        .iter()
        .map(|item| {
            item["memory"]["id"]
                .as_str()
                .expect("semantic memory ID")
                .to_owned()
        })
        .collect()
}

fn rrf(
    lexical: &[String],
    semantic: &[String],
    lexical_weight: f64,
    semantic_weight: f64,
    limit: usize,
) -> Vec<String> {
    let mut scores = HashMap::<String, f64>::new();
    for (ranking, weight) in [(lexical, lexical_weight), (semantic, semantic_weight)] {
        for (index, id) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_default() += weight / (RRF_K + (index + 1) as f64);
        }
    }

    let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(limit);
    ranked.into_iter().map(|(id, _)| id).collect()
}

fn rank_of(ranking: &[String], expected: &str) -> Option<usize> {
    ranking
        .iter()
        .position(|candidate| candidate == expected)
        .map(|index| index + 1)
}

fn display_rank(rank: Option<usize>) -> String {
    rank.map_or_else(|| "-".to_owned(), |rank| rank.to_string())
}

fn run_json(db: &Path, args: &[&str]) -> Value {
    run_json_env(db, args, &[])
}

fn run_json_env(db: &Path, args: &[&str], env: &[(&str, &str)]) -> Value {
    let mut command = mem_command(db, args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("run mem subprocess");
    assert!(
        output.status.success(),
        "mem {:?} failed:\nstdout: {}\nstderr: {}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("parse mem JSON output")
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
    let mut paths = HashSet::from([path.to_owned()]);
    paths.insert(PathBuf::from(format!("{}-shm", path.display())));
    paths.insert(PathBuf::from(format!("{}-wal", path.display())));
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}
