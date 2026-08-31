use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use uuid::Uuid;

const PROJECT: &str = "real-corpus-eval";
const WORKSPACE: &str = "main";
const LIMIT: usize = 10;
const RRF_K: f64 = 60.0;

// Real durable statements promoted from the mem project's own `ai/` context
// (brief/decisions/architecture), plus real cross-project distractors from
// the central agent-context checkout. Queries are realistic agent questions
// about that knowledge and deliberately avoid seed phrasing.
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
fn compare_retrieval_on_real_corpus() {
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

    let distractor_id = ids
        .get("other-project-distractor")
        .expect("cross-project distractor ID");
    let mut lexical_metrics = Metrics::default();
    let mut semantic_metrics = Metrics::default();
    let mut equal_rrf_metrics = Metrics::default();
    let mut context_default_metrics = Metrics::default();

    println!("case\tlexical\tsemantic\trrf\tcontext-default\tquery");
    for case in &cases {
        let expected = ids
            .get(case.key)
            .expect("case should reference seeded memory");
        let lexical = context_ids(&db, case.query, true);
        let semantic = semantic_ids(&db, case.query);
        let context_default = context_ids(&db, case.query, false);

        assert!(
            !lexical.contains(distractor_id),
            "lexical retrieval leaked cross-project memory"
        );
        assert!(
            !semantic.contains(distractor_id),
            "semantic retrieval leaked cross-project memory"
        );

        let equal_rrf = rrf(&lexical, &semantic, LIMIT);
        let lexical_rank = rank_of(&lexical, expected);
        let semantic_rank = rank_of(&semantic, expected);
        let equal_rrf_rank = rank_of(&equal_rrf, expected);
        let context_default_rank = rank_of(&context_default, expected);

        assert!(
            !context_default.contains(distractor_id),
            "default context ranking leaked cross-project memory"
        );

        lexical_metrics.observe(lexical_rank);
        semantic_metrics.observe(semantic_rank);
        equal_rrf_metrics.observe(equal_rrf_rank);
        context_default_metrics.observe(context_default_rank);

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            case.key,
            display_rank(lexical_rank),
            display_rank(semantic_rank),
            display_rank(equal_rrf_rank),
            display_rank(context_default_rank),
            case.query
        );
    }

    println!();
    lexical_metrics.print("lexical-or (context --lexical)");
    semantic_metrics.print("semantic (search --semantic)");
    equal_rrf_metrics.print("rrf (candidate context)");
    context_default_metrics.print("context default (semantic-first)");

    cleanup(&db);
}

fn corpus() -> Vec<Seed<'static>> {
    vec![
        Seed {
            key: "package-naming",
            text: "The crates.io package is named mem-cli while the user-facing executable and repository are named mem, so registry aliases can differ from the installed command.",
            kind: "fact",
            project: Some(PROJECT),
        },
        Seed {
            key: "toolchain",
            text: "Rust 1.98.0 is the pinned toolchain for this repository, and CI qualifies every change with rustfmt, tests, strict Clippy with -D warnings, and a release build.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "memory-models",
            text: "Semantic memory, episodic history, and continuation state are distinct models: current durable knowledge, past-session evidence projections, and the compact workspace cursor must not be collapsed into one log or document corpus.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "provenance",
            text: "Every promoted semantic memory requires a provenance row, and later Git/path freshness checks should mark implementation memories suspect when their supporting evidence changes rather than deleting them.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "fts-baseline",
            text: "FTS5 lexical search is the synchronous, always-available baseline; embeddings and vector retrieval are derived enhancements that must never become correctness dependencies.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "queue-fencing",
            text: "Derived embedding work is claimed with a generation, opaque lease token, and expiry; stale generations or tokens cannot complete newer work, and expired running leases can be reclaimed by another worker.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "episode-ownership",
            text: "An episode is uniquely owned by its source type and source reference; re-recording an existing source entry refreshes the projection instead of duplicating it.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "pi-adapter-boundary",
            text: "The Pi extension is a thin lifecycle adapter: recall and history policy stay in the Rust core, and the extension only maps Pi session events onto mem operations.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "semantic-only-wins",
            text: "On the first synthetic retrieval evaluation, exact semantic search beat the lexical OR baseline on hit@1 and MRR, but equal-weight RRF hybrid ranking demoted several semantic-only wins and reduced per-case quality on those queries.",
            kind: "fact",
            project: Some(PROJECT),
        },
        Seed {
            key: "oneshot-latency",
            text: "Measured one-shot semantic query latency is about 0.10 seconds per query, so a persistent serve --stdio daemon is not justified and one-shot CLI execution stays the default.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "no-llm-extraction",
            text: "Do not automatically promote every conversation turn into semantic memory; initial durable writes are explicit, and any later automatic extraction must produce candidates that pass dedupe and validation before promotion.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "no-storage-abstraction",
            text: "Do not introduce a generic storage-backend abstraction before a real second backend exists; keep SQLite-specific code isolated instead.",
            kind: "constraint",
            project: Some(PROJECT),
        },
        Seed {
            key: "correct-supersession",
            text: "Correcting a memory marks the predecessor superseded and links it to the replacement in the same SQLite transaction, preserving the old record and its evidence instead of destructively overwriting.",
            kind: "procedure",
            project: Some(PROJECT),
        },
        Seed {
            key: "workspace-scoping",
            text: "Continuation state is keyed by project and workspace so separate branches and worktrees keep independent resume points while sharing project-scoped semantic memory.",
            kind: "decision",
            project: Some(PROJECT),
        },
        Seed {
            key: "compactness",
            text: "Prefer concise durable memory over large ambient context dumps that duplicate source material the agent can re-read.",
            kind: "preference",
            project: None,
        },
        Seed {
            key: "confidence-decay",
            text: "Epistemic confidence of a memory must not decay merely because it has not been retrieved recently; retrieval utility and truth are separate concerns.",
            kind: "constraint",
            project: None,
        },
        Seed {
            key: "other-project-distractor",
            text: "The sy sync tool preserves setuid and setgid bits by masking received modes, and its SSH tests must run single-threaded because of shared remote state.",
            kind: "fact",
            project: Some("other-project"),
        },
        Seed {
            key: "other-project-distractor-2",
            text: "Every destination write in the sy sync engine uses an atomic temp-file-and-rename so an interrupted sync never leaves a partial file at the target.",
            kind: "constraint",
            project: Some("other-project"),
        },
    ]
}

fn cases() -> Vec<Case<'static>> {
    vec![
        Case {
            key: "package-naming",
            query: "Why does cargo install a differently named package than the command I end up with?",
        },
        Case {
            key: "toolchain",
            query: "Which Rust version does CI enforce here, and what checks gate a merge?",
        },
        Case {
            key: "memory-models",
            query: "Is there one big log with everything, or are different kinds of memory kept apart?",
        },
        Case {
            key: "provenance",
            query: "Can I store a conclusion without recording where it came from?",
        },
        Case {
            key: "fts-baseline",
            query: "What happens to search when the embedding model is missing or broken?",
        },
        Case {
            key: "queue-fencing",
            query: "How is stale worker state prevented from corrupting newer embedding work after a crash?",
        },
        Case {
            key: "episode-ownership",
            query: "If the same session content is ingested twice, do we get duplicate episode entries?",
        },
        Case {
            key: "pi-adapter-boundary",
            query: "Where does ranking policy live once Pi is only one of several clients?",
        },
        Case {
            key: "semantic-only-wins",
            query: "What did we learn about hybrid ranking on the first benchmark run?",
        },
        Case {
            key: "oneshot-latency",
            query: "Did we actually need to keep a resident process warm for memory queries?",
        },
        Case {
            key: "no-llm-extraction",
            query: "Does the system write new memories from every chat turn on its own?",
        },
        Case {
            key: "no-storage-abstraction",
            query: "Should I start building a backend trait so other databases can plug in?",
        },
        Case {
            key: "correct-supersession",
            query: "What is the procedure for fixing a memory that turned out to be wrong?",
        },
        Case {
            key: "workspace-scoping",
            query: "Two checkouts are on different branches; will they fight over the resume point?",
        },
        Case {
            key: "compactness",
            query: "Should I stuff whole documents into the recall context for safety?",
        },
        Case {
            key: "confidence-decay",
            query: "Does a fact become less true if the agent hasn't needed it for weeks?",
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
    run_json(db, &args)["memories"]
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

fn rrf(lexical: &[String], semantic: &[String], limit: usize) -> Vec<String> {
    let mut scores = HashMap::<String, f64>::new();
    for ranking in [lexical, semantic] {
        for (index, id) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_default() += 1.0 / (RRF_K + (index + 1) as f64);
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
    let output = mem_command(db, args).output().expect("run mem subprocess");
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
    std::env::temp_dir().join(format!("mem-real-corpus-eval-{}.db", Uuid::now_v7()))
}

fn cleanup(path: &Path) {
    let mut paths = HashSet::from([path.to_owned()]);
    paths.insert(PathBuf::from(format!("{}-shm", path.display())));
    paths.insert(PathBuf::from(format!("{}-wal", path.display())));
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}
