mod embedding;
mod embedding_worker;
mod id_resolve;
mod index_job;
mod project;
mod storage;
mod store;
mod vector_search;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use usage::{Args, Cli, Subcommands};

use crate::embedding_worker::{
    EMBEDDING_MODEL_ID, EmbeddingRunOptions, embed_query, embed_query_if_cached, model_cache_dir,
};
use crate::project::ProjectContext;
use crate::storage::StorageRouter;

/// Open the current database for a low-level index mutation.
fn require_exact_store(router: &StorageRouter) -> Result<Store> {
    Store::open(router.path())
}
use crate::store::{
    Memory, MemorySource, NewCorrection, NewMemory, NewWorkspaceState, NewWorkspaceStatePatch,
    SearchHit, Store, WorkspaceState,
};

#[derive(Cli)]
#[usage(bin = "mem", version, arg_required_else_help, unknown_flags = "error")]
struct MemCli {
    /// Emit machine-readable JSON.
    #[usage(long, global)]
    json: bool,

    /// Override the SQLite database path.
    #[usage(long, global, value_hint = usage::ValueHint::FilePath)]
    db: Option<PathBuf>,

    #[usage(subcommand)]
    command: Command,
}

#[derive(Subcommands)]
enum Command {
    Init(Init),
    Status(Status),
    Project(Project),
    State(State),
    Context(ContextCommand),
    Remember(Remember),
    Correct(Correct),
    Search(Search),
    Get(Get),
    Forget(Forget),
    Index(IndexCommand),
}

/// Initialize the local memory database.
#[derive(Args)]
struct Init;

/// Show database and memory counts.
#[derive(Args)]
struct Status;

/// Show the detected project and workspace identity.
#[derive(Args)]
struct Project {
    /// Override the project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the workspace identifier.
    #[usage(long)]
    workspace: Option<String>,
}

/// Read or update compact workspace continuation state.
#[derive(Args)]
struct State {
    #[usage(subcommand)]
    command: StateCommand,
}

#[derive(Subcommands)]
enum StateCommand {
    Show(StateShow),
    Set(StateSet),
    Patch(StatePatch),
    Clear(StateClear),
}

/// Show continuation state for the current workspace.
#[derive(Args)]
struct StateShow {
    /// Override the project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the workspace identifier.
    #[usage(long)]
    workspace: Option<String>,
}

/// Replace continuation state for the current workspace.
#[derive(Args)]
struct StateSet {
    /// Override the project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the workspace identifier.
    #[usage(long)]
    workspace: Option<String>,

    /// Last agent/session identifier associated with this checkpoint.
    #[usage(long)]
    session: Option<String>,

    /// Current high-level goal.
    #[usage(long)]
    goal: Option<String>,

    /// Optional external task reference.
    #[usage(long)]
    task: Option<String>,

    /// Compact resume checkpoint.
    #[usage(long)]
    checkpoint: Option<String>,
}

/// Update only the provided fields of the current workspace's continuation
/// state; omitted fields keep their current values.
#[derive(Args)]
struct StatePatch {
    /// Override the project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the workspace identifier.
    #[usage(long)]
    workspace: Option<String>,

    /// Last agent/session identifier associated with this checkpoint.
    #[usage(long)]
    session: Option<String>,

    /// Current high-level goal.
    #[usage(long)]
    goal: Option<String>,

    /// Optional external task reference.
    #[usage(long)]
    task: Option<String>,

    /// Compact resume checkpoint.
    #[usage(long)]
    checkpoint: Option<String>,

    /// Field to clear (set to nothing): one of session, goal, task, checkpoint.
    /// Repeatable.
    #[usage(long, var)]
    clear: Vec<String>,
}

/// Remove continuation state for the current workspace.
#[derive(Args)]
struct StateClear {
    /// Override the project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the workspace identifier.
    #[usage(long)]
    workspace: Option<String>,
}

/// Build adapter-facing context from state and broad semantic recall.
#[derive(Args)]
struct ContextCommand {
    /// Prompt or query to recall relevant memory for.
    query: String,

    /// Override the current project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Override the current workspace identifier.
    #[usage(long)]
    workspace: Option<String>,

    /// Maximum number of semantic memories.
    #[usage(short = 'n', long)]
    limit: Option<usize>,

    /// Maximum total memory-text bytes to return; recall stops at the first
    /// hit that would exceed it. Default 32768; 0 disables the budget.
    #[usage(long)]
    max_bytes: Option<usize>,
}

/// Store one durable semantic memory.
#[derive(Args)]
struct Remember {
    /// Memory text.
    text: String,

    /// Memory kind: fact, finding, decision, constraint, preference, or procedure.
    #[usage(long)]
    kind: Option<String>,

    /// Override the current project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Store user-wide memory instead of project memory.
    #[usage(long = "global")]
    global_memory: bool,

    /// Actor that established the memory, such as user or agent.
    #[usage(long)]
    actor: Option<String>,

    /// Provenance type, such as cli, session, git, file, or url.
    #[usage(long)]
    source_type: Option<String>,

    /// Optional source locator or stable reference.
    #[usage(long)]
    source_ref: Option<String>,
}

/// Replace one active memory while preserving the superseded record and provenance.
#[derive(Args)]
struct Correct {
    /// Full memory ID or an unambiguous ID prefix.
    id: String,

    /// Replacement memory text.
    text: String,

    /// Override the existing memory kind.
    #[usage(long)]
    kind: Option<String>,

    /// Actor that established the correction, such as user or agent.
    #[usage(long)]
    actor: Option<String>,

    /// Provenance type for the correction, such as cli, session, git, file, or url.
    #[usage(long)]
    source_type: Option<String>,

    /// Optional source locator or stable reference for the correction.
    #[usage(long)]
    source_ref: Option<String>,
}

/// Search active semantic memory.
#[derive(Args)]
struct Search {
    /// Search query.
    query: String,

    /// Override the current project identifier.
    #[usage(long)]
    project: Option<String>,

    /// Search only user-wide global memory.
    #[usage(long = "global")]
    global_memory: bool,

    /// Use local embeddings and exact cosine similarity instead of FTS5.
    #[usage(long)]
    semantic: bool,

    /// Maximum number of results.
    #[usage(short = 'n', long)]
    limit: Option<usize>,
}

/// Read one memory, provenance, and semantic relations.
#[derive(Args)]
struct Get {
    /// Full memory ID or an unambiguous ID prefix.
    id: String,
}

/// Soft-delete one memory.
#[derive(Args)]
struct Forget {
    /// Full memory ID or an unambiguous ID prefix.
    id: String,
}

/// Operate the durable derived-index work queue.
#[derive(Args)]
struct IndexCommand {
    #[usage(subcommand)]
    command: IndexSubcommand,
}

#[derive(Subcommands)]
enum IndexSubcommand {
    Status(IndexStatus),
    Run(IndexRun),
    Claim(IndexClaim),
    Commit(IndexCommit),
    Complete(IndexComplete),
    Retry(IndexRetry),
}

/// Show derived-index queue counts.
#[derive(Args)]
struct IndexStatus;

/// Process a bounded batch of pending embeddings with the built-in local model.
#[derive(Args)]
struct IndexRun {
    /// Maximum number of jobs to process in this invocation.
    #[usage(short = 'n', long)]
    limit: Option<usize>,

    /// Lease duration in seconds while model loading and inference run.
    #[usage(long)]
    lease_seconds: Option<u64>,

    /// Delay before retrying model/inference failures, in seconds.
    #[usage(long)]
    retry_seconds: Option<u64>,

    /// Do nothing when the embedding model is not already cached locally.
    /// Keeps automated runs (for example at session shutdown) from
    /// downloading the model.
    #[usage(long)]
    cached_only: bool,
}

/// Claim pending or expired derived-index work.
#[derive(Args)]
struct IndexClaim {
    /// Stable worker identifier.
    worker: String,

    /// Maximum number of jobs to claim.
    #[usage(short = 'n', long)]
    limit: Option<usize>,

    /// Lease duration in seconds.
    #[usage(long)]
    lease_seconds: Option<u64>,
}

/// Commit one embedding result and consume its exact claimed lease.
#[derive(Args)]
struct IndexCommit {
    /// Job identifier.
    job: String,

    /// Generation claimed by the worker.
    generation: i64,

    /// Opaque lease token returned by `index claim`.
    lease_token: String,

    /// Stable embedding model identifier.
    model: String,

    /// Embedding vector encoded as a JSON array of numbers.
    vector: String,
}

/// Complete one claimed non-embedding derived-index job.
#[derive(Args)]
struct IndexComplete {
    /// Job identifier.
    job: String,

    /// Generation claimed by the worker.
    generation: i64,

    /// Opaque lease token returned by `index claim`.
    lease_token: String,
}

/// Release one claimed job for retry after a delay.
#[derive(Args)]
struct IndexRetry {
    /// Job identifier.
    job: String,

    /// Generation claimed by the worker.
    generation: i64,

    /// Opaque lease token returned by `index claim`.
    lease_token: String,

    /// Diagnostic retained with the pending job.
    error: String,

    /// Delay before the job can be claimed again, in seconds.
    #[usage(long)]
    delay_seconds: Option<u64>,
}

#[derive(Serialize)]
struct StatusOutput {
    database: String,
    schema_version: i64,
    total: u64,
    active: u64,
    superseded: u64,
    deleted: u64,
}

#[derive(Serialize)]
struct ClearStateOutput<'a> {
    project_id: &'a str,
    workspace_id: &'a str,
    cleared: bool,
}

#[derive(Serialize)]
struct ContextOutput {
    project: Option<ProjectContext>,
    state: Option<WorkspaceState>,
    memories: Vec<ContextMemory>,
}

#[derive(Serialize)]
struct ContextMemory {
    memory: Memory,
    sources: Vec<MemorySource>,
}

fn main() {
    // Memory text can hold credentials and internal paths, and SQLite does
    // not let a caller set permissions on the WAL/shm sidecars it creates.
    // A one-shot CLI only writes store files, so a restrictive umask makes
    // every database and sidecar private to the user by default.
    #[cfg(unix)]
    {
        // safety: umask is a per-process value; changing it only affects
        // files this short-lived process creates.
        unsafe { libc::umask(0o077) };
    }

    let argv: Vec<String> = std::env::args().collect();
    let Some(cli) = parse_cli(&argv) else {
        return;
    };

    if let Err(error) = run(cli) {
        eprintln!("mem: {error:#}");
        process::exit(1);
    }
}

fn parse_cli(argv: &[String]) -> Option<MemCli> {
    let words: Vec<&OsStr> = argv.iter().skip(1).map(OsStr::new).collect();
    match MemCli::parse_from(&words) {
        Ok(cli) => Some(cli),
        Err(usage::Error::Help { cmd, long }) => {
            if let Some(page) =
                usage::help::render_styled(MemCli::spec(), cmd, long, usage::help::Style::auto())
            {
                print!("{page}");
            }
            None
        }
        Err(usage::Error::HelpAll { cmd }) => {
            if let Some(page) =
                usage::help::render_all_styled(MemCli::spec(), cmd, usage::help::Style::auto())
            {
                print!("{page}");
            }
            None
        }
        Err(usage::Error::Version { .. }) => {
            println!("mem {}", env!("CARGO_PKG_VERSION"));
            None
        }
        Err(error) => {
            eprint!("{}", usage::render_failure(MemCli::spec(), &words, &error));
            process::exit(2);
        }
    }
}

/// Default recall byte budget across all returned memory texts.
const RECALL_DEFAULT_MAX_BYTES: usize = 32 * 1024;

/// Minimum cosine similarity for automatic semantic context recall.
/// Calibrated against project-realistic positives and unrelated queries.
const SEMANTIC_CONTEXT_MIN_SCORE: f64 = 0.55;

fn run(cli: MemCli) -> Result<()> {
    let explicit_db = cli.db.is_some()
        || std::env::var_os("MEM_DB")
            .filter(|path| !path.is_empty())
            .is_some();
    let db_path = database_path(cli.db.as_deref())?;
    if !explicit_db && command_writes(&cli.command) {
        ensure_local_ignore(&db_path)?;
    }
    let router = StorageRouter::exact(db_path.clone());

    match cli.command {
        Command::Init(_) => {
            let store = Store::open(&db_path)?;
            let stats = store.stats()?;
            let output = StatusOutput {
                database: db_path.display().to_string(),
                schema_version: stats.schema_version,
                total: stats.total,
                active: stats.active,
                superseded: stats.superseded,
                deleted: stats.deleted,
            };
            if cli.json {
                print_json(&output)?;
            } else {
                println!("database: {}", output.database);
                println!("schema: {}", output.schema_version);
                println!(
                    "memories: {} active / {} superseded / {} deleted / {} total",
                    output.active, output.superseded, output.deleted, output.total
                );
            }
        }
        Command::Status(_) => {
            let output = match Store::open_existing(&db_path)? {
                Some(store) => {
                    let stats = store.stats()?;
                    StatusOutput {
                        database: db_path.display().to_string(),
                        schema_version: stats.schema_version,
                        total: stats.total,
                        active: stats.active,
                        superseded: stats.superseded,
                        deleted: stats.deleted,
                    }
                }
                None => StatusOutput {
                    database: db_path.display().to_string(),
                    schema_version: 0,
                    total: 0,
                    active: 0,
                    superseded: 0,
                    deleted: 0,
                },
            };
            if cli.json {
                print_json(&output)?;
            } else {
                println!("database: {}", output.database);
                if output.schema_version == 0 {
                    println!("store: not initialized");
                } else {
                    println!("schema: {}", output.schema_version);
                    println!(
                        "memories: {} active / {} superseded / {} deleted / {} total",
                        output.active, output.superseded, output.deleted, output.total
                    );
                }
            }
        }
        Command::Project(command) => {
            let context =
                project_context(command.project.as_deref(), command.workspace.as_deref())?;
            if cli.json {
                print_json(&context)?;
            } else {
                println!("project: {}", context.project_id);
                println!("workspace: {}", context.workspace_id);
                if let Some(root) = &context.root {
                    println!("root: {}", root.display());
                }
                if let Some(remote) = &context.remote {
                    println!("remote: {remote}");
                }
            }
        }
        Command::State(command) => match command.command {
            StateCommand::Show(command) => {
                let context =
                    project_context(command.project.as_deref(), command.workspace.as_deref())?;
                let state = crate::storage::routed_workspace_state(
                    &router,
                    &context.project_id,
                    &context.workspace_id,
                )?;
                if cli.json {
                    print_json(&state)?;
                } else if let Some(state) = state {
                    print_workspace_state(&state);
                } else {
                    println!(
                        "no state for {} {}",
                        context.project_id, context.workspace_id
                    );
                }
            }
            StateCommand::Set(command) => {
                let context =
                    project_context(command.project.as_deref(), command.workspace.as_deref())?;
                let store = router.write_store(Some(&context.project_id))?;
                let state = store.set_workspace_state(NewWorkspaceState {
                    project_id: context.project_id,
                    workspace_id: context.workspace_id,
                    last_session_id: command.session,
                    active_goal: command.goal,
                    active_task_ref: command.task,
                    checkpoint: command.checkpoint,
                })?;
                if cli.json {
                    print_json(&state)?;
                } else {
                    print_workspace_state(&state);
                }
            }
            StateCommand::Patch(command) => {
                let context =
                    project_context(command.project.as_deref(), command.workspace.as_deref())?;
                let store = router.write_store(Some(&context.project_id))?;
                let state = store.patch_workspace_state(NewWorkspaceStatePatch {
                    project_id: context.project_id,
                    workspace_id: context.workspace_id,
                    last_session_id: command.session,
                    active_goal: command.goal,
                    active_task_ref: command.task,
                    checkpoint: command.checkpoint,
                    clear_fields: command.clear,
                })?;
                if cli.json {
                    print_json(&state)?;
                } else {
                    print_workspace_state(&state);
                }
            }
            StateCommand::Clear(command) => {
                let context =
                    project_context(command.project.as_deref(), command.workspace.as_deref())?;
                let store = router.write_store(Some(&context.project_id))?;
                let cleared =
                    store.clear_workspace_state(&context.project_id, &context.workspace_id)?;
                if cli.json {
                    print_json(&ClearStateOutput {
                        project_id: &context.project_id,
                        workspace_id: &context.workspace_id,
                        cleared,
                    })?;
                } else if cleared {
                    println!(
                        "cleared state for {} {}",
                        context.project_id, context.workspace_id
                    );
                } else {
                    println!(
                        "no state for {} {}",
                        context.project_id, context.workspace_id
                    );
                }
            }
        },
        Command::Context(command) => {
            let project =
                optional_project_context(command.project.as_deref(), command.workspace.as_deref())?;
            let state = if let Some(project) = project.as_ref() {
                crate::storage::routed_workspace_state(
                    &router,
                    &project.project_id,
                    &project.workspace_id,
                )?
            } else {
                None
            };
            let project_id = project.as_ref().map(|project| project.project_id.as_str());
            let limit = command.limit.unwrap_or(10).clamp(1, 100);
            let max_bytes = command.max_bytes.unwrap_or(RECALL_DEFAULT_MAX_BYTES);
            let hits = crate::storage::routed_recall_hits(
                &router,
                project_id,
                &command.query,
                limit,
                false,
            )?;
            // Rank order, byte-bounded: a single huge memory can no longer
            // crowd an adapter's context window. The first hit is always
            // included when any budget allows it, so a query matching one
            // oversized memory still surfaces something.
            let mut memories = Vec::new();
            let mut total = 0usize;
            for hit in hits {
                let text_bytes = hit.memory.text.len();
                if max_bytes > 0 && !memories.is_empty() && total + text_bytes > max_bytes {
                    break;
                }
                total += text_bytes;
                let (store, id) =
                    crate::storage::routed_resolve_memory(&router, project_id, &hit.memory.id)?;
                let record = store.get(&id)?;
                memories.push(ContextMemory {
                    memory: record.memory,
                    sources: record.sources,
                });
            }
            let output = ContextOutput {
                project,
                state,
                memories,
            };
            if cli.json {
                print_json(&output)?;
            } else {
                print_context(&output);
            }
        }
        Command::Remember(command) => {
            let project_id = memory_project(command.project.as_deref(), command.global_memory)?;
            let mut store = router.write_store(project_id.as_deref())?;
            let memory = store.remember(NewMemory {
                text: command.text,
                kind: command.kind.unwrap_or_else(|| "fact".to_owned()),
                project_id,
                actor: command.actor.unwrap_or_else(|| "agent".to_owned()),
                source_type: command.source_type.unwrap_or_else(|| "cli".to_owned()),
                source_ref: command.source_ref,
            })?;
            if cli.json {
                print_json(&memory)?;
            } else {
                println!("{}\t{}\t{}", memory.id, memory.kind, memory.text);
            }
        }
        Command::Correct(command) => {
            let (mut store, id) =
                crate::storage::routed_resolve_memory(&router, None, &command.id)?;
            let result = store.correct(
                &id,
                NewCorrection {
                    text: command.text,
                    kind: command.kind,
                    actor: command.actor.unwrap_or_else(|| "agent".to_owned()),
                    source_type: command.source_type.unwrap_or_else(|| "cli".to_owned()),
                    source_ref: command.source_ref,
                },
            )?;
            if cli.json {
                print_json(&result)?;
            } else {
                println!(
                    "corrected {} -> {}",
                    result.previous.id, result.replacement.memory.id
                );
                println!(
                    "{}\t{}\t{}",
                    result.replacement.memory.id,
                    result.replacement.memory.kind,
                    result.replacement.memory.text
                );
            }
        }
        Command::Search(command) => {
            let project_id = memory_project(command.project.as_deref(), command.global_memory)?;
            let limit = command.limit.unwrap_or(10).clamp(1, 100);
            if command.semantic {
                let cache_dir = model_cache_dir()?;
                let query_vector = embed_query(&command.query, &cache_dir, !cli.json)?;
                let hits = crate::storage::routed_semantic_search(
                    &router,
                    project_id.as_deref(),
                    &query_vector,
                    limit,
                )?;
                if cli.json {
                    print_json(&hits)?;
                } else {
                    for hit in hits {
                        let scope = hit
                            .memory
                            .project_id
                            .as_deref()
                            .map_or("global", |project| project);
                        println!(
                            "{}\t{}\t{}\t{:.6}\t{}",
                            hit.memory.id, hit.memory.kind, scope, hit.score, hit.memory.text
                        );
                    }
                }
            } else {
                let hits = crate::storage::routed_lexical_search(
                    &router,
                    project_id.as_deref(),
                    &command.query,
                    limit,
                )?;
                if cli.json {
                    print_json(&hits)?;
                } else {
                    for hit in hits {
                        let scope = hit
                            .memory
                            .project_id
                            .as_deref()
                            .map_or("global", |project| project);
                        println!(
                            "{}\t{}\t{}\t{}",
                            hit.memory.id, hit.memory.kind, scope, hit.memory.text
                        );
                    }
                }
            }
        }
        Command::Get(command) => {
            let (store, id) = crate::storage::routed_resolve_memory(&router, None, &command.id)?;
            let record = store.get(&id)?;
            if cli.json {
                print_json(&record)?;
            } else {
                println!("id: {}", record.memory.id);
                println!("kind: {}", record.memory.kind);
                println!("scope: {}", record.memory.scope);
                if let Some(project) = &record.memory.project_id {
                    println!("project: {project}");
                }
                println!("actor: {}", record.memory.actor);
                println!("status: {}", record.memory.status);
                println!("text: {}", record.memory.text);
                for source in record.sources {
                    let locator = source.locator.as_deref().unwrap_or("-");
                    println!("source: {} {locator}", source.source_type);
                }
                for relation in record.relations {
                    println!(
                        "relation: {} {} -> {}",
                        relation.relation_type, relation.from_memory_id, relation.to_memory_id
                    );
                }
            }
        }
        Command::Forget(command) => {
            let (store, id) = crate::storage::routed_resolve_memory(&router, None, &command.id)?;
            let id = store.forget(&id)?;
            if cli.json {
                print_json(&serde_json::json!({ "id": id, "status": "deleted" }))?;
            } else {
                println!("forgot {id}");
            }
        }
        Command::Index(command) => match command.command {
            IndexSubcommand::Status(_) => {
                let context = optional_project_context(None, None)?;
                let project_id = context.as_ref().map(|c| c.project_id.as_str());
                let stats = crate::storage::routed_index_stats(&router, project_id)?;
                if cli.json {
                    print_json(&stats)?;
                } else {
                    println!("pending: {}", stats.pending);
                    println!("running: {}", stats.running);
                }
            }
            IndexSubcommand::Run(command) => {
                let context = optional_project_context(None, None)?;
                let project_id = context.as_ref().map(|c| c.project_id.as_str());
                let stats = crate::storage::routed_run_index(
                    &router,
                    project_id,
                    false,
                    EmbeddingRunOptions {
                        limit: command.limit.unwrap_or(64).min(1000),
                        lease_duration: Duration::from_secs(
                            command.lease_seconds.unwrap_or(1800).min(86_400),
                        ),
                        retry_delay: Duration::from_secs(
                            command.retry_seconds.unwrap_or(60).min(86_400),
                        ),
                        cache_dir: model_cache_dir()?,
                        show_download_progress: !cli.json,
                        cached_only: command.cached_only,
                    },
                )?;
                if cli.json {
                    print_json(&stats)?;
                } else {
                    println!("model: {}", stats.model);
                    println!("cache: {}", stats.cache_dir);
                    println!("stores: {}", stats.stores);
                    println!("claimed: {}", stats.claimed);
                    println!("eligible: {}", stats.eligible);
                    println!("committed: {}", stats.committed);
                    println!("stale: {}", stats.stale);
                    println!("retried: {}", stats.retried);
                }
            }
            IndexSubcommand::Claim(command) => {
                let mut store = require_exact_store(&router)?;
                let jobs = store.claim_index_jobs(
                    &command.worker,
                    command.limit.unwrap_or(32).clamp(1, 1000),
                    Duration::from_secs(command.lease_seconds.unwrap_or(60).clamp(1, 86_400)),
                )?;
                if cli.json {
                    print_json(&jobs)?;
                } else {
                    for job in jobs {
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                            job.id,
                            job.entity_type,
                            job.entity_id,
                            job.index_kind,
                            job.generation,
                            job.lease_token,
                            job.lease_until
                        );
                    }
                }
            }
            IndexSubcommand::Commit(command) => {
                let mut store = require_exact_store(&router)?;
                let vector: Vec<f32> = serde_json::from_str(&command.vector)
                    .context("embedding vector must be a JSON array of numbers")?;
                let committed = store.commit_embedding(
                    &command.job,
                    command.generation,
                    &command.lease_token,
                    &command.model,
                    &vector,
                )?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "job": command.job,
                        "generation": command.generation,
                        "model": command.model,
                        "committed": committed
                    }))?;
                } else if committed {
                    println!(
                        "committed embedding {} generation {} model {}",
                        command.job, command.generation, command.model
                    );
                } else {
                    println!(
                        "stale or unowned job {} generation {}",
                        command.job, command.generation
                    );
                }
            }
            IndexSubcommand::Complete(command) => {
                let store = require_exact_store(&router)?;
                let completed = store.complete_index_job(
                    &command.job,
                    command.generation,
                    &command.lease_token,
                )?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "job": command.job,
                        "generation": command.generation,
                        "completed": completed
                    }))?;
                } else if completed {
                    println!(
                        "completed {} generation {}",
                        command.job, command.generation
                    );
                } else {
                    println!(
                        "stale or unowned job {} generation {}",
                        command.job, command.generation
                    );
                }
            }
            IndexSubcommand::Retry(command) => {
                let store = require_exact_store(&router)?;
                let retried = store.retry_index_job(
                    &command.job,
                    command.generation,
                    &command.lease_token,
                    &command.error,
                    Duration::from_secs(command.delay_seconds.unwrap_or(30).min(86_400)),
                )?;
                if cli.json {
                    print_json(&serde_json::json!({
                        "job": command.job,
                        "generation": command.generation,
                        "retried": retried
                    }))?;
                } else if retried {
                    println!("retried {} generation {}", command.job, command.generation);
                } else {
                    println!(
                        "stale or unowned job {} generation {}",
                        command.job, command.generation
                    );
                }
            }
        },
    }

    Ok(())
}

fn optional_project_context(
    project: Option<&str>,
    workspace: Option<&str>,
) -> Result<Option<ProjectContext>> {
    let context = ProjectContext::detect(project, workspace)?;
    if context.is_none() && workspace.is_some() {
        bail!("--workspace requires a detected or explicit project");
    }
    Ok(context)
}

fn project_context(project: Option<&str>, workspace: Option<&str>) -> Result<ProjectContext> {
    optional_project_context(project, workspace)?
        .context("no project detected; run inside a Git repository or pass --project")
}

fn memory_project(explicit: Option<&str>, global: bool) -> Result<Option<String>> {
    if global && explicit.is_some() {
        bail!("--global conflicts with --project");
    }
    if global {
        return Ok(None);
    }
    if let Some(project) = explicit {
        let project = project.trim();
        if project.is_empty() {
            bail!("project identifier cannot be empty");
        }
        return Ok(Some(project.to_owned()));
    }

    Ok(ProjectContext::detect(None, None)?.map(|context| context.project_id))
}

/// Semantic-first recall with lexical-OR fallback for `mem context`.
/// Embeddings are a derived enhancement: when the model is not cached, the
/// cache directory cannot be determined, local embedding fails, or any
/// visible active memory still lacks a current-model vector, recall degrades
/// to the FTS5 baseline. Incomplete embedding coverage must never make a
/// canonical active memory disappear from recall.
pub fn recall_hits(
    store: &Store,
    query: &str,
    project_id: Option<&str>,
    limit: usize,
    force_lexical: bool,
) -> Result<Vec<SearchHit>> {
    if !force_lexical
        && store.has_complete_scope_coverage(EMBEDDING_MODEL_ID, project_id)?
        && let Some(query_vector) = cached_query_vector(query)?
    {
        let hits = store.semantic_search_by_vector(
            &query_vector,
            EMBEDDING_MODEL_ID,
            project_id,
            limit,
        )?;
        return Ok(hits
            .into_iter()
            .filter(|hit| hit.score >= SEMANTIC_CONTEXT_MIN_SCORE)
            .map(|hit| SearchHit {
                memory: hit.memory,
                rank: hit.score,
            })
            .collect());
    }
    store.recall(query, project_id, limit)
}

fn cached_query_vector(query: &str) -> Result<Option<Vec<f32>>> {
    let Ok(cache_dir) = model_cache_dir() else {
        return Ok(None);
    };
    // Fail open: a cached-but-broken model must not break adapter recall.
    Ok(embed_query_if_cached(query, &cache_dir).unwrap_or(None))
}

fn print_context(output: &ContextOutput) {
    if let Some(project) = &output.project {
        println!("project: {}", project.project_id);
        println!("workspace: {}", project.workspace_id);
    } else {
        println!("project: global");
    }

    if let Some(state) = &output.state {
        println!("state:");
        if let Some(session) = &state.last_session_id {
            println!("  session: {session}");
        }
        if let Some(goal) = &state.active_goal {
            println!("  goal: {goal}");
        }
        if let Some(task) = &state.active_task_ref {
            println!("  task: {task}");
        }
        if let Some(checkpoint) = &state.checkpoint {
            println!("  checkpoint: {checkpoint}");
        }
    }

    println!("memories:");
    for item in &output.memories {
        let scope = item
            .memory
            .project_id
            .as_deref()
            .map_or("global", |project| project);
        println!(
            "  {}\t{}\t{}\t{}",
            item.memory.id, item.memory.kind, scope, item.memory.text
        );
        for source in &item.sources {
            let locator = source.locator.as_deref().unwrap_or("-");
            println!("    source: {} {locator}", source.source_type);
        }
    }
}

fn print_workspace_state(state: &WorkspaceState) {
    println!("project: {}", state.project_id);
    println!("workspace: {}", state.workspace_id);
    if let Some(session) = &state.last_session_id {
        println!("session: {session}");
    }
    if let Some(goal) = &state.active_goal {
        println!("goal: {goal}");
    }
    if let Some(task) = &state.active_task_ref {
        println!("task: {task}");
    }
    if let Some(checkpoint) = &state.checkpoint {
        println!("checkpoint: {checkpoint}");
    }
    println!("updated_at: {}", state.updated_at);
}

/// Resolve the database for this invocation. Explicit overrides pin one
/// file. Otherwise `mem` uses `<project>/.mem/mem.db`. An existing
/// `.mem/` inside the current project wins; otherwise a Git root is the
/// project boundary, and non-Git use falls back to the current directory.
fn database_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_owned());
    }
    if let Some(path) = std::env::var_os("MEM_DB").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    let cwd = std::env::current_dir().context("determine current directory")?;
    let project = ProjectContext::detect(None, None)?;
    if let Some(root) = project.as_ref().and_then(|context| context.root.as_ref()) {
        for ancestor in cwd.ancestors() {
            let mem_dir = ancestor.join(".mem");
            if mem_dir.is_dir() {
                return Ok(mem_dir.join("mem.db"));
            }
            if ancestor == root {
                break;
            }
        }
        return Ok(root.join(".mem").join("mem.db"));
    }
    for ancestor in cwd.ancestors() {
        let mem_dir = ancestor.join(".mem");
        if mem_dir.is_dir() {
            return Ok(mem_dir.join("mem.db"));
        }
    }
    Ok(cwd.join(".mem").join("mem.db"))
}

fn command_writes(command: &Command) -> bool {
    match command {
        Command::Init(_) | Command::Remember(_) | Command::Correct(_) | Command::Forget(_) => true,
        Command::State(state) => !matches!(&state.command, StateCommand::Show(_)),
        Command::Index(index) => !matches!(&index.command, IndexSubcommand::Status(_)),
        _ => false,
    }
}

fn ensure_local_ignore(db_path: &Path) -> Result<()> {
    let directory = db_path
        .parent()
        .context("local memory database has no parent directory")?;
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create local memory directory {}", directory.display()))?;
    let ignore = directory.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(
            &ignore, "*
",
        )
        .with_context(|| format!("write {}", ignore.display()))?;
    }
    Ok(())
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    use std::io::Write as _;
    writeln!(lock)?;
    Ok(())
}

#[cfg(test)]
mod cli_parse_tests {
    //! Arg-matrix tests for the CLI surface: every command parses with its
    //! documented flags, defaults hold, and scope-flag coverage matches the
    //! help text. Parsing is pure, so no process or database is involved.

    use super::{Command, MemCli, parse_cli};

    fn parse(args: &str) -> MemCli {
        let argv = std::iter::once("mem".to_owned())
            .chain(args.split_whitespace().map(str::to_owned))
            .collect::<Vec<_>>();
        parse_cli(&argv).expect("parse")
    }

    #[test]
    fn remember_flags_and_defaults() {
        let cli = parse(
            "remember text-here --kind fact --global --actor agent --source-type cli --source-ref r",
        );
        let Command::Remember(command) = &cli.command else {
            panic!("expected remember");
        };
        assert_eq!(command.text, "text-here");
        assert_eq!(command.kind.as_deref(), Some("fact"));
        assert!(command.global_memory);
        assert_eq!(command.actor.as_deref(), Some("agent"));
        assert_eq!(command.source_type.as_deref(), Some("cli"));
        assert_eq!(command.source_ref.as_deref(), Some("r"));
        assert!(command.project.is_none());
    }

    #[test]
    fn search_scope_and_tier_flags() {
        let cli = parse("search q --global --semantic -n 3 --project proj");
        let Command::Search(command) = &cli.command else {
            panic!("expected search");
        };
        assert!(command.global_memory);
        assert!(command.semantic);
        assert_eq!(command.limit, Some(3));
        assert_eq!(command.project.as_deref(), Some("proj"));
    }

    #[test]
    fn id_commands_take_no_scope_flags() {
        // ID-directed operations resolve inside the current project store.
        let get = parse("get 01abc");
        assert!(matches!(get.command, super::Command::Get(_)));
        let forget = parse("forget 01abc");
        assert!(matches!(forget.command, super::Command::Forget(_)));
        let correct = parse("correct 01abc new-text");
        assert!(matches!(correct.command, super::Command::Correct(_)));
    }

    #[test]
    fn state_patch_accepts_repeatable_flags() {
        let cli = parse(
            "state patch --goal g --session s --task t1 --checkpoint c --clear session --clear goal",
        );
        let Command::State(state) = &cli.command else {
            panic!("expected state");
        };
        let super::StateCommand::Patch(patch) = &state.command else {
            panic!("expected patch");
        };
        assert_eq!(patch.goal.as_deref(), Some("g"));
        assert_eq!(patch.session.as_deref(), Some("s"));
        assert_eq!(patch.task.as_deref(), Some("t1"));
        assert_eq!(patch.checkpoint.as_deref(), Some("c"));
        assert_eq!(patch.clear, vec!["session".to_owned(), "goal".to_owned()]);
    }
}
