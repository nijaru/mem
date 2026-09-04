mod embedding;
mod embedding_worker;
mod id_resolve;
mod store;
mod vector_search;
mod workspace;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use usage::{Args, Cli, Subcommands};

use crate::embedding_worker::{
    EMBEDDING_MODEL_ID, EmbeddingRunOptions, EmbeddingRunStats, embed_query,
    embed_query_if_cached, model_cache_dir,
};
use crate::store::{
    Memory, NewCorrection, NewMemory, SearchHit, Store, WorkspaceState, WorkspaceStateUpdate,
};

#[derive(Cli)]
#[usage(bin = "mem", version, arg_required_else_help, unknown_flags = "error")]
struct MemCli {
    /// Emit machine-readable JSON.
    #[usage(long, global)]
    json: bool,

    /// Use an exact SQLite database instead of the repo-local store.
    #[usage(long, global, value_hint = usage::ValueHint::FilePath)]
    db: Option<PathBuf>,

    #[usage(subcommand)]
    command: Command,
}

#[derive(Subcommands)]
enum Command {
    Init(Init),
    Status(Status),
    State(State),
    Context(ContextCommand),
    Remember(Remember),
    Correct(Correct),
    Search(Search),
    Get(Get),
    Forget(Forget),
    Index(IndexCommand),
}

/// Initialize the repo-local memory store.
#[derive(Args)]
struct Init;

/// Show store, workspace, and embedding status.
#[derive(Args)]
struct Status;

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
    Clear(StateClear),
}

/// Show continuation state for the current workspace.
#[derive(Args)]
struct StateShow {
    /// Override the detected workspace identifier.
    #[usage(long)]
    workspace: Option<String>,
}

/// Update continuation state. Omitted fields are preserved.
#[derive(Args)]
struct StateSet {
    /// Override the detected workspace identifier.
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

    /// Field to clear: session, goal, task, or checkpoint. Repeatable.
    #[usage(long, var)]
    clear: Vec<String>,
}

/// Remove continuation state for the current workspace.
#[derive(Args)]
struct StateClear {
    /// Override the detected workspace identifier.
    #[usage(long)]
    workspace: Option<String>,
}

/// Build agent-facing context from continuation state and durable memory.
#[derive(Args)]
struct ContextCommand {
    /// Prompt or query to recall relevant memory for.
    query: String,

    /// Override the detected workspace identifier.
    #[usage(long)]
    workspace: Option<String>,

    /// Maximum number of memories.
    #[usage(short = 'n', long)]
    limit: Option<usize>,

    /// Maximum total memory-text bytes. Default 32768; 0 disables the budget.
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

/// Correct one active memory while preserving the superseded record.
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

    /// Provenance type for the correction.
    #[usage(long)]
    source_type: Option<String>,

    /// Optional source locator or stable reference for the correction.
    #[usage(long)]
    source_ref: Option<String>,
}

/// Search active memory. Lexical search is the default.
#[derive(Args)]
struct Search {
    /// Search query.
    query: String,

    /// Use local embeddings and exact cosine similarity.
    #[usage(long)]
    semantic: bool,

    /// Maximum number of results.
    #[usage(short = 'n', long)]
    limit: Option<usize>,
}

/// Read one memory and its provenance.
#[derive(Args)]
struct Get {
    /// Full memory ID or an unambiguous ID prefix.
    id: String,
}

/// Soft-delete one active memory from retrieval.
#[derive(Args)]
struct Forget {
    /// Full memory ID or an unambiguous ID prefix.
    id: String,
}

/// Build missing semantic embeddings for active memories.
#[derive(Args)]
struct IndexCommand {
    /// Maximum number of memories to embed.
    #[usage(short = 'n', long)]
    limit: Option<usize>,

    /// Do nothing when the embedding model is not already cached locally.
    #[usage(long)]
    cached_only: bool,
}

#[derive(Serialize)]
struct StatusOutput {
    database: String,
    initialized: bool,
    schema_version: i64,
    root: Option<PathBuf>,
    workspace: String,
    total: u64,
    active: u64,
    superseded: u64,
    deleted: u64,
    indexed: u64,
    unindexed: u64,
}

#[derive(Serialize)]
struct ClearStateOutput<'a> {
    workspace: &'a str,
    cleared: bool,
}

#[derive(Serialize)]
struct ContextOutput {
    root: Option<PathBuf>,
    workspace: String,
    state: Option<WorkspaceState>,
    memories: Vec<Memory>,
}

fn main() {
    #[cfg(unix)]
    {
        // Safety: this short-lived process only creates mem-owned files.
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

const RECALL_DEFAULT_MAX_BYTES: usize = 32 * 1024;
const SEMANTIC_CONTEXT_MIN_SCORE: f64 = 0.55;

fn run(cli: MemCli) -> Result<()> {
    let explicit_db = cli.db.is_some()
        || std::env::var_os("MEM_DB")
            .filter(|path| !path.is_empty())
            .is_some();
    let db_path = database_path(cli.db.as_deref())?;
    if !explicit_db && command_creates_store(&cli.command) {
        ensure_local_ignore(&db_path)?;
    }

    match cli.command {
        Command::Init(_) => {
            let store = Store::open(&db_path)?;
            print_status(&store_status(&db_path, Some(&store))?, cli.json)?;
        }
        Command::Status(_) => {
            let store = Store::open_existing(&db_path)?;
            print_status(&store_status(&db_path, store.as_ref())?, cli.json)?;
        }
        Command::State(command) => match command.command {
            StateCommand::Show(command) => {
                let workspace = crate::workspace::workspace_id(command.workspace.as_deref())?;
                let state = match Store::open_existing(&db_path)? {
                    Some(store) => store.workspace_state(&workspace)?,
                    None => None,
                };
                if cli.json {
                    print_json(&state)?;
                } else if let Some(state) = state {
                    print_workspace_state(&state);
                } else {
                    println!("no state for {workspace}");
                }
            }
            StateCommand::Set(command) => {
                let workspace = crate::workspace::workspace_id(command.workspace.as_deref())?;
                let store = Store::open(&db_path)?;
                let state = store.update_workspace_state(WorkspaceStateUpdate {
                    workspace,
                    session: command.session,
                    goal: command.goal,
                    task: command.task,
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
                let workspace = crate::workspace::workspace_id(command.workspace.as_deref())?;
                let cleared = match Store::open_existing(&db_path)? {
                    Some(store) => store.clear_workspace_state(&workspace)?,
                    None => false,
                };
                if cli.json {
                    print_json(&ClearStateOutput {
                        workspace: &workspace,
                        cleared,
                    })?;
                } else if cleared {
                    println!("cleared state for {workspace}");
                } else {
                    println!("no state for {workspace}");
                }
            }
        },
        Command::Context(command) => {
            let workspace = crate::workspace::workspace_id(command.workspace.as_deref())?;
            let root = crate::workspace::repo_root();
            let limit = command.limit.unwrap_or(10).clamp(1, 100);
            let max_bytes = command.max_bytes.unwrap_or(RECALL_DEFAULT_MAX_BYTES);
            let (state, hits) = match Store::open_existing(&db_path)? {
                Some(store) => (
                    store.workspace_state(&workspace)?,
                    recall_hits(&store, &command.query, limit)?,
                ),
                None => (None, Vec::new()),
            };
            let mut memories = Vec::new();
            let mut total = 0usize;
            for hit in hits {
                let bytes = hit.memory.text.len();
                if max_bytes > 0 && !memories.is_empty() && total + bytes > max_bytes {
                    break;
                }
                total += bytes;
                memories.push(hit.memory);
            }
            let output = ContextOutput {
                root,
                workspace,
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
            let store = Store::open(&db_path)?;
            let memory = store.remember(NewMemory {
                text: command.text,
                kind: command.kind.unwrap_or_else(|| "fact".to_owned()),
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
            let mut store = require_store(&db_path)?;
            let result = store.correct(
                &command.id,
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
                println!("corrected {} -> {}", result.previous.id, result.replacement.id);
                println!(
                    "{}\t{}\t{}",
                    result.replacement.id, result.replacement.kind, result.replacement.text
                );
            }
        }
        Command::Search(command) => {
            let limit = command.limit.unwrap_or(10).clamp(1, 100);
            let Some(store) = Store::open_existing(&db_path)? else {
                if cli.json {
                    print_json(&Vec::<SearchHit>::new())?;
                }
                return Ok(());
            };
            if command.semantic {
                if !store.has_complete_coverage(EMBEDDING_MODEL_ID)? {
                    bail!("semantic index is incomplete; run `mem index` first");
                }
                let cache_dir = model_cache_dir()?;
                let vector = embed_query(&command.query, &cache_dir, !cli.json)?;
                let hits = store.semantic_search_by_vector(&vector, EMBEDDING_MODEL_ID, limit)?;
                if cli.json {
                    print_json(&hits)?;
                } else {
                    for hit in hits {
                        println!(
                            "{}\t{}\t{:.6}\t{}",
                            hit.memory.id, hit.memory.kind, hit.score, hit.memory.text
                        );
                    }
                }
            } else {
                let hits = store.search(&command.query, limit)?;
                if cli.json {
                    print_json(&hits)?;
                } else {
                    for hit in hits {
                        println!("{}\t{}\t{}", hit.memory.id, hit.memory.kind, hit.memory.text);
                    }
                }
            }
        }
        Command::Get(command) => {
            let store = require_store(&db_path)?;
            let memory = store.get(&command.id)?;
            if cli.json {
                print_json(&memory)?;
            } else {
                print_memory(&memory);
            }
        }
        Command::Forget(command) => {
            let store = require_store(&db_path)?;
            let id = store.forget(&command.id)?;
            if cli.json {
                print_json(&serde_json::json!({ "id": id, "status": "deleted" }))?;
            } else {
                println!("forgot {id}");
            }
        }
        Command::Index(command) => {
            let cache_dir = model_cache_dir()?;
            let stats = match Store::open_existing(&db_path)? {
                Some(store) => store.run_embedding_index(EmbeddingRunOptions {
                    limit: command.limit.unwrap_or(64).clamp(1, 1000),
                    cache_dir,
                    show_download_progress: !cli.json,
                    cached_only: command.cached_only,
                })?,
                None => EmbeddingRunStats {
                    model: EMBEDDING_MODEL_ID,
                    cache_dir: cache_dir.display().to_string(),
                    indexed: 0,
                    stale: 0,
                    remaining: 0,
                },
            };
            if cli.json {
                print_json(&stats)?;
            } else {
                println!("model: {}", stats.model);
                println!("indexed: {}", stats.indexed);
                println!("stale: {}", stats.stale);
                println!("remaining: {}", stats.remaining);
            }
        }
    }
    Ok(())
}

pub fn recall_hits(store: &Store, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    if store.has_complete_coverage(EMBEDDING_MODEL_ID)?
        && let Some(query_vector) = cached_query_vector(query)?
    {
        let hits = store.semantic_search_by_vector(&query_vector, EMBEDDING_MODEL_ID, limit)?;
        return Ok(hits
            .into_iter()
            .filter(|hit| hit.score >= SEMANTIC_CONTEXT_MIN_SCORE)
            .map(|hit| SearchHit {
                memory: hit.memory,
                rank: hit.score,
            })
            .collect());
    }
    store.recall(query, limit)
}

fn cached_query_vector(query: &str) -> Result<Option<Vec<f32>>> {
    let Ok(cache_dir) = model_cache_dir() else {
        return Ok(None);
    };
    Ok(embed_query_if_cached(query, &cache_dir).unwrap_or(None))
}

fn store_status(path: &Path, store: Option<&Store>) -> Result<StatusOutput> {
    let workspace = crate::workspace::workspace_id(None)?;
    let root = crate::workspace::repo_root();
    let Some(store) = store else {
        return Ok(StatusOutput {
            database: path.display().to_string(),
            initialized: false,
            schema_version: 0,
            root,
            workspace,
            total: 0,
            active: 0,
            superseded: 0,
            deleted: 0,
            indexed: 0,
            unindexed: 0,
        });
    };
    let stats = store.stats()?;
    let coverage = store.embedding_coverage(EMBEDDING_MODEL_ID)?;
    Ok(StatusOutput {
        database: path.display().to_string(),
        initialized: true,
        schema_version: stats.schema_version,
        root,
        workspace,
        total: stats.total,
        active: stats.active,
        superseded: stats.superseded,
        deleted: stats.deleted,
        indexed: coverage.indexed,
        unindexed: coverage.unindexed,
    })
}

fn print_status(output: &StatusOutput, json: bool) -> Result<()> {
    if json {
        return print_json(output);
    }
    println!("database: {}", output.database);
    if let Some(root) = &output.root {
        println!("root: {}", root.display());
    }
    println!("workspace: {}", output.workspace);
    if !output.initialized {
        println!("store: not initialized");
        return Ok(());
    }
    println!("schema: {}", output.schema_version);
    println!(
        "memories: {} active / {} superseded / {} deleted / {} total",
        output.active, output.superseded, output.deleted, output.total
    );
    println!(
        "embeddings: {} indexed / {} unindexed",
        output.indexed, output.unindexed
    );
    Ok(())
}

fn print_memory(memory: &Memory) {
    println!("id: {}", memory.id);
    println!("kind: {}", memory.kind);
    println!("status: {}", memory.status);
    println!("actor: {}", memory.actor);
    println!("source: {}", memory.source_type);
    if let Some(source_ref) = &memory.source_ref {
        println!("source_ref: {source_ref}");
    }
    if let Some(replacement) = &memory.superseded_by {
        println!("superseded_by: {replacement}");
    }
    println!("text: {}", memory.text);
}

fn print_context(output: &ContextOutput) {
    if let Some(root) = &output.root {
        println!("root: {}", root.display());
    }
    println!("workspace: {}", output.workspace);
    if let Some(state) = &output.state {
        println!("state:");
        if let Some(session) = &state.session {
            println!("  session: {session}");
        }
        if let Some(goal) = &state.goal {
            println!("  goal: {goal}");
        }
        if let Some(task) = &state.task {
            println!("  task: {task}");
        }
        if let Some(checkpoint) = &state.checkpoint {
            println!("  checkpoint: {checkpoint}");
        }
    }
    println!("memories:");
    for memory in &output.memories {
        println!("  {}\t{}\t{}", memory.id, memory.kind, memory.text);
        let source_ref = memory.source_ref.as_deref().unwrap_or("-");
        println!(
            "    provenance: actor={} source={} ref={}",
            memory.actor, memory.source_type, source_ref
        );
    }
}

fn print_workspace_state(state: &WorkspaceState) {
    println!("workspace: {}", state.workspace);
    if let Some(session) = &state.session {
        println!("session: {session}");
    }
    if let Some(goal) = &state.goal {
        println!("goal: {goal}");
    }
    if let Some(task) = &state.task {
        println!("task: {task}");
    }
    if let Some(checkpoint) = &state.checkpoint {
        println!("checkpoint: {checkpoint}");
    }
    println!("updated_at: {}", state.updated_at);
}

fn require_store(path: &Path) -> Result<Store> {
    Store::open_existing(path)?.with_context(|| format!("memory store not found: {}", path.display()))
}

fn database_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_owned());
    }
    if let Some(path) = std::env::var_os("MEM_DB").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    let cwd = std::env::current_dir().context("determine current directory")?;
    if let Some(root) = crate::workspace::repo_root() {
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

fn command_creates_store(command: &Command) -> bool {
    matches!(command, Command::Init(_) | Command::Remember(_))
        || matches!(command, Command::State(State { command: StateCommand::Set(_) }))
}

fn ensure_local_ignore(db_path: &Path) -> Result<()> {
    let directory = db_path
        .parent()
        .context("local memory database has no parent directory")?;
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create local memory directory {}", directory.display()))?;
    let ignore = directory.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(&ignore, "*\n")
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
    use super::{Command, MemCli, StateCommand, parse_cli};

    fn parse(args: &str) -> MemCli {
        let argv = std::iter::once("mem".to_owned())
            .chain(args.split_whitespace().map(str::to_owned))
            .collect::<Vec<_>>();
        parse_cli(&argv).expect("parse")
    }

    #[test]
    fn remember_metadata_is_explicit() {
        let cli = parse(
            "remember text-here --kind finding --actor agent --source-type cli --source-ref r",
        );
        let Command::Remember(command) = &cli.command else {
            panic!("expected remember");
        };
        assert_eq!(command.kind.as_deref(), Some("finding"));
        assert_eq!(command.source_ref.as_deref(), Some("r"));
    }

    #[test]
    fn state_set_is_partial_and_supports_clear() {
        let cli = parse("state set --goal g --session s --clear checkpoint");
        let Command::State(state) = &cli.command else {
            panic!("expected state");
        };
        let StateCommand::Set(set) = &state.command else {
            panic!("expected set");
        };
        assert_eq!(set.goal.as_deref(), Some("g"));
        assert_eq!(set.clear, vec!["checkpoint".to_owned()]);
    }

    #[test]
    fn index_is_one_command_not_a_worker_protocol() {
        let cli = parse("index --cached-only -n 4");
        let Command::Index(index) = &cli.command else {
            panic!("expected index");
        };
        assert!(index.cached_only);
        assert_eq!(index.limit, Some(4));
    }
}
