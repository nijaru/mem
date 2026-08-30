mod project;
mod store;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use usage::{Args, Cli, Subcommands};

use crate::project::ProjectContext;
use crate::store::{
    Memory, MemorySource, NewCorrection, NewMemory, NewWorkspaceState, Store, WorkspaceState,
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
}

/// Store one durable semantic memory.
#[derive(Args)]
struct Remember {
    /// Memory text.
    text: String,

    /// Memory kind: fact, decision, constraint, preference, or procedure.
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

    /// Provenance type, such as cli, pi-session, git, file, or url.
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

    /// Provenance type for the correction, such as cli, pi-session, git, file, or url.
    #[usage(long)]
    source_type: Option<String>,

    /// Optional source locator or stable reference for the correction.
    #[usage(long)]
    source_ref: Option<String>,
}

/// Search active semantic memory with SQLite FTS5.
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

fn run(cli: MemCli) -> Result<()> {
    let db_path = database_path(cli.db.as_deref())?;
    let mut store = Store::open(&db_path)
        .with_context(|| format!("open memory database at {}", db_path.display()))?;

    match cli.command {
        Command::Init(_) | Command::Status(_) => {
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
                let state = store.workspace_state(&context.project_id, &context.workspace_id)?;
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
            StateCommand::Clear(command) => {
                let context =
                    project_context(command.project.as_deref(), command.workspace.as_deref())?;
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
                store.workspace_state(&project.project_id, &project.workspace_id)?
            } else {
                None
            };
            let project_id = project.as_ref().map(|project| project.project_id.as_str());
            let hits = store.recall(
                &command.query,
                project_id,
                command.limit.unwrap_or(10).clamp(1, 100),
            )?;
            let mut memories = Vec::with_capacity(hits.len());
            for hit in hits {
                let record = store.get(&hit.memory.id)?;
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
            let memory = store.remember(NewMemory {
                text: command.text,
                kind: command.kind.unwrap_or_else(|| "fact".to_owned()),
                project_id,
                actor: command.actor.unwrap_or_else(|| "user".to_owned()),
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
            let result = store.correct(
                &command.id,
                NewCorrection {
                    text: command.text,
                    kind: command.kind,
                    actor: command.actor.unwrap_or_else(|| "user".to_owned()),
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
            let hits = store.search(
                &command.query,
                project_id.as_deref(),
                command.limit.unwrap_or(10).clamp(1, 100),
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
        Command::Get(command) => {
            let record = store.get(&command.id)?;
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
            let id = store.forget(&command.id)?;
            if cli.json {
                print_json(&serde_json::json!({ "id": id, "status": "deleted" }))?;
            } else {
                println!("forgot {id}");
            }
        }
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

fn database_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_owned());
    }
    if let Some(path) = std::env::var_os("MEM_DB") {
        return Ok(PathBuf::from(path));
    }
    if let Some(home) = std::env::var_os("MEM_HOME") {
        return Ok(PathBuf::from(home).join("memory.db"));
    }

    let data_dir =
        dirs::data_local_dir().context("could not determine the local data directory")?;
    Ok(data_dir.join("mem").join("memory.db"))
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    use std::io::Write as _;
    writeln!(lock)?;
    Ok(())
}
