mod store;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result};
use serde::Serialize;
use usage::{Args, Cli, Subcommands};

use crate::store::{NewMemory, Store};

#[derive(Cli)]
#[usage(
    bin = "mem",
    version,
    arg_required_else_help,
    unknown_flags = "error"
)]
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
    Remember(Remember),
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

/// Store one durable semantic memory.
#[derive(Args)]
struct Remember {
    /// Memory text.
    text: String,

    /// Memory kind: fact, decision, constraint, preference, or procedure.
    #[usage(long)]
    kind: Option<String>,

    /// Project identifier. Omit for global memory.
    #[usage(long)]
    project: Option<String>,

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

/// Search active semantic memory with SQLite FTS5.
#[derive(Args)]
struct Search {
    /// Search query.
    query: String,

    /// Include global memory plus memory for this project.
    #[usage(long)]
    project: Option<String>,

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
    deleted: u64,
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
            if let Some(page) = usage::help::render_styled(
                MemCli::spec(),
                cmd,
                long,
                usage::help::Style::auto(),
            ) {
                print!("{page}");
            }
            None
        }
        Err(usage::Error::HelpAll { cmd }) => {
            if let Some(page) = usage::help::render_all_styled(
                MemCli::spec(),
                cmd,
                usage::help::Style::auto(),
            ) {
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
                deleted: stats.deleted,
            };
            if cli.json {
                print_json(&output)?;
            } else {
                println!("database: {}", output.database);
                println!("schema: {}", output.schema_version);
                println!("memories: {} active / {} total", output.active, output.total);
            }
        }
        Command::Remember(command) => {
            let memory = store.remember(NewMemory {
                text: command.text,
                kind: command.kind.unwrap_or_else(|| "fact".to_owned()),
                project_id: command.project,
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
        Command::Search(command) => {
            let hits = store.search(
                &command.query,
                command.project.as_deref(),
                command.limit.unwrap_or(10).clamp(1, 100),
            )?;
            if cli.json {
                print_json(&hits)?;
            } else {
                for hit in hits {
                    let scope = hit.memory.project_id.as_deref().unwrap_or("global");
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

    let data_dir = dirs::data_local_dir().context("could not determine the local data directory")?;
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
