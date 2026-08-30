# mem

`mem` is a local memory CLI for agents.

The project is agent-runtime agnostic: the Rust CLI owns durable storage, retrieval, project/workspace identity, and continuation state, while integrations such as Pi stay thin lifecycle adapters over the same core.

## Status

Early development. The current core provides durable semantic memory, SQLite FTS5 retrieval, automatic Git project/workspace identity, compact continuation state, and an adapter-facing `context` operation.

Episodic indexing, embeddings, automatic memory construction, and agent adapters are still intentionally deferred.

## Design

- **SQLite is canonical.** The database is local, transactional, and portable.
- **Semantic memory, episodic history, and continuation state are different data.** They do not share one generic log or `MEMORY.md` abstraction.
- **Provenance is first-class.** A semantic memory records who established it and at least one source type.
- **FTS is immediate.** SQLite FTS5 is updated with memory writes, so retrieval works without an embedding model.
- **Embeddings are derived data.** Vector indexing will be incremental/background work and must not be required for correctness.
- **Project knowledge and workspace state have different scope.** Durable memory is project/origin scoped; continuation is project + branch/detached-workspace scoped.
- **No daemon is required.** A warm stdio service may be added for agent integrations when it materially reduces model/index startup cost.
- **Sync is not part of the initial storage contract.** Stable IDs, timestamps, tombstones, relations, and migrations preserve room for it later.

## Build

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

Install the development build:

```bash
cargo install --path .
```

The crates.io package is `mem-cli`; the installed executable is `mem`.

## Usage

```bash
mem init
mem status

# Inspect automatic Git identity.
mem project

# Project-scoped by default when run inside a Git repository.
mem remember "Publication uses acquire/release ordering" \
  --kind decision \
  --source-type git \
  --source-ref abc123

# Explicit user-wide memory.
mem remember "Prefer source evidence over stale summaries" \
  --kind preference \
  --global

# Precise lexical search: all query terms must match a memory.
mem search "publication ordering"

# Compact per-workspace continuation state.
mem state set \
  --session pi-session-123 \
  --goal "qualify publication path" \
  --checkpoint "Loom is green; release build remains"
mem state show
mem state clear

# Adapter-facing retrieval: state + broad semantic recall + provenance.
mem context "publication handoff"

mem get <id-or-prefix>
mem forget <id-or-prefix>
```

Add `--json` for machine-readable output. `--db <path>` overrides the database for a command.

## Project and workspace identity

Inside Git, `mem` derives the project from the canonicalized `origin` remote when possible. Common SSH/HTTPS forms normalize to identities such as:

```text
github.com/nijaru/mem
```

Without an origin, it falls back to a local repository-root identity.

The default workspace identity is separate:

- attached branch: `branch:<name>`
- detached HEAD: `detached:<short-sha>`

This keeps durable project memory shared across branches while preventing independent workspaces from overwriting the same continuation cursor.

`--project` and `--workspace` can override detection where supported. `--global` opts memory operations into user-wide scope.

## Retrieval

`mem search` is intentionally precise: whitespace-separated terms are combined with FTS5 `AND`.

`mem context` is intended for agent recall and uses broader FTS5 `OR` retrieval. It returns:

- detected project/workspace identity when available;
- current continuation state for that workspace;
- matching active project + global semantic memories;
- provenance for each selected memory.

Current lexical retrieval does not perform stemming, semantic expansion, or embedding search. Those belong to later hybrid retrieval rather than being hidden inside the baseline.

## Data location

By default `mem` stores `memory.db` under the platform-local application data directory in a `mem` subdirectory.

Overrides:

- `MEM_DB=/path/to/memory.db` selects an exact database path.
- `MEM_HOME=/path/to/dir` stores the database at `$MEM_HOME/memory.db`.
- `--db /path/to/memory.db` takes precedence for one command.

## Semantic memory

Current memory kinds:

- `fact`
- `decision`
- `constraint`
- `preference`
- `procedure`

A memory is either global or project-scoped. Inside a Git repository, `remember` and `search` use the detected project by default; project retrieval also includes global memories. `--global` selects global-only behavior.

## Next

The next qualified slices are expected to be:

1. correction/supersession semantics and richer relation-aware reads;
2. episodic source/index primitives with exact session backreferences;
3. durable background indexing jobs and local embeddings;
4. hybrid FTS/vector retrieval;
5. a thin Pi extension, with a warm `mem serve --stdio` mode only if it materially helps latency.

ANN indexes, automatic LLM memory promotion, sync protocols, and custom database backends remain deferred until measured requirements justify them.

## License

MIT
