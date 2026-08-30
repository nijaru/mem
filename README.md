# mem

`mem` is a local memory CLI for agents.

The project is intentionally agent-runtime agnostic: the Rust CLI owns durable storage and retrieval, while integrations such as Pi can stay thin lifecycle adapters over the same core.

## Status

Early development. The initial vertical slice provides durable semantic memory and lexical retrieval; episodic indexing, continuation integration, embeddings, and agent adapters follow after the core is qualified.

## Design

- **SQLite is canonical.** The database is local, transactional, and portable.
- **Semantic memory, episodic history, and continuation state are different data.** They do not share one generic log or `MEMORY.md` abstraction.
- **Provenance is first-class.** A semantic memory records who established it and at least one source type.
- **FTS is immediate.** SQLite FTS5 is updated with memory writes, so search works without an embedding model.
- **Embeddings are derived data.** Vector indexing will be incremental/background work and must not be required for correctness.
- **No daemon is required.** A warm stdio service may be added for agent integrations when it materially reduces model/index startup cost.
- **Sync is not part of the initial storage contract.** Stable IDs, timestamps, tombstones, and migrations preserve room for it later.

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
mem remember "Prefer source evidence over stale summaries" --kind preference
mem remember "Publication uses acquire/release ordering" \
  --kind decision \
  --project github.com/nijaru/example \
  --source-type git \
  --source-ref abc123

mem search "source evidence"
mem search "publication ordering" --project github.com/nijaru/example
mem get <id-or-prefix>
mem forget <id-or-prefix>
mem status
```

Add `--json` for machine-readable output. `--db <path>` overrides the database for a command.

## Data location

By default `mem` stores `memory.db` under the platform-local application data directory in a `mem` subdirectory.

Overrides:

- `MEM_DB=/path/to/memory.db` selects an exact database path.
- `MEM_HOME=/path/to/dir` stores the database at `$MEM_HOME/memory.db`.
- `--db /path/to/memory.db` takes precedence for one command.

## Current model

Semantic memory kinds are deliberately small in the first version:

- `fact`
- `decision`
- `constraint`
- `preference`
- `procedure`

A memory is either global or project-scoped. Searching without `--project` searches global memory only; searching with `--project` searches that project plus global memory.

## Next

The next qualified slices are expected to be:

1. correction/supersession semantics and richer provenance;
2. project/workspace identity;
3. Pi session episode indexing with exact source backreferences;
4. background embedding jobs and hybrid FTS/vector retrieval;
5. a warm `mem serve --stdio` protocol for agent adapters.

The ordering can change based on measured use. ANN indexes, automatic LLM memory extraction, sync protocols, and custom database backends are intentionally deferred.

## License

MIT
