# mem

`mem` is a local memory CLI for agents.

The project is agent-runtime agnostic: the Rust CLI owns durable storage, retrieval, project/workspace identity, continuation state, source-backed episodic history, and durable derived-index scheduling, while integrations such as Pi stay thin lifecycle adapters over the same core.

## Status

Early development. The current core provides durable semantic memory, non-destructive correction/supersession, SQLite FTS5 retrieval, automatic Git project/workspace identity, compact continuation state, adapter-facing `context` retrieval, source-backed episodic history, and a crash-safe derived-index queue.

The actual local embedding model/vector retrieval and agent adapters are still intentionally deferred.

## Design

- **SQLite is canonical.** The database is local, transactional, and portable.
- **Semantic memory, episodic history, and continuation state are different data.** They do not share one generic log or `MEMORY.md` abstraction.
- **Provenance is first-class.** A semantic memory records who established it and at least one source type.
- **Corrections preserve history.** Replacements supersede active memories atomically instead of mutating or discarding prior evidence.
- **Episodes index original sources rather than replacing them.** Search hits retain both the source/session reference and the exact source-local entry reference.
- **FTS is immediate.** SQLite FTS5 is updated with semantic and episodic writes, so retrieval works without an embedding model.
- **Embeddings are derived data.** Canonical writes transactionally enqueue durable derived work; vector construction is allowed to lag or fail without affecting correctness.
- **Derived workers are fenced.** Generation numbers and opaque lease tokens prevent stale workers from completing work after the canonical source changed.
- **Project knowledge and workspace state have different scope.** Durable semantic memory is project/origin scoped; continuation is project + branch/detached-workspace scoped. Episodes can retain both project and workspace identity without making history itself the continuation cursor.
- **No daemon is required.** The queue has a one-shot worker protocol; a warm stdio service may be added only when it materially reduces model/index startup cost.
- **Sync is not part of the initial storage contract.** Stable IDs, timestamps, tombstones, relations, source references, generations, and migrations preserve room for it later.

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

# Correct an active memory without destroying its history.
mem correct <id-or-prefix> "Publication uses release/acquire ordering" \
  --source-type git \
  --source-ref def456

# Precise lexical semantic search: all query terms must match a memory.
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

# Index one original session and exact source-local entries.
mem episode create pi-session-123 --source-type pi-session
mem episode record <episode-id> message-42 "Publication handoff succeeded" \
  --kind message \
  --role assistant
mem episode end <episode-id>
mem episode get <episode-id>

# Search episode entries while retaining exact source backreferences.
mem history "publication handoff"

# Low-level derived-index worker protocol.
mem index status
mem index claim worker-1 --json
mem index complete <job-id> <generation> <lease-token>
mem index retry <job-id> <generation> <lease-token> "temporary failure"

# Includes provenance and semantic relations such as superseded_by.
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

`--project` and `--workspace` can override detection where supported. `--global` opts supported operations into user-wide/unscoped behavior.

## Semantic retrieval

`mem search` is intentionally precise: whitespace-separated terms are combined with FTS5 `AND`.

`mem context` is intended for agent recall and uses broader FTS5 `OR` retrieval. It returns:

- detected project/workspace identity when available;
- current continuation state for that workspace;
- matching active project + global semantic memories;
- provenance for each selected memory.

Superseded and deleted memories are retained in canonical storage but excluded from active search/recall.

Current retrieval remains lexical. The derived-index queue is present, but vector generation and hybrid ranking have not yet been enabled.

## Corrections and relations

`mem correct` only accepts an active memory. In one SQLite transaction it:

1. creates a replacement in the same global/project scope;
2. attaches new provenance to the replacement;
3. marks the previous record `superseded`;
4. records a `superseded_by` relation from old → new.

The replacement inherits the old memory kind unless `--kind` is supplied. It cannot silently move a memory between global and project scope.

`mem get` returns both incoming and outgoing semantic relations, so either side of a correction remains explainable.

## Episodic history

Schema v2 adds source-backed episodes and searchable episode entries. An episode represents one original session or event stream; `mem` does not copy that source into a new canonical transcript format.

`mem episode create` is idempotent for `(source_type, source_ref)`. Reusing the same source reference for a different project/workspace is rejected rather than silently rebinding history.

Each `mem episode record` entry stores:

- an exact source-local `source_ref`;
- source order (`ordinal`);
- entry kind and optional role;
- searchable text;
- optional occurrence timestamp and opaque JSON metadata.

Recording the same `(episode, source_ref)` again refreshes that indexed entry rather than creating a duplicate. `mem history` searches entry text with FTS5 `AND` and returns both the episode-level source reference and the exact entry-level source reference, so an adapter can progressively disclose the original evidence.

## Durable derived indexing

Schema v3 adds a durable queue for derived embedding work. Active semantic memories and episode entries are backfilled during migration, and future canonical writes enqueue or invalidate work through SQLite triggers in the same transaction as the source change.

A job has a stable entity key plus a monotonically increasing generation. Claims add an opaque lease token and expiry. This provides four important guarantees:

- a crash leaves running work reclaimable after lease expiry;
- a source update requeues the same entity at a newer generation;
- stale generations/lease tokens cannot complete newer work;
- retry can release a lease and defer the next claim without losing the diagnostic.

`mem index status|claim|complete|retry` is a low-level one-shot protocol for index workers and integrations. Normal memory workflows do not need to call it directly.

Opening older databases migrates sequentially through schema versions 1 → 2 → 3. Migration/backfill, enqueue, completion, stale-generation fencing, expired-lease recovery, and delayed retry are covered by tests.

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

Memory status is one of `active`, `superseded`, or `deleted`. `mem status` reports each category explicitly.

## Next

The next qualified slices are expected to be:

1. vector storage and stale-safe embedding result commit semantics;
2. a local embedding worker over the qualified queue;
3. exact vector retrieval followed by measured hybrid FTS/vector ranking;
4. a thin Pi extension, with a warm `mem serve --stdio` mode only if it materially helps latency.

ANN indexes, automatic LLM memory promotion, sync protocols, and custom database backends remain deferred until measured requirements justify them.

## License

MIT
