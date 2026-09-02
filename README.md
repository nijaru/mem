# mem

`mem` is a local memory CLI for agents and coding workflows.

It provides a small durable substrate for three different kinds of agent context:

- **semantic memory** — durable facts, decisions, constraints, preferences, and procedures;
- **episodic history** — searchable projections of past sessions with exact source backreferences;
- **continuation state** — compact per-workspace state describing where work should resume.

`mem` is agent-runtime agnostic. The Rust CLI owns persistence, retrieval, project/workspace identity, and memory semantics; integrations such as Pi can remain thin adapters over the same core.

## Features

- Single local SQLite database with bundled SQLite and FTS5.
- Project-scoped memory derived automatically from Git remotes.
- Separate branch/worktree continuation state.
- Global memory for user-wide preferences and facts.
- Provenance attached to semantic memories.
- Non-destructive correction and supersession.
- Source-backed episodic history with exact session/entry references.
- Machine-readable JSON output for agent integrations.
- No daemon required for normal operation.

## Installation

`mem` currently requires Rust 1.98 or newer.

```bash
cargo install --path .
```

The crates.io package name is `mem-cli`; the installed executable is `mem`.

For development:

```bash
cargo build
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

## Quick start

Initialize or inspect the local store:

```bash
mem init
mem status
```

Inside a Git repository, project scope is detected automatically:

```bash
mem project
```

Store a durable project memory:

```bash
mem remember "Publication uses release/acquire ordering" \
  --kind decision \
  --source-type git \
  --source-ref abc123
```

Store a user-wide memory instead:

```bash
mem remember "Prefer concise command output" \
  --kind preference \
  --global
```

Search semantic memory:

```bash
mem search "publication ordering"
```

Correct a memory without destroying the previous record:

```bash
mem correct <id-or-prefix> "Publication uses release/acquire ordering" \
  --source-type git \
  --source-ref def456
```

Read a memory together with its provenance and relations:

```bash
mem get <id-or-prefix>
```

Soft-delete it from active retrieval:

```bash
mem forget <id-or-prefix>
```

## Continuation state

Continuation state is keyed by project and workspace rather than shared across every branch:

```bash
mem state set \
  --session pi-session-123 \
  --goal "qualify publication path" \
  --checkpoint "Loom is green; release build remains"

mem state show
mem state clear
```

`state set` replaces the whole row; `state patch` updates only the provided fields atomically, so adapters never need read-merge-write (which races concurrent writers) and never wipe fields they did not read:

```bash
mem state patch --checkpoint "Loom is green; release build remains"
mem state patch --clear goal
```

This keeps durable project knowledge shared while allowing independent branches or worktrees to resume different work safely.

## Agent context

`mem context` is the primary adapter-facing read operation. It combines current workspace continuation state with broadly recalled semantic memory and provenance:

```bash
mem context "publication handoff"
```

Use `--json` for machine-readable output:

```bash
mem --json context "publication handoff"
```

Recall is byte-bounded as well as count-bounded: `--max-bytes` (default 32768, `0` disables) stops recall at the first hit whose memory text would push the total past the budget, in rank order. The top-ranked hit is always included when any budget allows it, so a query matching one oversized memory still surfaces that match instead of nothing.

## Episodic history

Episodes index an original session or event stream without replacing that source as the authoritative transcript.

```bash
mem episode create pi-session-123 --source-type pi-session

mem episode record <episode-id> message-42 \
  "Publication handoff succeeded" \
  --kind message \
  --role assistant

mem episode end <episode-id>
mem episode get <episode-id>
```

Search indexed history:

```bash
mem history "publication handoff"
```

History hits retain both the episode-level source reference and the exact source-local entry reference so integrations can expand a result back into the original evidence.

History search is lexical (FTS5) only. Episode entries are never embedded: no reader consumes episode-entry vectors yet, so maintaining them would spend model time and database space on derived data with no consumer. Embedding work covers semantic memories only.

## Project and workspace identity

When run inside Git, `mem` derives the project identity from the canonicalized `origin` remote where possible. Common SSH and HTTPS forms normalize to identities such as:

```text
github.com/nijaru/mem
```

Without an origin remote, `mem` falls back to a local repository-root identity.

Workspace identity is separate:

- attached branch: `branch:<name>`
- detached HEAD: `detached:<short-sha>`

Supported commands can override automatic detection with `--project` and `--workspace`. `--global` selects user-wide/unscoped behavior where applicable.

## Retrieval semantics

Semantic search uses SQLite FTS5 immediately; no external service or model is required.

- `mem search` uses precise lexical `AND` matching.
- `mem context` ranks semantically when the local embedding model is already cached and every active memory visible in the query scope has a current-model vector, and falls back to broader lexical `OR` recall otherwise. `--lexical` forces the lexical baseline.
- superseded and deleted memories remain in canonical storage but are excluded from active retrieval.

Vector retrieval is derived from the canonical SQLite data and is not required for correctness. Incomplete embedding coverage (for example, a just-remembered memory whose indexing job is still queued) makes `mem context` use lexical recall so no active memory can disappear. `mem context` never downloads the embedding model; run `mem index run` (or an explicit `mem search --semantic`) once to populate the local model cache and vectors. Embedding jobs whose canonical source lost its production-model vector (for example after a model change) are rediscovered by `mem index run`. `mem index run --cached-only` does nothing when the model is not already cached, so adapters can schedule indexing without risking a download. In the managed layout, `mem index run` covers the current project database plus the user database; `mem index run --all` also covers every other existing managed project database.

## Data location

By default, `mem` uses a managed per-project/user layout under the platform-local application data directory in a `mem` subdirectory:

```text
<managed root>/
  layout-v1/
    user.db                      global memories, global episodes
    projects/<encoded id>/mem.db one database per project
```

Overrides:

- `MEM_DB=/path/to/memory.db` selects an exact single database path for every operation (the bypass used by tests and isolated profiles).
- `--db /path/to/memory.db` overrides the database for one command.
- `MEM_HOME=/path/to/dir` moves the managed root (layout, and any legacy `memory.db` pending migration) to that directory.

Reads never create missing databases; writes create exactly the one database a row belongs in.

Created layout directories and store files are private to the current user (`0700` directories, `0600` databases and sidecars); memory text often holds credentials and internal paths, so a shared machine cannot read another user's agent memory. Existing files are never re-permissioned. Recall merges project and user stores under a complete-coverage semantic gate with a deterministic lexical fallback; `history` stays single-store by design.

### Legacy single file

Earlier versions stored everything in one `<managed root>/memory.db`. When that file exists without an active `layout-v1`, storage-touching commands refuse with guidance instead of silently splitting usage; run:

```text
mem storage status
mem storage migrate
```

Migration builds and verifies the full layout in a hidden staging directory and activates it with one atomic directory rename; the legacy file is left untouched for rollback. Re-running against an active layout is a no-op. `mem storage purge --project <id>` deletes exactly that project's managed database and SQLite sidecars after confirmation (`--yes` to skip).

### Managed layout (in progress)

A per-project/user managed layout (`layout-v1/` with `user.db` and `projects/<encoded>/mem.db`) is being rolled out in slices. `mem storage status` reports the managed-layout inventory (layout version and paths, legacy `memory.db` presence, per-store schema/counts/queue state, migration-needed state) without creating any files. Managed `index run` covers the current project database plus the user database; `index run --all` additionally covers every existing managed project database. The low-level worker protocol (`index claim|commit|complete|retry`) is pinned to one exact database. The remaining rollout item is dogfooding the managed default; the storage layout itself is complete: scoped routing, index routing, staged migration, and purge are all active.

## Memory model

Current semantic memory kinds are:

- `fact`
- `decision`
- `constraint`
- `preference`
- `procedure`

A semantic memory is either global or project-scoped and has one of three statuses: `active`, `superseded`, or `deleted`.

Corrections create a replacement record in the same scope, attach fresh provenance, mark the old memory `superseded`, and retain an explicit `superseded_by` relation. The previous evidence remains inspectable instead of being overwritten.

## Status

`mem` is pre-1.0 and under active development. The local semantic-memory, continuation-state, episodic-history, and lexical retrieval core is usable today; interfaces may still evolve before the first stable release.

## License

MIT
