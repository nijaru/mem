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
- `mem context` uses broader lexical `OR` recall for agent context construction.
- superseded and deleted memories remain in canonical storage but are excluded from active retrieval.

Vector retrieval is derived from the canonical SQLite data and is not required for correctness.

## Data location

By default, `mem` stores `memory.db` under the platform-local application data directory in a `mem` subdirectory.

Overrides:

- `MEM_DB=/path/to/memory.db` selects an exact database path.
- `MEM_HOME=/path/to/dir` stores the database at `$MEM_HOME/memory.db`.
- `--db /path/to/memory.db` overrides the database for one command.

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
