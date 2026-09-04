# mem

`mem` is a small repo-local memory CLI for coding agents. It preserves durable project knowledge after it falls out of the current context window.

## Core model

Each project owns one SQLite database at `.mem/mem.db`. Memories are concise durable facts, findings, decisions, constraints, preferences, and procedures. Each memory carries lightweight provenance (`actor`, `source_type`, and optional `source_ref`). Corrections preserve the old record as `superseded` and point it at the replacement.

Continuation state is separate: one compact resume cursor per workspace, normally the current Git branch or detached commit.

## Install

```bash
cargo install --path .
```

The crates.io package is `mem-cli`; the executable is `mem`. Rust 1.98 or newer is required.

## Commands

```bash
mem init
mem status

mem remember "Use release/acquire ordering for publication" \
  --kind decision --source-type git --source-ref abc123

mem context "publication handoff"
mem search "publication ordering"
mem search "how do readers observe a published value" --semantic

mem get <id-or-prefix>
mem correct <id-or-prefix> "Use release/acquire ordering on the publication path"
mem forget <id-or-prefix>

mem state show
mem state set --goal "qualify publication path" --checkpoint "loom is green"
mem state set --session session-123 --clear checkpoint
mem state clear

mem index
mem index --cached-only
```

`remember` and `state set` initialize the repo-local store automatically; `init` is the explicit setup command. `.mem/` is ignored by Git by default. `--db /path/to/file.db` and `MEM_DB` pin an exact database for tests or isolated profiles.

## Retrieval

FTS5 lexical search is synchronous and always available. `mem context` is the agent-facing retrieval path: when every active memory has a current embedding and the model is already cached, it uses semantic ranking; otherwise it falls back to lexical recall so incomplete derived state never hides canonical memories.

`mem search --semantic` is an explicit semantic-search tool. It requires complete current-model coverage and tells you to run `mem index` when coverage is incomplete. Semantic search uses exact cosine scoring; the expected local corpus is small enough that an ANN index would add complexity without demonstrated value.

Context is bounded by count and by memory-text bytes (`--max-bytes`, default 32768) to keep recalled context compact. Low-relevance semantic context is allowed to return no memories.

## Embeddings

Embeddings are rebuildable derived data. `mem index` directly selects active memories missing a vector for the current model, embeds a bounded batch, and commits each vector only if the memory is still active at the exact source version that was embedded. Failed work remains missing and a later `mem index` retries it naturally.

`mem index --cached-only` never downloads the model, which makes it safe for opportunistic agent hooks.

## Storage and safety

SQLite with bundled FTS5 is canonical. Normal storage is physically isolated per project; there is no automatic cross-project recall. Reads do not create missing stores. Created database files and SQLite sidecars are private to the current user on Unix.

`mem` is runtime-agnostic and requires no daemon. Task tracking, hosted sync, and transcript archives stay outside the memory core.

## Development

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release
```

`mem` is pre-1.0 and its interfaces may still change while the core is dogfooded.

## License

MIT
