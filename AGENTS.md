# Agent instructions

`mem` is a standalone Rust memory CLI for agents. Keep the core independent of any one agent runtime.

## Product direction

`mem` is a small persistent project-memory layer. Preserve high-value project knowledge after it falls out of the model's current context window.

- Normal storage is repo-local at `.mem/mem.db` (or an explicitly selected `--db` / `MEM_DB` file).
- Project isolation is physical: one database per project and no automatic cross-project recall.
- Keep one agent-facing retrieval path: `mem context`.
- Retrieved context stays small and high-signal; returning no memory is valid when relevance is weak.
- Persist promoted durable knowledge, not raw chain-of-thought, exhaustive transcripts, task state, or duplicated documents.
- Retain lightweight provenance so recalled knowledge can be traced to its source.
- Corrections are non-destructive: current knowledge remains distinguishable from superseded knowledge.
- Add sync, daemons, generic backends, richer provenance graphs, or approximate vector indexes only after a concrete workflow or profile demonstrates the need.

## Core implementation constraints

- The crates.io package is `mem-cli`; the executable is `mem`.
- Rust 1.98 is pinned.
- Bundled SQLite via `rusqlite` is canonical; FTS5 must work without embeddings.
- Embeddings are derived and rebuildable. Incomplete derived state must never hide active canonical memories.
- Multi-statement writes use IMMEDIATE transactions so concurrent writers serialize before doing work.
- Keep retrieval/storage policy in Rust and the CLI runtime-agnostic.
- Prefer direct code and deletion over speculative abstractions or compatibility machinery while pre-1.0.

## Development

Before landing code changes:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release
```

Retrieval changes should be justified with held-out/project-realistic evals, especially precision, supersession behavior, irrelevant-query abstention, and small context budgets.
