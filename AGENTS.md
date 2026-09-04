# Agent instructions

`mem` is a standalone Rust memory CLI for agents. Keep the core independent of any one agent runtime.

## Product direction

`mem` is a small persistent project-memory layer. Its job is to preserve high-value project knowledge after it falls out of the model's current context window.

Target the simplest design that improves cross-session continuity:

- normal storage is repo-local at `.mem/mem.db` (or an explicitly initialized local project directory outside Git);
- `.mem/` is ignored by Git by default; users may copy/snapshot stores explicitly when needed;
- `--db` / `MEM_DB` remain the explicit exact-database escape hatch;
- do not depend on GitHub, hosted project systems, `ai/`, `agent-context/`, or any agent-runtime-specific storage location;
- do not add sync, import/export formats, daemons, generic storage backends, or hosted services without demonstrated need;
- project isolation is physical by default: one local database per project, no automatic cross-project recall;
- keep one agent-facing retrieval path. The agent should not need to choose between lexical, semantic, fast, deep, or history retrieval modes;
- retrieved context must stay small and high-signal. It is valid to return no memories when relevance is weak;
- persist promoted durable knowledge, not raw chain-of-thought or exhaustive session transcripts. Useful kinds include facts/findings, decisions, constraints, research notes, preferences, procedures, and compact checkpoints;
- retain provenance when available so an agent can understand where a memory came from;
- corrections are non-destructive: current knowledge must be distinguishable from superseded knowledge.

The current centralized managed-layout, global/user store, Git-remote project identity, episodic session archive, and worker-style indexing machinery are implementation history, not product requirements. Prefer removing them when the repo-local design makes them unnecessary rather than preserving them for compatibility at v0.0.x.

## Core implementation constraints

- The crates.io package is `mem-cli`; the user-facing executable is `mem`.
- Rust 1.98 is the pinned toolchain.
- SQLite via bundled `rusqlite` is the canonical store.
- FTS5 must remain usable without embeddings.
- Embeddings/vector indexes are derived and rebuildable; incomplete derived state must never hide canonical memories.
- Every multi-statement write transaction uses IMMEDIATE, never deferred: a deferred upgrade can fail at COMMIT with `SQLITE_BUSY_SNAPSHOT`, which the busy timeout does not retry.
- Keep retrieval/storage policy in the Rust core and keep the CLI runtime-agnostic.
- Prefer deletion and simplification over preserving abstractions that no longer serve the product.

## Development

Before committing code changes, run when available:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Prefer small vertical slices with tests over speculative abstractions. Retrieval changes should be justified by held-out/project-realistic evals, especially precision, supersession behavior, irrelevant-query abstention, and performance under a small context budget.
