# Agent instructions

`mem` is a standalone Rust memory CLI for agents. Keep the core independent of any one agent runtime.

## Current architectural constraints

- The crates.io package is `mem-cli`; the user-facing executable is `mem`.
- Rust 1.98 is the pinned toolchain.
- SQLite via bundled `rusqlite` is the canonical v0.x store.
- Semantic memory, episodic history, and continuation state remain distinct models.
- Provenance is required for promoted semantic memory.
- FTS5 must remain usable without embeddings.
- Embeddings and vector indexes are derived/rebuildable data and must not be correctness dependencies.
- Do not add a daemon, ANN index, automatic LLM extraction, sync protocol, or alternative database backend without a demonstrated need.
- Keep Pi or other agent integrations thin; retrieval/storage policy belongs in the Rust core.

## Development

Before committing code changes, run when available:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Prefer small vertical slices with tests over speculative abstractions. Do not introduce a generic storage-backend framework merely to preserve hypothetical OmenDB compatibility; isolate SQLite-specific code cleanly and refactor only when another backend is real.
