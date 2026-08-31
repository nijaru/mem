# mem for Pi

Thin Pi integration for the `mem` CLI.

The extension keeps memory semantics in the Rust CLI. Pi is responsible only for lifecycle integration:

- before an agent turn, call `mem context` and add the result to transient LLM context;
- after settled turns and around compaction/shutdown, project Pi message/tool evidence into a `mem` episode;
- keep the original Pi JSONL session authoritative;
- fail open if `mem` is unavailable.

Retrieved `mem-recall` context is not written back into the Pi session.

## Requirements

Build or install `mem` first. By default the extension expects a `mem` executable on `PATH`.

For development from this repository:

```sh
cargo build --release
MEM_BIN="$PWD/target/release/mem" \
  pi --no-extensions -e "$PWD/integrations/pi/extensions/mem.ts"
```

Pi also supports installing a local package directory:

```sh
pi install ./integrations/pi
```

## Environment

- `MEM_BIN`: executable path or command name. Defaults to `mem`.
- `MEM_DB`: optional SQLite database path passed to every adapter invocation. If unset, `mem` uses its normal default database.

`MEM_DB` is useful for isolated tests or separate agent profiles without wrapping the CLI.

## Local smoke test

From the repository root:

```sh
bash scripts/pi-adapter-smoke.sh
```

The script uses a temporary `mem` database and Pi session directory while retaining your normal Pi authentication/settings. It walks through recall, tool evidence, compaction, shutdown, and resume, then verifies that:

- resume reuses the same Pi episode;
- projected source-entry IDs do not duplicate;
- resumed work adds new episodic entries;
- transient `mem-recall` messages are absent from the persisted Pi JSONL.

Set `MEM_PI_SMOKE_KEEP=1` to retain the temporary database and Pi session files for inspection.

## Current boundaries

The adapter does not automatically promote conversation text into durable semantic memory, does not update continuation state, and does not run a persistent `mem` daemon. Those remain separate policy/optimization decisions rather than Pi-specific behavior.
