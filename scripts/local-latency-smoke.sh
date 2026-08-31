#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo build --release
mem_bin="$repo_root/target/release/mem"

db="$(mktemp "${TMPDIR:-/tmp}/mem-local-smoke.XXXXXX.db")"
cleanup() {
  rm -f "$db" "$db-shm" "$db-wal"
}
trap cleanup EXIT

remember() {
  "$mem_bin" --db "$db" remember "$1" \
    --kind "$2" \
    --global \
    --source-type local-latency-smoke >/dev/null
}

remember \
  "Continuation state is isolated by project and workspace so separate branches and worktrees do not clobber each other's resume point." \
  decision
remember \
  "SQLite is the canonical local storage backend and FTS5 remains synchronously available." \
  decision
remember \
  "Embedding generation is derived work and must never block or invalidate a canonical memory write." \
  constraint
remember \
  "Use exact vector scanning first and add approximate nearest-neighbor indexing only after profiling demonstrates a need." \
  decision

printf '\n== index four memories ==\n'
time "$mem_bin" --db "$db" index run -n 4

query="how do multiple checkouts avoid overwriting the point an agent should resume from?"

printf '\n== lexical AND control (expected to miss or be weak) ==\n'
"$mem_bin" --db "$db" search "$query" --global -n 4 || true

for run in 1 2 3; do
  printf '\n== one-shot semantic query %s ==\n' "$run"
  time "$mem_bin" --db "$db" search "$query" --global --semantic -n 4
 done

printf '\nModel cache: %s\n' "${MEM_MODEL_CACHE:-${HF_HOME:-<mem default cache>}}"
printf 'DB was temporary and will be deleted: %s\n' "$db"
