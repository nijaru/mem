#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v pi >/dev/null 2>&1; then
  echo "error: pi is not on PATH" >&2
  exit 1
fi
if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "error: sqlite3 is required for smoke-test assertions" >&2
  exit 1
fi

cargo build --release
mem_bin="$repo_root/target/release/mem"
extension="$repo_root/integrations/pi/extensions/mem.ts"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/mem-pi-smoke.XXXXXX")"
db="$tmp_root/memory.db"
session_dir="$tmp_root/sessions"
mkdir -p "$session_dir"

cleanup() {
  if [[ "${MEM_PI_SMOKE_KEEP:-0}" == "1" ]]; then
    printf '\nSmoke artifacts kept at: %s\n' "$tmp_root"
  else
    rm -rf "$tmp_root"
  fi
}
trap cleanup EXIT

"$mem_bin" --db "$db" remember \
  "The mem Pi adapter smoke sentinel is cobalt-otter-731." \
  --kind fact \
  --global \
  --source-type pi-smoke \
  --source-ref sentinel >/dev/null

export MEM_BIN="$mem_bin"
export MEM_DB="$db"

cat <<'EOF'

== Pi adapter smoke: first launch ==
This run uses a temporary mem DB and temporary Pi session directory.
Your normal Pi auth/settings are still used, but installed extensions are disabled for the run.

Inside Pi, perform these steps:
  1. Ask: What is the mem Pi smoke sentinel? Answer with only the value.
     Expected: cobalt-otter-731
  2. Ask: Run `pwd` with the bash tool, then tell me you ran it.
  3. Ask: Run this exact command with the bash tool, then reply only with "done":
       head -c 45000 /dev/zero | tr '\0' 'a'
  4. Ask: Run this exact command with the bash tool, then reply only with "done":
       head -c 33000 /dev/urandom | base64
     These two turns raise the session past Pi's compaction minimum
     (keepRecentTokens defaults to 20000), so the next step can compact.
  5. Run: /compact smoke test
     If Pi reports "Nothing to compact (session too small)", repeat step 4
     once and try again before treating this as a failure.
  6. Ask the sentinel question again. Expected: cobalt-otter-731
  7. Run: /quit
EOF

pi \
  --session-dir "$session_dir" \
  --no-extensions \
  -e "$extension" \
  --name mem-pi-smoke

first_episode_count="$(sqlite3 "$db" "SELECT COUNT(*) FROM episodes WHERE source_type = 'pi-session';")"
first_entry_count="$(sqlite3 "$db" "SELECT COUNT(*) FROM episode_entries;")"

printf '\nAfter first launch: episodes=%s entries=%s\n' "$first_episode_count" "$first_entry_count"
if [[ "$first_episode_count" != "1" ]]; then
  echo "error: expected exactly one Pi episode after first launch" >&2
  exit 1
fi
if [[ "$first_entry_count" -eq 0 ]]; then
  echo "error: expected Pi message/tool entries to be projected" >&2
  exit 1
fi

cat <<'EOF'

== Pi adapter smoke: resume ==
Pi will now continue the same isolated session.

Inside Pi:
  1. Run: /session
  2. Ask: What is the mem Pi smoke sentinel? Answer with only the value.
     Expected: cobalt-otter-731
  3. Ask: Say "resume smoke complete" and nothing else.
  4. Run: /quit
EOF

pi \
  -c \
  --session-dir "$session_dir" \
  --no-extensions \
  -e "$extension"

episode_count="$(sqlite3 "$db" "SELECT COUNT(*) FROM episodes WHERE source_type = 'pi-session';")"
entry_count="$(sqlite3 "$db" "SELECT COUNT(*) FROM episode_entries;")"
duplicate_refs="$(sqlite3 "$db" "SELECT COUNT(*) FROM (SELECT episode_id, source_ref, COUNT(*) AS n FROM episode_entries GROUP BY episode_id, source_ref HAVING n > 1);")"
source_refs="$(sqlite3 "$db" "SELECT source_ref FROM episodes WHERE source_type = 'pi-session' ORDER BY source_ref;")"

printf '\n== projection checks ==\n'
printf 'Pi episodes: %s\n' "$episode_count"
printf 'Entries after first launch: %s\n' "$first_entry_count"
printf 'Entries after resume: %s\n' "$entry_count"
printf 'Duplicate source refs: %s\n' "$duplicate_refs"
printf 'Episode source ref: %s\n' "$source_refs"

if [[ "$episode_count" != "1" ]]; then
  echo "error: resume created a second Pi episode instead of reusing the session episode" >&2
  exit 1
fi
if [[ "$entry_count" -le "$first_entry_count" ]]; then
  echo "error: resumed session did not add new projected entries" >&2
  exit 1
fi
if [[ "$duplicate_refs" != "0" ]]; then
  echo "error: duplicate Pi source entry IDs were persisted" >&2
  exit 1
fi

if grep -R -q '"customType"[[:space:]]*:[[:space:]]*"mem-recall"' "$session_dir"; then
  echo "error: transient mem-recall message was persisted into the Pi session" >&2
  exit 1
fi
printf 'Persisted mem-recall messages: 0\n'

printf '\n== episodic history sample ==\n'
"$mem_bin" --db "$db" history sentinel -n 20 || true

printf '\nPi adapter smoke checks passed.\n'
printf 'Temporary DB: %s\n' "$db"
printf 'Temporary Pi sessions: %s\n' "$session_dir"
printf 'Set MEM_PI_SMOKE_KEEP=1 before running to keep these artifacts for inspection.\n'
