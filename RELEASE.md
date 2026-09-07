# mem release runbook (first publish — not yet executed)

Status: **prep only**. The qualification window is open; do not publish until the
user closes it. Nothing in this file has been run against the real registry.

## Preconditions (all verified 2026-09-05; re-verified 2026-09-07)

- [x] Gates green on HEAD: fmt, `cargo test --all-targets`, clippy `-D warnings`, release build.
- [x] `cargo publish --dry-run` packages and verifies cleanly (re-verified on 506408c, still 19 files / 178 KiB).
- [x] `mem-cli` and `mem` names both free (sparse index 404 on both).
- [x] Package excludes: `.github`, `.tasks`, `ai`, `scripts`, `tests` (verified in tarball list); `AGENTS.md` ships in the tarball (acceptable — repo instructions), consider whether to also exclude it at release time.
- [x] `Cargo.lock` tracked (binary-crate convention); `rust-toolchain.toml` ships in the tarball but only pins clones — `cargo install` from the registry uses the installer's own toolchain, with `rust-version = 1.98` as the floor cargo enforces.
- [x] Installed binary matches HEAD (`mem --version` = 506408cfaa4c at time of writing).
- [x] No git tags exist yet — `v0.0.1` will be the first.
- [x] Qualification evidence 2026-09-07: skill `NOTES.md` field log is empty after real dogfood — 9 project stores live, 63 memories, 5 corrections exercised, zero filed friction.

## Open questions for the user at release time

1. **Resolved 2026-09-07:** version is `0.0.1` — unreleased, no reason to jump to 0.x.0. `Cargo.toml` already carries it; no version-bump commit is needed.
2. **Resolved 2026-09-07:** keep `rust-version = 1.98`. It matches the verified dev toolchain; any lower claim would be untested. Edition 2024 (≥1.85) and let-chains (≥1.88) put the honest floor close by anyway, and lowering MSRV later is non-breaking while raising it is breaking — so start at the verified high-water mark and relax only when a real consumer needs it.
3. **Resolved 2026-09-07:** keep the "pre-1.0 interfaces may change" stance through 0.0.1 — no soft freeze, no README wording change.
4. Homebrew tap formula (`~/github/nijaru/homebrew-tap`) — add at release or after some soak time on crates.io?

## Steps (exact, in order)

1. Confirm the qualification window is closed: user says so, no open dogfood
   defects, and the `.tasks/mem-r8l2.json` task-log churn is either committed
   or explicitly left out of the release commit. (2026-09-07: the r8l2 task is
   `done`; only its log entries keep moving — commit as chore or leave out.)
2. Version is `0.0.1` (resolved above — no bump commit needed). The release
   commit is just the tag (open question 3 kept the README as-is):
   - tag `v0.0.1` (annotated, message = one-line summary of what v0.0.1 is).
3. Full gates on the tag commit, then:
   `cargo publish --dry-run` (must be clean at `0.0.1`).
4. Reinstall locally from source, confirm `mem --version` reports `0.0.1 (built from <head>)`.
5. `cargo publish` (real). Requires a login token in `~/.cargo/credentials` —
   check `cargo login --help` state first; the user must provide the token if absent.
   Publishing is irreversible: name + version are permanently claimed.
6. Post-publish checks:
   - `cargo install mem-cli` from the public index into a scratch `CARGO_HOME`
     (not the real one) and smoke-test `mem init/status/remember/context/export` in a
     temp repo — this catches what dry-run cannot (registry-side rendering).
   - Verify https://crates.io/crates/mem-cli renders, docs.rs builds.
7. Tag is already pushed; push the version-bump commit if not already.
8. Homebrew formula (if decided in step "open questions"): follow
   `~/github/nijaru/homebrew-tap/ADDING_FORMULA.md`; the tarball checksum
   comes from `cargo publish` output or the crates.io download URL.
9. If anything fails after the crate is live: 0.0.x/0.1.x can be yanked only if
   broken (`cargo yank`), and a bumped version must be published to fix.
   Do not delete tags already public.

## Known caveats carried into release

- The embedded-model blobs (66 MB ONNX) are **not** in the tarball; first
  `mem index` run downloads them on demand. This is intended, but means the
  tarball does not fully exercise semantic search — the smoke test in step 6
  uses lexical-only paths; `--cached-only` must report the model as uncached.
- `status`/`--version` report the source commit they were built from; binaries
  installed from crates.io will report a crates.io build environment's commit
  (`unknown` fallback) — expected, documented in README.
