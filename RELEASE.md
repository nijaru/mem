# mem release runbook (first publish — not yet executed)

Status: **prep only**. The qualification window is open; do not publish until the
user closes it. Nothing in this file has been run against the real registry.

## Preconditions (all verified 2026-09-05)

- [x] Gates green on HEAD: fmt, `cargo test --all-targets`, clippy `-D warnings`, release build.
- [x] `cargo publish --dry-run` packages and verifies cleanly: 19 files, 178 KiB.
- [x] `mem-cli` and `mem` names both free (sparse index 404 on both).
- [x] Package excludes: `.github`, `.tasks`, `ai`, `scripts`, `tests` (verified in tarball list); `AGENTS.md` ships in the tarball (acceptable — repo instructions), consider whether to also exclude it at release time.
- [x] `Cargo.lock` tracked (binary-crate convention); `rust-toolchain.toml` ships (pins 1.98 for installers).
- [x] Installed binary matches HEAD (`mem --version` = 797d6b01653e at time of writing).
- [x] No git tags exist yet — `v0.1.0` will be the first.

## Open questions for the user at release time

1. Version: `0.1.0` is the conventional first-stable. Package stays `mem-cli`, binary stays `mem`.
2. `rust-version = 1.98` — pinned toolchain also ships in the tarball. Fine, or relax to MSRV wording?
3. Docs say "pre-1.0 interfaces may change" — keep that stance through 0.1.0, or treat 0.1.0 as the soft freeze?
4. Homebrew tap formula (`~/github/nijaru/homebrew-tap`) — add at release or after some soak time on crates.io?

## Steps (exact, in order)

1. Confirm the qualification window is closed: user says so, no open dogfood
   defects, `.tasks/mem-r8l2.json` (currently staged, owned by another session)
   is either merged or explicitly left out of the release.
2. Pick the version, then in one commit:
   - `Cargo.toml`: `version = "0.1.0"`;
   - `README.md`: drop "pre-1.0" wording per decision above;
   - tag `v0.1.0` (annotated, message = one-line summary of what v0.1.0 is).
3. Full gates on the bumped commit, then:
   `cargo publish --dry-run` (must be clean at the new version).
4. Reinstall locally from source, confirm `mem --version` reports `0.1.0 (built from <head>)`.
5. `cargo publish` (real). Requires a login token in `~/.cargo/credentials` —
   check `cargo login --help` state first; the user must provide the token if absent.
   Publishing is irreversible: name + version are permanently claimed.
6. Post-publish checks:
   - `cargo install mem-cli` from the public index into a scratch `CARGO_HOME`
     (not the real one) and smoke-test `mem init/status/remember/context` in a
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
