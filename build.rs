use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Build identity for staleness checks: agents comparing an installed mem
    // against the repository can see which commit the binary was built from.
    // Falls back to "unknown" outside a Git checkout so builds stay hermetic.
    let identity = git_head()
        .or_else(env_commit)
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=MEM_BUILD_COMMIT={identity}");
    println!("cargo:rerun-if-env-changed=MEM_BUILD_COMMIT");
    // Git resolves shared refs correctly for both normal and linked worktrees.
    watch_git_path("HEAD");
    watch_git_path("packed-refs");
    if let Some(reference) = git_output(&["symbolic-ref", "--quiet", "HEAD"]) {
        watch_git_path(&reference);
    }
}

fn watch_git_path(name: &str) {
    if let Some(path) = git_output(&["rev-parse", "--git-path", name]) {
        // Watch the containing directory until an absent ref is created.
        // Watching a missing file directly would make Cargo rebuild every time.
        if let Some(existing) = Path::new(&path).ancestors().find(|path| path.exists()) {
            println!("cargo:rerun-if-changed={}", existing.display());
        }
    }
}

fn git_head() -> Option<String> {
    git_output(&["rev-parse", "--short=12", "HEAD"])
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn env_commit() -> Option<String> {
    let value = env::var("MEM_BUILD_COMMIT").ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}
