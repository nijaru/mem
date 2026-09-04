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
    // Re-run only when the HEAD ref moves, not on every rebuild.
    let git_dir = Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    if let Some(git_dir) = git_dir {
        let head_file = Path::new(&git_dir).join("HEAD");
        if head_file.is_file() {
            println!("cargo:rerun-if-changed={}", head_file.display());
        }
    }
}

fn git_head() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
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
