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
        let git_dir = Path::new(&git_dir);
        let head_file = git_dir.join("HEAD");
        if head_file.is_file() {
            println!("cargo:rerun-if-changed={}", head_file.display());
            // On a branch HEAD is a symref whose content never changes as
            // commits land, so also watch the ref it points at. Watch
            // packed-refs too: the branch ref may be packed rather than a
            // loose file. Detached HEAD needs no extra watch: its content
            // is the sha itself and changes on every checkout.
            if let Ok(head) = std::fs::read_to_string(&head_file)
                && let Some(ref_path) = head.strip_prefix("ref:").map(str::trim)
            {
                let target = git_dir.join(ref_path);
                if target.is_file() {
                    println!("cargo:rerun-if-changed={}", target.display());
                }
            }
            let packed = git_dir.join("packed-refs");
            if packed.is_file() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
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
