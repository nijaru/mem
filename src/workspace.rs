use std::path::PathBuf;
use std::process::Command;

use anyhow::{Result, bail};

pub fn repo_root() -> Option<PathBuf> {
    git_output(&["rev-parse", "--show-toplevel"]).map(PathBuf::from)
}

pub fn workspace_id(override_id: Option<&str>) -> Result<String> {
    if let Some(value) = override_id {
        let value = value.trim();
        if value.is_empty() {
            bail!("workspace identifier cannot be empty");
        }
        return Ok(value.to_owned());
    }
    if let Some(branch) = git_output(&["symbolic-ref", "--quiet", "--short", "HEAD"]) {
        return Ok(format!("branch:{branch}"));
    }
    if let Some(commit) = git_output(&["rev-parse", "--short=12", "HEAD"]) {
        return Ok(format!("detached:{commit}"));
    }
    Ok("default".to_owned())
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

#[cfg(test)]
mod tests {
    use super::workspace_id;

    #[test]
    fn workspace_override_is_validated() {
        assert_eq!(
            workspace_id(Some("worktree:feature")).expect("workspace"),
            "worktree:feature"
        );
        assert!(workspace_id(Some("   ")).is_err());
    }
}
