use std::path::PathBuf;
use std::process::Command;

use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectContext {
    pub workspace_id: String,
    pub root: Option<PathBuf>,
}

impl ProjectContext {
    pub fn detect(workspace_override: Option<&str>) -> Result<Self> {
        let root = git_output(&["rev-parse", "--show-toplevel"]).map(PathBuf::from);
        let workspace_id = match workspace_override {
            Some(workspace) => validate_override(workspace, "workspace")?.to_owned(),
            None => detect_workspace(),
        };
        Ok(Self { workspace_id, root })
    }
}

fn detect_workspace() -> String {
    if let Some(branch) = git_output(&["symbolic-ref", "--quiet", "--short", "HEAD"]) {
        return format!("branch:{branch}");
    }
    if let Some(commit) = git_output(&["rev-parse", "--short=12", "HEAD"]) {
        return format!("detached:{commit}");
    }
    "default".to_owned()
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

fn validate_override<'a>(value: &'a str, kind: &str) -> Result<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{kind} identifier cannot be empty");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::ProjectContext;

    #[test]
    fn workspace_override_is_validated_without_project_identity() {
        let context = ProjectContext::detect(Some("worktree:feature")).expect("context");
        assert_eq!(context.workspace_id, "worktree:feature");
        assert!(ProjectContext::detect(Some("   ")).is_err());
    }
}
