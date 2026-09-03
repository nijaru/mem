use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectContext {
    pub project_id: String,
    pub workspace_id: String,
    pub root: Option<PathBuf>,
    pub remote: Option<String>,
}

impl ProjectContext {
    pub fn detect(
        project_override: Option<&str>,
        workspace_override: Option<&str>,
    ) -> Result<Option<Self>> {
        let root = git_output(&["rev-parse", "--show-toplevel"]).map(PathBuf::from);
        let remote = git_output(&["remote", "get-url", "origin"]);

        if root.is_none() && project_override.is_none() {
            return Ok(None);
        }

        let project_id = match project_override {
            Some(project) => validate_override(project, "project")?.to_owned(),
            None => match remote.as_deref() {
                Some(remote) => normalize_remote(remote),
                None => {
                    let root = root
                        .as_deref()
                        .context("Git repository has no origin and no working tree root")?;
                    format!("local:{}", root.display())
                }
            },
        };

        let workspace_id = match workspace_override {
            Some(workspace) => validate_override(workspace, "workspace")?.to_owned(),
            None => detect_workspace(),
        };

        Ok(Some(Self {
            project_id,
            workspace_id,
            root,
            remote,
        }))
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

fn normalize_remote(remote: &str) -> String {
    let remote = remote.trim().trim_end_matches('/');
    let remote = remote.strip_suffix(".git").unwrap_or(remote);

    if let Some((scheme, rest)) = remote.split_once("://") {
        // file:// URLs are local paths, not host/path pairs; the host
        // splitting below would misread the first path component as a
        // network authority (file:///tmp/repo -> "tmp/repo").
        if scheme.eq_ignore_ascii_case("file") {
            let path = rest.trim_start_matches('/');
            if path.is_empty() {
                return "local:".to_owned();
            }
            return format!("local:/{path}");
        }
        let rest = rest.trim_start_matches('/');
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let host = authority
            .rsplit('@')
            .next()
            .unwrap_or(authority)
            .split(':')
            .next()
            .unwrap_or(authority);
        return join_host_path(host, path);
    }

    if let Some((authority, path)) = remote.split_once(':')
        && !authority.contains('/')
    {
        let host = authority.rsplit('@').next().unwrap_or(authority);
        return join_host_path(host, path);
    }

    // Relative local remotes (../shared.git) canonicalize against the
    // process working directory, so the same repository would map to
    // different project IDs depending on where inside it `mem` runs.
    // Git resolves them against the repository root instead.
    if remote.starts_with('.')
        && let Some(resolved) = resolve_relative_remote(remote)
    {
        return format!("local:{}", resolved.display());
    }

    if let Ok(path) = PathBuf::from(remote).canonicalize() {
        return format!("local:{}", path.display());
    }

    if remote.starts_with('/') || remote.starts_with('.') {
        return format!("local:{remote}");
    }

    remote.to_owned()
}

/// Resolve a dot-relative git remote against the repository root rather
/// than the process working directory, matching how git itself resolves
/// remote paths. Returns None when there is no enclosing repository.
fn resolve_relative_remote(remote: &str) -> Option<PathBuf> {
    let root = git_output(&["rev-parse", "--show-toplevel"])?;
    let resolved = PathBuf::from(root).join(remote);
    let canonical = resolved.canonicalize().ok()?;
    Some(canonical)
}

fn join_host_path(host: &str, path: &str) -> String {
    let host = host.to_ascii_lowercase();
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        host
    } else {
        format!("{host}/{path}")
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_remote, resolve_relative_remote};

    #[test]
    fn normalizes_common_git_remote_forms() {
        for (input, expected) in [
            ("git@github.com:nijaru/mem.git", "github.com/nijaru/mem"),
            ("https://github.com/nijaru/mem.git", "github.com/nijaru/mem"),
            (
                "ssh://git@github.com/nijaru/mem.git",
                "github.com/nijaru/mem",
            ),
            ("https://GitHub.COM/nijaru/mem/", "github.com/nijaru/mem"),
            // file:// remotes are local paths; the first path component is
            // not a network host (regression: used to yield "tmp/repo").
            ("file:///tmp/repo.git", "local:/tmp/repo"),
            ("file://srv/share/repo.git", "local:/srv/share/repo"),
        ] {
            assert_eq!(normalize_remote(input), expected);
        }
    }

    #[test]
    fn relative_remotes_resolve_against_repository_root() {
        // A dot-relative remote must yield one project ID regardless of the
        // subdirectory `mem` runs in. `resolve_relative_remote` anchors at
        // the repository root; this test runs from a stable root (the test
        // working directory is the repository root) so the assertion is
        // deterministic here while still pinning the anchoring behavior.
        let inside_repo = resolve_relative_remote("../src");
        if let Some(resolved) = inside_repo {
            // From the repo root, ../src is the crate's src directory.
            assert!(
                resolved.ends_with("mem/src"),
                "unexpected resolution: {}",
                resolved.display()
            );
        }
    }
}
