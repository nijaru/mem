//! Managed storage layout: per-project SQLite routing with safe path
//! encoding. Exact-file operation (`--db`/`MEM_DB`) bypasses this entirely.
//! Slice A establishes the resolver and path-safety invariants; routed
//! operations consume it starting in Slice B, so some items are unused until
//! then.

#![allow(dead_code)]

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

pub const LAYOUT_VERSION_DIR: &str = "layout-v1";
const LEGACY_DB_FILENAME: &str = "memory.db";
const USER_DB_FILENAME: &str = "user.db";
const PROJECT_DB_FILENAME: &str = "mem.db";

/// Managed layout root: `<MEM_HOME>` or `<local-data>/mem`. The legacy
/// combined database sits directly under the root; the active split layout
/// lives in a versioned subdirectory.
#[derive(Debug, Clone, Serialize)]
pub struct ManagedLayout {
    root: PathBuf,
}

impl ManagedLayout {
    pub fn resolve() -> Result<Self> {
        if let Some(home) = std::env::var_os("MEM_HOME").filter(|home| !home.is_empty()) {
            return Ok(Self::at(PathBuf::from(home)));
        }
        let data_dir =
            dirs::data_local_dir().context("could not determine the local data directory")?;
        Ok(Self::at(data_dir.join("mem")))
    }

    pub fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn legacy_db(&self) -> PathBuf {
        self.root.join(LEGACY_DB_FILENAME)
    }

    pub fn layout_dir(&self) -> PathBuf {
        self.root.join(LAYOUT_VERSION_DIR)
    }

    pub fn user_db(&self) -> PathBuf {
        self.layout_dir().join(USER_DB_FILENAME)
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.layout_dir().join("projects")
    }

    /// Resolve one project's database inside the managed layout. The encoded
    /// path can never traverse outside the projects directory by construction;
    /// containment is verified again after joining as a defensive invariant.
    pub fn project_db(&self, project_id: &str) -> Result<ProjectDb> {
        if project_id.trim().is_empty() {
            bail!("project identifier cannot be empty");
        }
        let projects = self.projects_dir();
        let mut path = projects.clone();
        for component in project_id.split('/') {
            path = path.join(encode_component(component));
        }
        path = path.join(PROJECT_DB_FILENAME);
        verify_contained(&projects, &path)?;
        Ok(ProjectDb {
            path,
            project_id: project_id.to_owned(),
        })
    }

    /// Recover the logical project identity from an existing managed project
    /// database directory. Only canonical encodings are accepted.
    pub fn decode_project_dir(&self, dir: &Path) -> Result<String> {
        let relative = dir.strip_prefix(self.projects_dir()).with_context(|| {
            format!(
                "{} is outside the managed projects directory",
                dir.display()
            )
        })?;
        let mut components = Vec::new();
        for component in relative.components() {
            let Component::Normal(part) = component else {
                bail!(
                    "{} is not a canonical managed project directory",
                    dir.display()
                );
            };
            let Some(encoded) = part.to_str() else {
                bail!(
                    "{} is not a canonical managed project directory",
                    dir.display()
                );
            };
            components.push(decode_component(encoded)?);
        }
        if components.is_empty() {
            bail!("{} is not a managed project directory", dir.display());
        }
        Ok(components.join("/"))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectDb {
    pub path: PathBuf,
    pub project_id: String,
}

/// Safe-set encoding for one filesystem component. Bytes outside
/// `[A-Za-z0-9_]` — including `-`, `.`, native separators, and all non-ASCII
/// bytes — are escaped as `-XX` (uppercase hex). The `-` escape character is
/// itself always escaped, so a lone `-` is unambiguously reserved for the
/// empty component.
pub fn encode_component(component: &str) -> String {
    if component.is_empty() {
        return "-".to_owned();
    }
    let mut encoded = String::with_capacity(component.len());
    for byte in component.bytes() {
        if is_safe_byte(byte) {
            encoded.push(byte as char);
        } else {
            encoded.push('-');
            encoded.push(hex_digit(byte >> 4));
            encoded.push(hex_digit(byte & 0x0F));
        }
    }
    encoded
}

/// Inverse of [`encode_component`]. Accepts only canonical encodings: every
/// `-` must be followed by two uppercase hex digits, except the single `-`
/// that marks an empty component.
pub fn decode_component(encoded: &str) -> Result<String> {
    if encoded == "-" {
        return Ok(String::new());
    }
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if is_safe_byte(byte) {
            decoded.push(byte);
            index += 1;
            continue;
        }
        if byte != b'-' || index + 2 >= bytes.len() {
            bail!("invalid encoded path component: {encoded}");
        }
        let high = hex_value(bytes[index + 1]);
        let low = hex_value(bytes[index + 2]);
        let (Some(high), Some(low)) = (high, low) else {
            bail!("invalid encoded path component: {encoded}");
        };
        decoded.push(high << 4 | low);
        index += 3;
    }
    String::from_utf8(decoded)
        .with_context(|| format!("encoded component {encoded} is not valid UTF-8"))
}

fn is_safe_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn hex_digit(nibble: u8) -> char {
    const HEX: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    ];
    HEX[nibble as usize]
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Defensive containment invariant: every resolved managed path must be a
/// strict descendant of the managed projects directory built only from normal
/// components. Encoding already guarantees this; verification catches
/// regressions.
fn verify_contained(projects_dir: &Path, path: &Path) -> Result<()> {
    let relative = path
        .strip_prefix(projects_dir)
        .with_context(|| format!("{} escapes the managed projects directory", path.display()))?;
    if relative.components().next().is_none() {
        bail!(
            "{} is the managed projects directory itself",
            path.display()
        );
    }
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("{} escapes the managed projects directory", path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ManagedLayout, decode_component, encode_component};

    #[test]
    fn encoding_covers_hostile_and_unusual_inputs() {
        assert_eq!(encode_component("github.com"), "github-2Ecom");
        assert_eq!(encode_component("."), "-2E");
        assert_eq!(encode_component(".."), "-2E-2E");
        assert_eq!(
            encode_component("nijaru"),
            "nijaru",
            "safe bytes pass through unchanged"
        );
        // Escape character and separators are always escaped.
        assert_eq!(encode_component("-"), "-2D");
        assert_eq!(encode_component("_"), "_");
        assert_eq!(encode_component("a\\..\\b"), "a-5C-2E-2E-5Cb");
        assert_eq!(encode_component("C:\\Users"), "C-3A-5CUsers");
        assert_eq!(encode_component("café"), "caf-C3-A9");
        // Empty component has an explicit, unambiguous encoding.
        assert_eq!(encode_component(""), "-");
    }

    #[test]
    fn encoding_is_injective_on_reserved_sequences() {
        // A lone "-" can only be the empty component; a real dash or NUL
        // byte always carries its hex payload.
        assert_ne!(encode_component(""), encode_component("-"));
        assert_ne!(encode_component(""), encode_component("\0"));
        assert_eq!(encode_component("\0"), "-00");
        // Long components stay linear and deterministic.
        let long = "x".repeat(300);
        assert_eq!(encode_component(&long), long);
        let hostile = "\u{1F600}".repeat(100);
        assert_eq!(
            encode_component(&hostile).matches('-').count(),
            hostile.chars().count() * 4 // 4 bytes per emoji, each escaped
        );
    }

    #[test]
    fn decoding_round_trips_and_rejects_non_canonical_forms() {
        for value in [
            "github.com",
            ".",
            "..",
            "-",
            "",
            "a\\..\\b",
            "C:\\Users",
            "café",
            "\0",
            "a-5Cb",
        ] {
            let encoded = encode_component(value);
            assert_eq!(decode_component(&encoded).expect("decode"), value);
        }
        // Lowercase hex, truncated escapes, and raw unsafe bytes are not
        // canonical encodings.
        assert!(decode_component("a-2eb").is_err());
        assert!(decode_component("a-2").is_err());
        assert!(decode_component("a.b").is_err());
        assert!(decode_component("a-2E-").is_err());
    }

    #[test]
    fn project_paths_stay_contained_for_hostile_ids() {
        let layout = ManagedLayout::at(std::path::PathBuf::from("/data/mem"));
        let projects = layout.projects_dir();

        for hostile in [
            "../../outside",
            "/absolute",
            "a//b",
            "a\\..\\b",
            "C:\\Users\\nick",
            "..",
            ".",
            "café/ünïcode",
        ] {
            let project_db = layout
                .project_db(hostile)
                .expect("hostile id must still resolve");
            assert!(
                project_db.path.starts_with(&projects),
                "{} escaped the projects directory",
                project_db.path.display()
            );
            // No component can be a traversal or separator: encoding maps
            // every such byte to an escape.
            let relative = project_db.path.strip_prefix(&projects).expect("prefix");
            for component in relative.components() {
                assert!(
                    matches!(component, std::path::Component::Normal(_)),
                    "non-normal component in {}",
                    project_db.path.display()
                );
            }
            // The logical identity survives the round trip.
            let dir = project_db
                .path
                .parent()
                .expect("project db has a parent directory");
            assert_eq!(layout.decode_project_dir(dir).expect("decode"), hostile);
        }

        assert!(layout.project_db("  ").is_err());
    }

    #[test]
    fn layout_paths_have_the_reviewed_shape() {
        let layout = ManagedLayout::at(std::path::PathBuf::from("/data/mem"));
        assert_eq!(
            layout.legacy_db(),
            std::path::PathBuf::from("/data/mem/memory.db")
        );
        assert_eq!(
            layout.user_db(),
            std::path::PathBuf::from("/data/mem/layout-v1/user.db")
        );
        let project = layout
            .project_db("github.com/nijaru/mem")
            .expect("resolve project");
        assert_eq!(
            project.path,
            std::path::PathBuf::from("/data/mem/layout-v1/projects/github-2Ecom/nijaru/mem/mem.db")
        );
    }
}
