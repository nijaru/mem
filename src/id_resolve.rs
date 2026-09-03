//! Shared exact-ID-or-prefix resolution and FTS query construction.
//!
//! One implementation so memories and episodes cannot drift apart on the
//! same rules — the duplicated builders previously produced paired defects
//! (unescaped LIKE prefixes, divergent AND/OR joining).

use anyhow::{Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

/// Escape SQL LIKE wildcards (`_`, `%`, and the escape character itself)
/// so an ID-prefix candidate matches literally instead of any character.
fn escape_like_for_prefix(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for character in input.chars() {
        if matches!(character, '_' | '%' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}
/// Candidate IDs matching an exact ID or a prefix of one, exact matches
/// first, at most `limit` rows. Used by routed resolution, which must see
/// every store's matches to enforce cross-store uniqueness.
fn id_candidates(
    connection: &Connection,
    table: &str,
    id_or_prefix: &str,
    limit: usize,
) -> Result<Vec<String>> {
    let candidate = id_or_prefix.trim();
    let prefix = format!("{}%", escape_like_for_prefix(candidate));
    let sql = format!(
        "SELECT id FROM {table}\n\
         WHERE id = ?1 OR id LIKE ?2 ESCAPE '\\'\n\
         ORDER BY CASE WHEN id = ?1 THEN 0 ELSE 1 END, id\n\
         LIMIT ?3"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![candidate, prefix, limit as i64], |row| {
        row.get::<_, String>(0)
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Resolve an exact ID or an unambiguous prefix within one store. An exact
/// ID always wins, even when it is also a prefix of other IDs.
fn resolve_id(
    connection: &Connection,
    table: &str,
    kind: &str,
    id_or_prefix: &str,
) -> Result<String> {
    let candidate = id_or_prefix.trim();
    if candidate.is_empty() {
        bail!("{kind} ID cannot be empty");
    }
    if let Some(id) = connection
        .query_row(
            &format!("SELECT id FROM {table} WHERE id = ?1"),
            [candidate],
            |row| row.get::<_, String>(0),
        )
        .optional()?
    {
        return Ok(id);
    }
    let ids = id_candidates(connection, table, candidate, 2)?;
    match ids.as_slice() {
        [] => bail!("{kind} not found: {candidate}"),
        [id] => Ok(id.clone()),
        _ => bail!("ambiguous {kind} ID prefix: {candidate}"),
    }
}

/// Build an FTS5 query string joining each whitespace-separated term as a
/// quoted phrase with OR, so partial-term matches stay ranked-visible
/// instead of being AND-filtered out of existence.
pub(crate) fn fts_query(input: &str) -> Result<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        bail!("query cannot be empty");
    }
    Ok(terms.join(" OR "))
}

impl super::store::Store {
    /// Memory IDs matching an exact ID or prefix, for routed resolution.
    pub fn memory_id_candidates(&self, id_or_prefix: &str) -> Result<Vec<String>> {
        id_candidates(&self.connection, "memories", id_or_prefix, 3)
    }

    pub(crate) fn resolve_memory_id(&self, id_or_prefix: &str) -> Result<String> {
        resolve_id(&self.connection, "memories", "memory", id_or_prefix)
    }

    /// Episode IDs matching an exact ID or prefix, for routed resolution.
    pub fn episode_id_candidates(&self, id_or_prefix: &str) -> Result<Vec<String>> {
        id_candidates(&self.connection, "episodes", id_or_prefix, 3)
    }

    pub(crate) fn resolve_episode_id(&self, id_or_prefix: &str) -> Result<String> {
        resolve_id(&self.connection, "episodes", "episode", id_or_prefix)
    }
}
#[cfg(test)]
mod tests {
    use super::super::store::Store;

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mem-id-resolve-{}-{}.db",
            std::process::id(),
            line!()
        ))
    }

    #[test]
    fn wildcards_in_prefixes_match_literally_for_both_entities() {
        // A `_` or `%` in a candidate must not act as a SQL LIKE wildcard —
        // the defect class where episode resolution drifted from memory
        // resolution before this module unified them.
        let path = test_path();
        let mut store = Store::open(&path).expect("open store");
        store
            .remember(super::super::store::NewMemory {
                text: "literal wildcard memory".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("seed memory");
        store
            .ensure_episode(super::super::episode::NewEpisode {
                project_id: None,
                workspace_id: None,
                source_type: "test".to_owned(),
                source_ref: "wildcard-episode".to_owned(),
                started_at: None,
                metadata_json: None,
            })
            .expect("seed episode");

        for wildcard in ["_", "%", "\\".to_string().as_str()] {
            assert!(
                store.get(wildcard).is_err(),
                "memory wildcard {wildcard:?} must not resolve"
            );
            assert!(
                store.resolve_episode_id(wildcard).is_err(),
                "episode wildcard {wildcard:?} must not resolve"
            );
        }
        // Sanity: a real exact prefix still resolves for both entities.
        let memories = store.memory_id_candidates("").expect("memory candidates");
        assert_eq!(memories.len(), 1);
        let episodes = store.episode_id_candidates("").expect("episode candidates");
        assert_eq!(episodes.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exact_id_wins_over_ambiguity() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open store");
        let memory = store
            .remember(super::super::store::NewMemory {
                text: "exact id wins".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("seed memory");
        // The memory's own ID is trivially its exact match.
        assert!(store.resolve_memory_id(&memory.id).is_ok());
        let _ = std::fs::remove_file(&path);
    }
}
