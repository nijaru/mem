use anyhow::{Result, bail};
use rusqlite::{OptionalExtension, params};

fn escape_like_prefix(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for character in input.chars() {
        if matches!(character, '_' | '%' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

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

impl crate::store::Store {
    pub(crate) fn resolve_memory_id(&self, id_or_prefix: &str) -> Result<String> {
        let candidate = id_or_prefix.trim();
        if candidate.is_empty() {
            bail!("memory ID cannot be empty");
        }
        if let Some(id) = self
            .connection
            .query_row(
                "SELECT id FROM memories WHERE id = ?1",
                [candidate],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            return Ok(id);
        }

        let prefix = format!("{}%", escape_like_prefix(candidate));
        let mut statement = self.connection.prepare(
            "SELECT id FROM memories\n\
             WHERE id LIKE ?1 ESCAPE '\\'\n\
             ORDER BY id\n\
             LIMIT 2",
        )?;
        let ids = statement
            .query_map(params![prefix], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        match ids.as_slice() {
            [] => bail!("memory not found: {candidate}"),
            [id] => Ok(id.clone()),
            _ => bail!("ambiguous memory ID prefix: {candidate}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::store::{NewMemory, Store};

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-id-test-{}.db", uuid::Uuid::now_v7()))
    }

    fn remember(store: &Store) -> crate::store::Memory {
        store
            .remember(NewMemory {
                text: "literal wildcard memory".to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("remember")
    }

    #[test]
    fn wildcard_characters_are_literal_in_prefixes() {
        let path = test_path();
        let store = Store::open(&path).expect("open");
        let memory = remember(&store);
        for wildcard in ["_", "%", "\\"] {
            assert!(store.get(wildcard).is_err());
        }
        assert_eq!(
            store.get(&memory.id[..8]).expect("resolve prefix").id,
            memory.id
        );
        let _ = std::fs::remove_file(path);
    }
}
