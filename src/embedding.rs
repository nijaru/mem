use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::params;

use crate::store::Store;

impl Store {
    pub(crate) fn upsert_embedding_if_current(
        &self,
        memory_id: &str,
        source_updated_at: i64,
        model: &str,
        vector: &[f32],
    ) -> Result<bool> {
        if memory_id.trim().is_empty() {
            bail!("memory ID cannot be empty");
        }
        if model.trim().is_empty() {
            bail!("embedding model cannot be empty");
        }
        let normalized = normalize(vector)?;
        let encoded = encode_vector(&normalized);
        let dimensions = i64::try_from(normalized.len())?;
        let now = unix_millis()?;

        let changed = self.connection.execute(
            "INSERT INTO embeddings (\n\
                 memory_id, model, dimensions, vector, source_updated_at, updated_at\n\
             )\n\
             SELECT id, ?3, ?4, ?5, updated_at, ?6\n\
             FROM memories\n\
             WHERE id = ?1 AND status = 'active' AND updated_at = ?2\n\
             ON CONFLICT(memory_id, model) DO UPDATE SET\n\
                 dimensions = excluded.dimensions,\n\
                 vector = excluded.vector,\n\
                 source_updated_at = excluded.source_updated_at,\n\
                 updated_at = excluded.updated_at",
            params![
                memory_id,
                source_updated_at,
                model,
                dimensions,
                encoded,
                now
            ],
        )?;
        Ok(changed == 1)
    }
}

fn normalize(vector: &[f32]) -> Result<Vec<f32>> {
    if vector.is_empty() {
        bail!("embedding vector cannot be empty");
    }
    if vector.len() > 65_536 {
        bail!("embedding vector is unreasonably large");
    }
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("embedding vector must contain only finite values");
    }
    let squared_norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if !squared_norm.is_finite() || squared_norm <= f64::EPSILON {
        bail!("embedding vector must have non-zero finite magnitude");
    }
    let norm = squared_norm.sqrt();
    Ok(vector
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect())
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

fn unix_millis() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}

#[cfg(test)]
mod tests {
    use crate::store::{NewMemory, Store};

    #[test]
    fn stale_source_versions_cannot_overwrite_embeddings() {
        let path =
            std::env::temp_dir().join(format!("mem-embedding-test-{}.db", uuid::Uuid::now_v7()));
        let store = Store::open(&path).expect("open");
        let memory = store
            .remember(NewMemory {
                text: "first text".to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("remember");
        assert!(
            store
                .upsert_embedding_if_current(&memory.id, memory.updated_at, "model", &[3.0, 4.0])
                .expect("commit")
        );
        store
            .connection
            .execute(
                "UPDATE memories SET text = 'second text', updated_at = updated_at + 1 WHERE id = ?1",
                [&memory.id],
            )
            .expect("refresh");
        assert!(
            !store
                .upsert_embedding_if_current(&memory.id, memory.updated_at, "model", &[1.0, 0.0])
                .expect("reject stale")
        );
        let count: i64 = store
            .connection
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(path);
    }
}
