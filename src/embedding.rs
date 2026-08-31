use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{OptionalExtension, TransactionBehavior, params};

use crate::store::Store;

impl Store {
    pub fn commit_embedding(
        &mut self,
        job_id: &str,
        generation: i64,
        lease_token: &str,
        model: &str,
        vector: &[f32],
    ) -> Result<bool> {
        validate_commit(job_id, generation, lease_token, model, vector)?;
        let normalized = normalize(vector)?;
        let encoded = encode_vector(&normalized);
        let dimensions = i64::try_from(normalized.len())?;
        let now = unix_millis()?;

        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let claim: Option<(String, String)> = transaction
            .query_row(
                "SELECT entity_type, entity_id\n\
                 FROM index_jobs\n\
                 WHERE id = ?1\n\
                   AND generation = ?2\n\
                   AND state = 'running'\n\
                   AND lease_token = ?3\n\
                   AND index_kind = 'embedding'",
                params![job_id, generation, lease_token],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let Some((entity_type, entity_id)) = claim else {
            return Ok(false);
        };

        transaction.execute(
            "INSERT INTO embeddings (\n\
                 entity_type, entity_id, model, dimensions, vector, source_generation, updated_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)\n\
             ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET\n\
                 dimensions = excluded.dimensions,\n\
                 vector = excluded.vector,\n\
                 source_generation = excluded.source_generation,\n\
                 updated_at = excluded.updated_at",
            params![
                entity_type,
                entity_id,
                model,
                dimensions,
                encoded,
                generation,
                now
            ],
        )?;
        let consumed = transaction.execute(
            "DELETE FROM index_jobs\n\
             WHERE id = ?1\n\
               AND generation = ?2\n\
               AND state = 'running'\n\
               AND lease_token = ?3",
            params![job_id, generation, lease_token],
        )?;
        if consumed != 1 {
            bail!("embedding lease changed while result was being committed");
        }
        transaction.commit()?;
        Ok(true)
    }
}

fn validate_commit(
    job_id: &str,
    generation: i64,
    lease_token: &str,
    model: &str,
    vector: &[f32],
) -> Result<()> {
    if job_id.trim().is_empty() {
        bail!("index job identifier cannot be empty");
    }
    if generation <= 0 {
        bail!("index job generation must be greater than zero");
    }
    if lease_token.trim().is_empty() {
        bail!("index job lease token cannot be empty");
    }
    if model.trim().is_empty() {
        bail!("embedding model identifier cannot be empty");
    }
    if vector.is_empty() {
        bail!("embedding vector cannot be empty");
    }
    if vector.len() > 65_536 {
        bail!("embedding vector is unreasonably large");
    }
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("embedding vector must contain only finite values");
    }
    Ok(())
}

fn normalize(vector: &[f32]) -> Result<Vec<f32>> {
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
    let mut encoded = Vec::with_capacity(vector.len() * size_of::<f32>());
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
    use std::time::Duration;

    use uuid::Uuid;

    use super::Store;
    use crate::episode::{NewEpisode, NewEpisodeEntry};
    use crate::store::NewMemory;

    #[test]
    fn embedding_commit_is_atomic_normalized_and_consumes_lease() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let memory = store
            .remember(NewMemory {
                text: "normalize this embedding".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
        let job = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim embedding job")
            .pop()
            .expect("claimed job");

        assert!(
            store
                .commit_embedding(
                    &job.id,
                    job.generation,
                    &job.lease_token,
                    "test-model",
                    &[3.0, 4.0],
                )
                .expect("commit embedding")
        );
        assert_eq!(job.entity_id, memory.id);
        let stats = store.index_job_stats().expect("read queue stats");
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.running, 0);

        let (dimensions, bytes, source_generation): (i64, Vec<u8>, i64) = store
            .connection
            .query_row(
                "SELECT dimensions, vector, source_generation\n\
                 FROM embeddings\n\
                 WHERE entity_type = 'memory' AND entity_id = ?1 AND model = 'test-model'",
                [&memory.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read stored embedding");
        assert_eq!(dimensions, 2);
        assert_eq!(source_generation, job.generation);
        let values = decode(&bytes);
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn stale_episode_claim_cannot_write_after_source_refresh() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let episode = store
            .ensure_episode(NewEpisode {
                project_id: None,
                workspace_id: None,
                source_type: "pi-session".to_owned(),
                source_ref: "session-1".to_owned(),
                started_at: Some(10),
                metadata_json: None,
            })
            .expect("create episode");
        let entry = store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: Some("assistant".to_owned()),
                    text: "first source text".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record entry");
        let stale = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim first generation")
            .pop()
            .expect("claimed job");

        store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: Some("assistant".to_owned()),
                    text: "refreshed source text".to_owned(),
                    occurred_at: Some(30),
                    metadata_json: None,
                },
            )
            .expect("refresh entry");

        assert!(
            !store
                .commit_embedding(
                    &stale.id,
                    stale.generation,
                    &stale.lease_token,
                    "test-model",
                    &[1.0, 0.0],
                )
                .expect("reject stale embedding")
        );
        let embedding_count: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE entity_id = ?1",
                [&entry.id],
                |row| row.get(0),
            )
            .expect("count embeddings");
        assert_eq!(embedding_count, 0);
        let queued = store
            .claim_index_jobs("worker-b", 1, Duration::from_secs(30))
            .expect("claim refreshed work")
            .pop()
            .expect("refreshed job");
        assert_eq!(queued.entity_id, entry.id);
        assert_ne!(queued.lease_token, stale.lease_token);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn source_refresh_invalidates_a_previously_committed_vector() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let episode = store
            .ensure_episode(NewEpisode {
                project_id: None,
                workspace_id: None,
                source_type: "pi-session".to_owned(),
                source_ref: "session-1".to_owned(),
                started_at: Some(10),
                metadata_json: None,
            })
            .expect("create episode");
        let entry = store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: None,
                    text: "before refresh".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record entry");
        let job = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim work")
            .pop()
            .expect("claimed work");
        assert!(
            store
                .commit_embedding(
                    &job.id,
                    job.generation,
                    &job.lease_token,
                    "test-model",
                    &[1.0, 1.0],
                )
                .expect("commit vector")
        );

        store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: None,
                    text: "after refresh".to_owned(),
                    occurred_at: Some(30),
                    metadata_json: None,
                },
            )
            .expect("refresh source");
        let embedding_count: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE entity_id = ?1",
                [&entry.id],
                |row| row.get(0),
            )
            .expect("count invalidated vectors");
        assert_eq!(embedding_count, 0);
        assert_eq!(store.index_job_stats().expect("read queue").pending, 1);

        drop(store);
        cleanup(&path);
    }

    fn decode(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-embedding-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
