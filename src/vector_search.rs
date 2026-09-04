use std::mem::size_of;

use anyhow::{Result, bail};
use rusqlite::Row;
use serde::Serialize;

use crate::store::{Memory, Store};

#[derive(Debug, Clone, Serialize)]
pub struct SemanticSearchHit {
    pub memory: Memory,
    pub score: f64,
}

struct VectorCandidate {
    memory: Memory,
    dimensions: i64,
    vector: Vec<u8>,
}

impl Store {
    /// Whether every active memory in this local project store has a current-model vector.
    pub fn has_complete_coverage(&self, model: &str) -> Result<bool> {
        if model.trim().is_empty() {
            bail!("embedding model identifier cannot be empty");
        }
        let covered: i64 = self.connection.query_row(
            "SELECT COUNT(*)

             FROM memories AS m

             WHERE m.status = 'active'

 AND EXISTS (

     SELECT 1 FROM embeddings AS e

     WHERE e.entity_type = 'memory'

       AND e.entity_id = m.id

       AND e.model = ?1

 )",
            [model],
            |row| row.get(0),
        )?;
        let visible: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'",
            [],
            |row| row.get(0),
        )?;
        Ok(covered == visible)
    }

    pub fn semantic_search_by_vector(
        &self,
        query_vector: &[f32],
        model: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>> {
        if model.trim().is_empty() {
            bail!("embedding model identifier cannot be empty");
        }
        if limit == 0 {
            bail!("semantic search limit must be greater than zero");
        }
        let query = normalize_query(query_vector)?;
        let mut statement = self.connection.prepare(
            "SELECT m.id, m.scope, m.project_id, m.kind, m.text, m.actor, m.status,
\
      m.created_at, m.updated_at, m.deleted_at, e.dimensions, e.vector
\
             FROM embeddings AS e
\
             JOIN memories AS m ON m.id = e.entity_id
\
             WHERE e.entity_type = 'memory'
\
 AND e.model = ?1
\
 AND m.status = 'active'",
        )?;
        let rows = statement.query_map([model], vector_candidate_from_row)?;
        let mut candidates = Vec::new();
        for row in rows {
            candidates.push(row?);
        }

        let mut hits = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let dimensions = usize::try_from(candidate.dimensions)?;
            if dimensions != query.len() {
                bail!(
                    "embedding dimension mismatch for {}: stored {}, query {}",
                    candidate.memory.id,
                    dimensions,
                    query.len()
                );
            }
            let vector = decode_vector(&candidate.vector, dimensions, &candidate.memory.id)?;
            let score = query
                .iter()
                .zip(vector.iter())
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum();
            hits.push(SemanticSearchHit {
                memory: candidate.memory,
                score,
            });
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.memory.updated_at.cmp(&left.memory.updated_at))
                .then_with(|| left.memory.id.cmp(&right.memory.id))
        });
        hits.truncate(limit);
        Ok(hits)
    }
}

fn vector_candidate_from_row(row: &Row<'_>) -> rusqlite::Result<VectorCandidate> {
    Ok(VectorCandidate {
        memory: Memory {
            id: row.get(0)?,
            scope: row.get(1)?,
            project_id: row.get(2)?,
            kind: row.get(3)?,
            text: row.get(4)?,
            actor: row.get(5)?,
            status: row.get(6)?,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
            deleted_at: row.get(9)?,
        },
        dimensions: row.get(10)?,
        vector: row.get(11)?,
    })
}

fn normalize_query(vector: &[f32]) -> Result<Vec<f32>> {
    if vector.is_empty() {
        bail!("semantic query vector cannot be empty");
    }
    if vector.len() > 65_536 {
        bail!("semantic query vector is unreasonably large");
    }
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("semantic query vector must contain only finite values");
    }

    let squared_norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if !squared_norm.is_finite() || squared_norm <= f64::EPSILON {
        bail!("semantic query vector must have non-zero finite magnitude");
    }
    let norm = squared_norm.sqrt();
    Ok(vector
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect())
}

fn decode_vector(bytes: &[u8], dimensions: usize, entity_id: &str) -> Result<Vec<f32>> {
    let expected = dimensions
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("embedding dimensions overflow for {entity_id}"))?;
    if bytes.len() != expected {
        bail!(
            "invalid embedding byte length for {entity_id}: expected {expected}, got {}",
            bytes.len()
        );
    }
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        bail!("invalid embedding byte alignment for {entity_id}");
    }
    let vector: Vec<f32> = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("stored embedding contains non-finite values for {entity_id}");
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::Store;
    use crate::store::NewMemory;

    #[test]
    fn exact_scan_scores_all_active_memories() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let first = remember(&mut store, "first candidate", None);
        let best = remember(&mut store, "best candidate", Some("legacy-alpha"));
        let third = remember(&mut store, "third candidate", Some("legacy-beta"));
        let jobs = store
            .claim_index_jobs("worker", 3, Duration::from_secs(30))
            .expect("claim embedding work");
        for job in jobs {
            let vector = if job.entity_id == best.id {
                [1.0, 0.0]
            } else if job.entity_id == first.id {
                [0.8, 0.2]
            } else {
                assert_eq!(job.entity_id, third.id);
                [0.5, 0.5]
            };
            seed_vector(
                &store,
                &job.entity_id,
                "test-model",
                &vector,
                job.generation,
            );
            store
                .connection
                .execute(
                    "DELETE FROM index_jobs WHERE id = ?1",
                    rusqlite::params![job.id],
                )
                .expect("consume seeded job");
        }
        let hits = store
            .semantic_search_by_vector(&[2.0, 0.0], "test-model", 10)
            .expect("search local store");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].memory.id, best.id);
        assert!(hits.iter().any(|hit| hit.memory.id == first.id));
        assert!(hits.iter().any(|hit| hit.memory.id == third.id));
        drop(store);
        cleanup(&path);
    }

    #[test]
    fn coverage_gate_tracks_all_active_memories() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let first = remember(&mut store, "first candidate", None);
        let second = remember(&mut store, "second candidate", Some("legacy-alpha"));
        assert!(
            !store
                .has_complete_coverage("prod-model")
                .expect("partial gate")
        );

        let empty_path = test_path();
        let empty = Store::open(&empty_path).expect("open empty store");
        assert!(
            empty
                .has_complete_coverage("prod-model")
                .expect("empty gate")
        );
        drop(empty);
        cleanup(&empty_path);

        let jobs = store
            .claim_index_jobs("worker", 2, Duration::from_secs(30))
            .expect("claim embedding work");
        for job in jobs {
            seed_vector(
                &store,
                &job.entity_id,
                "prod-model",
                &[1.0, 0.0],
                job.generation,
            );
            store
                .connection
                .execute(
                    "DELETE FROM index_jobs WHERE id = ?1",
                    rusqlite::params![job.id],
                )
                .expect("consume seeded job");
        }
        assert!(
            store
                .has_complete_coverage("prod-model")
                .expect("full gate")
        );
        assert!(
            !store
                .has_complete_coverage("other-model")
                .expect("wrong-model gate")
        );

        let fresh = remember(&mut store, "fresh fact", Some("legacy-other"));
        assert!(
            !store
                .has_complete_coverage("prod-model")
                .expect("fresh gate")
        );
        let job = store
            .claim_index_jobs("worker", 1, Duration::from_secs(30))
            .expect("claim fresh work")
            .pop()
            .expect("fresh job");
        assert_eq!(job.entity_id, fresh.id);
        seed_vector(
            &store,
            &job.entity_id,
            "prod-model",
            &[0.0, 1.0],
            job.generation,
        );
        store
            .connection
            .execute(
                "DELETE FROM index_jobs WHERE id = ?1",
                rusqlite::params![job.id],
            )
            .expect("consume fresh job");
        assert!(
            store
                .has_complete_coverage("prod-model")
                .expect("restored gate")
        );

        store.forget(&fresh.id).expect("delete fresh memory");
        assert!(
            store
                .has_complete_coverage("prod-model")
                .expect("inactive ignored")
        );
        let hits = store
            .semantic_search_by_vector(&[1.0, 0.0], "prod-model", 10)
            .expect("search still works");
        assert!(hits.iter().all(|hit| hit.memory.id != fresh.id));
        assert!(hits.iter().any(|hit| hit.memory.id == first.id));
        assert!(hits.iter().any(|hit| hit.memory.id == second.id));
        drop(store);
        cleanup(&path);
    }

    fn seed_vector(store: &Store, entity_id: &str, model: &str, vector: &[f32], generation: i64) {
        let mut encoded = Vec::with_capacity(std::mem::size_of_val(vector));
        for value in vector {
            encoded.extend_from_slice(&value.to_le_bytes());
        }
        store
            .connection
            .execute(
                "INSERT INTO embeddings (
       entity_type, entity_id, model, dimensions, vector, source_generation, updated_at
   ) VALUES ('memory', ?1, ?2, ?3, ?4, ?5, 0)
   ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET
       dimensions = excluded.dimensions,
       vector = excluded.vector,
       source_generation = excluded.source_generation",
                rusqlite::params![entity_id, model, vector.len() as i64, encoded, generation],
            )
            .expect("seed test vector");
    }

    fn remember(store: &mut Store, text: &str, project_id: Option<&str>) -> crate::store::Memory {
        store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                project_id: project_id.map(str::to_owned),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory")
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-vector-search-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
