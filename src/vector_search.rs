use std::mem::size_of;

use anyhow::{Result, bail};
use rusqlite::{Row, params};
use serde::Serialize;

use crate::store::{Memory, Store};

#[derive(Debug, Serialize)]
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
    /// Whether every active memory visible in this scope has a stored vector
    /// for the model (complete coverage). Adapter-facing semantic-first recall
    /// uses this as a correctness gate: semantic ranking joins from the
    /// embeddings table, so any visible active memory without a vector would
    /// silently disappear from recall. Incomplete coverage must fall back to
    /// lexical retrieval instead.
    pub fn has_complete_scope_coverage(
        &self,
        model: &str,
        project_id: Option<&str>,
    ) -> Result<bool> {
        if model.trim().is_empty() {
            bail!("embedding model identifier cannot be empty");
        }
        let covered: i64 = self.connection.query_row(
            "SELECT COUNT(*)
             FROM memories AS m
             WHERE m.status = 'active'
               AND (m.project_id IS NULL OR m.project_id = ?2)
               AND EXISTS (
                   SELECT 1 FROM embeddings AS e
                   WHERE e.entity_type = 'memory'
                     AND e.entity_id = m.id
                     AND e.model = ?1
               )",
            params![model, project_id],
            |row| row.get(0),
        )?;
        if covered == 0 {
            return Ok(false);
        }
        let visible: i64 = self.connection.query_row(
            "SELECT COUNT(*)
             FROM memories AS m
             WHERE m.status = 'active'
               AND (m.project_id IS NULL OR m.project_id = ?1)",
            params![project_id],
            |row| row.get(0),
        )?;
        Ok(covered == visible)
    }

    pub fn semantic_search_by_vector(
        &self,
        query_vector: &[f32],
        model: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>> {
        if model.trim().is_empty() {
            bail!("embedding model identifier cannot be empty");
        }
        if limit == 0 {
            bail!("semantic search limit must be greater than zero");
        }
        let query = normalize_query(query_vector)?;
        let mut candidates = Vec::new();

        if let Some(project_id) = project_id {
            let mut statement = self.connection.prepare(
                "SELECT m.id, m.scope, m.project_id, m.kind, m.text, m.actor, m.status,\n\
                        m.created_at, m.updated_at, m.deleted_at, e.dimensions, e.vector\n\
                 FROM embeddings AS e\n\
                 JOIN memories AS m ON m.id = e.entity_id\n\
                 WHERE e.entity_type = 'memory'\n\
                   AND e.model = ?1\n\
                   AND m.status = 'active'\n\
                   AND (m.project_id IS NULL OR m.project_id = ?2)",
            )?;
            let rows =
                statement.query_map(params![model, project_id], vector_candidate_from_row)?;
            for row in rows {
                candidates.push(row?);
            }
        } else {
            let mut statement = self.connection.prepare(
                "SELECT m.id, m.scope, m.project_id, m.kind, m.text, m.actor, m.status,\n\
                        m.created_at, m.updated_at, m.deleted_at, e.dimensions, e.vector\n\
                 FROM embeddings AS e\n\
                 JOIN memories AS m ON m.id = e.entity_id\n\
                 WHERE e.entity_type = 'memory'\n\
                   AND e.model = ?1\n\
                   AND m.status = 'active'\n\
                   AND m.project_id IS NULL",
            )?;
            let rows = statement.query_map([model], vector_candidate_from_row)?;
            for row in rows {
                candidates.push(row?);
            }
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
    fn exact_scan_scores_current_model_and_scope() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let global = remember(&mut store, "global candidate", None);
        let alpha = remember(&mut store, "alpha candidate", Some("alpha"));
        let beta = remember(&mut store, "beta candidate", Some("beta"));

        let jobs = store
            .claim_index_jobs("worker", 3, Duration::from_secs(30))
            .expect("claim embedding work");
        for job in jobs {
            let vector = if job.entity_id == alpha.id {
                [1.0, 0.0]
            } else if job.entity_id == global.id {
                [0.8, 0.2]
            } else {
                assert_eq!(job.entity_id, beta.id);
                [1.0, 0.0]
            };
            assert!(
                store
                    .commit_embedding(
                        &job.id,
                        job.generation,
                        &job.lease_token,
                        "test-model",
                        &vector,
                    )
                    .expect("commit embedding")
            );
        }

        let alpha_hits = store
            .semantic_search_by_vector(&[2.0, 0.0], "test-model", Some("alpha"), 10)
            .expect("search alpha scope");
        assert_eq!(alpha_hits.len(), 2);
        assert_eq!(alpha_hits[0].memory.id, alpha.id);
        assert_eq!(alpha_hits[1].memory.id, global.id);
        assert!(alpha_hits[0].score > alpha_hits[1].score);
        assert!(alpha_hits.iter().all(|hit| hit.memory.id != beta.id));

        let global_hits = store
            .semantic_search_by_vector(&[1.0, 0.0], "test-model", None, 10)
            .expect("search global scope");
        assert_eq!(global_hits.len(), 1);
        assert_eq!(global_hits[0].memory.id, global.id);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn coverage_gate_counts_only_active_visible_current_model_vectors() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let _global = remember(&mut store, "global candidate", None);
        let _alpha = remember(&mut store, "alpha candidate", Some("alpha"));
        let _beta = remember(&mut store, "beta candidate", Some("beta"));

        // Empty scope: no vectors, no active visible rows -> no semantic mode.
        assert!(
            !store
                .has_complete_scope_coverage("prod-model", None)
                .expect("gate global empty")
        );
        assert!(
            !store
                .has_complete_scope_coverage("prod-model", Some("alpha"))
                .expect("gate alpha empty")
        );

        // Embed each memory through the queue protocol with production-model
        // vectors.
        let jobs = store
            .claim_index_jobs("worker", 3, Duration::from_secs(30))
            .expect("claim embedding work");
        assert_eq!(jobs.len(), 3);
        for job in &jobs {
            assert!(
                store
                    .commit_embedding(
                        &job.id,
                        job.generation,
                        &job.lease_token,
                        "prod-model",
                        &[1.0, 0.0],
                    )
                    .expect("commit embedding")
            );
        }

        // Full current-model coverage across the visible scope (project +
        // global memories) unlocks semantic mode; the other project is
        // invisible to this scope and irrelevant.
        assert!(
            store
                .has_complete_scope_coverage("prod-model", Some("alpha"))
                .expect("gate alpha full")
        );
        assert!(
            store
                .has_complete_scope_coverage("prod-model", None)
                .expect("gate global full")
        );

        // A vector under another model ID does not count toward coverage of
        // the production model.
        assert!(
            !store
                .has_complete_scope_coverage("other-model", Some("alpha"))
                .expect("gate wrong model")
        );

        // A newly remembered memory in the same scope breaks coverage until
        // its embedding job is consumed.
        let fresh = remember(&mut store, "fresh alpha fact", Some("alpha"));
        assert!(
            !store
                .has_complete_scope_coverage("prod-model", Some("alpha"))
                .expect("gate partial")
        );
        // The global-only scope is unaffected by the alpha-scoped addition.
        assert!(
            store
                .has_complete_scope_coverage("prod-model", None)
                .expect("gate global still full")
        );

        // Consuming the fresh job restores complete coverage.
        let job = store
            .claim_index_jobs("worker", 1, Duration::from_secs(30))
            .expect("claim fresh work")
            .pop()
            .expect("fresh job");
        assert_eq!(job.entity_id, fresh.id);
        assert!(
            store
                .commit_embedding(
                    &job.id,
                    job.generation,
                    &job.lease_token,
                    "prod-model",
                    &[0.0, 1.0],
                )
                .expect("commit fresh embedding")
        );
        assert!(
            store
                .has_complete_scope_coverage("prod-model", Some("alpha"))
                .expect("gate full again")
        );

        // Superseding a memory removes it from the visible active set; the
        // still-covered remainder keeps semantic mode unlocked.
        store.forget(&fresh.id).expect("delete fresh memory");
        assert!(
            store
                .has_complete_scope_coverage("prod-model", Some("alpha"))
                .expect("gate ignores inactive rows")
        );
        let alpha_hits = store
            .semantic_search_by_vector(&[1.0, 0.0], "prod-model", Some("alpha"), 10)
            .expect("search still works");
        assert!(alpha_hits.iter().all(|hit| hit.memory.id != fresh.id));

        drop(store);
        cleanup(&path);
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
