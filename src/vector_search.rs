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
    pub fn has_complete_coverage(&self, model: &str) -> Result<bool> {
        Ok(self.embedding_coverage(model)?.unindexed == 0)
    }

    pub fn semantic_search_by_vector(
        &self,
        query_vector: &[f32],
        model: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>> {
        if model.trim().is_empty() {
            bail!("embedding model cannot be empty");
        }
        if limit == 0 {
            bail!("semantic search limit must be greater than zero");
        }
        let query = normalize_query(query_vector)?;
        let mut statement = self.connection.prepare(
            "SELECT m.id, m.kind, m.text, m.actor, m.source_type, m.source_ref,\n\
                    m.status, m.superseded_by, m.created_at, m.updated_at, m.deleted_at,\n\
                    e.dimensions, e.vector\n\
             FROM embeddings AS e\n\
             JOIN memories AS m ON m.id = e.memory_id\n\
             WHERE e.model = ?1\n\
               AND e.source_updated_at = m.updated_at\n\
               AND m.status = 'active'",
        )?;
        let rows = statement.query_map([model], candidate_from_row)?;
        let mut hits = Vec::new();
        for row in rows {
            let candidate = row?;
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

fn candidate_from_row(row: &Row<'_>) -> rusqlite::Result<VectorCandidate> {
    Ok(VectorCandidate {
        memory: Memory {
            id: row.get(0)?,
            kind: row.get(1)?,
            text: row.get(2)?,
            actor: row.get(3)?,
            source_type: row.get(4)?,
            source_ref: row.get(5)?,
            status: row.get(6)?,
            superseded_by: row.get(7)?,
            created_at: row.get(8)?,
            updated_at: row.get(9)?,
            deleted_at: row.get(10)?,
        },
        dimensions: row.get(11)?,
        vector: row.get(12)?,
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

fn decode_vector(bytes: &[u8], dimensions: usize, memory_id: &str) -> Result<Vec<f32>> {
    let expected = dimensions
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("embedding dimensions overflow for {memory_id}"))?;
    if bytes.len() != expected {
        bail!(
            "invalid embedding byte length for {memory_id}: expected {expected}, got {}",
            bytes.len()
        );
    }
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        bail!("invalid embedding byte alignment for {memory_id}");
    }
    let vector: Vec<f32> = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("stored embedding contains non-finite values for {memory_id}");
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use crate::store::{NewMemory, Store};

    #[test]
    fn exact_scan_ranks_all_current_active_vectors() {
        let path =
            std::env::temp_dir().join(format!("mem-vector-test-{}.db", uuid::Uuid::now_v7()));
        let store = Store::open(&path).expect("open");
        let first = remember(&store, "first");
        let best = remember(&store, "best");
        store
            .upsert_embedding_if_current(&first.id, first.updated_at, "test", &[0.8, 0.2])
            .expect("seed first");
        store
            .upsert_embedding_if_current(&best.id, best.updated_at, "test", &[1.0, 0.0])
            .expect("seed best");
        let hits = store
            .semantic_search_by_vector(&[2.0, 0.0], "test", 10)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory.id, best.id);
        assert!(store.has_complete_coverage("test").expect("coverage"));
        let _ = std::fs::remove_file(path);
    }

    fn remember(store: &Store, text: &str) -> crate::store::Memory {
        store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("remember")
    }
}
