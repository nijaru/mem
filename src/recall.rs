use anyhow::Result;

use crate::embedding_worker::{EMBEDDING_MODEL_ID, embed_query_if_cached, model_cache_dir};
use crate::store::{SearchHit, Store};

const SEMANTIC_CONTEXT_MIN_SCORE: f64 = 0.55;

pub fn recall_hits(store: &Store, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    recall_with_query_vector(store, query, limit, || cached_query_vector(query))
}

fn recall_with_query_vector(
    store: &Store,
    query: &str,
    limit: usize,
    query_vector: impl FnOnce() -> Result<Option<Vec<f32>>>,
) -> Result<Vec<SearchHit>> {
    if store.has_complete_coverage(EMBEDDING_MODEL_ID)?
        && let Some(query_vector) = query_vector()?
        && let Ok(hits) = store.semantic_search_by_vector(&query_vector, EMBEDDING_MODEL_ID, limit)
    {
        return Ok(hits
            .into_iter()
            .filter(|hit| hit.score >= SEMANTIC_CONTEXT_MIN_SCORE)
            .map(|hit| SearchHit {
                memory: hit.memory,
                rank: hit.score,
            })
            .collect());
    }
    // Missing, concurrently invalidated, or malformed derived data must not
    // disable canonical recall. Explicit semantic search still reports errors.
    store.search(query, limit)
}

fn cached_query_vector(query: &str) -> Result<Option<Vec<f32>>> {
    let Ok(cache_dir) = model_cache_dir() else {
        return Ok(None);
    };
    Ok(embed_query_if_cached(query, &cache_dir).unwrap_or(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::NewMemory;

    fn seed(store: &Store, text: &str) -> crate::store::Memory {
        let memory = store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .unwrap();
        Store::upsert_embedding_if_current(
            &store.connection,
            &memory.id,
            memory.updated_at,
            EMBEDDING_MODEL_ID,
            &[1.0, 0.0],
        )
        .unwrap();
        memory
    }

    #[test]
    fn fresh_memory_during_query_embedding_forces_lexical_fallback() {
        let path = std::env::temp_dir().join(format!("mem-recall-{}.db", uuid::Uuid::now_v7()));
        let store = Store::open(&path).unwrap();
        let writer = Store::open(&path).unwrap();
        seed(&store, "old knowledge");
        let hits = recall_with_query_vector(&store, "fresh", 10, || {
            writer.remember(NewMemory {
                text: "fresh knowledge".to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })?;
            Ok(Some(vec![1.0, 0.0]))
        })
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.text, "fresh knowledge");
        assert!(
            store
                .semantic_search_by_vector(&[1.0, 0.0], EMBEDDING_MODEL_ID, 10)
                .is_err()
        );
        drop(writer);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_derived_vectors_do_not_disable_context() {
        let path = std::env::temp_dir().join(format!("mem-recall-{}.db", uuid::Uuid::now_v7()));
        let store = Store::open(&path).unwrap();
        let memory = seed(&store, "lexical fallback");
        for bytes in [
            vec![0_u8],
            [f32::NAN.to_le_bytes(), 0.0_f32.to_le_bytes()].concat(),
        ] {
            store
                .connection
                .execute("UPDATE embeddings SET vector = ?1", [bytes])
                .unwrap();
            let hits = recall_with_query_vector(&store, "lexical", 10, || Ok(Some(vec![1.0, 0.0])))
                .unwrap();
            assert_eq!(hits[0].memory.id, memory.id);
            assert!(
                store
                    .semantic_search_by_vector(&[1.0, 0.0], EMBEDDING_MODEL_ID, 10)
                    .is_err()
            );
        }
        drop(store);
        std::fs::remove_file(path).unwrap();
    }
}
