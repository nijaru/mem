//! Single-store routing for the project-local SQLite database.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::Serialize;

use crate::embedding_worker::{EMBEDDING_MODEL_ID, EmbeddingRunOptions};
use crate::episode::HistoryHit;
use crate::index_job::IndexJobStats;
use crate::store::{SearchHit, Store, WorkspaceState};
use crate::vector_search::SemanticSearchHit;

#[derive(Debug, Clone)]
pub struct StorageRouter {
    path: PathBuf,
}

impl StorageRouter {
    pub fn exact(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_store(&self, _project_id: Option<&str>) -> Result<Store> {
        Store::open(&self.path)
    }

    fn read_store(&self) -> Result<Option<Store>> {
        Store::open_existing(&self.path)
    }
}

pub fn routed_resolve_memory(
    router: &StorageRouter,
    _project_hint: Option<&str>,
    id_or_prefix: &str,
) -> Result<(Store, String)> {
    resolve_id(router, id_or_prefix, "memory", |store, candidate| {
        store.memory_id_candidates(candidate)
    })
}

pub fn routed_resolve_episode(
    router: &StorageRouter,
    _project_hint: Option<&str>,
    id_or_prefix: &str,
) -> Result<(Store, String)> {
    resolve_id(router, id_or_prefix, "episode", |store, candidate| {
        store.episode_id_candidates(candidate)
    })
}

fn resolve_id<F>(
    router: &StorageRouter,
    id_or_prefix: &str,
    kind: &str,
    candidates: F,
) -> Result<(Store, String)>
where
    F: FnOnce(&Store, &str) -> Result<Vec<String>>,
{
    let candidate = id_or_prefix.trim();
    if candidate.is_empty() {
        bail!("{kind} ID cannot be empty");
    }
    let Some(store) = router.read_store()? else {
        bail!("{kind} not found: {candidate}");
    };
    let matches = candidates(&store, candidate)?;
    if let Some(exact) = matches.iter().find(|id| id.as_str() == candidate) {
        return Ok((store, exact.clone()));
    }
    match matches.as_slice() {
        [] => bail!("{kind} not found: {candidate}"),
        [id] => Ok((store, id.clone())),
        _ => bail!("ambiguous {kind} ID prefix: {candidate}"),
    }
}

pub fn routed_recall_hits(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
    force_lexical: bool,
) -> Result<Vec<SearchHit>> {
    let Some(store) = router.read_store()? else {
        return Ok(Vec::new());
    };
    crate::recall_hits(&store, query, project_id, limit, force_lexical)
}

pub fn routed_lexical_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchHit>> {
    let Some(store) = router.read_store()? else {
        return Ok(Vec::new());
    };
    store.search(query, project_id, limit)
}

pub fn routed_semantic_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<SemanticSearchHit>> {
    let Some(store) = router.read_store()? else {
        return Ok(Vec::new());
    };
    store.semantic_search_by_vector(query_vector, EMBEDDING_MODEL_ID, project_id, limit)
}

pub fn routed_history_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<HistoryHit>> {
    let Some(store) = router.read_store()? else {
        return Ok(Vec::new());
    };
    store.history_search(query, project_id, limit)
}

pub fn routed_workspace_state(
    router: &StorageRouter,
    project_id: &str,
    workspace_id: &str,
) -> Result<Option<WorkspaceState>> {
    let Some(store) = router.read_store()? else {
        return Ok(None);
    };
    store.workspace_state(project_id, workspace_id)
}

#[derive(Debug, Serialize)]
pub struct RoutedRunStats {
    pub model: &'static str,
    pub cache_dir: String,
    pub stores: usize,
    pub claimed: usize,
    pub eligible: usize,
    pub committed: usize,
    pub stale: usize,
    pub retried: usize,
}

pub fn routed_run_index(
    router: &StorageRouter,
    _project_id: Option<&str>,
    _all: bool,
    options: EmbeddingRunOptions,
) -> Result<RoutedRunStats> {
    let mut stats = RoutedRunStats {
        model: EMBEDDING_MODEL_ID,
        cache_dir: options.cache_dir.display().to_string(),
        stores: 1,
        claimed: 0,
        eligible: 0,
        committed: 0,
        stale: 0,
        retried: 0,
    };
    let Some(mut store) = router.read_store()? else {
        return Ok(stats);
    };
    let run = store.run_embedding_jobs(options)?;
    stats.claimed = run.claimed;
    stats.eligible = run.eligible;
    stats.committed = run.committed;
    stats.stale = run.stale;
    stats.retried = run.retried;
    Ok(stats)
}

pub fn routed_index_stats(
    router: &StorageRouter,
    _project_id: Option<&str>,
) -> Result<IndexJobStats> {
    let Some(store) = router.read_store()? else {
        return Ok(IndexJobStats {
            pending: 0,
            running: 0,
        });
    };
    store.index_job_stats()
}
