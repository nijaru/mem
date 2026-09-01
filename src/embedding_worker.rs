use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use uuid::Uuid;

use crate::index_job::ClaimedIndexJob;
use crate::store::Store;

pub const EMBEDDING_MODEL_ID: &str =
    "fastembed-5.3.0:Qdrant/bge-small-en-v1.5-onnx-Q/model_optimized.onnx";

pub struct EmbeddingRunOptions {
    pub limit: usize,
    pub lease_duration: Duration,
    pub retry_delay: Duration,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingRunStats {
    pub model: &'static str,
    pub cache_dir: String,
    pub claimed: usize,
    pub eligible: usize,
    pub committed: usize,
    pub stale: usize,
    pub retried: usize,
}

impl Store {
    pub fn run_embedding_jobs(
        &mut self,
        options: EmbeddingRunOptions,
    ) -> Result<EmbeddingRunStats> {
        validate_options(&options)?;
        ensure_cache_dir(&options.cache_dir)?;

        self.backfill_missing_production_embeddings()?;

        let worker_id = format!("mem-{}", Uuid::now_v7());
        let jobs = self.claim_index_jobs(&worker_id, options.limit, options.lease_duration)?;
        let mut stats = EmbeddingRunStats {
            model: EMBEDDING_MODEL_ID,
            cache_dir: options.cache_dir.display().to_string(),
            claimed: jobs.len(),
            eligible: 0,
            committed: 0,
            stale: 0,
            retried: 0,
        };
        if jobs.is_empty() {
            return Ok(stats);
        }

        let mut work = Vec::with_capacity(jobs.len());
        for job in jobs {
            if job.index_kind != "embedding" {
                bail!("unsupported derived index kind: {}", job.index_kind);
            }
            if let Some(text) = self.embedding_source_text(&job)? {
                work.push((job, text));
            } else {
                stats.stale += 1;
            }
        }
        stats.eligible = work.len();
        if work.is_empty() {
            return Ok(stats);
        }

        let mut model = match load_model(&options.cache_dir, options.show_download_progress) {
            Ok(model) => model,
            Err(error) => {
                let message = format!("embedding model initialization failed: {error:#}");
                stats.retried += self.retry_embedding_work(&work, &message, options.retry_delay)?;
                bail!(message);
            }
        };

        let texts: Vec<&str> = work.iter().map(|(_, text)| text.as_str()).collect();
        let batch_size = texts.len();
        let embeddings = match model.embed(texts, Some(batch_size)) {
            Ok(embeddings) => embeddings,
            Err(error) => {
                let message = format!("embedding inference failed: {error}");
                stats.retried += self.retry_embedding_work(&work, &message, options.retry_delay)?;
                bail!(message);
            }
        };
        if embeddings.len() != work.len() {
            let message = format!(
                "embedding inference returned {} vectors for {} source texts",
                embeddings.len(),
                work.len()
            );
            stats.retried += self.retry_embedding_work(&work, &message, options.retry_delay)?;
            bail!(message);
        }

        for ((job, _), embedding) in work.iter().zip(embeddings) {
            match self.commit_embedding(
                &job.id,
                job.generation,
                &job.lease_token,
                EMBEDDING_MODEL_ID,
                &embedding,
            ) {
                Ok(true) => stats.committed += 1,
                Ok(false) => stats.stale += 1,
                Err(error) => {
                    let message = format!("embedding commit failed: {error:#}");
                    if self.retry_index_job(
                        &job.id,
                        job.generation,
                        &job.lease_token,
                        &message,
                        options.retry_delay,
                    )? {
                        stats.retried += 1;
                    } else {
                        stats.stale += 1;
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Re-enqueue embedding work for active semantic memories that have no
    /// vector for the current production model and no outstanding job row.
    /// A production model change (or a historical job consumed before this
    /// invariant existed) must not leave canonical sources permanently
    /// unembedded: `index run` rediscovers the missing work. Job rows reuse
    /// the trigger ID scheme (`memory:<id>:embedding`) with generation 1 so a
    /// later canonical write conflict-bumps instead of duplicating.
    fn backfill_missing_production_embeddings(&self) -> Result<()> {
        let now = unix_millis_for_backfill()?;
        self.connection.execute(
            "INSERT INTO index_jobs (
                 id, entity_type, entity_id, index_kind, generation, state, attempts,
                 available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
             )
             SELECT
                 'memory:' || m.id || ':embedding', 'memory', m.id, 'embedding', 1, 'pending', 0,
                 ?1, NULL, NULL, NULL, NULL, m.created_at, m.updated_at
             FROM memories AS m
             WHERE m.status = 'active'
               AND NOT EXISTS (
                   SELECT 1 FROM embeddings AS e
                   WHERE e.entity_type = 'memory'
                     AND e.entity_id = m.id
                     AND e.model = ?2
               )
               AND NOT EXISTS (
                   SELECT 1 FROM index_jobs AS j
                   WHERE j.entity_type = 'memory'
                     AND j.entity_id = m.id
                     AND j.index_kind = 'embedding'
               )
             ON CONFLICT(entity_type, entity_id, index_kind) DO NOTHING",
            params![now, EMBEDDING_MODEL_ID],
        )?;
        Ok(())
    }

    fn embedding_source_text(&self, job: &ClaimedIndexJob) -> Result<Option<String>> {
        match job.entity_type.as_str() {
            "memory" => Ok(self
                .connection
                .query_row(
                    "SELECT text FROM memories WHERE id = ?1 AND status = 'active'",
                    [&job.entity_id],
                    |row| row.get(0),
                )
                .optional()?),
            entity_type => bail!("unsupported embedding entity type: {entity_type}"),
        }
    }

    fn retry_embedding_work(
        &self,
        work: &[(ClaimedIndexJob, String)],
        message: &str,
        retry_delay: Duration,
    ) -> Result<usize> {
        let mut retried = 0;
        for (job, _) in work {
            if self.retry_index_job(
                &job.id,
                job.generation,
                &job.lease_token,
                message,
                retry_delay,
            )? {
                retried += 1;
            }
        }
        Ok(retried)
    }
}

pub fn embed_query(
    query: &str,
    cache_dir: &Path,
    show_download_progress: bool,
) -> Result<Vec<f32>> {
    if query.trim().is_empty() {
        bail!("semantic search query cannot be empty");
    }
    ensure_cache_dir(cache_dir)?;
    let mut model = load_model(cache_dir, show_download_progress)?;
    let mut embeddings = model
        .embed(vec![query], Some(1))
        .context("embedding semantic search query")?;
    if embeddings.len() != 1 {
        bail!(
            "embedding inference returned {} vectors for one search query",
            embeddings.len()
        );
    }
    embeddings
        .pop()
        .context("embedding inference returned no query vector")
}

/// Embed a query only when the embedding model is already fully cached.
/// Adapter-facing recall must never trigger a model download: a fresh machine
/// falls back to lexical retrieval instead of blocking on the network.
pub fn embed_query_if_cached(query: &str, cache_dir: &Path) -> Result<Option<Vec<f32>>> {
    if query.trim().is_empty() {
        bail!("semantic recall query cannot be empty");
    }
    if !model_is_cached(cache_dir) {
        return Ok(None);
    }
    embed_query(query, cache_dir, false).map(Some)
}

const EMBEDDING_MODEL_REPO: &str = "Qdrant/bge-small-en-v1.5-onnx-Q";
const EMBEDDING_MODEL_FILES: [&str; 5] = [
    "model_optimized.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// Mirrors the fastembed/hf-hub cache layout: one snapshot directory holds all
/// model files, addressed by refs/<revision>.
fn model_is_cached(cache_dir: &Path) -> bool {
    let snapshot = cached_model_snapshot(cache_dir);
    EMBEDDING_MODEL_FILES.iter().all(|file| {
        snapshot
            .as_deref()
            .map(|snapshot| snapshot.join(file).is_file())
            .unwrap_or(false)
    })
}

fn cached_model_snapshot(cache_dir: &Path) -> Option<std::path::PathBuf> {
    // hf-hub cache layout: the repo folder is one path component,
    // e.g. models--Qdrant--bge-small-en-v1.5-onnx-Q.
    let repo_dir = cache_dir.join(format!(
        "models--{}",
        EMBEDDING_MODEL_REPO.replace('/', "--")
    ));
    let commit = std::fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        return None;
    }
    Some(repo_dir.join("snapshots").join(commit))
}

pub fn model_cache_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("HF_HOME") {
        let path = PathBuf::from(path);
        if path.as_os_str().is_empty() {
            bail!("HF_HOME cannot be empty");
        }
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("MEM_MODEL_CACHE") {
        let path = PathBuf::from(path);
        if path.as_os_str().is_empty() {
            bail!("MEM_MODEL_CACHE cannot be empty");
        }
        return Ok(path);
    }

    let cache_dir = dirs::cache_dir().context("could not determine the local cache directory")?;
    Ok(cache_dir.join("mem").join("models"))
}

fn load_model(cache_dir: &Path, show_download_progress: bool) -> Result<TextEmbedding> {
    let init_options = TextInitOptions::new(EmbeddingModel::BGESmallENV15Q)
        .with_cache_dir(cache_dir.to_owned())
        .with_show_download_progress(show_download_progress);
    TextEmbedding::try_new(init_options).context("initialize local embedding model")
}

fn ensure_cache_dir(cache_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("create embedding model cache {}", cache_dir.display()))?;
    Ok(())
}

fn validate_options(options: &EmbeddingRunOptions) -> Result<()> {
    if options.limit == 0 {
        bail!("embedding worker limit must be greater than zero");
    }
    if options.lease_duration.is_zero() {
        bail!("embedding worker lease duration must be greater than zero");
    }
    Ok(())
}

fn unix_millis_for_backfill() -> Result<i64> {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::{
        EMBEDDING_MODEL_FILES, EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REPO, Store, model_is_cached,
    };
    use crate::store::NewMemory;

    #[test]
    fn index_run_backfills_missing_production_model_vectors() {
        // Production-failure regression: a job consumed before this invariant
        // existed could leave an active memory with only a wrong-model vector
        // (or none at all) and no pending work. `index run` must rediscover
        // the missing production-model embedding work before claiming.
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let covered = remember(&mut store, "already covered by production model");
        let legacy = remember(&mut store, "historically embedded under an old model");

        // Simulate the historical wrong-model consumption: claim both jobs
        // and consume them as a pre-guard database would — `covered` got a
        // production-model vector, `legacy` only an old-model one.
        let claimed = store
            .claim_index_jobs("worker-a", 10, Duration::from_secs(30))
            .expect("claim all jobs");
        assert_eq!(claimed.len(), 2);
        for job in &claimed {
            if job.entity_id == covered.id {
                store
                    .connection
                    .execute(
                        "INSERT INTO embeddings (
                             entity_type, entity_id, model, dimensions, vector, source_generation, updated_at
                         ) VALUES ('memory', ?1, ?2, 2, X'0000803F00000000', ?3, 0)
                         ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET
                             source_generation = excluded.source_generation",
                        rusqlite::params![job.entity_id, EMBEDDING_MODEL_ID, job.generation],
                    )
                    .expect("insert production vector for covered");
            } else {
                store
                    .connection
                    .execute(
                        "INSERT INTO embeddings (
                             entity_type, entity_id, model, dimensions, vector, source_generation, updated_at
                         ) VALUES (?1, ?2, 'old-model', 2, X'0000803F00000000', ?3, 0)
                         ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET
                             source_generation = excluded.source_generation",
                        rusqlite::params![job.entity_type, job.entity_id, job.generation],
                    )
                    .expect("insert legacy vector");
            }
            store
                .connection
                .execute(
                    "DELETE FROM index_jobs WHERE id = ?1",
                    rusqlite::params![job.id],
                )
                .expect("consume job without the guard");
        }
        let stats = store.index_job_stats().expect("queue is empty");
        assert_eq!(stats.pending + stats.running, 0);

        // The production-model vector only exists for `covered`; `legacy`
        // holds a wrong-model vector. Backfill must requeue `legacy` only.
        store
            .backfill_missing_production_embeddings()
            .expect("run backfill");
        let pending: Vec<String> = store
            .connection
            .prepare("SELECT entity_id FROM index_jobs WHERE state = 'pending'")
            .expect("prepare pending query")
            .query_map([], |row| row.get::<_, String>(0))
            .expect("map pending rows")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("collect pending rows");
        let legacy_id = legacy.id.clone();
        assert_eq!(
            pending,
            vec![legacy_id.clone()],
            "only the uncovered memory is requeued"
        );

        // Claiming the backfilled job and committing a production-model
        // vector completes the recovery.
        let job = store
            .claim_index_jobs("worker-b", 1, Duration::from_secs(30))
            .expect("claim backfilled work")
            .pop()
            .expect("backfilled job");
        assert_eq!(job.entity_id, legacy_id);
        assert!(
            store
                .commit_embedding(
                    &job.id,
                    job.generation,
                    &job.lease_token,
                    EMBEDDING_MODEL_ID,
                    &[1.0, 1.0],
                )
                .expect("commit production vector")
        );
        let covered_count: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE model = ?1 AND entity_type = 'memory'",
                rusqlite::params![EMBEDDING_MODEL_ID],
                |row| row.get(0),
            )
            .expect("count production vectors");
        assert_eq!(covered_count, 2, "both memories are now production-covered");

        // A second backfill is a no-op: the vector now exists.
        store
            .backfill_missing_production_embeddings()
            .expect("run backfill again");
        let stats = store.index_job_stats().expect("read stats");
        assert_eq!(stats.pending + stats.running, 0, "no duplicate work");

        drop(store);
        cleanup(&path);
    }

    fn remember(store: &mut Store, text: &str) -> crate::store::Memory {
        store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory")
    }

    #[test]
    fn claimed_jobs_resolve_only_current_canonical_text() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let memory = store
            .remember(NewMemory {
                text: "current semantic text".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
        let second = store
            .remember(NewMemory {
                text: "current second text".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store second memory");
        // Direct canonical text refresh so claimed jobs must resolve only the
        // current text — the path the source-resolution guards.
        store
            .connection
            .execute(
                "UPDATE memories\n\
                 SET text = 'refreshed second text', updated_at = updated_at + 1 WHERE id = ?1",
                rusqlite::params![&second.id],
            )
            .expect("refresh canonical text");

        let jobs = store
            .claim_index_jobs("worker-a", 2, Duration::from_secs(30))
            .expect("claim jobs");
        assert_eq!(jobs.len(), 2);
        for job in &jobs {
            let text = store
                .embedding_source_text(job)
                .expect("resolve source")
                .expect("eligible source");
            if job.entity_id == memory.id {
                assert_eq!(text, "current semantic text");
            } else {
                assert_eq!(job.entity_id, second.id);
                assert_eq!(text, "refreshed second text");
            }
        }

        store.forget(&memory.id).expect("delete memory");
        let memory_job = jobs
            .iter()
            .find(|job| job.entity_id == memory.id)
            .expect("memory job");
        assert!(
            store
                .embedding_source_text(memory_job)
                .expect("resolve deleted memory")
                .is_none()
        );

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn model_id_matches_cached_repo_layout() {
        // EMBEDDING_MODEL_ID is the vector-store key derived from the same
        // repo and file the cache-presence check must recognize.
        let repo = "Qdrant/bge-small-en-v1.5-onnx-Q";
        let model_file = "model_optimized.onnx";
        assert_eq!(
            EMBEDDING_MODEL_ID,
            format!("fastembed-5.3.0:{repo}/{model_file}")
        );
        assert_eq!(EMBEDDING_MODEL_REPO, repo);
        assert_eq!(EMBEDDING_MODEL_FILES[0], model_file);
    }

    #[test]
    fn cache_presence_check_uses_hf_hub_layout() {
        let dir = std::env::temp_dir().join(format!("mem-model-cache-test-{}", Uuid::now_v7()));
        let repo_dir = dir.join("models--Qdrant--bge-small-en-v1.5-onnx-Q");
        let refs_dir = repo_dir.join("refs");
        let snapshot = repo_dir.join("snapshots").join("abc123");
        std::fs::create_dir_all(&refs_dir).expect("create refs dir");
        std::fs::create_dir_all(&snapshot).expect("create snapshot dir");
        std::fs::write(refs_dir.join("main"), "abc123\n").expect("write ref");

        assert!(!model_is_cached(&dir));
        for file in EMBEDDING_MODEL_FILES {
            std::fs::write(snapshot.join(file), "stub").expect("write model file");
        }
        assert!(model_is_cached(&dir));

        std::fs::remove_file(snapshot.join(EMBEDDING_MODEL_FILES[0])).expect("remove model file");
        assert!(!model_is_cached(&dir));

        std::fs::remove_dir_all(&dir).expect("remove cache dir");
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-embedding-worker-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
