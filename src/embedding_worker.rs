use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::OptionalExtension;
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
        fs::create_dir_all(&options.cache_dir).with_context(|| {
            format!(
                "create embedding model cache {}",
                options.cache_dir.display()
            )
        })?;

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

        let init_options = TextInitOptions::new(EmbeddingModel::BGESmallENV15Q)
            .with_cache_dir(options.cache_dir)
            .with_show_download_progress(options.show_download_progress);
        let mut model = match TextEmbedding::try_new(init_options) {
            Ok(model) => model,
            Err(error) => {
                let message = format!("embedding model initialization failed: {error}");
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
            "episode_entry" => Ok(self
                .connection
                .query_row(
                    "SELECT text FROM episode_entries WHERE id = ?1",
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

fn validate_options(options: &EmbeddingRunOptions) -> Result<()> {
    if options.limit == 0 {
        bail!("embedding worker limit must be greater than zero");
    }
    if options.lease_duration.is_zero() {
        bail!("embedding worker lease duration must be greater than zero");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::Store;
    use crate::episode::{NewEpisode, NewEpisodeEntry};
    use crate::store::NewMemory;

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
        let episode = store
            .ensure_episode(NewEpisode {
                project_id: None,
                workspace_id: None,
                source_type: "test-session".to_owned(),
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
                    text: "current episode text".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record episode entry");

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
                assert_eq!(job.entity_id, entry.id);
                assert_eq!(text, "current episode text");
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

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-embedding-worker-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
