use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::Serialize;

use crate::store::Store;

pub const EMBEDDING_MODEL_ID: &str =
    "fastembed-5.3.0:Qdrant/bge-small-en-v1.5-onnx-Q/model_optimized.onnx";

pub struct EmbeddingRunOptions {
    pub limit: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    pub cached_only: bool,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingRunStats {
    pub model: &'static str,
    pub cache_dir: String,
    pub indexed: usize,
    pub stale: usize,
    pub remaining: u64,
}

impl Store {
    pub fn run_embedding_index(&self, options: EmbeddingRunOptions) -> Result<EmbeddingRunStats> {
        if options.limit == 0 {
            bail!("embedding limit must be greater than zero");
        }

        let coverage = self.embedding_coverage(EMBEDDING_MODEL_ID)?;
        let mut stats = EmbeddingRunStats {
            model: EMBEDDING_MODEL_ID,
            cache_dir: options.cache_dir.display().to_string(),
            indexed: 0,
            stale: 0,
            remaining: coverage.unindexed,
        };
        if coverage.unindexed == 0 {
            return Ok(stats);
        }
        if options.cached_only && !model_is_cached(&options.cache_dir) {
            return Ok(stats);
        }

        let sources = self.pending_embedding_sources(EMBEDDING_MODEL_ID, options.limit)?;
        if sources.is_empty() {
            return Ok(stats);
        }

        ensure_cache_dir(&options.cache_dir)?;
        let mut model = load_model(&options.cache_dir, options.show_download_progress)?;
        let texts: Vec<&str> = sources.iter().map(|source| source.text.as_str()).collect();
        let embeddings = model
            .embed(texts, Some(sources.len()))
            .context("embed pending memories")?;
        if embeddings.len() != sources.len() {
            bail!(
                "embedding inference returned {} vectors for {} memories",
                embeddings.len(),
                sources.len()
            );
        }

        for (source, vector) in sources.iter().zip(embeddings) {
            if self.upsert_embedding_if_current(
                &source.id,
                source.updated_at,
                EMBEDDING_MODEL_ID,
                &vector,
            )? {
                stats.indexed += 1;
            } else {
                stats.stale += 1;
            }
        }
        stats.remaining = self.embedding_coverage(EMBEDDING_MODEL_ID)?.unindexed;
        Ok(stats)
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
        .context("embed semantic search query")?;
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

pub fn model_is_cached(cache_dir: &Path) -> bool {
    let Some(snapshot) = cached_model_snapshot(cache_dir) else {
        return false;
    };
    EMBEDDING_MODEL_FILES
        .iter()
        .all(|file| snapshot.join(file).is_file())
}

fn cached_model_snapshot(cache_dir: &Path) -> Option<PathBuf> {
    let repo_dir = cache_dir.join(format!(
        "models--{}",
        EMBEDDING_MODEL_REPO.replace('/', "--")
    ));
    let commit = fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        return None;
    }
    Some(repo_dir.join("snapshots").join(commit))
}

pub fn model_cache_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MEM_MODEL_CACHE") {
        let path = PathBuf::from(path);
        if path.as_os_str().is_empty() {
            bail!("MEM_MODEL_CACHE cannot be empty");
        }
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        let path = PathBuf::from(path);
        if path.as_os_str().is_empty() {
            bail!("HF_HOME cannot be empty");
        }
        return Ok(path);
    }
    let cache = dirs::cache_dir().context("could not determine the local cache directory")?;
    Ok(cache.join("mem").join("models"))
}

fn load_model(cache_dir: &Path, show_download_progress: bool) -> Result<TextEmbedding> {
    let options = TextInitOptions::new(EmbeddingModel::BGESmallENV15Q)
        .with_cache_dir(cache_dir.to_owned())
        .with_show_download_progress(show_download_progress);
    TextEmbedding::try_new(options).context("initialize local embedding model")
}

fn ensure_cache_dir(cache_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("create embedding model cache {}", cache_dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EMBEDDING_MODEL_FILES, EmbeddingRunOptions, model_is_cached,
    };
    use crate::store::{NewMemory, Store};
    use uuid::Uuid;

    #[test]
    fn cached_only_index_is_a_noop_when_model_is_absent() {
        let path = std::env::temp_dir().join(format!("mem-index-test-{}.db", Uuid::now_v7()));
        let store = Store::open(&path).expect("open");
        store
            .remember(NewMemory {
                text: "pending semantic memory".to_owned(),
                kind: "fact".to_owned(),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("remember");
        let cache = std::env::temp_dir().join(format!("mem-cache-empty-{}", Uuid::now_v7()));
        let stats = store
            .run_embedding_index(EmbeddingRunOptions {
                limit: 16,
                cache_dir: cache.clone(),
                show_download_progress: false,
                cached_only: true,
            })
            .expect("index");
        assert_eq!(stats.indexed, 0);
        assert_eq!(stats.remaining, 1);
        assert!(!cache.exists());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cache_presence_uses_hf_hub_layout() {
        let dir = std::env::temp_dir().join(format!("mem-model-cache-test-{}", Uuid::now_v7()));
        let repo = dir.join("models--Qdrant--bge-small-en-v1.5-onnx-Q");
        let refs = repo.join("refs");
        let snapshot = repo.join("snapshots").join("abc123");
        std::fs::create_dir_all(&refs).expect("refs");
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(refs.join("main"), "abc123\n").expect("ref");
        assert!(!model_is_cached(&dir));
        for file in EMBEDDING_MODEL_FILES {
            std::fs::write(snapshot.join(file), "stub").expect("model file");
        }
        assert!(model_is_cached(&dir));
        std::fs::remove_dir_all(dir).expect("cleanup");
    }
}
