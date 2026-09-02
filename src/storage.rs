//! Managed storage layout: per-project SQLite routing with safe path
//! encoding. Exact-file operation (`--db`/`MEM_DB`) bypasses this entirely.

#![allow(dead_code)]

use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;

use crate::embedding_worker::{EMBEDDING_MODEL_ID, EmbeddingRunOptions};
use crate::episode::HistoryHit;
use crate::index_job::IndexJobStats;
use crate::store::{SearchHit, Store, WorkspaceState};
use crate::vector_search::SemanticSearchHit;

pub const LAYOUT_VERSION_DIR: &str = "layout-v1";
const LEGACY_DB_FILENAME: &str = "memory.db";
const USER_DB_FILENAME: &str = "user.db";
const PROJECT_DB_FILENAME: &str = "mem.db";

/// Managed layout root: `<MEM_HOME>` or `<local-data>/mem`. The legacy
/// combined database sits directly under the root; the active split layout
/// lives in a versioned subdirectory.
#[derive(Debug, Clone, Serialize)]
pub struct ManagedLayout {
    root: PathBuf,
}

impl ManagedLayout {
    pub fn resolve() -> Result<Self> {
        if let Some(home) = std::env::var_os("MEM_HOME").filter(|home| !home.is_empty()) {
            return Ok(Self::at(PathBuf::from(home)));
        }
        let data_dir =
            dirs::data_local_dir().context("could not determine the local data directory")?;
        Ok(Self::at(data_dir.join("mem")))
    }

    pub fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn legacy_db(&self) -> PathBuf {
        self.root.join(LEGACY_DB_FILENAME)
    }

    pub fn layout_dir(&self) -> PathBuf {
        self.root.join(LAYOUT_VERSION_DIR)
    }

    pub fn user_db(&self) -> PathBuf {
        self.layout_dir().join(USER_DB_FILENAME)
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.layout_dir().join("projects")
    }

    /// Resolve one project's database inside the managed layout. The encoded
    /// path can never traverse outside the projects directory by construction;
    /// containment is verified again after joining as a defensive invariant.
    pub fn project_db(&self, project_id: &str) -> Result<ProjectDb> {
        if project_id.trim().is_empty() {
            bail!("project identifier cannot be empty");
        }
        let projects = self.projects_dir();
        let mut path = projects.clone();
        for component in project_id.split('/') {
            path = path.join(encode_component(component));
        }
        path = path.join(PROJECT_DB_FILENAME);
        verify_contained(&projects, &path)?;
        Ok(ProjectDb {
            path,
            project_id: project_id.to_owned(),
        })
    }

    /// Recover the logical project identity from an existing managed project
    /// database directory. Only canonical encodings are accepted.
    pub fn decode_project_dir(&self, dir: &Path) -> Result<String> {
        let relative = dir.strip_prefix(self.projects_dir()).with_context(|| {
            format!(
                "{} is outside the managed projects directory",
                dir.display()
            )
        })?;
        let mut components = Vec::new();
        for component in relative.components() {
            let Component::Normal(part) = component else {
                bail!(
                    "{} is not a canonical managed project directory",
                    dir.display()
                );
            };
            let Some(encoded) = part.to_str() else {
                bail!(
                    "{} is not a canonical managed project directory",
                    dir.display()
                );
            };
            components.push(decode_component(encoded)?);
        }
        if components.is_empty() {
            bail!("{} is not a managed project directory", dir.display());
        }
        Ok(components.join("/"))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectDb {
    pub path: PathBuf,
    pub project_id: String,
}

/// Safe-set encoding for one filesystem component. Bytes outside
/// `[A-Za-z0-9_]` — including `-`, `.`, native separators, and all non-ASCII
/// bytes — are escaped as `-XX` (uppercase hex). The `-` escape character is
/// itself always escaped, so a lone `-` is unambiguously reserved for the
/// empty component.
pub fn encode_component(component: &str) -> String {
    if component.is_empty() {
        return "-".to_owned();
    }
    let mut encoded = String::with_capacity(component.len());
    for byte in component.bytes() {
        if is_safe_byte(byte) {
            encoded.push(byte as char);
        } else {
            encoded.push('-');
            encoded.push(hex_digit(byte >> 4));
            encoded.push(hex_digit(byte & 0x0F));
        }
    }
    encoded
}

/// Inverse of [`encode_component`]. Accepts only canonical encodings: every
/// `-` must be followed by two uppercase hex digits, except the single `-`
/// that marks an empty component.
pub fn decode_component(encoded: &str) -> Result<String> {
    if encoded == "-" {
        return Ok(String::new());
    }
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if is_safe_byte(byte) {
            decoded.push(byte);
            index += 1;
            continue;
        }
        if byte != b'-' || index + 2 >= bytes.len() {
            bail!("invalid encoded path component: {encoded}");
        }
        let high = hex_value(bytes[index + 1]);
        let low = hex_value(bytes[index + 2]);
        let (Some(high), Some(low)) = (high, low) else {
            bail!("invalid encoded path component: {encoded}");
        };
        decoded.push(high << 4 | low);
        index += 3;
    }
    String::from_utf8(decoded)
        .with_context(|| format!("encoded component {encoded} is not valid UTF-8"))
}

fn is_safe_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn hex_digit(nibble: u8) -> char {
    const HEX: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    ];
    HEX[nibble as usize]
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Defensive containment invariant: every resolved managed path must be a
/// strict descendant of the managed projects directory built only from normal
/// components. Encoding already guarantees this; verification catches
/// regressions.
fn verify_contained(projects_dir: &Path, path: &Path) -> Result<()> {
    let relative = path
        .strip_prefix(projects_dir)
        .with_context(|| format!("{} escapes the managed projects directory", path.display()))?;
    if relative.components().next().is_none() {
        bail!(
            "{} is the managed projects directory itself",
            path.display()
        );
    }
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("{} escapes the managed projects directory", path.display());
        }
    }
    Ok(())
}

impl ManagedLayout {
    /// Every existing managed project database, in deterministic order.
    /// Only canonical encodings are accepted; unrecognizable directories are
    /// skipped rather than failing the whole scan (they may be stale staging
    /// or foreign files, and `storage status` reports them separately).
    pub fn existing_project_dbs(&self) -> Result<Vec<ProjectDb>> {
        let projects = self.projects_dir();
        if !projects.is_dir() {
            return Ok(Vec::new());
        }
        let mut dbs = Vec::new();
        collect_project_dbs(self, &projects, &mut dbs)?;
        dbs.sort_by(|left, right| left.project_id.cmp(&right.project_id));
        Ok(dbs)
    }
}

fn collect_project_dbs(layout: &ManagedLayout, dir: &Path, dbs: &mut Vec<ProjectDb>) -> Result<()> {
    let entries = fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_project_dbs(layout, &path, dbs)?;
        } else {
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name != PROJECT_DB_FILENAME {
                continue;
            }
            let Some(parent) = path.parent() else {
                continue;
            };
            if let Ok(project_id) = layout.decode_project_dir(parent) {
                dbs.push(ProjectDb { path, project_id });
            }
        }
    }
    Ok(())
}

/// One store paired with the scope filter that selects exactly its share of
/// a query scope. Routing construction guarantees a project database holds
/// only that project's rows and the user database only global rows, so the
/// pair (store, filter) covers the store's whole active set.
pub struct ScopedStore {
    pub store: Store,
    pub project_filter: Option<String>,
}

/// One command's store routing. Exact-file operation (`--db`/`MEM_DB`, and
/// the not-yet-activated default) pins every operation to one database and
/// keeps today's single-store semantics. Managed operation routes writes to
/// the owning database and reads across the visible scope.
pub enum StorageRouter {
    Exact(PathBuf),
    Managed(ManagedLayout),
}

impl StorageRouter {
    pub fn exact(path: PathBuf) -> Self {
        Self::Exact(path)
    }

    pub fn managed() -> Result<Self> {
        Ok(Self::Managed(ManagedLayout::resolve()?))
    }

    pub fn managed_at(home: PathBuf) -> Self {
        Self::Managed(ManagedLayout::at(home))
    }

    /// The managed layout this router uses. Exact-file routers have none.
    pub fn managed_layout(&self) -> &ManagedLayout {
        match self {
            Self::Managed(layout) => layout,
            Self::Exact(_) => panic!("managed_layout on an exact-file router"),
        }
    }

    /// The single store a canonical write must go to, created on demand.
    pub fn write_store(&self, project_id: Option<&str>) -> Result<Store> {
        match self {
            Self::Exact(path) => Store::open(path),
            Self::Managed(layout) => match project_id {
                Some(project) => Store::open(&layout.project_db(project)?.path),
                None => Store::open(&layout.user_db()),
            },
        }
    }

    /// Stores for an explicitly selected scope. `None` selects the user
    /// store only (global scope). Project stores carry their own project as
    /// the query filter; the user store carries the global filter.
    pub fn scope_stores(&self, project_id: Option<&str>) -> Result<Vec<ScopedStore>> {
        match self {
            Self::Exact(path) => Ok(vec![ScopedStore {
                store: Store::open(path)?,
                project_filter: project_id.map(str::to_owned),
            }]),
            Self::Managed(layout) => {
                let mut scoped = Vec::new();
                if let Some(project) = project_id
                    && let Some(store) = open_if_exists(&layout.project_db(project)?.path)?
                {
                    scoped.push(ScopedStore {
                        store,
                        project_filter: Some(project.to_owned()),
                    });
                }
                if let Some(store) = open_if_exists(&layout.user_db())? {
                    scoped.push(ScopedStore {
                        store,
                        project_filter: None,
                    });
                }
                Ok(scoped)
            }
        }
    }

    /// Stores searched for ID-directed operations: the hint project and the
    /// user store first, then every existing managed project database.
    fn id_search_stores(&self, project_hint: Option<&str>) -> Result<Vec<Store>> {
        match self {
            Self::Exact(path) => Ok(vec![Store::open(path)?]),
            Self::Managed(layout) => {
                let mut stores = Vec::new();
                if let Some(project_id) = project_hint
                    && let Some(store) = open_if_exists(&layout.project_db(project_id)?.path)?
                {
                    stores.push(store);
                }
                if let Some(store) = open_if_exists(&layout.user_db())? {
                    stores.push(store);
                }
                for project_db in layout.existing_project_dbs()? {
                    if stores
                        .iter()
                        .any(|store| store.path() == project_db.path.to_str())
                    {
                        continue;
                    }
                    if let Some(store) = open_if_exists(&project_db.path)? {
                        stores.push(store);
                    }
                }
                Ok(stores)
            }
        }
    }
}

fn open_if_exists(path: &Path) -> Result<Option<Store>> {
    Store::open_existing(path)
}

/// Resolve a memory ID or prefix across the search set. An exact ID resolves
/// immediately (exact IDs must resolve once globally); otherwise the union of
/// prefix matches must be unique or the resolution fails as ambiguous.
pub fn routed_resolve_memory(
    router: &StorageRouter,
    project_hint: Option<&str>,
    id_or_prefix: &str,
) -> Result<(Store, String)> {
    let candidate = id_or_prefix.trim();
    if candidate.is_empty() {
        bail!("memory ID cannot be empty");
    }
    let stores = router.id_search_stores(project_hint)?;
    let mut prefix_matches: Vec<(usize, String)> = Vec::new();
    let mut exact_match: Option<(usize, String)> = None;
    for (index, store) in stores.iter().enumerate() {
        for id in store.memory_id_candidates(candidate)? {
            if id == candidate {
                exact_match = Some((index, id));
            } else {
                prefix_matches.push((index, id));
            }
        }
    }
    resolve_candidate(stores, exact_match, prefix_matches, "memory", candidate)
}

/// Resolve an episode ID or prefix across the search set.
pub fn routed_resolve_episode(
    router: &StorageRouter,
    project_hint: Option<&str>,
    id_or_prefix: &str,
) -> Result<(Store, String)> {
    let candidate = id_or_prefix.trim();
    if candidate.is_empty() {
        bail!("episode ID cannot be empty");
    }
    let stores = router.id_search_stores(project_hint)?;
    let mut prefix_matches: Vec<(usize, String)> = Vec::new();
    let mut exact_match: Option<(usize, String)> = None;
    for (index, store) in stores.iter().enumerate() {
        for id in store.episode_id_candidates(candidate)? {
            if id == candidate {
                exact_match = Some((index, id));
            } else {
                prefix_matches.push((index, id));
            }
        }
    }
    resolve_candidate(stores, exact_match, prefix_matches, "episode", candidate)
}

fn resolve_candidate(
    stores: Vec<Store>,
    exact_match: Option<(usize, String)>,
    prefix_matches: Vec<(usize, String)>,
    kind: &str,
    candidate: &str,
) -> Result<(Store, String)> {
    if let Some((index, id)) = exact_match {
        let store = stores
            .into_iter()
            .nth(index)
            .with_context(|| format!("{kind} store vanished during resolution"))?;
        return Ok((store, id));
    }
    match prefix_matches.as_slice() {
        [] => bail!("{kind} not found: {candidate}"),
        [(index, id)] => {
            let store = stores
                .into_iter()
                .nth(*index)
                .with_context(|| format!("{kind} store vanished during resolution"))?;
            Ok((store, id.clone()))
        }
        _ => bail!("ambiguous {kind} ID prefix: {candidate}"),
    }
}

/// Routed recall. Exact-file operation keeps single-store semantics
/// unchanged (same `recall_hits` path the CLI used before routing). Managed
/// operation selects the scope's stores, applies the complete-coverage
/// semantic gate across the whole selected scope, and falls back to
/// deterministic lexical rank interleave otherwise.
pub fn routed_recall_hits(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
    force_lexical: bool,
) -> Result<Vec<SearchHit>> {
    if let StorageRouter::Exact(path) = router {
        let store = Store::open(path)?;
        return crate::recall_hits(&store, query, project_id, limit, force_lexical);
    }

    let stores = router.scope_stores(project_id)?;
    if stores.is_empty() {
        return Ok(Vec::new());
    }

    // Each scoped store's whole active set is exactly its share of the
    // selected scope, so the complete-coverage gate applies per store with
    // that store's own filter.
    if !force_lexical
        && stores
            .iter()
            .map(|scoped| {
                scoped.store.has_complete_scope_coverage(
                    crate::embedding_worker::EMBEDDING_MODEL_ID,
                    scoped.project_filter.as_deref(),
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .all(|covered| covered)
        && let Some(query_vector) = cached_query_vector(query)?
    {
        let mut hits = Vec::new();
        for scoped in &stores {
            hits.extend(
                scoped
                    .store
                    .semantic_search_by_vector(
                        &query_vector,
                        crate::embedding_worker::EMBEDDING_MODEL_ID,
                        scoped.project_filter.as_deref(),
                        limit,
                    )?
                    .into_iter()
                    .map(|hit| SearchHit {
                        memory: hit.memory,
                        rank: hit.score,
                    }),
            );
        }
        sort_and_truncate(&mut hits, limit);
        return Ok(hits);
    }

    // Lexical fallback: deterministic rank interleave across stores. Each
    // store's FTS ranking is only corpus-local, so merging by rank keeps the
    // merge deterministic without pretending scores are globally comparable.
    // Project stores come first, so project hits win equal-rank ties.
    let mut per_store: Vec<Vec<SearchHit>> = Vec::with_capacity(stores.len());
    for scoped in &stores {
        per_store.push(
            scoped
                .store
                .recall(query, scoped.project_filter.as_deref(), limit)?,
        );
    }
    Ok(interleave_hits(per_store, limit))
}

/// Routed explicit lexical search (`mem search`): bm25-ranked OR matching
/// per store, merged by the same deterministic interleave. Full term
/// matches rank above partial ones, so ranking carries the precision
/// instead of discarding partial matches.
pub fn routed_lexical_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchHit>> {
    if let StorageRouter::Exact(path) = router {
        let store = Store::open(path)?;
        return store.search(query, project_id, limit);
    }
    let stores = router.scope_stores(project_id)?;
    let mut per_store: Vec<Vec<SearchHit>> = Vec::with_capacity(stores.len());
    for scoped in &stores {
        per_store.push(
            scoped
                .store
                .search(query, scoped.project_filter.as_deref(), limit)?,
        );
    }
    Ok(interleave_hits(per_store, limit))
}

/// Routed explicit semantic search (`mem search --semantic`): the query
/// vector is embedded once, every selected store is scanned with its own
/// filter, and hits merge globally by score with the deterministic tie
/// policy. Explicit semantic search has no coverage gate — the gate belongs
/// to `context` recall.
pub fn routed_semantic_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<SemanticSearchHit>> {
    if let StorageRouter::Exact(path) = router {
        let store = Store::open(path)?;
        return store.semantic_search_by_vector(
            query_vector,
            crate::embedding_worker::EMBEDDING_MODEL_ID,
            project_id,
            limit,
        );
    }
    let stores = router.scope_stores(project_id)?;
    let mut hits = Vec::new();
    for scoped in &stores {
        hits.extend(scoped.store.semantic_search_by_vector(
            query_vector,
            crate::embedding_worker::EMBEDDING_MODEL_ID,
            scoped.project_filter.as_deref(),
            limit,
        )?);
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

/// Routed history search. History never merges across stores: a project
/// scope searches that project's database, `--global` searches the user
/// database.
pub fn routed_history_search(
    router: &StorageRouter,
    project_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<HistoryHit>> {
    let store = match (router, project_id) {
        (StorageRouter::Exact(path), _) => Store::open(path)?,
        (StorageRouter::Managed(layout), Some(project)) => {
            let path = layout.project_db(project)?.path;
            let Some(store) = open_if_exists(&path)? else {
                return Ok(Vec::new());
            };
            store
        }
        (StorageRouter::Managed(layout), None) => {
            let Some(store) = open_if_exists(&layout.user_db())? else {
                return Ok(Vec::new());
            };
            store
        }
    };
    store.history_search(query, project_id, limit)
}

/// Routed continuation-state read. State lives in the project database in
/// managed mode; exact mode keeps single-store behavior.
pub fn routed_workspace_state(
    router: &StorageRouter,
    project_id: &str,
    workspace_id: &str,
) -> Result<Option<WorkspaceState>> {
    match router {
        StorageRouter::Exact(path) => {
            let store = Store::open(path)?;
            store.workspace_state(project_id, workspace_id)
        }
        StorageRouter::Managed(layout) => {
            let path = layout.project_db(project_id)?.path;
            match open_if_exists(&path)? {
                Some(store) => store.workspace_state(project_id, workspace_id),
                None => Ok(None),
            }
        }
    }
}

fn sort_and_truncate(hits: &mut Vec<SearchHit>, limit: usize) {
    hits.sort_by(|left, right| {
        right
            .rank
            .total_cmp(&left.rank)
            .then_with(|| right.memory.updated_at.cmp(&left.memory.updated_at))
            .then_with(|| left.memory.id.cmp(&right.memory.id))
    });
    hits.truncate(limit);
}

/// Deterministic rank interleave: take the next unseen hit from each store in
/// order each cycle; project stores come before the user store, so project
/// hits win equal-rank ties. Truncate once, after the merge.
fn interleave_hits(per_store: Vec<Vec<SearchHit>>, limit: usize) -> Vec<SearchHit> {
    let mut merged = Vec::with_capacity(limit.min(per_store.iter().map(Vec::len).sum()));
    let mut indices = vec![0usize; per_store.len()];
    while merged.len() < limit {
        let mut advanced = false;
        for (store_index, hits) in per_store.iter().enumerate() {
            let index = indices[store_index];
            if index < hits.len() {
                merged.push(hits[index].clone());
                indices[store_index] += 1;
                advanced = true;
                if merged.len() == limit {
                    break;
                }
            }
        }
        if !advanced {
            break;
        }
    }
    merged
}

fn cached_query_vector(query: &str) -> Result<Option<Vec<f32>>> {
    let Ok(cache_dir) = crate::embedding_worker::model_cache_dir() else {
        return Ok(None);
    };
    // Fail open: a cached-but-broken model must not break adapter recall.
    Ok(crate::embedding_worker::embed_query_if_cached(query, &cache_dir).unwrap_or(None))
}

/// Aggregate of `EmbeddingRunStats` across the stores one routed index run
/// covered, so `index run` reports one object regardless of store count.
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

impl StorageRouter {
    /// Aggregate `index run` across the selected stores. Runs proceed store
    /// by store; a failure on one store aborts the run (leases already
    /// claimed elsewhere keep running until expiry, which matches how an
    /// interrupted single-store run behaves today).
    fn run_index_across(
        &self,
        stores: Vec<PathBuf>,
        options: EmbeddingRunOptions,
    ) -> Result<RoutedRunStats> {
        let mut stats = RoutedRunStats {
            model: EMBEDDING_MODEL_ID,
            cache_dir: options.cache_dir.display().to_string(),
            stores: stores.len(),
            claimed: 0,
            eligible: 0,
            committed: 0,
            stale: 0,
            retried: 0,
        };
        if stores.is_empty() {
            return Ok(stats);
        }
        for path in &stores {
            if !path.is_file() {
                continue;
            }
            let Some(mut store) = Store::open_existing(path)? else {
                continue;
            };
            let run = store.run_embedding_jobs(options.clone())?;
            stats.claimed += run.claimed;
            stats.eligible += run.eligible;
            stats.committed += run.committed;
            stats.stale += run.stale;
            stats.retried += run.retried;
        }
        Ok(stats)
    }

    /// The databases a managed `index run` covers by default: the current
    /// project's database and the user database. Normal runs must not scan
    /// every project; `--all` exists for that maintenance mode.
    fn managed_index_store_paths(&self, project_id: Option<&str>) -> Result<Vec<PathBuf>> {
        match self {
            Self::Exact(path) => Ok(vec![path.clone()]),
            Self::Managed(layout) => {
                let mut paths = Vec::new();
                if let Some(project) = project_id {
                    paths.push(layout.project_db(project)?.path);
                }
                paths.push(layout.user_db());
                Ok(paths)
            }
        }
    }

    /// All managed databases for `index run --all`: user plus every existing
    /// project database.
    fn all_managed_store_paths(&self) -> Result<Vec<PathBuf>> {
        match self {
            Self::Exact(path) => Ok(vec![path.clone()]),
            Self::Managed(layout) => {
                let mut paths = vec![layout.user_db()];
                paths.extend(
                    layout
                        .existing_project_dbs()?
                        .into_iter()
                        .map(|ProjectDb { path, .. }| path),
                );
                Ok(paths)
            }
        }
    }
}

/// `index run` through the router. Managed default covers the current
/// project database plus the user database; `all` additionally covers every
/// existing managed project database. Stores that do not exist are skipped:
/// indexing must never create a database.
pub fn routed_run_index(
    router: &StorageRouter,
    project_id: Option<&str>,
    all: bool,
    options: EmbeddingRunOptions,
) -> Result<RoutedRunStats> {
    let paths = if all {
        router.all_managed_store_paths()?
    } else {
        router.managed_index_store_paths(project_id)?
    };
    router.run_index_across(paths, options)
}

/// `index status` through the router: pending/running counts aggregated over
/// the same stores a normal `index run` would cover.
pub fn routed_index_stats(
    router: &StorageRouter,
    project_id: Option<&str>,
) -> Result<IndexJobStats> {
    let paths = router.managed_index_store_paths(project_id)?;
    let mut stats = IndexJobStats {
        pending: 0,
        running: 0,
    };
    for path in paths {
        if !path.is_file() {
            continue;
        }
        let Some(store) = Store::open_existing(&path)? else {
            continue;
        };
        let run = store.index_job_stats()?;
        stats.pending += run.pending;
        stats.running += run.running;
    }
    Ok(stats)
}

/// Read-only store inventory for `storage status`. Opening a store for the
/// inventory must never create the file: absent DBs are reported as missing,
/// not opened.
#[derive(Debug, Serialize)]
pub struct StoreInventory {
    pub path: String,
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memories: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_memories: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episodes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_jobs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub running_jobs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Managed-layout inventory for `storage status`. Pure read: no file is
/// created or modified.
#[derive(Debug, Serialize)]
pub struct StorageStatusReport {
    pub managed_root: String,
    pub layout_version: &'static str,
    pub layout_dir: String,
    pub layout_exists: bool,
    pub legacy_db: String,
    pub legacy_db_exists: bool,
    pub migration_needed: bool,
    pub user_store: StoreInventory,
    pub project_stores: Vec<ProjectStoreInventory>,
}

#[derive(Debug, Serialize)]
pub struct ProjectStoreInventory {
    pub project_id: String,
    pub path: String,
    #[serde(flatten)]
    pub inventory: StoreInventory,
}

fn inventory_store(path: &Path) -> StoreInventory {
    let exists = path.is_file();
    let mut inventory = StoreInventory {
        path: path.display().to_string(),
        exists,
        schema_version: None,
        memories: None,
        active_memories: None,
        episodes: None,
        pending_jobs: None,
        running_jobs: None,
        error: None,
    };
    if !exists {
        return inventory;
    }
    let report = |error: anyhow::Error, mut failed: StoreInventory| {
        failed.error = Some(format!("{error:#}"));
        failed
    };
    let Ok(connection) = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY) else {
        return report(anyhow::anyhow!("open failed"), inventory);
    };
    let schema_version =
        match connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0)) {
            Ok(version) => version,
            Err(error) => return report(error.into(), inventory),
        };
    let counts = || -> rusqlite::Result<(i64, i64, i64, i64, i64)> {
        Ok((
            connection.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?,
            connection.query_row(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'",
                [],
                |row| row.get(0),
            )?,
            connection.query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?,
            connection.query_row(
                "SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'",
                [],
                |row| row.get(0),
            )?,
            connection.query_row(
                "SELECT COUNT(*) FROM index_jobs WHERE state = 'running'",
                [],
                |row| row.get(0),
            )?,
        ))
    };
    let Ok((memories, active_memories, episodes, pending_jobs, running_jobs)) = counts() else {
        return report(anyhow::anyhow!("inventory query failed"), inventory);
    };
    inventory.schema_version = Some(schema_version);
    inventory.memories = Some(memories.unsigned_abs());
    inventory.active_memories = Some(active_memories.unsigned_abs());
    inventory.episodes = Some(episodes.unsigned_abs());
    inventory.pending_jobs = Some(pending_jobs.unsigned_abs());
    inventory.running_jobs = Some(running_jobs.unsigned_abs());
    inventory
}

pub fn storage_status_report(layout: &ManagedLayout) -> Result<StorageStatusReport> {
    let user_db = layout.user_db();
    let layout_dir = layout.layout_dir();
    let legacy_db = layout.legacy_db();
    let project_dbs = layout.existing_project_dbs()?;
    let project_stores = project_dbs
        .into_iter()
        .map(|ProjectDb { path, project_id }| ProjectStoreInventory {
            project_id,
            path: path.display().to_string(),
            inventory: inventory_store(&path),
        })
        .collect();
    let user_store = inventory_store(&user_db);
    Ok(StorageStatusReport {
        managed_root: layout.root().display().to_string(),
        layout_version: LAYOUT_VERSION_DIR,
        layout_dir: layout_dir.display().to_string(),
        layout_exists: layout_dir.is_dir(),
        legacy_db: legacy_db.display().to_string(),
        legacy_db_exists: legacy_db.is_file(),
        migration_needed: legacy_db.is_file() && !layout_dir.is_dir(),
        user_store,
        project_stores,
    })
}

#[cfg(test)]
mod tests {
    use super::{ManagedLayout, decode_component, encode_component};

    #[test]
    fn encoding_covers_hostile_and_unusual_inputs() {
        assert_eq!(encode_component("github.com"), "github-2Ecom");
        assert_eq!(encode_component("."), "-2E");
        assert_eq!(encode_component(".."), "-2E-2E");
        assert_eq!(
            encode_component("nijaru"),
            "nijaru",
            "safe bytes pass through unchanged"
        );
        // Escape character and separators are always escaped.
        assert_eq!(encode_component("-"), "-2D");
        assert_eq!(encode_component("_"), "_");
        assert_eq!(encode_component("a\\..\\b"), "a-5C-2E-2E-5Cb");
        assert_eq!(encode_component("C:\\Users"), "C-3A-5CUsers");
        assert_eq!(encode_component("café"), "caf-C3-A9");
        // Empty component has an explicit, unambiguous encoding.
        assert_eq!(encode_component(""), "-");
    }

    #[test]
    fn encoding_is_injective_on_reserved_sequences() {
        // A lone "-" can only be the empty component; a real dash or NUL
        // byte always carries its hex payload.
        assert_ne!(encode_component(""), encode_component("-"));
        assert_ne!(encode_component(""), encode_component("\0"));
        assert_eq!(encode_component("\0"), "-00");
        // Long components stay linear and deterministic.
        let long = "x".repeat(300);
        assert_eq!(encode_component(&long), long);
        let hostile = "\u{1F600}".repeat(100);
        assert_eq!(
            encode_component(&hostile).matches('-').count(),
            hostile.chars().count() * 4 // 4 bytes per emoji, each escaped
        );
    }

    #[test]
    fn decoding_round_trips_and_rejects_non_canonical_forms() {
        for value in [
            "github.com",
            ".",
            "..",
            "-",
            "",
            "a\\..\\b",
            "C:\\Users",
            "café",
            "\0",
            "a-5Cb",
        ] {
            let encoded = encode_component(value);
            assert_eq!(decode_component(&encoded).expect("decode"), value);
        }
        // Lowercase hex, truncated escapes, and raw unsafe bytes are not
        // canonical encodings.
        assert!(decode_component("a-2eb").is_err());
        assert!(decode_component("a-2").is_err());
        assert!(decode_component("a.b").is_err());
        assert!(decode_component("a-2E-").is_err());
    }

    #[test]
    fn project_paths_stay_contained_for_hostile_ids() {
        let layout = ManagedLayout::at(std::path::PathBuf::from("/data/mem"));
        let projects = layout.projects_dir();

        for hostile in [
            "../../outside",
            "/absolute",
            "a//b",
            "a\\..\\b",
            "C:\\Users\\nick",
            "..",
            ".",
            "café/ünïcode",
        ] {
            let project_db = layout
                .project_db(hostile)
                .expect("hostile id must still resolve");
            assert!(
                project_db.path.starts_with(&projects),
                "{} escaped the projects directory",
                project_db.path.display()
            );
            let relative = project_db.path.strip_prefix(&projects).expect("prefix");
            for component in relative.components() {
                assert!(
                    matches!(component, std::path::Component::Normal(_)),
                    "non-normal component in {}",
                    project_db.path.display()
                );
            }
            let dir = project_db
                .path
                .parent()
                .expect("project db has a parent directory");
            assert_eq!(layout.decode_project_dir(dir).expect("decode"), hostile);
        }

        assert!(layout.project_db("  ").is_err());
    }

    #[test]
    fn layout_paths_have_the_reviewed_shape() {
        let layout = ManagedLayout::at(std::path::PathBuf::from("/data/mem"));
        assert_eq!(
            layout.legacy_db(),
            std::path::PathBuf::from("/data/mem/memory.db")
        );
        assert_eq!(
            layout.user_db(),
            std::path::PathBuf::from("/data/mem/layout-v1/user.db")
        );
        let project = layout
            .project_db("github.com/nijaru/mem")
            .expect("resolve project");
        assert_eq!(
            project.path,
            std::path::PathBuf::from("/data/mem/layout-v1/projects/github-2Ecom/nijaru/mem/mem.db")
        );
    }
}

#[cfg(test)]
mod routing_tests {
    use super::StorageRouter;
    use crate::embedding_worker::EmbeddingRunOptions;
    use crate::store::NewMemory;

    fn test_home() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("mem-storage-routing-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&path).expect("create test home");
        path
    }

    fn remember(store: &mut crate::store::Store, text: &str, project_id: Option<&str>) {
        store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                project_id: project_id.map(str::to_owned),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
    }

    fn recall_texts(router: &StorageRouter, project: Option<&str>, query: &str) -> Vec<String> {
        super::routed_recall_hits(router, project, query, 10, false)
            .expect("routed recall")
            .into_iter()
            .map(|hit| hit.memory.text)
            .collect()
    }

    #[test]
    fn managed_routing_splits_and_merges_scopes() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        // Global memory lands in the user store; project memory in the
        // project database. Reads inside the project see both.
        let mut user = router.write_store(None).expect("open user store");
        remember(&mut user, "global fact about routing", None);
        drop(user);
        let mut project = router
            .write_store(Some("github.com/nijaru/mem"))
            .expect("open project store");
        remember(
            &mut project,
            "project fact about routing",
            Some("github.com/nijaru/mem"),
        );
        drop(project);

        let router = StorageRouter::managed_at(home.clone());
        let mut texts = recall_texts(&router, Some("github.com/nijaru/mem"), "routing");
        texts.sort();
        assert_eq!(
            texts,
            vec![
                "global fact about routing".to_owned(),
                "project fact about routing".to_owned(),
            ]
        );

        // Global scope sees only the user store.
        let global_router = StorageRouter::managed_at(home.clone());
        assert_eq!(
            recall_texts(&global_router, None, "routing"),
            vec!["global fact about routing".to_owned()]
        );

        // Another project's scope includes global memories but never the
        // first project's own memories.
        let other = StorageRouter::managed_at(home.clone());
        assert_eq!(
            recall_texts(&other, Some("github.com/nijaru/other"), "routing"),
            vec!["global fact about routing".to_owned()]
        );

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn reads_never_create_managed_databases() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());
        assert_eq!(
            recall_texts(&router, Some("github.com/nijaru/absent"), "anything"),
            Vec::<String>::new()
        );
        assert!(
            !home
                .join("layout-v1/projects/github-2Ecom/nijaru/absent/mem.db")
                .exists()
        );
        assert!(!home.join("layout-v1/user.db").exists());

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn lexical_interleave_project_store_wins_ties() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());
        let mut user = router.write_store(None).expect("open user store");
        remember(&mut user, "user alpha about interleave", None);
        remember(&mut user, "user beta about interleave", None);
        drop(user);
        let mut project = router
            .write_store(Some("github.com/x/y"))
            .expect("open project store");
        remember(
            &mut project,
            "project alpha about interleave",
            Some("github.com/x/y"),
        );
        drop(project);

        let router = StorageRouter::managed_at(home.clone());
        // No vectors exist, so recall falls back to lexical interleave.
        let texts = recall_texts(&router, Some("github.com/x/y"), "interleave");
        assert_eq!(texts.len(), 3, "both stores contribute");
        // The first hit is the project store's best hit: project wins ties
        // because project stores precede the user store in scope_stores.
        assert_eq!(texts[0], "project alpha about interleave");
        let remaining: Vec<&str> = texts[1..].iter().map(String::as_str).collect();
        assert!(
            remaining.contains(&"user alpha about interleave")
                && remaining.contains(&"user beta about interleave"),
            "user hits follow: {texts:?}"
        );

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn id_resolution_spans_managed_stores() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        let mut user = router.write_store(None).expect("user store");
        remember(&mut user, "user memory one", None);
        drop(user);
        let mut project = router
            .write_store(Some("github.com/x/y"))
            .expect("project store");
        remember(&mut project, "project memory one", Some("github.com/x/y"));
        drop(project);

        // Force colliding prefixes across the two stores.
        let project = router
            .write_store(Some("github.com/x/y"))
            .expect("project store");
        project
            .connection
            .execute(
                "INSERT INTO memories (id, scope, project_id, kind, text, actor, status, created_at, updated_at)\n\
                 VALUES ('collide-0000-aaaa-bbbb', 'project', 'github.com/x/y', 'fact', 'forced a', 'agent', 'active', 1, 1)",
                [],
            )
            .expect("insert forced a");
        drop(project);
        let user = router.write_store(None).expect("user store");
        user.connection
            .execute(
                "INSERT INTO memories (id, scope, project_id, kind, text, actor, status, created_at, updated_at)\n\
                 VALUES ('collide-0000-aaaa-cccc', 'global', NULL, 'fact', 'forced b', 'agent', 'active', 1, 1)",
                [],
            )
            .expect("insert forced b");
        drop(user);

        // A cross-store ambiguous prefix fails.
        let error = super::routed_resolve_memory(&router, None, "collide-0000-aaaa")
            .expect_err("cross-store prefix collision must fail");
        assert!(
            format!("{error:#}").contains("ambiguous memory ID prefix"),
            "unexpected error: {error:#}"
        );

        // The full exact ID resolves in its owning store.
        let (store, id) = super::routed_resolve_memory(&router, None, "collide-0000-aaaa-bbbb")
            .expect("resolve exact id");
        assert_eq!(id, "collide-0000-aaaa-bbbb");
        let record = store.get(&id).expect("read resolved memory");
        assert_eq!(record.memory.text, "forced a");

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn episode_id_resolution_spans_managed_stores() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        let user = router.write_store(None).expect("user store");
        user.connection
            .execute(
                "INSERT INTO episodes (id, project_id, source_type, source_ref, started_at, ended_at, summary, metadata_json, workspace_id)\n\
                 VALUES ('eeee-0000-aaaa-bbbb', NULL, 'transcript', 's1', 10, NULL, NULL, NULL, NULL)",
                [],
            )
            .expect("insert user episode");
        drop(user);
        let project = router
            .write_store(Some("github.com/x/y"))
            .expect("project store");
        project
            .connection
            .execute(
                "INSERT INTO episodes (id, project_id, source_type, source_ref, started_at, ended_at, summary, metadata_json, workspace_id)\n\
                 VALUES ('eeee-0000-aaaa-cccc', 'github.com/x/y', 'transcript', 's2', 10, NULL, NULL, NULL, NULL)",
                [],
            )
            .expect("insert project episode");
        drop(project);

        let error = super::routed_resolve_episode(&router, None, "eeee-0000-aaaa")
            .expect_err("cross-store episode prefix collision must fail");
        assert!(
            format!("{error:#}").contains("ambiguous episode ID prefix"),
            "unexpected error: {error:#}"
        );

        let (_store, id) = super::routed_resolve_episode(&router, None, "eeee-0000-aaaa-cccc")
            .expect("resolve exact episode id");
        assert_eq!(id, "eeee-0000-aaaa-cccc");

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn managed_index_run_selects_current_project_and_user_stores() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        // Current project + user stores have pending embedding jobs; another
        // project's store has one too but must NOT be touched by a scoped run.
        let mut project = router
            .write_store(Some("github.com/x/current"))
            .expect("open project store");
        remember(
            &mut project,
            "current project memory",
            Some("github.com/x/current"),
        );
        let mut user = router.write_store(None).expect("open user store");
        remember(&mut user, "user memory", None);
        let mut other = router
            .write_store(Some("github.com/x/other"))
            .expect("open other project store");
        remember(
            &mut other,
            "other project memory",
            Some("github.com/x/other"),
        );
        drop(project);
        drop(user);
        drop(other);

        // Run with cached_only against an empty cache dir: zero claims, and
        // pending counts prove the scoped stores (not the other project)
        // would be covered by a full run.
        let options = EmbeddingRunOptions {
            limit: 16,
            lease_duration: std::time::Duration::from_secs(60),
            retry_delay: std::time::Duration::from_secs(60),
            cache_dir: home.join("empty-model-cache"),
            show_download_progress: false,
            cached_only: true,
        };
        let stats = super::routed_run_index(&router, Some("github.com/x/current"), false, options)
            .expect("routed index run");
        assert_eq!(stats.stores, 2, "project + user only");
        assert_eq!(stats.claimed, 0);
        assert_eq!(stats.committed, 0);

        // index status aggregates over the same scoped stores.
        let stats = super::routed_index_stats(&router, Some("github.com/x/current"))
            .expect("routed index stats");
        assert_eq!(stats.pending, 2, "one job in each scoped store");
        assert_eq!(stats.running, 0);

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn index_run_all_covers_every_managed_store() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        for project in ["github.com/x/a", "github.com/x/b"] {
            let mut store = router
                .write_store(Some(project))
                .expect("open project store");
            remember(&mut store, "memory", Some(project));
        }
        let mut user = router.write_store(None).expect("open user store");
        remember(&mut user, "user memory", None);
        drop(user);

        let options = EmbeddingRunOptions {
            limit: 16,
            lease_duration: std::time::Duration::from_secs(60),
            retry_delay: std::time::Duration::from_secs(60),
            cache_dir: home.join("empty-model-cache"),
            show_download_progress: false,
            cached_only: true,
        };
        let stats =
            super::routed_run_index(&router, None, true, options).expect("routed all index run");
        assert_eq!(stats.stores, 3, "user + two projects");
        assert_eq!(stats.claimed, 0);

        let stats =
            super::routed_index_stats(&router, Some("github.com/x/a")).expect("routed index stats");
        // Scoped stats cover the current project + user stores.
        assert_eq!(stats.pending, 2);

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn managed_index_run_never_creates_missing_stores() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        // Only the user store exists. A scoped run referencing a project
        // whose DB was never created must not create it.
        let options = EmbeddingRunOptions {
            limit: 16,
            lease_duration: std::time::Duration::from_secs(60),
            retry_delay: std::time::Duration::from_secs(60),
            cache_dir: home.join("empty-model-cache"),
            show_download_progress: false,
            cached_only: true,
        };
        let stats =
            super::routed_run_index(&router, Some("github.com/x/never-created"), false, options)
                .expect("routed index run");
        assert_eq!(stats.stores, 2, "scoped run still counts both paths");
        assert!(
            !router
                .managed_layout()
                .project_db("github.com/x/never-created")
                .expect("resolve project db")
                .path
                .is_file(),
            "indexing must not create a project database"
        );

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn storage_status_reports_without_creating_files() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());

        let mut project = router
            .write_store(Some("github.com/x/inv"))
            .expect("open project store");
        remember(&mut project, "memory", Some("github.com/x/inv"));
        let mut user = router.write_store(None).expect("open user store");
        remember(&mut user, "user memory", None);
        drop(project);
        drop(user);

        let layout = router.managed_layout();
        let report = super::storage_status_report(layout).expect("status report");
        assert!(report.layout_exists);
        assert_eq!(report.layout_version, "layout-v1");
        assert!(!report.legacy_db_exists);
        assert!(!report.migration_needed);
        assert!(report.user_store.exists);
        assert_eq!(report.user_store.schema_version, Some(5));
        assert_eq!(report.project_stores.len(), 1);
        assert_eq!(report.project_stores[0].project_id, "github.com/x/inv");
        assert_eq!(report.project_stores[0].inventory.active_memories, Some(1));
        assert_eq!(report.project_stores[0].inventory.pending_jobs, Some(1));

        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn storage_status_flags_legacy_migration_needed() {
        let home = test_home();
        let layout = crate::storage::ManagedLayout::at(home.clone());
        // Legacy DB present, layout absent → migration_needed.
        let legacy = layout.legacy_db();
        std::fs::write(&legacy, b"not a real db, presence only").expect("write legacy marker");

        let report = super::storage_status_report(&layout).expect("status report");
        assert!(report.legacy_db_exists);
        assert!(!report.layout_exists);
        assert!(report.migration_needed);
        assert!(!report.user_store.exists);
        assert!(report.project_stores.is_empty());
        // Inventory must not fail on the non-database legacy file.
        assert!(report.user_store.error.is_none());

        std::fs::remove_dir_all(&home).expect("cleanup");
    }
}
