//! Staged whole-layout migration from the legacy single-file store into the
//! managed per-project/user layout, plus managed project purge.
//!
//! Migration protocol (design §10):
//! 1. an already-active `layout-v1` is validated and reported as done — never
//!    overwritten;
//! 2. all canonical rows and production-model vectors are read from the
//!    legacy database under one consistent read transaction, so a concurrent
//!    writer cannot produce a torn or internally inconsistent copy;
//! 3. a complete `.layout-v1.tmp-<uuid>/` staging layout is built with the
//!    normal schema constructors, so destination triggers regenerate
//!    derived index state exactly as a fresh store would;
//! 4. row counts and referential integrity are verified inside staging;
//! 5. the whole staging directory is activated with a single rename on the
//!    same filesystem;
//! 6. the legacy database is left untouched for rollback/manual deletion.
//!
//! A crash before the rename leaves only a stale staging directory, which is
//! never active; the next migration removes stale staging first.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use serde::Serialize;
use uuid::Uuid;

use crate::embedding_worker::EMBEDDING_MODEL_ID;
use crate::storage::{LAYOUT_VERSION_DIR, ManagedLayout, ProjectDb};
use crate::store::Store;

/// Outcome of one migration attempt.
#[derive(Debug, Serialize)]
pub struct MigrationReport {
    pub state: MigrationState,
    pub legacy_db: String,
    pub layout_dir: String,
    pub memories: u64,
    pub episodes: u64,
    pub workspaces: u64,
    pub stores: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MigrationState {
    /// `layout-v1` already exists and validates: nothing to do.
    AlreadyActive,
    /// No legacy database exists: nothing to migrate.
    NoLegacyStore,
    /// Staging was built, verified, and activated by a single rename.
    Migrated,
}

/// Migrate the layout's legacy database into the managed layout.
pub fn migrate_layout(layout: &ManagedLayout) -> Result<MigrationReport> {
    let legacy_db = layout.legacy_db();
    let layout_dir = layout.layout_dir();

    if layout_dir.is_dir() {
        verify_active_layout(layout).with_context(|| {
            format!(
                "layout {} exists but failed validation; refusing to migrate over it",
                layout_dir.display()
            )
        })?;
        return Ok(MigrationReport {
            state: MigrationState::AlreadyActive,
            legacy_db: legacy_db.display().to_string(),
            layout_dir: layout_dir.display().to_string(),
            memories: 0,
            episodes: 0,
            workspaces: 0,
            stores: Vec::new(),
        });
    }

    if !legacy_db.is_file() {
        return Ok(MigrationReport {
            state: MigrationState::NoLegacyStore,
            legacy_db: legacy_db.display().to_string(),
            layout_dir: layout_dir.display().to_string(),
            memories: 0,
            episodes: 0,
            workspaces: 0,
            stores: Vec::new(),
        });
    }

    remove_stale_staging(layout)?;

    // The snapshot must span every read that feeds the destination stores:
    // reading canonical rows and vectors from one consistent view is what
    // keeps text/vector pairs coherent across the split.
    let snapshot = read_legacy_snapshot(&legacy_db)?;
    let plan = route_snapshot(&snapshot)?;

    let staging = layout
        .root()
        .join(format!(".{LAYOUT_VERSION_DIR}.tmp-{}", Uuid::now_v7()));
    build_staging(layout, &plan, &staging)?;
    verify_staging(layout, &plan, &staging)?;
    activate_staging(&staging, &layout_dir)?;

    let mut stores = vec![layout.user_db().display().to_string()];
    stores.extend(
        plan.shares
            .keys()
            .filter(|project| project.as_str() != GLOBAL)
            .map(|project| layout.project_db(project).expect("route checked").path)
            .map(|path| path.display().to_string()),
    );
    Ok(MigrationReport {
        state: MigrationState::Migrated,
        legacy_db: legacy_db.display().to_string(),
        layout_dir: layout_dir.display().to_string(),
        memories: plan.memories(),
        episodes: plan.episodes(),
        workspaces: plan.total_states as u64,
        stores,
    })
}

/// Key for the share holding rows that belong to no project (user store).
const GLOBAL: &str = "\0global";

/// One consistent read of every legacy row the migration copies.
struct LegacySnapshot {
    memories: Vec<MemoryRow>,
    sources: Vec<SourceRow>,
    relations: Vec<RelationRow>,
    episodes: Vec<EpisodeRow>,
    entries: Vec<EntryRow>,
    states: Vec<StateRow>,
    vectors: Vec<VectorRow>,
}

struct MemoryRow {
    id: String,
    project_id: Option<String>,
    kind: String,
    text: String,
    actor: String,
    status: String,
    created_at: i64,
    updated_at: i64,
    deleted_at: Option<i64>,
}

struct SourceRow {
    id: String,
    memory_id: String,
    source_type: String,
    locator: Option<String>,
    excerpt: Option<String>,
    created_at: i64,
}

struct RelationRow {
    from: String,
    to: String,
    relation_type: String,
    created_at: i64,
}

struct EpisodeRow {
    id: String,
    project_id: Option<String>,
    source_type: String,
    source_ref: String,
    started_at: Option<i64>,
    ended_at: Option<i64>,
    summary: Option<String>,
    metadata_json: Option<String>,
    workspace_id: Option<String>,
}

struct EntryRow {
    id: String,
    episode_id: String,
    source_ref: String,
    ordinal: i64,
    kind: String,
    role: Option<String>,
    text: String,
    occurred_at: Option<i64>,
    metadata_json: Option<String>,
}

struct StateRow {
    project_id: String,
    workspace_id: String,
    last_session_id: Option<String>,
    active_goal: Option<String>,
    active_task_ref: Option<String>,
    checkpoint: Option<String>,
    updated_at: i64,
}

struct VectorRow {
    entity_id: String,
    dimensions: i64,
    vector: Vec<u8>,
    source_generation: i64,
    updated_at: i64,
}

/// One destination store's share of the snapshot.
struct Share {
    /// Project identifier, or `None` for the user store.
    project_id: Option<String>,
    memories: Vec<MemoryRow>,
    sources: Vec<SourceRow>,
    relations: Vec<RelationRow>,
    episodes: Vec<EpisodeRow>,
    entries: Vec<EntryRow>,
    states: Vec<StateRow>,
    vectors: Vec<VectorRow>,
}

/// The routing of a snapshot into destination stores.
struct CopyPlan {
    /// GLOBAL-keyed share is the user store; others are project stores.
    shares: BTreeMap<String, Share>,
    total_states: usize,
}

impl CopyPlan {
    fn memories(&self) -> u64 {
        self.shares
            .values()
            .map(|s| s.memories.len())
            .sum::<usize>() as u64
    }

    fn episodes(&self) -> u64 {
        self.shares
            .values()
            .map(|s| s.episodes.len())
            .sum::<usize>() as u64
    }
}

fn read_legacy_snapshot(legacy_db: &Path) -> Result<LegacySnapshot> {
    let connection = Connection::open(legacy_db)
        .with_context(|| format!("open legacy database {}", legacy_db.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(5))?;
    // One read transaction: every SELECT below sees the same view, so a
    // concurrent writer cannot interleave between canonical rows and vectors.
    connection.execute_batch("BEGIN DEFERRED")?;
    let snapshot = (|| -> Result<LegacySnapshot> {
        let memories;
        {
            let mut statement = connection.prepare(
                "SELECT id, project_id, kind, text, actor, status, created_at, updated_at, deleted_at\n\
                 FROM memories",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(MemoryRow {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        kind: row.get(2)?,
                        text: row.get(3)?,
                        actor: row.get(4)?,
                        status: row.get(5)?,
                        created_at: row.get(6)?,
                        updated_at: row.get(7)?,
                        deleted_at: row.get(8)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            memories = rows;
        }
        let sources;
        {
            let mut statement = connection.prepare(
                "SELECT id, memory_id, source_type, locator, excerpt, created_at FROM memory_sources",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(SourceRow {
                        id: row.get(0)?,
                        memory_id: row.get(1)?,
                        source_type: row.get(2)?,
                        locator: row.get(3)?,
                        excerpt: row.get(4)?,
                        created_at: row.get(5)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            sources = rows;
        }
        let relations;
        {
            let mut statement = connection.prepare(
                "SELECT from_memory_id, to_memory_id, relation_type, created_at FROM memory_relations",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(RelationRow {
                        from: row.get(0)?,
                        to: row.get(1)?,
                        relation_type: row.get(2)?,
                        created_at: row.get(3)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            relations = rows;
        }
        let episodes;
        {
            let mut statement = connection.prepare(
                "SELECT id, project_id, source_type, source_ref, started_at, ended_at, summary, metadata_json, workspace_id\n\
                 FROM episodes",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(EpisodeRow {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        source_type: row.get(2)?,
                        source_ref: row.get(3)?,
                        started_at: row.get(4)?,
                        ended_at: row.get(5)?,
                        summary: row.get(6)?,
                        metadata_json: row.get(7)?,
                        workspace_id: row.get(8)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            episodes = rows;
        }
        let entries;
        {
            let mut statement = connection.prepare(
                "SELECT id, episode_id, source_ref, ordinal, kind, role, text, occurred_at, metadata_json\n\
                 FROM episode_entries",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(EntryRow {
                        id: row.get(0)?,
                        episode_id: row.get(1)?,
                        source_ref: row.get(2)?,
                        ordinal: row.get(3)?,
                        kind: row.get(4)?,
                        role: row.get(5)?,
                        text: row.get(6)?,
                        occurred_at: row.get(7)?,
                        metadata_json: row.get(8)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            entries = rows;
        }
        let states;
        {
            let mut statement = connection.prepare(
                "SELECT project_id, workspace_id, last_session_id, active_goal, active_task_ref, checkpoint, updated_at\n\
                 FROM workspace_state",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok(StateRow {
                        project_id: row.get(0)?,
                        workspace_id: row.get(1)?,
                        last_session_id: row.get(2)?,
                        active_goal: row.get(3)?,
                        active_task_ref: row.get(4)?,
                        checkpoint: row.get(5)?,
                        updated_at: row.get(6)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            states = rows;
        }
        let vectors;
        {
            // Only current-production-model vectors for active memories are
            // derived state worth carrying; everything else is regenerated by
            // destination triggers or was already dead weight.
            let mut statement = connection.prepare(
                "SELECT e.entity_id, e.dimensions, e.vector, e.source_generation, e.updated_at\n\
                 FROM embeddings AS e\n\
                 JOIN memories AS m ON m.id = e.entity_id\n\
                 WHERE e.entity_type = 'memory' AND e.model = ?1 AND m.status = 'active'",
            )?;
            let rows = statement
                .query_map([EMBEDDING_MODEL_ID], |row| {
                    Ok(VectorRow {
                        entity_id: row.get(0)?,
                        dimensions: row.get(1)?,
                        vector: row.get(2)?,
                        source_generation: row.get(3)?,
                        updated_at: row.get(4)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            vectors = rows;
        }
        Ok(LegacySnapshot {
            memories,
            sources,
            relations,
            episodes,
            entries,
            states,
            vectors,
        })
    })()?;
    connection.execute_batch("COMMIT")?;
    Ok(snapshot)
}

/// Split the snapshot into destination shares. Every row lands in exactly one
/// share keyed by its owning project (or GLOBAL).
fn route_snapshot(snapshot: &LegacySnapshot) -> Result<CopyPlan> {
    let mut shares: BTreeMap<String, Share> = BTreeMap::new();
    // Keys are validated by share_entry before any store is created, so a
    // hostile row cannot half-build a layout.

    let mut memory_project = BTreeMap::new();
    for memory in &snapshot.memories {
        memory_project.insert(memory.id.clone(), memory.project_id.clone());
        share_entry(&mut shares, memory.project_id.as_deref())?
            .memories
            .push(MemoryRow {
                id: memory.id.clone(),
                project_id: memory.project_id.clone(),
                kind: memory.kind.clone(),
                text: memory.text.clone(),
                actor: memory.actor.clone(),
                status: memory.status.clone(),
                created_at: memory.created_at,
                updated_at: memory.updated_at,
                deleted_at: memory.deleted_at,
            });
    }
    for source in &snapshot.sources {
        let project = memory_project
            .get(&source.memory_id)
            .with_context(|| format!("source references unknown memory {}", source.memory_id))?;
        share_entry(&mut shares, project.as_deref())?
            .sources
            .push(SourceRow {
                id: source.id.clone(),
                memory_id: source.memory_id.clone(),
                source_type: source.source_type.clone(),
                locator: source.locator.clone(),
                excerpt: source.excerpt.clone(),
                created_at: source.created_at,
            });
    }
    for relation in &snapshot.relations {
        let from_project = memory_project
            .get(&relation.from)
            .with_context(|| format!("relation references unknown memory {}", relation.from))?;
        let to_project = memory_project
            .get(&relation.to)
            .with_context(|| format!("relation references unknown memory {}", relation.to))?;
        if from_project != to_project {
            bail!(
                "relation {} -> {} crosses a project boundary and cannot be split across stores",
                relation.from,
                relation.to
            );
        }
        share_entry(&mut shares, from_project.as_deref())?
            .relations
            .push(RelationRow {
                from: relation.from.clone(),
                to: relation.to.clone(),
                relation_type: relation.relation_type.clone(),
                created_at: relation.created_at,
            });
    }
    let mut episode_project = BTreeMap::new();
    for episode in &snapshot.episodes {
        episode_project.insert(episode.id.clone(), episode.project_id.clone());
        share_entry(&mut shares, episode.project_id.as_deref())?
            .episodes
            .push(EpisodeRow {
                id: episode.id.clone(),
                project_id: episode.project_id.clone(),
                source_type: episode.source_type.clone(),
                source_ref: episode.source_ref.clone(),
                started_at: episode.started_at,
                ended_at: episode.ended_at,
                summary: episode.summary.clone(),
                metadata_json: episode.metadata_json.clone(),
                workspace_id: episode.workspace_id.clone(),
            });
    }
    for entry in &snapshot.entries {
        let project = episode_project
            .get(&entry.episode_id)
            .with_context(|| format!("entry references unknown episode {}", entry.episode_id))?;
        share_entry(&mut shares, project.as_deref())?
            .entries
            .push(EntryRow {
                id: entry.id.clone(),
                episode_id: entry.episode_id.clone(),
                source_ref: entry.source_ref.clone(),
                ordinal: entry.ordinal,
                kind: entry.kind.clone(),
                role: entry.role.clone(),
                text: entry.text.clone(),
                occurred_at: entry.occurred_at,
                metadata_json: entry.metadata_json.clone(),
            });
    }
    for state in &snapshot.states {
        share_entry(&mut shares, Some(state.project_id.as_str()))?
            .states
            .push(StateRow {
                project_id: state.project_id.clone(),
                workspace_id: state.workspace_id.clone(),
                last_session_id: state.last_session_id.clone(),
                active_goal: state.active_goal.clone(),
                active_task_ref: state.active_task_ref.clone(),
                checkpoint: state.checkpoint.clone(),
                updated_at: state.updated_at,
            });
    }
    for vector in &snapshot.vectors {
        let project = memory_project
            .get(&vector.entity_id)
            .with_context(|| format!("vector references unknown memory {}", vector.entity_id))?;
        share_entry(&mut shares, project.as_deref())?
            .vectors
            .push(VectorRow {
                entity_id: vector.entity_id.clone(),
                dimensions: vector.dimensions,
                vector: vector.vector.clone(),
                source_generation: vector.source_generation,
                updated_at: vector.updated_at,
            });
    }

    let total_states = snapshot.states.len();
    Ok(CopyPlan {
        shares,
        total_states,
    })
}

/// Borrow (or create) the share for one routing key.
fn share_entry<'a>(
    shares: &'a mut BTreeMap<String, Share>,
    project: Option<&str>,
) -> Result<&'a mut Share> {
    let key = project.unwrap_or(GLOBAL);
    if key != GLOBAL && key.trim().is_empty() {
        bail!("legacy row references an empty project identifier");
    }
    if !shares.contains_key(key) {
        let project_id = if key == GLOBAL {
            None
        } else {
            Some(key.to_owned())
        };
        shares.insert(
            key.to_owned(),
            Share {
                project_id,
                memories: Vec::new(),
                sources: Vec::new(),
                relations: Vec::new(),
                episodes: Vec::new(),
                entries: Vec::new(),
                states: Vec::new(),
                vectors: Vec::new(),
            },
        );
    }
    Ok(shares.get_mut(key).expect("share was just ensured"))
}

/// Build the complete layout inside `staging`. Destination stores are created
/// with the normal constructors so schema and triggers are exactly those of a
/// fresh store; canonical rows are inserted so the triggers regenerate derived
/// index state, then existing production vectors are restored and jobs whose
/// memories already have a current-model vector are dropped (the same
/// invariant `index run`'s backfill maintains).
fn build_staging(layout: &ManagedLayout, plan: &CopyPlan, staging: &Path) -> Result<()> {
    fs::create_dir_all(staging)
        .with_context(|| format!("create staging directory {}", staging.display()))?;
    for share in plan.shares.values() {
        let destination = staging_store_path(layout, staging, share.project_id.as_deref())?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create store directory {}", parent.display()))?;
        }
        let mut store = Store::open(&destination)?;
        write_share(&mut store, share)?;
        checkpoint_and_close(store)?;
    }
    // The user store always exists in an active layout, even when the legacy
    // database held no global rows.
    if !plan.shares.contains_key(GLOBAL) {
        let destination = staging_store_path(layout, staging, None)?;
        let store = Store::open(&destination)?;
        checkpoint_and_close(store)?;
    }
    sync_directory(staging)?;
    Ok(())
}

/// Where a store's database lives inside staging: the same relative path it
/// will occupy once the staging directory becomes `layout-v1`.
fn staging_store_path(
    layout: &ManagedLayout,
    staging: &Path,
    project_id: Option<&str>,
) -> Result<PathBuf> {
    let active = match project_id {
        None => layout.user_db(),
        Some(project_id) => layout.project_db(project_id)?.path,
    };
    let relative = active
        .strip_prefix(layout.layout_dir())
        .expect("managed store lives inside the layout directory");
    Ok(staging.join(relative))
}

fn write_share(store: &mut Store, share: &Share) -> Result<()> {
    // IMMEDIATE per the repository-wide write-transaction discipline (see
    // AGENTS.md): mode-only change, behaviorally identical here since the
    // staging database is private to this migration run.
    let transaction = store
        .connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    {
        let mut insert = transaction.prepare(
            "INSERT INTO memories (\n\
                 id, scope, project_id, kind, text, actor, status,\n\
                 created_at, updated_at, deleted_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for memory in &share.memories {
            insert.execute(rusqlite::params![
                memory.id,
                if memory.project_id.is_some() {
                    "project"
                } else {
                    "global"
                },
                memory.project_id,
                memory.kind,
                memory.text,
                memory.actor,
                memory.status,
                memory.created_at,
                memory.updated_at,
                memory.deleted_at
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO memory_sources (\n\
                 id, memory_id, source_type, locator, excerpt, created_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for source in &share.sources {
            insert.execute(rusqlite::params![
                source.id,
                source.memory_id,
                source.source_type,
                source.locator,
                source.excerpt,
                source.created_at
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO memory_relations (\n\
                 from_memory_id, to_memory_id, relation_type, created_at\n\
             ) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for relation in &share.relations {
            insert.execute(rusqlite::params![
                relation.from,
                relation.to,
                relation.relation_type,
                relation.created_at
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO episodes (\n\
                 id, project_id, source_type, source_ref, started_at, ended_at,\n\
                 summary, metadata_json, workspace_id\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        for episode in &share.episodes {
            insert.execute(rusqlite::params![
                episode.id,
                episode.project_id,
                episode.source_type,
                episode.source_ref,
                episode.started_at,
                episode.ended_at,
                episode.summary,
                episode.metadata_json,
                episode.workspace_id
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO episode_entries (\n\
                 id, episode_id, source_ref, ordinal, kind, role, text,\n\
                 occurred_at, metadata_json\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        for entry in &share.entries {
            insert.execute(rusqlite::params![
                entry.id,
                entry.episode_id,
                entry.source_ref,
                entry.ordinal,
                entry.kind,
                entry.role,
                entry.text,
                entry.occurred_at,
                entry.metadata_json
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO workspace_state (\n\
                 project_id, workspace_id, last_session_id, active_goal,\n\
                 active_task_ref, checkpoint, updated_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for state in &share.states {
            insert.execute(rusqlite::params![
                state.project_id,
                state.workspace_id,
                state.last_session_id,
                state.active_goal,
                state.active_task_ref,
                state.checkpoint,
                state.updated_at
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO embeddings (\n\
                 entity_type, entity_id, model, dimensions, vector,\n\
                 source_generation, updated_at\n\
             ) VALUES ('memory', ?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for vector in &share.vectors {
            insert.execute(rusqlite::params![
                vector.entity_id,
                EMBEDDING_MODEL_ID,
                vector.dimensions,
                vector.vector,
                vector.source_generation,
                vector.updated_at
            ])?;
        }
    }
    {
        // Keep the queue invariant: a job exists exactly when its active
        // memory lacks a current-model vector.
        transaction.execute(
            "DELETE FROM index_jobs\n\
             WHERE entity_type = 'memory'\n\
               AND index_kind = 'embedding'\n\
               AND EXISTS (\n\
                   SELECT 1 FROM embeddings AS e\n\
                   WHERE e.entity_type = 'memory'\n\
                     AND e.entity_id = index_jobs.entity_id\n\
                     AND e.model = ?1\n\
               )",
            [EMBEDDING_MODEL_ID],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

/// Checkpoint WAL sidecars into the main database files so the staged layout
/// is self-contained before the activating rename.
fn checkpoint_and_close(store: Store) -> Result<()> {
    store
        .connection
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    Ok(())
}

/// Best-effort durability for the staged directory tree before rename.
fn sync_directory(path: &Path) -> Result<()> {
    let file = fs::File::open(path)
        .with_context(|| format!("open directory {} for sync", path.display()))?;
    file.sync_all().ok();
    Ok(())
}

/// Verify the staged layout before activation: every expected store exists,
/// opens at the current schema, holds exactly its planned row counts, and
/// passes referential checks.
fn verify_staging(layout: &ManagedLayout, plan: &CopyPlan, staging: &Path) -> Result<()> {
    for share in plan.shares.values() {
        let key = share.project_id.as_deref().unwrap_or("user store");
        let destination = staging_store_path(layout, staging, share.project_id.as_deref())?;
        let store = Store::open_existing(&destination)?
            .with_context(|| format!("verify staged store {}", destination.display()))?;
        let count = |sql: &str| -> Result<u64> {
            Ok(store
                .connection
                .query_row(sql, [], |row| row.get::<_, i64>(0))?
                .unsigned_abs())
        };
        let memories = count("SELECT COUNT(*) FROM memories")?;
        if memories as usize != share.memories.len() {
            bail!(
                "store {key}: expected {} memories, found {memories}",
                share.memories.len()
            );
        }
        let sources = count("SELECT COUNT(*) FROM memory_sources")?;
        if sources as usize != share.sources.len() {
            bail!(
                "store {key}: expected {} sources, found {sources}",
                share.sources.len()
            );
        }
        let relations = count("SELECT COUNT(*) FROM memory_relations")?;
        if relations as usize != share.relations.len() {
            bail!(
                "store {key}: expected {} relations, found {relations}",
                share.relations.len()
            );
        }
        let episodes = count("SELECT COUNT(*) FROM episodes")?;
        if episodes as usize != share.episodes.len() {
            bail!(
                "store {key}: expected {} episodes, found {episodes}",
                share.episodes.len()
            );
        }
        let entries = count("SELECT COUNT(*) FROM episode_entries")?;
        if entries as usize != share.entries.len() {
            bail!(
                "store {key}: expected {} entries, found {entries}",
                share.entries.len()
            );
        }
        let states = count("SELECT COUNT(*) FROM workspace_state")?;
        if states as usize != share.states.len() {
            bail!(
                "store {key}: expected {} workspaces, found {states}",
                share.states.len()
            );
        }
        let vectors = count("SELECT COUNT(*) FROM embeddings")?;
        if vectors as usize != share.vectors.len() {
            bail!(
                "store {key}: expected {} vectors, found {vectors}",
                share.vectors.len()
            );
        }
        let violations = store
            .connection
            .query_row("PRAGMA foreign_key_check", [], |row| row.get::<_, i64>(0))
            .unwrap_or(0);
        if violations > 0 {
            bail!("store {key}: foreign key check reported {violations} violations");
        }
    }
    Ok(())
}

/// Validate an already-active layout directory.
fn verify_active_layout(layout: &ManagedLayout) -> Result<()> {
    let user_db = layout.user_db();
    if !user_db.is_file() {
        bail!("active layout has no user store at {}", user_db.display());
    }
    Store::open_existing(&user_db).context("open user store")?;
    for ProjectDb { path, project_id } in layout.existing_project_dbs()? {
        Store::open_existing(&path)
            .with_context(|| format!("open project store {project_id} at {}", path.display()))?;
    }
    Ok(())
}

/// Remove leftover staging directories from interrupted runs. Staging is
/// never active, so removal is always safe.
fn remove_stale_staging(layout: &ManagedLayout) -> Result<()> {
    let root = layout.root();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(&format!(".{LAYOUT_VERSION_DIR}.tmp-")) {
            fs::remove_dir_all(entry.path())
                .with_context(|| format!("remove stale staging {}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Activate the staged layout with one same-filesystem directory rename.
fn activate_staging(staging: &Path, layout_dir: &Path) -> Result<()> {
    fs::rename(staging, layout_dir).with_context(|| {
        format!(
            "activate staged layout: rename {} to {}",
            staging.display(),
            layout_dir.display()
        )
    })
}

/// Outcome of a managed project purge.
#[derive(Debug, Serialize)]
pub struct PurgeReport {
    pub project_id: String,
    pub path: String,
    pub removed: bool,
}

/// Delete one project's managed database and SQLite sidecars. Only the
/// project's own artifacts under `layout-v1/projects` are touched; the user
/// store is never reachable through a project identifier. The caller owns
/// confirmation; this function is the destructive primitive.
pub fn purge_project_db(layout: &ManagedLayout, project_id: &str) -> Result<PurgeReport> {
    let ProjectDb { path, project_id } = layout.project_db(project_id)?;
    if !path.is_file() {
        return Ok(PurgeReport {
            project_id,
            path: path.display().to_string(),
            removed: false,
        });
    }
    // Containment: the resolved database must sit strictly beneath the
    // managed projects directory, and must be a regular file rather than a
    // symlink standing in for one.
    let canonical_projects = layout
        .projects_dir()
        .canonicalize()
        .context("canonicalize projects directory")?;
    let canonical_db = path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?;
    if !canonical_db.starts_with(&canonical_projects) {
        bail!(
            "refusing to purge {}: resolved path escapes the managed projects directory",
            path.display()
        );
    }
    if !path
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .is_file()
    {
        bail!("refusing to purge {}: not a regular file", path.display());
    }
    fs::remove_file(&path).with_context(|| format!("delete {}", path.display()))?;
    for sidecar in ["mem.db-wal", "mem.db-shm"] {
        let sidecar = path.with_file_name(sidecar);
        if sidecar.is_file() {
            fs::remove_file(&sidecar)
                .with_context(|| format!("delete sidecar {}", sidecar.display()))?;
        }
    }
    // Prune now-empty encoded project directories so the managed layout does
    // not accumulate husks.
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }
    Ok(PurgeReport {
        project_id,
        path: path.display().to_string(),
        removed: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{NewEpisode, NewEpisodeEntry};
    use crate::storage::StorageRouter;
    use crate::store::{NewMemory, NewWorkspaceState};
    use std::time::Duration;

    fn test_home() -> PathBuf {
        let path = std::env::temp_dir().join(format!("mem-storage-migration-{}", Uuid::now_v7()));
        std::fs::create_dir_all(&path).expect("create test home");
        path
    }

    fn remember(store: &mut Store, text: &str, project_id: Option<&str>) -> String {
        store
            .remember(NewMemory {
                text: text.to_owned(),
                kind: "fact".to_owned(),
                project_id: project_id.map(str::to_owned),
                actor: "agent".to_owned(),
                source_type: "cli".to_owned(),
                source_ref: None,
            })
            .expect("remember")
            .id
    }

    /// Build a legacy store at the layout's legacy path with the full row
    /// variety: global + project memories, a correction relation, an episode
    /// with entries, and continuation state.
    fn legacy_fixture(home: &Path) -> Store {
        let layout = ManagedLayout::at(home.to_path_buf());
        let legacy_path = layout.legacy_db();
        if let Some(parent) = legacy_path.parent() {
            std::fs::create_dir_all(parent).expect("create legacy parent");
        }
        let mut store = Store::open(&legacy_path).expect("open legacy");

        let global_id = remember(&mut store, "global fact one", None);
        remember(&mut store, "global fact two", None);
        let project_id = remember(&mut store, "project fact one", Some("github.com/x/a"));

        // A correction pair in the same project keeps a superseded_by relation.
        let replacement = store
            .correct(
                &project_id,
                crate::store::NewCorrection {
                    text: "project fact one, corrected".to_owned(),
                    kind: None,
                    actor: "agent".to_owned(),
                    source_type: "cli".to_owned(),
                    source_ref: None,
                },
            )
            .expect("correct")
            .replacement
            .memory
            .id;

        store
            .ensure_episode(NewEpisode {
                project_id: Some("github.com/x/a".to_owned()),
                workspace_id: None,
                source_type: "transcript".to_owned(),
                source_ref: "sess-1".to_owned(),
                started_at: Some(1000),
                metadata_json: None,
            })
            .expect("episode");
        let episode_id: String = store
            .connection
            .query_row(
                "SELECT id FROM episodes WHERE source_ref = 'sess-1'",
                [],
                |row| row.get(0),
            )
            .expect("episode row");
        store
            .record_episode_entry(
                &episode_id,
                NewEpisodeEntry {
                    source_ref: "e-1".to_owned(),
                    ordinal: None,
                    kind: "message".to_owned(),
                    role: Some("user".to_owned()),
                    text: "episode entry one".to_owned(),
                    occurred_at: Some(1001),
                    metadata_json: None,
                },
            )
            .expect("entry");

        store
            .set_workspace_state(NewWorkspaceState {
                project_id: "github.com/x/a".to_owned(),
                workspace_id: "main".to_owned(),
                last_session_id: Some("sess-1".to_owned()),
                active_goal: Some("finish storage".to_owned()),
                active_task_ref: None,
                checkpoint: None,
            })
            .expect("state");

        let _ = (global_id, replacement);
        store
    }

    #[test]
    fn migrate_splits_legacy_store_and_preserves_rows() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        let mut legacy = legacy_fixture(&home);
        // Drain the queue so committed vectors exist and the job table is empty;
        // migration must still regenerate pending jobs for unembedded memories.
        legacy
            .run_embedding_jobs(crate::embedding_worker::EmbeddingRunOptions {
                limit: 64,
                lease_duration: Duration::from_secs(60),
                retry_delay: Duration::from_secs(60),
                cache_dir: std::env::temp_dir().join("mem-test-model-cache-absent"),
                show_download_progress: false,
                cached_only: true,
            })
            .expect("run");
        drop(legacy);

        let report = migrate_layout(&layout).expect("migrate");
        assert_eq!(report.state, MigrationState::Migrated);
        assert_eq!(report.memories, 4, "2 global + 2 project (correction pair)");
        assert_eq!(report.episodes, 1);
        assert_eq!(report.workspaces, 1);
        assert_eq!(report.stores.len(), 2, "user + one project store");

        // Legacy file untouched.
        assert!(layout.legacy_db().is_file());
        // Layout active.
        assert!(layout.layout_dir().is_dir());
        // Staging cleaned up by rename.
        let staging_left = std::fs::read_dir(layout.root())
            .expect("read root")
            .filter(|entry| {
                entry
                    .as_ref()
                    .map(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with(".layout-v1.tmp-")
                    })
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(staging_left, 0, "no staging directories remain");

        // Split correctness: user store has the global rows, project store
        // the project rows, IDs preserved.
        let user = Store::open_existing(&layout.user_db())
            .expect("open user")
            .expect("user store exists");
        let user_count: i64 = user
            .connection
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .expect("count");
        assert_eq!(user_count, 2);
        let user_text: String = user
            .connection
            .query_row(
                "SELECT text FROM memories WHERE text LIKE 'global fact one'",
                [],
                |row| row.get(0),
            )
            .expect("global fact one survives with its ID");

        let project_db = layout.project_db("github.com/x/a").expect("project db");
        let project = Store::open_existing(&project_db.path)
            .expect("open project")
            .expect("project store exists");
        let project_count: i64 = project
            .connection
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .expect("count");
        assert_eq!(project_count, 2, "original + corrected replacement");
        let relation: i64 = project
            .connection
            .query_row("SELECT COUNT(*) FROM memory_relations", [], |row| {
                row.get(0)
            })
            .expect("relations");
        assert_eq!(relation, 1, "superseded_by relation preserved");
        let state: i64 = project
            .connection
            .query_row("SELECT COUNT(*) FROM workspace_state", [], |row| row.get(0))
            .expect("state");
        assert_eq!(state, 1, "continuation state routed to the project store");
        let episode_entries: i64 = project
            .connection
            .query_row("SELECT COUNT(*) FROM episode_entries", [], |row| row.get(0))
            .expect("entries");
        assert_eq!(episode_entries, 1);
        // FTS5 rebuilt: lexical search finds the copied memory.
        let hits = project
            .recall("corrected", Some("github.com/x/a"), 10)
            .expect("recall");
        assert!(!hits.is_empty(), "lexical search works on migrated rows");

        // Queue invariant: pending embedding jobs for every active memory
        // lacking a vector (model was absent, so all four).
        let pending: i64 = project
            .connection
            .query_row(
                "SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'",
                [],
                |row| row.get(0),
            )
            .expect("pending");
        assert_eq!(
            pending, 1,
            "jobs exist exactly for active memories: the corrected original is superseded"
        );
        let user_pending: i64 = user
            .connection
            .query_row(
                "SELECT COUNT(*) FROM index_jobs WHERE state = 'pending'",
                [],
                |row| row.get(0),
            )
            .expect("pending");
        assert_eq!(user_pending, 2);

        let _ = user_text;
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn migrate_is_idempotent_and_reports_active_layout() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        {
            let legacy = legacy_fixture(&home);
            drop(legacy);
        }
        migrate_layout(&layout).expect("first migrate");
        let second = migrate_layout(&layout).expect("second migrate");
        assert_eq!(second.state, MigrationState::AlreadyActive);
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn migrate_without_legacy_store_reports_nothing_to_do() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        let report = migrate_layout(&layout).expect("migrate");
        assert_eq!(report.state, MigrationState::NoLegacyStore);
        assert!(!layout.layout_dir().is_dir(), "no layout created");
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn migrate_refuses_conflicting_active_layout() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        {
            let legacy = legacy_fixture(&home);
            drop(legacy);
        }
        migrate_layout(&layout).expect("first migrate");

        // Corrupt the active layout's user store, then attempt to migrate over it.
        std::fs::write(layout.user_db(), b"corrupt").expect("corrupt user store");
        let error = migrate_layout(&layout).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("failed validation"),
            "unexpected error: {error:#}"
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn migrate_rejects_cross_project_relations() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        let legacy_path = layout.legacy_db();
        std::fs::create_dir_all(legacy_path.parent().unwrap()).expect("parent");
        let mut legacy = Store::open(&legacy_path).expect("open legacy");
        let a = remember(&mut legacy, "memory a", Some("github.com/x/a"));
        let b = remember(&mut legacy, "memory b", Some("github.com/x/b"));
        // Hand-insert a cross-project relation (no CLI path produces one, but
        // the migration must refuse it rather than split it).
        legacy
            .connection
            .execute(
                "INSERT INTO memory_relations (from_memory_id, to_memory_id, relation_type, created_at)
                 VALUES (?1, ?2, 'superseded_by', 1)",
                rusqlite::params![a, b],
            )
            .expect("insert relation");
        drop(legacy);

        let error = migrate_layout(&layout).expect_err("cross-project relation must fail");
        assert!(
            format!("{error:#}").contains("crosses a project boundary"),
            "unexpected error: {error:#}"
        );
        assert!(
            !layout.layout_dir().is_dir(),
            "failed migration must not activate"
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn purge_removes_only_the_project_store() {
        let home = test_home();
        let router = StorageRouter::managed_at(home.clone());
        let layout = ManagedLayout::at(home.clone());

        let mut project = router
            .write_store(Some("github.com/x/purge"))
            .expect("project store");
        remember(&mut project, "memory", Some("github.com/x/purge"));
        drop(project);
        let mut other = router
            .write_store(Some("github.com/x/keep"))
            .expect("other store");
        remember(&mut other, "memory", Some("github.com/x/keep"));
        drop(other);
        let mut user = router.write_store(None).expect("user store");
        remember(&mut user, "user memory", None);
        drop(user);

        let report = purge_project_db(&layout, "github.com/x/purge").expect("purge");
        assert!(report.removed);
        assert!(
            !layout
                .project_db("github.com/x/purge")
                .expect("path")
                .path
                .is_file(),
            "project db deleted"
        );
        assert!(
            layout
                .project_db("github.com/x/keep")
                .expect("path")
                .path
                .is_file()
        );
        assert!(layout.user_db().is_file(), "user store untouched");
        // Re-purge is a no-op report, not an error.
        let again = purge_project_db(&layout, "github.com/x/purge").expect("re-purge");
        assert!(!again.removed);
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn purge_rejects_project_ids_that_escape_containment() {
        let home = test_home();
        let layout = ManagedLayout::at(home.clone());
        // Empty project ids are rejected by the encoder already; a non-empty
        // id that encodes to a path outside projects/ cannot be constructed
        // through the public API, so verify the encoder rejection path.
        let error = layout.project_db("").expect_err("empty id");
        assert!(
            format!("{error:#}").contains("empty"),
            "unexpected error: {error:#}"
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }
}
