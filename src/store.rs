use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;
use uuid::Uuid;

const SCHEMA_VERSION: i64 = 5;
const MEMORY_KINDS: &[&str] = &["fact", "decision", "constraint", "preference", "procedure"];

#[derive(Debug)]
pub struct Store {
    pub(crate) connection: Connection,
}

pub struct NewMemory {
    pub text: String,
    pub kind: String,
    pub project_id: Option<String>,
    pub actor: String,
    pub source_type: String,
    pub source_ref: Option<String>,
}

pub struct NewCorrection {
    pub text: String,
    pub kind: Option<String>,
    pub actor: String,
    pub source_type: String,
    pub source_ref: Option<String>,
}

pub struct NewWorkspaceState {
    pub project_id: String,
    pub workspace_id: String,
    pub last_session_id: Option<String>,
    pub active_goal: Option<String>,
    pub active_task_ref: Option<String>,
    pub checkpoint: Option<String>,
}

/// Fields `state patch` may update. Session/goal/task/checkpoint are the
/// mutable payload; project and workspace identify the state row.
pub const PATCHABLE_FIELDS: &[&str] = &["session", "goal", "task", "checkpoint"];

/// Partial continuation-state update: only provided fields change.
#[derive(Debug, Default)]
pub struct NewWorkspaceStatePatch {
    pub project_id: String,
    pub workspace_id: String,
    pub last_session_id: Option<String>,
    pub active_goal: Option<String>,
    pub active_task_ref: Option<String>,
    pub checkpoint: Option<String>,
    /// Field names to explicitly null out.
    pub clear_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: String,
    pub scope: String,
    pub project_id: Option<String>,
    pub kind: String,
    pub text: String,
    pub actor: String,
    pub status: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub deleted_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySource {
    pub id: String,
    pub source_type: String,
    pub locator: Option<String>,
    pub excerpt: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryRelation {
    pub from_memory_id: String,
    pub to_memory_id: String,
    pub relation_type: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryRecord {
    pub memory: Memory,
    pub sources: Vec<MemorySource>,
    pub relations: Vec<MemoryRelation>,
}

#[derive(Debug, Serialize)]
pub struct CorrectionResult {
    pub previous: Memory,
    pub replacement: MemoryRecord,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub memory: Memory,
    pub rank: f64,
}

#[derive(Debug, Serialize)]
pub struct WorkspaceState {
    pub project_id: String,
    pub workspace_id: String,
    pub last_session_id: Option<String>,
    pub active_goal: Option<String>,
    pub active_task_ref: Option<String>,
    pub checkpoint: Option<String>,
    pub updated_at: i64,
}

pub struct StoreStats {
    pub schema_version: i64,
    pub total: u64,
    pub active: u64,
    pub superseded: u64,
    pub deleted: u64,
}

impl Store {
    /// The database file path backing this store, if known.
    pub fn path(&self) -> Option<&str> {
        self.connection.path()
    }

    pub fn open(path: &Path) -> Result<Self> {
        let created = !path.exists();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            let parent_created = !parent.exists();
            fs::create_dir_all(parent)
                .with_context(|| format!("create data directory {}", parent.display()))?;
            if parent_created {
                restrict_permissions(parent, 0o700)?;
            }
        }

        let connection = Connection::open(path)?;
        if created {
            restrict_permissions(path, 0o600)?;
        }
        let store = Self::connect(connection)?;
        store.migrate()?;
        Ok(store)
    }

    /// Open a store that must already exist. Read-only invocations use this
    /// so visiting a project never creates an empty database.
    pub fn open_existing(path: &Path) -> Result<Option<Self>> {
        if !path.is_file() {
            return Ok(None);
        }
        let connection = Connection::open(path)?;
        let store = Self::connect(connection)?;
        store.migrate()?;
        Ok(Some(store))
    }

    fn connect(connection: Connection) -> Result<Self> {
        connection.busy_timeout(Duration::from_secs(5))?;
        // Switching into WAL needs a brief exclusive moment that does not
        // honor the busy handler, so concurrent openers race the flip and
        // one fails instantly. Only flip when the database is not already
        // in WAL; a concurrent flip that lost the race reads back as WAL
        // and continues.
        let mode: String = connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        if mode != "wal" {
            // First open races another first open: the WAL flip needs a brief
            // exclusive moment that ignores the busy handler, so the loser
            // gets an instant "database is locked". Retry a few times — the
            // winner's flip makes the mode read back "wal" and both proceed.
            let mut flip_error = None;
            for _ in 0..20 {
                match connection.execute_batch("PRAGMA journal_mode=WAL;") {
                    Ok(()) => {
                        flip_error = None;
                        break;
                    }
                    Err(error) => {
                        let mode: String =
                            connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
                        if mode == "wal" {
                            flip_error = None;
                            break;
                        }
                        flip_error = Some(error);
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            if let Some(error) = flip_error {
                return Err(error.into());
            }
        }
        connection.execute_batch(
            "PRAGMA foreign_keys=ON;\n\
             PRAGMA synchronous=NORMAL;",
        )?;
        Ok(Self { connection })
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let schema_version = self.schema_version()?;
        let (total, active, superseded, deleted): (i64, i64, i64, i64) =
            self.connection.query_row(
                "SELECT COUNT(*),\n\
                        COALESCE(SUM(status = 'active'), 0),\n\
                        COALESCE(SUM(status = 'superseded'), 0),\n\
                        COALESCE(SUM(status = 'deleted'), 0)\n\
                 FROM memories",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;

        Ok(StoreStats {
            schema_version,
            total: total.try_into()?,
            active: active.try_into()?,
            superseded: superseded.try_into()?,
            deleted: deleted.try_into()?,
        })
    }

    pub fn remember(&mut self, input: NewMemory) -> Result<Memory> {
        validate_new_memory(&input)?;

        let id = Uuid::now_v7().to_string();
        let source_id = Uuid::now_v7().to_string();
        let now = unix_millis()?;
        let scope = if input.project_id.is_some() {
            "project"
        } else {
            "global"
        };

        let transaction = self.connection.transaction()?;
        insert_memory(&transaction, &id, scope, &input, now)?;
        insert_source(&transaction, &source_id, &id, &input, now)?;
        transaction.commit()?;

        self.memory_by_id(&id)
    }

    pub fn correct(
        &mut self,
        id_or_prefix: &str,
        input: NewCorrection,
    ) -> Result<CorrectionResult> {
        let previous_id = self.resolve_id(id_or_prefix)?;
        let previous = self.memory_by_id(&previous_id)?;
        if previous.status != "active" {
            bail!(
                "memory {} is {}; only active memory can be corrected",
                previous.id,
                previous.status
            );
        }

        let replacement_input = NewMemory {
            text: input.text,
            kind: input.kind.unwrap_or_else(|| previous.kind.clone()),
            project_id: previous.project_id.clone(),
            actor: input.actor,
            source_type: input.source_type,
            source_ref: input.source_ref,
        };
        validate_new_memory(&replacement_input)?;

        let replacement_id = Uuid::now_v7().to_string();
        let source_id = Uuid::now_v7().to_string();
        let now = unix_millis()?;
        let transaction = self.connection.transaction()?;
        insert_memory(
            &transaction,
            &replacement_id,
            &previous.scope,
            &replacement_input,
            now,
        )?;
        insert_source(
            &transaction,
            &source_id,
            &replacement_id,
            &replacement_input,
            now,
        )?;
        let changed = transaction.execute(
            "UPDATE memories\n\
             SET status = 'superseded', updated_at = ?2\n\
             WHERE id = ?1 AND status = 'active'",
            params![previous_id, now],
        )?;
        if changed != 1 {
            bail!("memory changed while correction was being applied");
        }
        transaction.execute(
            "INSERT INTO memory_relations (\n\
                 from_memory_id, to_memory_id, relation_type, created_at\n\
             ) VALUES (?1, ?2, 'superseded_by', ?3)",
            params![previous_id, replacement_id, now],
        )?;
        transaction.commit()?;

        Ok(CorrectionResult {
            previous: self.memory_by_id(&previous_id)?,
            replacement: self.get(&replacement_id)?,
        })
    }

    /// Lexical search over the memory corpus. Ranked with bm25 across all
    /// query terms (OR semantics), so partial matches stay visible and
    /// rank below full matches instead of being discarded; explicit search
    /// must never return empty while the semantic tier finds the same
    /// corpus relevant.
    pub fn search(
        &self,
        query: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_fts(&fts_query(query, " OR ")?, project_id, limit)
    }

    /// Broad lexical recall used by `mem context` when semantic ranking is
    /// unavailable or coverage is incomplete. Same OR matching as [`search`].
    pub fn recall(
        &self,
        query: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_fts(&fts_query(query, " OR ")?, project_id, limit)
    }

    pub fn get(&self, id_or_prefix: &str) -> Result<MemoryRecord> {
        let id = self.resolve_id(id_or_prefix)?;
        let memory = self.memory_by_id(&id)?;

        let mut source_statement = self.connection.prepare(
            "SELECT id, source_type, locator, excerpt, created_at\n\
             FROM memory_sources\n\
             WHERE memory_id = ?1\n\
             ORDER BY created_at, id",
        )?;
        let source_rows = source_statement.query_map([&id], |row| {
            Ok(MemorySource {
                id: row.get(0)?,
                source_type: row.get(1)?,
                locator: row.get(2)?,
                excerpt: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        let mut sources = Vec::new();
        for row in source_rows {
            sources.push(row?);
        }

        let mut relation_statement = self.connection.prepare(
            "SELECT from_memory_id, to_memory_id, relation_type, created_at\n\
             FROM memory_relations\n\
             WHERE from_memory_id = ?1 OR to_memory_id = ?1\n\
             ORDER BY created_at, from_memory_id, to_memory_id, relation_type",
        )?;
        let relation_rows = relation_statement.query_map([&id], |row| {
            Ok(MemoryRelation {
                from_memory_id: row.get(0)?,
                to_memory_id: row.get(1)?,
                relation_type: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        let mut relations = Vec::new();
        for row in relation_rows {
            relations.push(row?);
        }

        Ok(MemoryRecord {
            memory,
            sources,
            relations,
        })
    }

    pub fn forget(&self, id_or_prefix: &str) -> Result<String> {
        let id = self.resolve_id(id_or_prefix)?;
        let now = unix_millis()?;
        self.connection.execute(
            "UPDATE memories\n\
             SET status = 'deleted', deleted_at = ?2, updated_at = ?2\n\
             WHERE id = ?1",
            params![id, now],
        )?;
        Ok(id)
    }

    pub fn workspace_state(
        &self,
        project_id: &str,
        workspace_id: &str,
    ) -> Result<Option<WorkspaceState>> {
        validate_identity(project_id, "project")?;
        validate_identity(workspace_id, "workspace")?;

        Ok(self
            .connection
            .query_row(
                "SELECT project_id, workspace_id, last_session_id, active_goal,\n\
                        active_task_ref, checkpoint, updated_at\n\
                 FROM workspace_state\n\
                 WHERE project_id = ?1 AND workspace_id = ?2",
                params![project_id, workspace_id],
                workspace_state_from_row,
            )
            .optional()?)
    }

    pub fn set_workspace_state(&self, input: NewWorkspaceState) -> Result<WorkspaceState> {
        validate_identity(&input.project_id, "project")?;
        validate_identity(&input.workspace_id, "workspace")?;
        validate_optional(&input.last_session_id, "session")?;
        validate_optional(&input.active_goal, "goal")?;
        validate_optional(&input.active_task_ref, "task")?;
        validate_optional(&input.checkpoint, "checkpoint")?;

        let now = unix_millis()?;
        self.connection.execute(
            "INSERT INTO workspace_state (\n\
                 project_id, workspace_id, last_session_id, active_goal, active_task_ref,\n\
                 checkpoint, updated_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)\n\
             ON CONFLICT(project_id, workspace_id) DO UPDATE SET\n\
                 last_session_id = excluded.last_session_id,\n\
                 active_goal = excluded.active_goal,\n\
                 active_task_ref = excluded.active_task_ref,\n\
                 checkpoint = excluded.checkpoint,\n\
                 updated_at = excluded.updated_at",
            params![
                &input.project_id,
                &input.workspace_id,
                &input.last_session_id,
                &input.active_goal,
                &input.active_task_ref,
                &input.checkpoint,
                now
            ],
        )?;

        self.workspace_state(&input.project_id, &input.workspace_id)?
            .context("workspace state disappeared after write")
    }

    /// Atomically update only the provided fields of one workspace's
    /// continuation state. Absent fields keep their current values (or stay
    /// absent for a new state row); `None` in `clear_fields` explicitly
    /// nulls that field. This is the partial-update primitive adapters use
    /// instead of read-merge-write, which races concurrent writers.
    pub fn patch_workspace_state(&self, input: NewWorkspaceStatePatch) -> Result<WorkspaceState> {
        validate_identity(&input.project_id, "project")?;
        validate_identity(&input.workspace_id, "workspace")?;
        validate_optional(&input.last_session_id, "session")?;
        validate_optional(&input.active_goal, "goal")?;
        validate_optional(&input.active_task_ref, "task")?;
        validate_optional(&input.checkpoint, "checkpoint")?;
        for field in &input.clear_fields {
            if !PATCHABLE_FIELDS.contains(&field.as_str()) {
                bail!("unknown state field: {field}");
            }
        }

        let now = unix_millis()?;
        self.connection.execute(
            "INSERT INTO workspace_state (
                 project_id, workspace_id, last_session_id, active_goal, active_task_ref,
                 checkpoint, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(project_id, workspace_id) DO UPDATE SET
                 last_session_id = COALESCE(?3, CASE WHEN ?8 THEN NULL ELSE last_session_id END),
                 active_goal = COALESCE(?4, CASE WHEN ?9 THEN NULL ELSE active_goal END),
                 active_task_ref = COALESCE(?5, CASE WHEN ?10 THEN NULL ELSE active_task_ref END),
                 checkpoint = COALESCE(?6, CASE WHEN ?11 THEN NULL ELSE checkpoint END),
                 updated_at = ?7",
            params![
                &input.project_id,
                &input.workspace_id,
                &input.last_session_id,
                &input.active_goal,
                &input.active_task_ref,
                &input.checkpoint,
                now,
                input.clear_fields.iter().any(|field| field == "session"),
                input.clear_fields.iter().any(|field| field == "goal"),
                input.clear_fields.iter().any(|field| field == "task"),
                input.clear_fields.iter().any(|field| field == "checkpoint"),
            ],
        )?;

        self.workspace_state(&input.project_id, &input.workspace_id)?
            .context("workspace state disappeared after write")
    }

    pub fn clear_workspace_state(&self, project_id: &str, workspace_id: &str) -> Result<bool> {
        validate_identity(project_id, "project")?;
        validate_identity(workspace_id, "workspace")?;
        Ok(self.connection.execute(
            "DELETE FROM workspace_state WHERE project_id = ?1 AND workspace_id = ?2",
            params![project_id, workspace_id],
        )? > 0)
    }

    fn search_fts(
        &self,
        query: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        let limit = i64::try_from(limit)?;
        let mut hits = Vec::new();

        if let Some(project_id) = project_id {
            let mut statement = self.connection.prepare(
                "SELECT m.id, m.scope, m.project_id, m.kind, m.text, m.actor, m.status,\n\
                        m.created_at, m.updated_at, m.deleted_at, bm25(memories_fts)\n\
                 FROM memories_fts\n\
                 JOIN memories AS m ON m.rowid = memories_fts.rowid\n\
                 WHERE memories_fts MATCH ?1\n\
                   AND m.status = 'active'\n\
                   AND (m.project_id IS NULL OR m.project_id = ?2)\n\
                 ORDER BY bm25(memories_fts), m.updated_at DESC\n\
                 LIMIT ?3",
            )?;
            let rows =
                statement.query_map(params![query, project_id, limit], search_hit_from_row)?;
            for row in rows {
                hits.push(row?);
            }
        } else {
            let mut statement = self.connection.prepare(
                "SELECT m.id, m.scope, m.project_id, m.kind, m.text, m.actor, m.status,\n\
                        m.created_at, m.updated_at, m.deleted_at, bm25(memories_fts)\n\
                 FROM memories_fts\n\
                 JOIN memories AS m ON m.rowid = memories_fts.rowid\n\
                 WHERE memories_fts MATCH ?1\n\
                   AND m.status = 'active'\n\
                   AND m.project_id IS NULL\n\
                 ORDER BY bm25(memories_fts), m.updated_at DESC\n\
                 LIMIT ?2",
            )?;
            let rows = statement.query_map(params![query, limit], search_hit_from_row)?;
            for row in rows {
                hits.push(row?);
            }
        }

        Ok(hits)
    }

    fn migrate(&self) -> Result<()> {
        let version = self.schema_version()?;
        if version > SCHEMA_VERSION {
            bail!(
                "database schema version {version} is newer than supported version {SCHEMA_VERSION}"
            );
        }

        // Concurrent creators (an adapter session plus a background indexer,
        // say) can both observe an old version and both apply the next
        // migration; the loser's CREATE fails after the winner commits. On
        // that failure the version has already advanced past the step, so
        // re-reading it and continuing converges instead of crashing.
        let apply = |sql: &'static str, from_version: i64| -> Result<()> {
            match self.connection.execute_batch(sql) {
                Ok(()) => Ok(()),
                Err(error) => {
                    // A batch that fails mid-script leaves its transaction
                    // open. Close it before deciding: continuing with an
                    // open transaction would let later writes "succeed"
                    // invisibly inside a transaction nobody commits.
                    let _ = self.connection.execute_batch("ROLLBACK");
                    let current = self.schema_version()?;
                    if current > from_version {
                        Ok(())
                    } else {
                        Err(error.into())
                    }
                }
            }
        };

        if version == 0 {
            apply(include_str!("../migrations/0001_initial.sql"), 0)?;
        }
        if self.schema_version()? == 1 {
            apply(include_str!("../migrations/0002_episode_entries.sql"), 1)?;
        }
        if self.schema_version()? == 2 {
            apply(include_str!("../migrations/0003_index_jobs.sql"), 2)?;
        }
        if self.schema_version()? == 3 {
            apply(include_str!("../migrations/0004_embeddings.sql"), 3)?;
        }
        if self.schema_version()? == 4 {
            apply(
                include_str!("../migrations/0005_episode_vectors_disabled.sql"),
                4,
            )?;
        }

        let version = self.schema_version()?;
        if version != SCHEMA_VERSION {
            bail!("failed to migrate database to schema version {SCHEMA_VERSION}; got {version}");
        }
        Ok(())
    }

    fn schema_version(&self) -> Result<i64> {
        Ok(self
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))?)
    }

    fn memory_by_id(&self, id: &str) -> Result<Memory> {
        Ok(self.connection.query_row(
            "SELECT id, scope, project_id, kind, text, actor, status,\n\
                    created_at, updated_at, deleted_at\n\
             FROM memories WHERE id = ?1",
            [id],
            memory_from_row,
        )?)
    }

    /// Candidate IDs matching an exact ID or prefix. Used by routed
    /// resolution, which must see every store's matches to enforce
    /// cross-store uniqueness.
    pub fn memory_id_candidates(&self, id_or_prefix: &str) -> Result<Vec<String>> {
        let candidate = id_or_prefix.trim();
        let prefix = format!("{candidate}%");
        let mut statement = self.connection.prepare(
            "SELECT id\n\
             FROM memories\n\
             WHERE id = ?1 OR id LIKE ?2\n\
             ORDER BY CASE WHEN id = ?1 THEN 0 ELSE 1 END, id\n\
             LIMIT 3",
        )?;
        let rows =
            statement.query_map(params![candidate, prefix], |row| row.get::<_, String>(0))?;
        let ids = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    fn resolve_id(&self, id_or_prefix: &str) -> Result<String> {
        let candidate = id_or_prefix.trim();
        if candidate.is_empty() {
            bail!("memory ID cannot be empty");
        }

        // An exact ID always resolves, even when it is also a prefix of other
        // IDs; only genuine prefix lookups can be ambiguous.
        if let Some(id) = self
            .connection
            .query_row(
                "SELECT id FROM memories WHERE id = ?1",
                [candidate],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            return Ok(id);
        }

        let prefix = format!("{candidate}%");
        let mut statement = self.connection.prepare(
            "SELECT id\n\
             FROM memories\n\
             WHERE id = ?1 OR id LIKE ?2\n\
             ORDER BY CASE WHEN id = ?1 THEN 0 ELSE 1 END, id\n\
             LIMIT 2",
        )?;
        let rows =
            statement.query_map(params![candidate, prefix], |row| row.get::<_, String>(0))?;
        let ids: Vec<String> = rows.collect::<rusqlite::Result<_>>()?;

        match ids.as_slice() {
            [] => bail!("memory not found: {candidate}"),
            [id] => Ok(id.clone()),
            _ => bail!("ambiguous memory ID prefix: {candidate}"),
        }
    }
}

fn insert_memory(
    connection: &Connection,
    id: &str,
    scope: &str,
    input: &NewMemory,
    now: i64,
) -> Result<()> {
    connection.execute(
        "INSERT INTO memories (\n\
             id, scope, project_id, kind, text, actor, status, created_at, updated_at\n\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?7)",
        params![
            id,
            scope,
            &input.project_id,
            &input.kind,
            &input.text,
            &input.actor,
            now
        ],
    )?;
    Ok(())
}

fn insert_source(
    connection: &Connection,
    source_id: &str,
    memory_id: &str,
    input: &NewMemory,
    now: i64,
) -> Result<()> {
    connection.execute(
        "INSERT INTO memory_sources (\n\
             id, memory_id, source_type, locator, excerpt, created_at\n\
         ) VALUES (?1, ?2, ?3, ?4, NULL, ?5)",
        params![
            source_id,
            memory_id,
            &input.source_type,
            &input.source_ref,
            now
        ],
    )?;
    Ok(())
}

fn validate_new_memory(input: &NewMemory) -> Result<()> {
    if input.text.trim().is_empty() {
        bail!("memory text cannot be empty");
    }
    if !MEMORY_KINDS.contains(&input.kind.as_str()) {
        bail!(
            "unknown memory kind '{}'; expected one of: {}",
            input.kind,
            MEMORY_KINDS.join(", ")
        );
    }
    if input.actor.trim().is_empty() {
        bail!("memory actor cannot be empty");
    }
    if input.source_type.trim().is_empty() {
        bail!("memory source type cannot be empty");
    }
    if let Some(project_id) = input.project_id.as_deref() {
        validate_identity(project_id, "project")?;
    }
    Ok(())
}

fn validate_identity(value: &str, kind: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{kind} identifier cannot be empty");
    }
    Ok(())
}

fn validate_optional(value: &Option<String>, kind: &str) -> Result<()> {
    if value
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        bail!("{kind} value cannot be empty");
    }
    Ok(())
}

fn memory_from_row(row: &Row<'_>) -> rusqlite::Result<Memory> {
    Ok(Memory {
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
    })
}

fn search_hit_from_row(row: &Row<'_>) -> rusqlite::Result<SearchHit> {
    Ok(SearchHit {
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
        rank: row.get(10)?,
    })
}

fn workspace_state_from_row(row: &Row<'_>) -> rusqlite::Result<WorkspaceState> {
    Ok(WorkspaceState {
        project_id: row.get(0)?,
        workspace_id: row.get(1)?,
        last_session_id: row.get(2)?,
        active_goal: row.get(3)?,
        active_task_ref: row.get(4)?,
        checkpoint: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

fn fts_query(input: &str, operator: &str) -> Result<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        bail!("search query cannot be empty");
    }
    Ok(terms.join(operator))
}

/// Memory text can hold credentials and internal paths, so created stores
/// and their directories are private to the current user. Only creations are
/// restricted; existing paths keep whatever the user or umask chose.
fn restrict_permissions(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))
            .with_context(|| format!("restrict permissions on {}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
    Ok(())
}

fn unix_millis() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}

#[cfg(test)]
mod tests {
    use super::{NewCorrection, NewMemory, NewWorkspaceState, NewWorkspaceStatePatch, Store};
    use uuid::Uuid;

    #[test]
    fn workspace_state_round_trips_and_clears() {
        let path = test_path();
        let store = Store::open(&path).expect("open test store");

        assert!(
            store
                .workspace_state("github.com/nijaru/mem", "branch:main")
                .expect("read empty state")
                .is_none()
        );

        let state = store
            .set_workspace_state(NewWorkspaceState {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                last_session_id: Some("session-1".to_owned()),
                active_goal: Some("build continuation state".to_owned()),
                active_task_ref: None,
                checkpoint: Some("project identity is validated".to_owned()),
            })
            .expect("write state");
        assert_eq!(state.last_session_id.as_deref(), Some("session-1"));
        assert_eq!(
            state.active_goal.as_deref(),
            Some("build continuation state")
        );

        assert!(
            store
                .clear_workspace_state("github.com/nijaru/mem", "branch:main")
                .expect("clear state")
        );
        assert!(
            store
                .workspace_state("github.com/nijaru/mem", "branch:main")
                .expect("read cleared state")
                .is_none()
        );

        let store = Store::open(&test_path()).expect("open patch store");
        store
            .set_workspace_state(NewWorkspaceState {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                last_session_id: Some("session-1".to_owned()),
                active_goal: Some("build continuation state".to_owned()),
                active_task_ref: Some("tk-42".to_owned()),
                checkpoint: Some("v1".to_owned()),
            })
            .expect("seed state for patch");

        // Patch only the checkpoint: every other field must survive.
        let state = store
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                checkpoint: Some("v2".to_owned()),
                ..Default::default()
            })
            .expect("patch checkpoint");
        assert_eq!(state.checkpoint.as_deref(), Some("v2"));
        assert_eq!(state.last_session_id.as_deref(), Some("session-1"));
        assert_eq!(
            state.active_goal.as_deref(),
            Some("build continuation state")
        );
        assert_eq!(state.active_task_ref.as_deref(), Some("tk-42"));

        // Explicit clear nulls exactly the named fields.
        let state = store
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                clear_fields: vec!["goal".to_owned(), "task".to_owned()],
                ..Default::default()
            })
            .expect("clear goal and task");
        assert_eq!(state.active_goal, None);
        assert_eq!(state.active_task_ref, None);
        assert_eq!(state.last_session_id.as_deref(), Some("session-1"));
        assert_eq!(state.checkpoint.as_deref(), Some("v2"));

        // Patch on a missing row creates it with only the provided fields.
        let state = store
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:feature".to_owned(),
                active_goal: Some("resume work".to_owned()),
                ..Default::default()
            })
            .expect("patch new row");
        assert_eq!(state.active_goal.as_deref(), Some("resume work"));
        assert_eq!(state.last_session_id, None);

        // Unknown field names are rejected instead of silently ignored.
        let error = store
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                clear_fields: vec!["project".to_owned()],
                ..Default::default()
            })
            .expect_err("unknown field must fail");
        assert!(
            format!("{error:#}").contains("unknown state field"),
            "unexpected error: {error:#}"
        );

        // A patch must not lose another writer's concurrent field update:
        // the read-merge-write anti-pattern this API exists to prevent.
        let db_path = store.path().expect("test store path");
        let writer = Store::open(std::path::Path::new(db_path)).expect("open second writer");
        writer
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                active_goal: Some("writer goal".to_owned()),
                ..Default::default()
            })
            .expect("concurrent goal patch");
        let state = store
            .patch_workspace_state(NewWorkspaceStatePatch {
                project_id: "github.com/nijaru/mem".to_owned(),
                workspace_id: "branch:main".to_owned(),
                active_task_ref: Some("tk-7".to_owned()),
                ..Default::default()
            })
            .expect("second patch");
        assert_eq!(
            state.active_goal.as_deref(),
            Some("writer goal"),
            "concurrent field update must survive this writer's patch"
        );
        assert_eq!(state.active_task_ref.as_deref(), Some("tk-7"));

        cleanup(&path);
    }

    #[test]
    fn recall_is_broader_than_explicit_search() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let project = "github.com/nijaru/mem";

        store
            .remember(NewMemory {
                text: "publication notification handoff".to_owned(),
                kind: "fact".to_owned(),
                project_id: Some(project.to_owned()),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store project memory");
        store
            .remember(NewMemory {
                text: "user preference favors concise memory tooling".to_owned(),
                kind: "preference".to_owned(),
                project_id: None,
                actor: "user".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store global memory");

        // Dogfood regression (omengrep session, 2026-09-02): a query whose
        // terms never co-occur literally in any memory must still return
        // ranked partial matches from explicit lexical search. Strict AND
        // semantics made `mem search` return [] while the semantic tier
        // found five hits on the same corpus — the two tiers must never
        // silently disagree about whether relevant memory exists.
        let ranked = store
            .search("publication preference", Some(project), 10)
            .expect("ranked lexical search");
        assert_eq!(ranked.len(), 2);
        // Full term matches rank above partial matches under bm25.
        assert!(ranked[0].rank <= ranked[1].rank, "hits must be ranked");
        let recalled = store
            .recall("publication preference", Some(project), 10)
            .expect("broad recall");
        assert_eq!(recalled.len(), 2);

        cleanup(&path);
    }

    #[test]
    fn colliding_id_prefixes_stay_individually_addressable() {
        // Dogfood regression (omengrep session, 2026-09-02): UUIDv7 IDs minted
        // in the same millisecond share long prefixes, and `context` can
        // legitimately return both. Displayed prefixes must never collide in
        // practice, but when they do, resolution must fail explicitly rather
        // than silently returning either memory.
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let first = store
            .remember(NewMemory {
                text: "first colliding memory".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store first memory")
            .id;
        let second = store
            .remember(NewMemory {
                text: "second colliding memory".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store second memory")
            .id;

        if first.len() >= 8 && second.len() >= 8 && first[..8] == second[..8] {
            let ambiguous = &first[..8];
            assert!(store.get(ambiguous).is_err(), "ambiguous prefix must error");
        }
        // Exact IDs always resolve even when they share a prefix.
        assert_eq!(
            store.get(&first).expect("exact first").memory.id,
            first,
            "exact ID must always resolve"
        );
        assert_eq!(
            store.get(&second).expect("exact second").memory.id,
            second,
            "exact ID must always resolve"
        );

        cleanup(&path);
    }

    #[test]
    fn open_existing_never_creates_a_database() {
        let path = test_path();
        assert!(
            Store::open_existing(&path)
                .expect("probe absent store")
                .is_none(),
            "absent path must not be created"
        );
        assert!(!path.exists());

        let mut store = Store::open(&path).expect("create store");
        store
            .remember(NewMemory {
                text: "existing memory".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
        drop(store);

        let reopened = Store::open_existing(&path)
            .expect("probe existing store")
            .expect("existing file opens");
        assert_eq!(reopened.stats().expect("read stats").schema_version, 5);

        drop(reopened);
        cleanup(&path);
    }

    #[test]
    fn correction_supersedes_without_losing_history() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let project = "github.com/nijaru/mem";

        let original = store
            .remember(NewMemory {
                text: "alpha policy remains active".to_owned(),
                kind: "decision".to_owned(),
                project_id: Some(project.to_owned()),
                actor: "user".to_owned(),
                source_type: "test".to_owned(),
                source_ref: Some("original".to_owned()),
            })
            .expect("store original memory");

        let result = store
            .correct(
                &original.id,
                NewCorrection {
                    text: "beta policy replaces alpha".to_owned(),
                    kind: None,
                    actor: "user".to_owned(),
                    source_type: "test".to_owned(),
                    source_ref: Some("correction".to_owned()),
                },
            )
            .expect("correct memory");

        assert_eq!(result.previous.status, "superseded");
        assert_eq!(result.replacement.memory.status, "active");
        assert_eq!(result.replacement.memory.kind, "decision");
        assert_eq!(
            result.replacement.memory.project_id.as_deref(),
            Some(project)
        );
        assert!(
            result
                .replacement
                .relations
                .iter()
                .any(|relation| relation.from_memory_id == original.id
                    && relation.to_memory_id == result.replacement.memory.id
                    && relation.relation_type == "superseded_by")
        );

        let original_record = store.get(&original.id).expect("read original memory");
        assert_eq!(original_record.memory.status, "superseded");
        assert_eq!(original_record.relations.len(), 1);
        assert!(
            store
                .search("active", Some(project), 10)
                .expect("search superseded wording")
                .is_empty()
        );
        assert_eq!(
            store
                .search("beta", Some(project), 10)
                .expect("search replacement")
                .len(),
            1
        );

        let stats = store.stats().expect("read stats");
        assert_eq!(stats.total, 2);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.superseded, 1);
        assert_eq!(stats.deleted, 0);

        cleanup(&path);
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
