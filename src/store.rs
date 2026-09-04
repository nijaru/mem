use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Row, TransactionBehavior, params};
use serde::Serialize;
use uuid::Uuid;

const SCHEMA_VERSION: i64 = 7;
const MEMORY_KINDS: &[&str] = &[
    "fact",
    "finding",
    "decision",
    "constraint",
    "preference",
    "procedure",
];
pub const STATE_FIELDS: &[&str] = &["session", "goal", "task", "checkpoint"];

#[derive(Debug)]
pub struct Store {
    pub(crate) connection: Connection,
}

pub struct NewMemory {
    pub text: String,
    pub kind: String,
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

#[derive(Debug, Default)]
pub struct WorkspaceStateUpdate {
    pub workspace: String,
    pub session: Option<String>,
    pub goal: Option<String>,
    pub task: Option<String>,
    pub checkpoint: Option<String>,
    pub clear_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub actor: String,
    pub source_type: String,
    pub source_ref: Option<String>,
    pub status: String,
    pub superseded_by: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub deleted_at: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CorrectionResult {
    pub previous: Memory,
    pub replacement: Memory,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub memory: Memory,
    pub rank: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceState {
    pub workspace: String,
    pub session: Option<String>,
    pub goal: Option<String>,
    pub task: Option<String>,
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

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EmbeddingCoverage {
    pub indexed: u64,
    pub unindexed: u64,
}

#[derive(Debug)]
pub(crate) struct EmbeddingSource {
    pub id: String,
    pub text: String,
    pub updated_at: i64,
}

impl Store {
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
        let mode: String = connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        if mode != "wal" {
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

    pub fn remember(&self, input: NewMemory) -> Result<Memory> {
        validate_new_memory(&input)?;
        let id = Uuid::now_v7().to_string();
        let now = unix_millis()?;
        insert_memory(&self.connection, &id, &input, now)?;
        self.memory_by_id(&id)
    }

    pub fn correct(
        &mut self,
        id_or_prefix: &str,
        input: NewCorrection,
    ) -> Result<CorrectionResult> {
        let previous_id = self.resolve_memory_id(id_or_prefix)?;
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
            actor: input.actor,
            source_type: input.source_type,
            source_ref: input.source_ref,
        };
        validate_new_memory(&replacement_input)?;

        let replacement_id = Uuid::now_v7().to_string();
        let now = unix_millis()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        insert_memory(&transaction, &replacement_id, &replacement_input, now)?;
        let changed = transaction.execute(
            "UPDATE memories\n\
             SET status = 'superseded', superseded_by = ?2, updated_at = ?3\n\
             WHERE id = ?1 AND status = 'active'",
            params![previous_id, replacement_id, now],
        )?;
        if changed != 1 {
            bail!("memory changed while correction was being applied");
        }
        transaction.commit()?;

        Ok(CorrectionResult {
            previous: self.memory_by_id(&previous_id)?,
            replacement: self.memory_by_id(&replacement_id)?,
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        self.search_fts(&crate::id_resolve::fts_query(query)?, limit)
    }

    pub fn get(&self, id_or_prefix: &str) -> Result<Memory> {
        let id = self.resolve_memory_id(id_or_prefix)?;
        self.memory_by_id(&id)
    }

    pub fn forget(&self, id_or_prefix: &str) -> Result<String> {
        let id = self.resolve_memory_id(id_or_prefix)?;
        let memory = self.memory_by_id(&id)?;
        if memory.status != "active" {
            bail!("memory {} is already {}", memory.id, memory.status);
        }
        let now = unix_millis()?;
        let changed = self.connection.execute(
            "UPDATE memories\n\
             SET status = 'deleted', deleted_at = ?2, updated_at = ?2\n\
             WHERE id = ?1 AND status = 'active'",
            params![id, now],
        )?;
        if changed != 1 {
            bail!("memory changed while it was being forgotten");
        }
        Ok(id)
    }

    pub fn workspace_state(&self, workspace: &str) -> Result<Option<WorkspaceState>> {
        validate_identity(workspace, "workspace")?;
        Ok(self
            .connection
            .query_row(
                "SELECT workspace, session, goal, task, checkpoint, updated_at\n\
                 FROM workspace_state WHERE workspace = ?1",
                [workspace],
                workspace_state_from_row,
            )
            .optional()?)
    }

    pub fn update_workspace_state(
        &mut self,
        input: WorkspaceStateUpdate,
    ) -> Result<WorkspaceState> {
        validate_identity(&input.workspace, "workspace")?;
        validate_optional(&input.session, "session")?;
        validate_optional(&input.goal, "goal")?;
        validate_optional(&input.task, "task")?;
        validate_optional(&input.checkpoint, "checkpoint")?;

        for field in &input.clear_fields {
            if !STATE_FIELDS.contains(&field.as_str()) {
                bail!("unknown state field: {field}");
            }
        }
        for (name, value) in [
            ("session", &input.session),
            ("goal", &input.goal),
            ("task", &input.task),
            ("checkpoint", &input.checkpoint),
        ] {
            if value.is_some() && input.clear_fields.iter().any(|field| field == name) {
                bail!("state field {name} cannot be set and cleared together");
            }
        }
        let has_value = input.session.is_some()
            || input.goal.is_some()
            || input.task.is_some()
            || input.checkpoint.is_some();
        if !has_value && input.clear_fields.is_empty() {
            bail!("state set requires at least one field to set or clear");
        }
        if !has_value
            && self.workspace_state(&input.workspace)?.is_none()
            && !input.clear_fields.is_empty()
        {
            bail!("no state for {}", input.workspace);
        }

        let now = unix_millis()?;
        // Validation reads and the upsert must serialize as one unit: without
        // the transaction a concurrent `state clear` between the existence
        // check and the upsert can resurrect an emptied state row, and the
        // "no state" error can go stale.
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO workspace_state (workspace, session, goal, task, checkpoint, updated_at)\n\
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)\n\
             ON CONFLICT(workspace) DO UPDATE SET\n\
                 session = COALESCE(?2, CASE WHEN ?7 THEN NULL ELSE session END),\n\
                 goal = COALESCE(?3, CASE WHEN ?8 THEN NULL ELSE goal END),\n\
                 task = COALESCE(?4, CASE WHEN ?9 THEN NULL ELSE task END),\n\
                 checkpoint = COALESCE(?5, CASE WHEN ?10 THEN NULL ELSE checkpoint END),\n\
                 updated_at = ?6",
            params![
                &input.workspace,
                &input.session,
                &input.goal,
                &input.task,
                &input.checkpoint,
                now,
                input.clear_fields.iter().any(|field| field == "session"),
                input.clear_fields.iter().any(|field| field == "goal"),
                input.clear_fields.iter().any(|field| field == "task"),
                input.clear_fields.iter().any(|field| field == "checkpoint"),
            ],
        )?;
        let state = transaction
            .query_row(
                "SELECT workspace, session, goal, task, checkpoint, updated_at\n\
                 FROM workspace_state WHERE workspace = ?1",
                [&input.workspace],
                workspace_state_from_row,
            )
            .context("workspace state disappeared after write")?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn clear_workspace_state(&self, workspace: &str) -> Result<bool> {
        validate_identity(workspace, "workspace")?;
        Ok(self.connection.execute(
            "DELETE FROM workspace_state WHERE workspace = ?1",
            [workspace],
        )? > 0)
    }

    pub fn embedding_coverage(&self, model: &str) -> Result<EmbeddingCoverage> {
        validate_identity(model, "embedding model")?;
        let (active, indexed): (i64, i64) = self.connection.query_row(
            "SELECT COUNT(*),\n\
                    COALESCE(SUM(EXISTS (\n\
                        SELECT 1 FROM embeddings AS e\n\
                        WHERE e.memory_id = m.id\n\
                          AND e.model = ?1\n\
                          AND e.source_updated_at = m.updated_at\n\
                    )), 0)\n\
             FROM memories AS m\n\
             WHERE m.status = 'active'",
            [model],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(EmbeddingCoverage {
            indexed: indexed.try_into()?,
            unindexed: (active - indexed).try_into()?,
        })
    }

    pub(crate) fn pending_embedding_sources(
        &self,
        model: &str,
        limit: usize,
    ) -> Result<Vec<EmbeddingSource>> {
        validate_identity(model, "embedding model")?;
        if limit == 0 {
            bail!("embedding limit must be greater than zero");
        }
        let limit = i64::try_from(limit)?;
        let mut statement = self.connection.prepare(
            "SELECT m.id, m.text, m.updated_at\n\
             FROM memories AS m\n\
             WHERE m.status = 'active'\n\
               AND NOT EXISTS (\n\
                   SELECT 1 FROM embeddings AS e\n\
                   WHERE e.memory_id = m.id\n\
                     AND e.model = ?1\n\
                     AND e.source_updated_at = m.updated_at\n\
               )\n\
             ORDER BY m.updated_at DESC, m.id\n\
             LIMIT ?2",
        )?;
        let rows = statement.query_map(params![model, limit], |row| {
            Ok(EmbeddingSource {
                id: row.get(0)?,
                text: row.get(1)?,
                updated_at: row.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        if limit == 0 {
            bail!("search limit must be greater than zero");
        }
        let limit = i64::try_from(limit)?;
        let mut statement = self.connection.prepare(
            "SELECT m.id, m.kind, m.text, m.actor, m.source_type, m.source_ref,\n\
                    m.status, m.superseded_by, m.created_at, m.updated_at, m.deleted_at,\n\
                    bm25(memories_fts)\n\
             FROM memories_fts\n\
             JOIN memories AS m ON m.rowid = memories_fts.rowid\n\
             WHERE memories_fts MATCH ?1 AND m.status = 'active'\n\
             ORDER BY bm25(memories_fts), m.updated_at DESC\n\
             LIMIT ?2",
        )?;
        let rows = statement.query_map(params![query, limit], |row| {
            Ok(SearchHit {
                memory: memory_from_row(row)?,
                rank: row.get(11)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn migrate(&self) -> Result<()> {
        self.connection.execute_batch("BEGIN IMMEDIATE;")?;
        let result = (|| -> Result<()> {
            let version = self.schema_version()?;
            if version > SCHEMA_VERSION {
                bail!(
                    "database schema version {version} is newer than supported version {SCHEMA_VERSION}"
                );
            }
            match version {
                0 => self
                    .connection
                    .execute_batch(include_str!("../migrations/schema.sql"))?,
                6 => self
                    .connection
                    .execute_batch(include_str!("../migrations/0007_lean_core.sql"))?,
                7 => {}
                1..=5 => bail!(
                    "database schema version {version} is from an unsupported pre-1.0 build; upgrade it to v6 with the previous mem build first"
                ),
                _ => unreachable!(),
            }
            let version = self.schema_version()?;
            if version != SCHEMA_VERSION {
                bail!(
                    "failed to migrate database to schema version {SCHEMA_VERSION}; got {version}"
                );
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.connection.execute_batch("COMMIT;")?;
                Ok(())
            }
            Err(error) => {
                let _ = self.connection.execute_batch("ROLLBACK;");
                Err(error)
            }
        }
    }

    fn schema_version(&self) -> Result<i64> {
        Ok(self
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))?)
    }

    fn memory_by_id(&self, id: &str) -> Result<Memory> {
        Ok(self.connection.query_row(
            "SELECT id, kind, text, actor, source_type, source_ref, status, superseded_by,\n\
                    created_at, updated_at, deleted_at\n\
             FROM memories WHERE id = ?1",
            [id],
            memory_from_row,
        )?)
    }
}

fn insert_memory(connection: &Connection, id: &str, input: &NewMemory, now: i64) -> Result<()> {
    connection.execute(
        "INSERT INTO memories (\n\
             id, kind, text, actor, source_type, source_ref, status, created_at, updated_at\n\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?7)",
        params![
            id,
            &input.kind,
            &input.text,
            &input.actor,
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
    validate_identity(&input.actor, "memory actor")?;
    validate_identity(&input.source_type, "memory source type")?;
    validate_optional(&input.source_ref, "source reference")?;
    Ok(())
}

fn validate_identity(value: &str, kind: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{kind} cannot be empty");
    }
    Ok(())
}

fn validate_optional(value: &Option<String>, kind: &str) -> Result<()> {
    if value
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        bail!("{kind} cannot be empty");
    }
    Ok(())
}

fn memory_from_row(row: &Row<'_>) -> rusqlite::Result<Memory> {
    Ok(Memory {
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
    })
}

fn workspace_state_from_row(row: &Row<'_>) -> rusqlite::Result<WorkspaceState> {
    Ok(WorkspaceState {
        workspace: row.get(0)?,
        session: row.get(1)?,
        goal: row.get(2)?,
        task: row.get(3)?,
        checkpoint: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

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
    use super::{NewCorrection, NewMemory, Store, WorkspaceStateUpdate};
    use uuid::Uuid;

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-store-test-{}.db", Uuid::now_v7()))
    }

    fn remember(store: &Store, text: &str) -> super::Memory {
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

    #[test]
    fn correction_preserves_history_and_provenance() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open");
        let first = remember(&store, "old statement");
        let result = store
            .correct(
                &first.id,
                NewCorrection {
                    text: "new statement".to_owned(),
                    kind: None,
                    actor: "agent".to_owned(),
                    source_type: "test".to_owned(),
                    source_ref: Some("correction".to_owned()),
                },
            )
            .expect("correct");
        assert_eq!(result.previous.status, "superseded");
        assert_eq!(
            result.previous.superseded_by.as_deref(),
            Some(result.replacement.id.as_str())
        );
        assert_eq!(result.replacement.source_ref.as_deref(), Some("correction"));
        assert_eq!(store.search("statement", 10).expect("search").len(), 1);
        cleanup(&path);
    }

    #[test]
    fn workspace_state_set_is_partial_and_clear_is_explicit() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open");
        store
            .update_workspace_state(WorkspaceStateUpdate {
                workspace: "branch:main".to_owned(),
                goal: Some("ship".to_owned()),
                checkpoint: Some("tests green".to_owned()),
                ..Default::default()
            })
            .expect("set state");
        let state = store
            .update_workspace_state(WorkspaceStateUpdate {
                workspace: "branch:main".to_owned(),
                session: Some("s2".to_owned()),
                clear_fields: vec!["checkpoint".to_owned()],
                ..Default::default()
            })
            .expect("patch state");
        assert_eq!(state.goal.as_deref(), Some("ship"));
        assert_eq!(state.session.as_deref(), Some("s2"));
        assert!(state.checkpoint.is_none());
        cleanup(&path);
    }

    #[test]
    fn concurrent_remembers_do_not_lose_writes() {
        let path = test_path();
        let mut joins = Vec::new();
        for index in 0..8 {
            let path = path.clone();
            joins.push(std::thread::spawn(move || {
                let store = Store::open(&path).expect("open writer");
                remember(&store, &format!("memory {index}"));
            }));
        }
        for join in joins {
            join.join().expect("join");
        }
        let store = Store::open(&path).expect("open final");
        assert_eq!(store.stats().expect("stats").active, 8);
        cleanup(&path);
    }

    #[test]
    fn concurrent_state_sets_do_not_lose_writes() {
        let path = test_path();
        let mut joins = Vec::new();
        for index in 0..8 {
            let path = path.clone();
            joins.push(std::thread::spawn(move || {
                let mut store = Store::open(&path).expect("open writer");
                store
                    .update_workspace_state(WorkspaceStateUpdate {
                        workspace: format!("branch:ws{index}").to_owned(),
                        goal: Some(format!("goal {index}")).to_owned(),
                        checkpoint: Some(format!("checkpoint {index}")).to_owned(),
                        ..Default::default()
                    })
                    .expect("set state");
            }));
        }
        for join in joins {
            join.join().expect("join");
        }
        let store = Store::open(&path).expect("open final");
        for index in 0..8 {
            let state = store
                .workspace_state(&format!("branch:ws{index}"))
                .expect("read state")
                .expect("state row survived concurrent writes");
            assert_eq!(
                state.goal.as_deref(),
                Some(format!("goal {index}").as_str())
            );
        }
        cleanup(&path);
    }

    #[test]
    fn state_clear_cannot_race_set_into_resurrecting_a_row() {
        // A clear-only update on an absent row must fail rather than
        // recreate the row the clear just removed; the IMMEDIATE
        // transaction keeps the existence check and the upsert consistent.
        let path = test_path();
        let mut store = Store::open(&path).expect("open");
        store
            .update_workspace_state(WorkspaceStateUpdate {
                workspace: "branch:main".to_owned(),
                goal: Some("ship".to_owned()),
                ..Default::default()
            })
            .expect("seed");
        assert!(store.clear_workspace_state("branch:main").expect("clear"));
        // A clear-only update on an absent row must fail, not recreate it.
        let result = store.update_workspace_state(WorkspaceStateUpdate {
            workspace: "branch:main".to_owned(),
            clear_fields: vec!["goal".to_owned()],
            ..Default::default()
        });
        assert!(result.is_err());
        assert!(
            store
                .workspace_state("branch:main")
                .expect("read")
                .is_none()
        );
        cleanup(&path);
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    }
}
