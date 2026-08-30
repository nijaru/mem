use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;
use uuid::Uuid;

const SCHEMA_VERSION: i64 = 1;
const MEMORY_KINDS: &[&str] = &["fact", "decision", "constraint", "preference", "procedure"];

pub struct Store {
    connection: Connection,
}

pub struct NewMemory {
    pub text: String,
    pub kind: String,
    pub project_id: Option<String>,
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

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
pub struct MemorySource {
    pub id: String,
    pub source_type: String,
    pub locator: Option<String>,
    pub excerpt: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Serialize)]
pub struct MemoryRecord {
    pub memory: Memory,
    pub sources: Vec<MemorySource>,
}

#[derive(Debug, Serialize)]
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
    pub deleted: u64,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create data directory {}", parent.display()))?;
        }

        let connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;\n\
             PRAGMA foreign_keys=ON;\n\
             PRAGMA synchronous=NORMAL;",
        )?;

        let store = Self { connection };
        store.migrate()?;
        Ok(store)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let schema_version = self.schema_version()?;
        let (total, active, deleted): (i64, i64, i64) = self.connection.query_row(
            "SELECT COUNT(*),\n\
                    COALESCE(SUM(status = 'active'), 0),\n\
                    COALESCE(SUM(status = 'deleted'), 0)\n\
             FROM memories",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;

        Ok(StoreStats {
            schema_version,
            total: total.try_into()?,
            active: active.try_into()?,
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
        transaction.execute(
            "INSERT INTO memories (\n\
                 id, scope, project_id, kind, text, actor, status, created_at, updated_at\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?7)",
            params![
                id,
                scope,
                input.project_id,
                input.kind,
                input.text,
                input.actor,
                now
            ],
        )?;
        transaction.execute(
            "INSERT INTO memory_sources (\n\
                 id, memory_id, source_type, locator, excerpt, created_at\n\
             ) VALUES (?1, ?2, ?3, ?4, NULL, ?5)",
            params![source_id, id, input.source_type, input.source_ref, now],
        )?;
        transaction.commit()?;

        self.memory_by_id(&id)
    }

    pub fn search(
        &self,
        query: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_fts(&fts_query(query, " AND ")?, project_id, limit)
    }

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

        let mut statement = self.connection.prepare(
            "SELECT id, source_type, locator, excerpt, created_at\n\
             FROM memory_sources\n\
             WHERE memory_id = ?1\n\
             ORDER BY created_at, id",
        )?;
        let rows = statement.query_map([&id], |row| {
            Ok(MemorySource {
                id: row.get(0)?,
                source_type: row.get(1)?,
                locator: row.get(2)?,
                excerpt: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        let mut sources = Vec::new();
        for row in rows {
            sources.push(row?);
        }

        Ok(MemoryRecord { memory, sources })
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
        match self.schema_version()? {
            0 => {
                self.connection
                    .execute_batch(include_str!("../migrations/0001_initial.sql"))?;
                self.connection
                    .pragma_update(None, "user_version", SCHEMA_VERSION)?;
                Ok(())
            }
            SCHEMA_VERSION => Ok(()),
            version => bail!(
                "database schema version {version} is newer than supported version {SCHEMA_VERSION}"
            ),
        }
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

    fn resolve_id(&self, id_or_prefix: &str) -> Result<String> {
        let candidate = id_or_prefix.trim();
        if candidate.is_empty() {
            bail!("memory ID cannot be empty");
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

fn unix_millis() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}

#[cfg(test)]
mod tests {
    use super::{NewMemory, NewWorkspaceState, Store};
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

        assert!(
            store
                .search("publication preference", Some(project), 10)
                .expect("strict search")
                .is_empty()
        );
        let recalled = store
            .recall("publication preference", Some(project), 10)
            .expect("broad recall");
        assert_eq!(recalled.len(), 2);

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
