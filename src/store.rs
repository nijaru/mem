use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, Row, params};
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

pub struct StoreStats {
    pub schema_version: i64,
    pub total: u64,
    pub active: u64,
    pub deleted: u64,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
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
        let query = fts_query(query)?;
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
            let rows = statement.query_map(params![query, project_id, limit], search_hit_from_row)?;
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
        let rows = statement.query_map(params![candidate, prefix], |row| row.get::<_, String>(0))?;
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
    if input.project_id.as_deref().is_some_and(str::is_empty) {
        bail!("project identifier cannot be empty");
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

fn fts_query(input: &str) -> Result<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        bail!("search query cannot be empty");
    }
    Ok(terms.join(" AND "))
}

fn unix_millis() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}
