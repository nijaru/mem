use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{OptionalExtension, Row, params};
use serde::Serialize;
use uuid::Uuid;

use crate::store::Store;

#[derive(Clone)]
pub struct NewEpisode {
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub source_type: String,
    pub source_ref: String,
    pub started_at: Option<i64>,
    pub metadata_json: Option<String>,
}

pub struct NewEpisodeEntry {
    pub source_ref: String,
    pub ordinal: Option<i64>,
    pub kind: String,
    pub role: Option<String>,
    pub text: String,
    pub occurred_at: Option<i64>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Episode {
    pub id: String,
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub source_type: String,
    pub source_ref: String,
    pub started_at: Option<i64>,
    pub ended_at: Option<i64>,
    pub summary: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EpisodeEntry {
    pub id: String,
    pub episode_id: String,
    pub source_ref: String,
    pub ordinal: i64,
    pub kind: String,
    pub role: Option<String>,
    pub text: String,
    pub occurred_at: Option<i64>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EpisodeRecord {
    pub episode: Episode,
    pub entries: Vec<EpisodeEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryHit {
    pub episode_id: String,
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub source_type: String,
    pub episode_source_ref: String,
    pub entry_id: String,
    pub entry_source_ref: String,
    pub ordinal: i64,
    pub kind: String,
    pub role: Option<String>,
    pub text: String,
    pub occurred_at: Option<i64>,
    pub rank: f64,
}

impl Store {
    pub fn ensure_episode(&self, input: NewEpisode) -> Result<Episode> {
        validate_episode(&input)?;

        // One conflict-safe statement: the source's identity decides. Two
        // adapters creating the same episode concurrently converge on one
        // row; the first binding wins and a later mismatched binding is
        // rejected by reading back what actually landed.
        let id = Uuid::now_v7().to_string();
        let started_at = input.started_at.or(Some(unix_millis()?));
        self.connection.execute(
            "INSERT INTO episodes (\n\
                 id, project_id, source_type, source_ref, started_at, ended_at, summary,\n\
                 metadata_json, workspace_id\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL, ?6, ?7)\n\
             ON CONFLICT(source_type, source_ref) DO NOTHING",
            params![
                id,
                input.project_id,
                input.source_type,
                input.source_ref,
                started_at,
                input.metadata_json,
                input.workspace_id
            ],
        )?;
        let stored = self
            .episode_by_source(&input.source_type, &input.source_ref)?
            .context("episode disappeared after upsert")?;
        if stored.project_id != input.project_id || stored.workspace_id != input.workspace_id {
            bail!(
                "episode source {}:{} is already bound to a different project/workspace",
                input.source_type,
                input.source_ref
            );
        }
        Ok(stored)
    }

    pub fn record_episode_entry(
        &self,
        episode_id_or_prefix: &str,
        input: NewEpisodeEntry,
    ) -> Result<EpisodeEntry> {
        validate_episode_entry(&input)?;
        let episode_id = self.resolve_episode_id(episode_id_or_prefix)?;

        // First record attempts a conflict-safe insert; the loser of a
        // concurrent first-record falls through to the refresh below, which
        // updates the winner's row instead of failing on the unique index.
        if input.ordinal.is_some() {
            let id = Uuid::now_v7().to_string();
            let ordinal = input.ordinal.unwrap_or_default();
            let inserted = self.connection.execute(
                "INSERT INTO episode_entries (\n\
                     id, episode_id, source_ref, ordinal, kind, role, text, occurred_at, metadata_json\n\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)\n\
                 ON CONFLICT(episode_id, source_ref) DO NOTHING",
                params![
                    id,
                    episode_id,
                    input.source_ref,
                    ordinal,
                    input.kind,
                    input.role,
                    input.text,
                    input.occurred_at,
                    input.metadata_json
                ],
            )?;
            if inserted > 0 {
                return self.episode_entry_by_id(&id);
            }
        }

        // Refresh (or first-record without an explicit ordinal): read the
        // current row state inside the same statement flow, then update.
        let existing: Option<(String, i64)> = self
            .connection
            .query_row(
                "SELECT id, ordinal\n\
                 FROM episode_entries\n\
                 WHERE episode_id = ?1 AND source_ref = ?2",
                params![episode_id, input.source_ref],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((id, existing_ordinal)) = existing {
            let ordinal = input.ordinal.unwrap_or(existing_ordinal);
            self.connection.execute(
                "UPDATE episode_entries\n\
                 SET ordinal = ?2, kind = ?3, role = ?4, text = ?5, occurred_at = ?6,\n\
                     metadata_json = ?7\n\
                 WHERE id = ?1",
                params![
                    id,
                    ordinal,
                    input.kind,
                    input.role,
                    input.text,
                    input.occurred_at,
                    input.metadata_json
                ],
            )?;
            return self.episode_entry_by_id(&id);
        }

        let ordinal = match input.ordinal {
            Some(ordinal) => ordinal,
            None => self.connection.query_row(
                "SELECT COALESCE(MAX(ordinal) + 1, 0)\n\
                 FROM episode_entries\n\
                 WHERE episode_id = ?1",
                [&episode_id],
                |row| row.get(0),
            )?,
        };
        let id = Uuid::now_v7().to_string();
        self.connection.execute(
            "INSERT INTO episode_entries (\n\
                 id, episode_id, source_ref, ordinal, kind, role, text, occurred_at, metadata_json\n\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)\n\
             ON CONFLICT(episode_id, source_ref) DO NOTHING",
            params![
                id,
                episode_id,
                input.source_ref,
                ordinal,
                input.kind,
                input.role,
                input.text,
                input.occurred_at,
                input.metadata_json
            ],
        )?;
        self.episode_entry_by_id(&id)
    }

    pub fn end_episode(
        &self,
        episode_id_or_prefix: &str,
        ended_at: Option<i64>,
    ) -> Result<Episode> {
        let id = self.resolve_episode_id(episode_id_or_prefix)?;
        let ended_at = ended_at.unwrap_or(unix_millis()?);
        self.connection.execute(
            "UPDATE episodes SET ended_at = ?2 WHERE id = ?1",
            params![id, ended_at],
        )?;
        self.episode_by_id(&id)
    }

    pub fn get_episode(&self, episode_id_or_prefix: &str) -> Result<EpisodeRecord> {
        let id = self.resolve_episode_id(episode_id_or_prefix)?;
        let episode = self.episode_by_id(&id)?;
        let mut statement = self.connection.prepare(
            "SELECT id, episode_id, source_ref, ordinal, kind, role, text, occurred_at,\n\
                    metadata_json\n\
             FROM episode_entries\n\
             WHERE episode_id = ?1\n\
             ORDER BY ordinal, id",
        )?;
        let rows = statement.query_map([&id], episode_entry_from_row)?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(EpisodeRecord { episode, entries })
    }

    pub fn history_search(
        &self,
        query: &str,
        project_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<HistoryHit>> {
        let query = crate::id_resolve::fts_query(query)?;
        let limit = i64::try_from(limit)?;
        let mut hits = Vec::new();

        if let Some(project_id) = project_id {
            let mut statement = self.connection.prepare(
                "SELECT e.id, e.project_id, e.workspace_id, e.source_type, e.source_ref,\n\
                        ee.id, ee.source_ref, ee.ordinal, ee.kind, ee.role, ee.text,\n\
                        ee.occurred_at, bm25(episode_entries_fts)\n\
                 FROM episode_entries_fts\n\
                 JOIN episode_entries AS ee ON ee.rowid = episode_entries_fts.rowid\n\
                 JOIN episodes AS e ON e.id = ee.episode_id\n\
                 WHERE episode_entries_fts MATCH ?1 AND e.project_id = ?2\n\
                 ORDER BY bm25(episode_entries_fts), e.started_at DESC, ee.ordinal\n\
                 LIMIT ?3",
            )?;
            let rows =
                statement.query_map(params![query, project_id, limit], history_hit_from_row)?;
            for row in rows {
                hits.push(row?);
            }
        } else {
            let mut statement = self.connection.prepare(
                "SELECT e.id, e.project_id, e.workspace_id, e.source_type, e.source_ref,\n\
                        ee.id, ee.source_ref, ee.ordinal, ee.kind, ee.role, ee.text,\n\
                        ee.occurred_at, bm25(episode_entries_fts)\n\
                 FROM episode_entries_fts\n\
                 JOIN episode_entries AS ee ON ee.rowid = episode_entries_fts.rowid\n\
                 JOIN episodes AS e ON e.id = ee.episode_id\n\
                 WHERE episode_entries_fts MATCH ?1 AND e.project_id IS NULL\n\
                 ORDER BY bm25(episode_entries_fts), e.started_at DESC, ee.ordinal\n\
                 LIMIT ?2",
            )?;
            let rows = statement.query_map(params![query, limit], history_hit_from_row)?;
            for row in rows {
                hits.push(row?);
            }
        }

        Ok(hits)
    }

    fn episode_by_source(&self, source_type: &str, source_ref: &str) -> Result<Option<Episode>> {
        Ok(self
            .connection
            .query_row(
                "SELECT id, project_id, workspace_id, source_type, source_ref, started_at,\n\
                        ended_at, summary, metadata_json\n\
                 FROM episodes\n\
                 WHERE source_type = ?1 AND source_ref = ?2",
                params![source_type, source_ref],
                episode_from_row,
            )
            .optional()?)
    }

    fn episode_by_id(&self, id: &str) -> Result<Episode> {
        Ok(self.connection.query_row(
            "SELECT id, project_id, workspace_id, source_type, source_ref, started_at,\n\
                    ended_at, summary, metadata_json\n\
             FROM episodes WHERE id = ?1",
            [id],
            episode_from_row,
        )?)
    }

    fn episode_entry_by_id(&self, id: &str) -> Result<EpisodeEntry> {
        Ok(self.connection.query_row(
            "SELECT id, episode_id, source_ref, ordinal, kind, role, text, occurred_at,\n\
                    metadata_json\n\
             FROM episode_entries WHERE id = ?1",
            [id],
            episode_entry_from_row,
        )?)
    }
}

fn validate_episode(input: &NewEpisode) -> Result<()> {
    validate_nonempty(&input.source_type, "episode source type")?;
    validate_nonempty(&input.source_ref, "episode source reference")?;
    if input.project_id.is_none() && input.workspace_id.is_some() {
        bail!("episode workspace requires a project");
    }
    if let Some(project) = input.project_id.as_deref() {
        validate_nonempty(project, "project identifier")?;
    }
    if let Some(workspace) = input.workspace_id.as_deref() {
        validate_nonempty(workspace, "workspace identifier")?;
    }
    validate_json(input.metadata_json.as_deref(), "episode metadata")
}

fn validate_episode_entry(input: &NewEpisodeEntry) -> Result<()> {
    validate_nonempty(&input.source_ref, "episode entry source reference")?;
    validate_nonempty(&input.kind, "episode entry kind")?;
    validate_nonempty(&input.text, "episode entry text")?;
    if input.ordinal.is_some_and(|ordinal| ordinal < 0) {
        bail!("episode entry ordinal cannot be negative");
    }
    if let Some(role) = input.role.as_deref() {
        validate_nonempty(role, "episode entry role")?;
    }
    validate_json(input.metadata_json.as_deref(), "episode entry metadata")
}

fn validate_nonempty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{name} cannot be empty");
    }
    Ok(())
}

fn validate_json(value: Option<&str>, name: &str) -> Result<()> {
    if let Some(value) = value {
        serde_json::from_str::<serde_json::Value>(value)
            .with_context(|| format!("{name} must be valid JSON"))?;
    }
    Ok(())
}

fn episode_from_row(row: &Row<'_>) -> rusqlite::Result<Episode> {
    Ok(Episode {
        id: row.get(0)?,
        project_id: row.get(1)?,
        workspace_id: row.get(2)?,
        source_type: row.get(3)?,
        source_ref: row.get(4)?,
        started_at: row.get(5)?,
        ended_at: row.get(6)?,
        summary: row.get(7)?,
        metadata_json: row.get(8)?,
    })
}

fn episode_entry_from_row(row: &Row<'_>) -> rusqlite::Result<EpisodeEntry> {
    Ok(EpisodeEntry {
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
}

fn history_hit_from_row(row: &Row<'_>) -> rusqlite::Result<HistoryHit> {
    Ok(HistoryHit {
        episode_id: row.get(0)?,
        project_id: row.get(1)?,
        workspace_id: row.get(2)?,
        source_type: row.get(3)?,
        episode_source_ref: row.get(4)?,
        entry_id: row.get(5)?,
        entry_source_ref: row.get(6)?,
        ordinal: row.get(7)?,
        kind: row.get(8)?,
        role: row.get(9)?,
        text: row.get(10)?,
        occurred_at: row.get(11)?,
        rank: row.get(12)?,
    })
}

fn unix_millis() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    Ok(i64::try_from(duration.as_millis())?)
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;
    use uuid::Uuid;

    use super::{NewEpisode, NewEpisodeEntry};
    use crate::store::Store;

    #[test]
    fn history_preserves_exact_source_backreferences() {
        let path = test_path();
        let store = Store::open(&path).expect("open test store");
        let project = "github.com/nijaru/mem";

        let episode = store
            .ensure_episode(NewEpisode {
                project_id: Some(project.to_owned()),
                workspace_id: Some("branch:main".to_owned()),
                source_type: "session".to_owned(),
                source_ref: "session-1".to_owned(),
                started_at: Some(10),
                metadata_json: None,
            })
            .expect("create episode");
        let duplicate = store
            .ensure_episode(NewEpisode {
                project_id: Some(project.to_owned()),
                workspace_id: Some("branch:main".to_owned()),
                source_type: "session".to_owned(),
                source_ref: "session-1".to_owned(),
                started_at: Some(10),
                metadata_json: None,
            })
            .expect("ensure episode idempotently");
        assert_eq!(duplicate.id, episode.id);

        store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "entry-7".to_owned(),
                    ordinal: Some(7),
                    kind: "message".to_owned(),
                    role: Some("assistant".to_owned()),
                    text: "publication handoff succeeded".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record episode entry");

        let hits = store
            .history_search("publication handoff", Some(project), 10)
            .expect("search history");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].episode_source_ref, "session-1");
        assert_eq!(hits[0].entry_source_ref, "entry-7");

        // Ranked visibility: a term that appears in no entry text must not
        // discard the entry that matches the remaining terms.
        let partial = store
            .history_search("publication nonexistentterm", Some(project), 10)
            .expect("search history with a non-matching term");
        assert_eq!(
            partial.len(),
            1,
            "partial term matches must stay visible under bm25 ranking"
        );

        let record = store.get_episode(&episode.id).expect("read episode");
        assert_eq!(record.entries.len(), 1);
        assert_eq!(record.entries[0].source_ref, "entry-7");
        assert_eq!(store.stats().expect("read stats").schema_version, 5);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn recording_entries_never_queues_embedding_work() {
        // Episodic history is lexical-only in v0.x; no trigger or reader
        // maintains episode-entry vectors, so recording entries must leave
        // the embedding queue untouched.
        let path = test_path();
        let store = Store::open(&path).expect("open test store");
        let episode = store
            .ensure_episode(NewEpisode {
                project_id: None,
                workspace_id: None,
                source_type: "transcript".to_owned(),
                source_ref: "session-1".to_owned(),
                started_at: Some(10),
                metadata_json: None,
            })
            .expect("create episode");
        store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: None,
                    text: "recorded without embedding work".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record entry");
        let stats = store.index_job_stats().expect("read job stats");
        assert_eq!(stats.pending + stats.running, 0);
        let episode_vectors: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE entity_type = 'episode_entry'",
                [],
                |row| row.get(0),
            )
            .expect("count episode vectors");
        assert_eq!(episode_vectors, 0);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn opening_a_v1_database_runs_episode_migration() {
        let path = test_path();
        let connection = Connection::open(&path).expect("open v1 database");
        connection
            .execute_batch(include_str!("../migrations/0001_initial.sql"))
            .expect("create v1 schema");
        drop(connection);

        let store = Store::open(&path).expect("migrate v1 database");
        assert_eq!(store.stats().expect("read stats").schema_version, 5);
        drop(store);
        cleanup(&path);
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-episode-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }

    #[test]
    fn concurrent_ensure_and_entry_upserts_are_idempotent() {
        let path = test_path();
        // Two real threads hitting the same episode source simultaneously:
        // the check-then-insert window is what a sequential test cannot
        // compress.
        let db = path.display().to_string();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let make_input = |source_ref: &'static str| NewEpisode {
            project_id: Some("github.com/x/race".to_owned()),
            workspace_id: None,
            source_type: "session".to_owned(),
            source_ref: source_ref.to_owned(),
            started_at: Some(1),
            metadata_json: None,
        };
        let episodes: Vec<std::sync::Arc<NewEpisode>> = (0..2)
            .map(|_| std::sync::Arc::new(make_input("threaded")))
            .collect();
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let db = db.clone();
                let barrier = barrier.clone();
                let input = episodes[i].clone();
                std::thread::spawn(move || {
                    let store = Store::open(std::path::Path::new(&db))
                        .unwrap_or_else(|error| panic!("open store: {error:#}"));
                    barrier.wait();
                    store.ensure_episode((*input).clone())
                })
            })
            .collect();
        let mut ids = Vec::new();
        for handle in handles {
            match handle.join().expect("thread panicked") {
                Ok(episode) => ids.push(episode.id),
                Err(error) => panic!("concurrent ensure failed: {error:#}"),
            }
        }
        assert_eq!(ids[0], ids[1], "both threads must converge on one episode");

        let a = Store::open(&path).expect("open writer a");
        let b = Store::open(std::path::Path::new(a.path().expect("path"))).expect("open writer b");

        let input = NewEpisode {
            project_id: Some("github.com/x/race".to_owned()),
            workspace_id: None,
            source_type: "session".to_owned(),
            source_ref: "shared".to_owned(),
            started_at: Some(1),
            metadata_json: None,
        };
        let first = a.ensure_episode(input.clone()).expect("writer a creates");
        let second = b
            .ensure_episode(input)
            .expect("writer b must upsert, not fail");
        assert_eq!(first.id, second.id, "both writers must see one episode");

        // Entries: concurrent first-record of the same source_ref upserts.
        let entry = |text: &str| NewEpisodeEntry {
            source_ref: "e-1".to_owned(),
            ordinal: None,
            kind: "message".to_owned(),
            role: Some("user".to_owned()),
            text: text.to_owned(),
            occurred_at: Some(2),
            metadata_json: None,
        };
        let first_entry = a
            .record_episode_entry(&first.id, entry("from a"))
            .expect("writer a records entry");
        let second_entry = b
            .record_episode_entry(&first.id, entry("from b"))
            .expect("writer b must upsert, not fail");
        assert_eq!(first_entry.id, second_entry.id, "one entry row, not two");
        assert_eq!(
            second_entry.text, "from b",
            "last write wins, matching the sequential upsert"
        );

        cleanup(&path);
    }
}
