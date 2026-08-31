use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rusqlite::{TransactionBehavior, params};
use serde::Serialize;
use uuid::Uuid;

use crate::store::Store;

#[derive(Debug, Serialize)]
pub struct IndexJobStats {
    pub pending: u64,
    pub running: u64,
}

#[derive(Debug, Serialize)]
pub struct ClaimedIndexJob {
    pub id: String,
    pub entity_type: String,
    pub entity_id: String,
    pub index_kind: String,
    pub generation: i64,
    pub attempts: i64,
    pub lease_owner: String,
    pub lease_token: String,
    pub lease_until: i64,
}

struct ClaimCandidate {
    id: String,
    entity_type: String,
    entity_id: String,
    index_kind: String,
    generation: i64,
    attempts: i64,
}

impl Store {
    pub fn index_job_stats(&self) -> Result<IndexJobStats> {
        let (pending, running): (i64, i64) = self.connection.query_row(
            "SELECT COALESCE(SUM(state = 'pending'), 0),\n\
                    COALESCE(SUM(state = 'running'), 0)\n\
             FROM index_jobs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(IndexJobStats {
            pending: pending.try_into()?,
            running: running.try_into()?,
        })
    }

    pub fn claim_index_jobs(
        &mut self,
        worker_id: &str,
        limit: usize,
        lease_duration: Duration,
    ) -> Result<Vec<ClaimedIndexJob>> {
        let worker_id = worker_id.trim();
        if worker_id.is_empty() {
            bail!("index worker identifier cannot be empty");
        }
        if limit == 0 {
            bail!("index job claim limit must be greater than zero");
        }
        let lease_millis = i64::try_from(lease_duration.as_millis())?;
        if lease_millis <= 0 {
            bail!("index job lease duration must be greater than zero");
        }

        let now = unix_millis()?;
        let lease_until = now
            .checked_add(lease_millis)
            .context("index job lease timestamp overflow")?;
        let limit = i64::try_from(limit)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;

        let candidates = {
            let mut statement = transaction.prepare(
                "SELECT id, entity_type, entity_id, index_kind, generation, attempts\n\
                 FROM index_jobs\n\
                 WHERE (state = 'pending' AND available_at <= ?1)\n\
                    OR (state = 'running' AND lease_until <= ?1)\n\
                 ORDER BY CASE state WHEN 'pending' THEN 0 ELSE 1 END,\n\
                          available_at, updated_at, id\n\
                 LIMIT ?2",
            )?;
            let rows = statement.query_map(params![now, limit], |row| {
                Ok(ClaimCandidate {
                    id: row.get(0)?,
                    entity_type: row.get(1)?,
                    entity_id: row.get(2)?,
                    index_kind: row.get(3)?,
                    generation: row.get(4)?,
                    attempts: row.get(5)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut claimed = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let lease_token = Uuid::now_v7().to_string();
            let changed = transaction.execute(
                "UPDATE index_jobs\n\
                 SET state = 'running',\n\
                     attempts = attempts + 1,\n\
                     lease_owner = ?2,\n\
                     lease_token = ?3,\n\
                     lease_until = ?4,\n\
                     updated_at = ?5\n\
                 WHERE id = ?1\n\
                   AND generation = ?6\n\
                   AND ((state = 'pending' AND available_at <= ?5)\n\
                     OR (state = 'running' AND lease_until <= ?5))",
                params![
                    &candidate.id,
                    worker_id,
                    &lease_token,
                    lease_until,
                    now,
                    candidate.generation
                ],
            )?;
            if changed == 1 {
                claimed.push(ClaimedIndexJob {
                    id: candidate.id,
                    entity_type: candidate.entity_type,
                    entity_id: candidate.entity_id,
                    index_kind: candidate.index_kind,
                    generation: candidate.generation,
                    attempts: candidate.attempts + 1,
                    lease_owner: worker_id.to_owned(),
                    lease_token,
                    lease_until,
                });
            }
        }

        transaction.commit()?;
        Ok(claimed)
    }

    pub fn complete_index_job(
        &self,
        job_id: &str,
        generation: i64,
        lease_token: &str,
    ) -> Result<bool> {
        validate_completion(job_id, generation, lease_token)?;
        Ok(self.connection.execute(
            "DELETE FROM index_jobs\n\
             WHERE id = ?1 AND generation = ?2 AND state = 'running' AND lease_token = ?3",
            params![job_id, generation, lease_token],
        )? == 1)
    }

    pub fn retry_index_job(
        &self,
        job_id: &str,
        generation: i64,
        lease_token: &str,
        error: &str,
        retry_delay: Duration,
    ) -> Result<bool> {
        validate_completion(job_id, generation, lease_token)?;
        let now = unix_millis()?;
        let delay_millis = i64::try_from(retry_delay.as_millis())?;
        let available_at = now
            .checked_add(delay_millis)
            .context("index job retry timestamp overflow")?;
        Ok(self.connection.execute(
            "UPDATE index_jobs\n\
             SET state = 'pending',\n\
                 available_at = ?4,\n\
                 lease_owner = NULL,\n\
                 lease_token = NULL,\n\
                 lease_until = NULL,\n\
                 last_error = ?5,\n\
                 updated_at = ?6\n\
             WHERE id = ?1 AND generation = ?2 AND state = 'running' AND lease_token = ?3",
            params![job_id, generation, lease_token, available_at, error, now],
        )? == 1)
    }
}

fn validate_completion(job_id: &str, generation: i64, lease_token: &str) -> Result<()> {
    if job_id.trim().is_empty() {
        bail!("index job identifier cannot be empty");
    }
    if generation <= 0 {
        bail!("index job generation must be greater than zero");
    }
    if lease_token.trim().is_empty() {
        bail!("index job lease token cannot be empty");
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
    use std::time::Duration;

    use rusqlite::{Connection, params};
    use uuid::Uuid;

    use super::Store;
    use crate::episode::{NewEpisode, NewEpisodeEntry};
    use crate::store::NewMemory;

    #[test]
    fn canonical_writes_enqueue_and_completion_removes_jobs() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let memory = store
            .remember(NewMemory {
                text: "derived index queue".to_owned(),
                kind: "fact".to_owned(),
                project_id: Some("github.com/nijaru/mem".to_owned()),
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");

        let stats = store.index_job_stats().expect("read job stats");
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.running, 0);

        let claimed = store
            .claim_index_jobs("worker-a", 10, Duration::from_secs(30))
            .expect("claim jobs");
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].entity_type, "memory");
        assert_eq!(claimed[0].entity_id, memory.id);
        assert_eq!(claimed[0].generation, 1);
        assert_eq!(claimed[0].attempts, 1);

        assert!(
            store
                .complete_index_job(
                    &claimed[0].id,
                    claimed[0].generation,
                    &claimed[0].lease_token,
                )
                .expect("complete job")
        );
        let stats = store.index_job_stats().expect("read completed stats");
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.running, 0);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn requeue_fences_a_stale_worker_generation() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        let episode = store
            .ensure_episode(NewEpisode {
                project_id: Some("github.com/nijaru/mem".to_owned()),
                workspace_id: Some("branch:main".to_owned()),
                source_type: "pi-session".to_owned(),
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
                    role: Some("assistant".to_owned()),
                    text: "first indexed text".to_owned(),
                    occurred_at: Some(20),
                    metadata_json: None,
                },
            )
            .expect("record entry");

        let first = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim first generation")
            .pop()
            .expect("first job");
        assert_eq!(first.generation, 1);

        store
            .record_episode_entry(
                &episode.id,
                NewEpisodeEntry {
                    source_ref: "message-1".to_owned(),
                    ordinal: Some(0),
                    kind: "message".to_owned(),
                    role: Some("assistant".to_owned()),
                    text: "updated indexed text".to_owned(),
                    occurred_at: Some(30),
                    metadata_json: None,
                },
            )
            .expect("refresh entry");

        assert!(
            !store
                .complete_index_job(&first.id, first.generation, &first.lease_token)
                .expect("reject stale completion")
        );
        let second = store
            .claim_index_jobs("worker-b", 1, Duration::from_secs(30))
            .expect("claim refreshed generation")
            .pop()
            .expect("refreshed job");
        assert_eq!(second.id, first.id);
        assert_eq!(second.generation, 2);
        assert_ne!(second.lease_token, first.lease_token);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn expired_running_job_can_be_reclaimed() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        store
            .remember(NewMemory {
                text: "lease recovery".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
        let first = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim job")
            .pop()
            .expect("claimed job");
        store
            .connection
            .execute(
                "UPDATE index_jobs SET lease_until = 0 WHERE id = ?1",
                [&first.id],
            )
            .expect("expire lease");

        let second = store
            .claim_index_jobs("worker-b", 1, Duration::from_secs(30))
            .expect("reclaim expired job")
            .pop()
            .expect("reclaimed job");
        assert_eq!(second.id, first.id);
        assert_eq!(second.generation, first.generation);
        assert_eq!(second.attempts, 2);
        assert_ne!(second.lease_token, first.lease_token);
        assert!(
            !store
                .complete_index_job(&first.id, first.generation, &first.lease_token)
                .expect("old lease cannot complete")
        );

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn v2_migration_backfills_existing_derived_work() {
        let path = test_path();
        let connection = Connection::open(&path).expect("open v2 database");
        connection
            .execute_batch(include_str!("../migrations/0001_initial.sql"))
            .expect("create v1 schema");
        connection
            .execute_batch(include_str!("../migrations/0002_episode_entries.sql"))
            .expect("create v2 schema");
        connection
            .execute(
                "INSERT INTO memories (\n\
                     id, scope, project_id, kind, text, actor, status, created_at, updated_at\n\
                 ) VALUES ('memory-1', 'global', NULL, 'fact', 'existing memory', 'agent',\n\
                           'active', 10, 10)",
                [],
            )
            .expect("insert existing memory");
        connection
            .execute(
                "INSERT INTO episodes (\n\
                     id, project_id, source_type, source_ref, started_at, ended_at, summary,\n\
                     metadata_json, workspace_id\n\
                 ) VALUES ('episode-1', NULL, 'pi-session', 'session-1', 20, NULL, NULL, NULL, NULL)",
                [],
            )
            .expect("insert existing episode");
        connection
            .execute(
                "INSERT INTO episode_entries (\n\
                     id, episode_id, source_ref, ordinal, kind, role, text, occurred_at, metadata_json\n\
                 ) VALUES ('entry-1', 'episode-1', 'message-1', 0, 'message', 'assistant',\n\
                           'existing entry', 30, NULL)",
                [],
            )
            .expect("insert existing entry");
        drop(connection);

        let store = Store::open(&path).expect("migrate v2 database");
        assert_eq!(store.stats().expect("read store stats").schema_version, 3);
        let stats = store.index_job_stats().expect("read job stats");
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.running, 0);
        let count: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM index_jobs WHERE generation = 1 AND state = 'pending'",
                [],
                |row| row.get(0),
            )
            .expect("count backfilled jobs");
        assert_eq!(count, 2);

        drop(store);
        cleanup(&path);
    }

    #[test]
    fn retry_releases_lease_and_defers_reclaim() {
        let path = test_path();
        let mut store = Store::open(&path).expect("open test store");
        store
            .remember(NewMemory {
                text: "retry indexing".to_owned(),
                kind: "fact".to_owned(),
                project_id: None,
                actor: "agent".to_owned(),
                source_type: "test".to_owned(),
                source_ref: None,
            })
            .expect("store memory");
        let claimed = store
            .claim_index_jobs("worker-a", 1, Duration::from_secs(30))
            .expect("claim job")
            .pop()
            .expect("claimed job");
        assert!(
            store
                .retry_index_job(
                    &claimed.id,
                    claimed.generation,
                    &claimed.lease_token,
                    "temporary failure",
                    Duration::from_secs(60),
                )
                .expect("retry job")
        );
        assert!(
            store
                .claim_index_jobs("worker-b", 1, Duration::from_secs(30))
                .expect("respect retry delay")
                .is_empty()
        );
        let row: (
            String,
            Option<String>,
            Option<String>,
            Option<i64>,
            Option<String>,
        ) = store
            .connection
            .query_row(
                "SELECT state, lease_owner, lease_token, lease_until, last_error\n\
                 FROM index_jobs WHERE id = ?1",
                params![claimed.id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .expect("read retried job");
        assert_eq!(row.0, "pending");
        assert!(row.1.is_none());
        assert!(row.2.is_none());
        assert!(row.3.is_none());
        assert_eq!(row.4.as_deref(), Some("temporary failure"));

        drop(store);
        cleanup(&path);
    }

    fn test_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mem-index-job-test-{}.db", Uuid::now_v7()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
    }
}
