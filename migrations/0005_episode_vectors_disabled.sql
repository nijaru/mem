BEGIN IMMEDIATE;

-- Episodic history is lexical-only in v0.x: no reader consumes episode-entry
-- vectors, so maintaining them spends model time and database space on derived
-- data with no consumer and lets episode jobs compete with semantic-memory
-- jobs for the bounded worker limit. This migration stops enqueueing episode
-- embedding work, drains whatever is queued, and removes existing episode
-- vectors. The embeddings/index_jobs CHECK constraints still admit
-- 'episode_entry' (SQLite cannot alter a CHECK without a table rebuild); no
-- trigger can produce such rows anymore.

-- Narrow the delete-guard to memories first: it must stop blocking the drain
-- of running episode jobs. The "no embedding work discarded without a
-- persisted result" invariant stays enforced where derived data is read.
DROP TRIGGER index_jobs_embedding_requires_result_before_delete;
CREATE TRIGGER index_jobs_embedding_requires_result_before_delete
BEFORE DELETE ON index_jobs
WHEN old.index_kind = 'embedding'
  AND old.state = 'running'
  AND old.entity_type = 'memory'
  AND EXISTS (
      SELECT 1 FROM memories AS m
      WHERE m.id = old.entity_id AND m.status = 'active'
  )
  AND NOT EXISTS (
      SELECT 1 FROM embeddings AS emb
      WHERE emb.entity_type = old.entity_type
        AND emb.entity_id = old.entity_id
        AND emb.source_generation = old.generation
  )
BEGIN
    SELECT RAISE(ABORT, 'embedding job cannot complete without a persisted result');
END;

DROP TRIGGER episode_entries_index_job_insert;
DROP TRIGGER episode_entries_index_job_update;
DROP TRIGGER episode_entries_index_job_delete;
DROP TRIGGER episode_entries_embedding_invalidate_update;
DROP TRIGGER episode_entries_embedding_invalidate_delete;

DELETE FROM index_jobs WHERE entity_type = 'episode_entry';
DELETE FROM embeddings WHERE entity_type = 'episode_entry';

PRAGMA user_version = 5;
COMMIT;
