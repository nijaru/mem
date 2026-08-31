BEGIN IMMEDIATE;

CREATE TABLE embeddings (
    entity_type TEXT NOT NULL CHECK (entity_type IN ('memory', 'episode_entry')),
    entity_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    source_generation INTEGER NOT NULL CHECK (source_generation > 0),
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (entity_type, entity_id, model)
);

CREATE INDEX embeddings_model_entity_idx
    ON embeddings (model, entity_type, entity_id);

-- A running embedding job may only be acknowledged while its canonical source
-- remains eligible if the exact generation has first produced a vector. Source
-- deletion/supersession is still allowed to cancel the job because the source
-- is no longer eligible when the v3 cancellation trigger deletes it.
CREATE TRIGGER index_jobs_embedding_requires_result_before_delete
BEFORE DELETE ON index_jobs
WHEN old.index_kind = 'embedding'
  AND old.state = 'running'
  AND (
      (old.entity_type = 'memory' AND EXISTS (
          SELECT 1 FROM memories AS m
          WHERE m.id = old.entity_id AND m.status = 'active'
      ))
      OR
      (old.entity_type = 'episode_entry' AND EXISTS (
          SELECT 1 FROM episode_entries AS ee
          WHERE ee.id = old.entity_id
      ))
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

-- A canonical source change makes all previous vectors for that entity stale
-- immediately. The v3 queue triggers independently advance/requeue the job.
CREATE TRIGGER memories_embedding_invalidate_active_update
AFTER UPDATE OF text, status ON memories
WHEN new.status = 'active'
BEGIN
    DELETE FROM embeddings
    WHERE entity_type = 'memory' AND entity_id = new.id;
END;

CREATE TRIGGER memories_embedding_invalidate_inactive_update
AFTER UPDATE OF status ON memories
WHEN new.status != 'active'
BEGIN
    DELETE FROM embeddings
    WHERE entity_type = 'memory' AND entity_id = new.id;
END;

CREATE TRIGGER memories_embedding_invalidate_delete
AFTER DELETE ON memories
BEGIN
    DELETE FROM embeddings
    WHERE entity_type = 'memory' AND entity_id = old.id;
END;

CREATE TRIGGER episode_entries_embedding_invalidate_update
AFTER UPDATE OF text ON episode_entries
BEGIN
    DELETE FROM embeddings
    WHERE entity_type = 'episode_entry' AND entity_id = new.id;
END;

CREATE TRIGGER episode_entries_embedding_invalidate_delete
AFTER DELETE ON episode_entries
BEGIN
    DELETE FROM embeddings
    WHERE entity_type = 'episode_entry' AND entity_id = old.id;
END;

-- A v3 database could have had queue rows manually acknowledged before vector
-- storage existed. Ensure every eligible source without outstanding work is
-- re-enqueued when v4 is first opened.
INSERT INTO index_jobs (
    id, entity_type, entity_id, index_kind, generation, state, attempts,
    available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
)
SELECT
    'memory:' || m.id || ':embedding',
    'memory',
    m.id,
    'embedding',
    1,
    'pending',
    0,
    m.updated_at,
    NULL,
    NULL,
    NULL,
    NULL,
    m.created_at,
    m.updated_at
FROM memories AS m
WHERE m.status = 'active'
  AND NOT EXISTS (
      SELECT 1 FROM index_jobs AS j
      WHERE j.entity_type = 'memory'
        AND j.entity_id = m.id
        AND j.index_kind = 'embedding'
  );

INSERT INTO index_jobs (
    id, entity_type, entity_id, index_kind, generation, state, attempts,
    available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
)
SELECT
    'episode_entry:' || ee.id || ':embedding',
    'episode_entry',
    ee.id,
    'embedding',
    1,
    'pending',
    0,
    COALESCE(ee.occurred_at, e.started_at, 0),
    NULL,
    NULL,
    NULL,
    NULL,
    COALESCE(ee.occurred_at, e.started_at, 0),
    COALESCE(ee.occurred_at, e.started_at, 0)
FROM episode_entries AS ee
JOIN episodes AS e ON e.id = ee.episode_id
WHERE NOT EXISTS (
    SELECT 1 FROM index_jobs AS j
    WHERE j.entity_type = 'episode_entry'
      AND j.entity_id = ee.id
      AND j.index_kind = 'embedding'
);

PRAGMA user_version = 4;
COMMIT;
