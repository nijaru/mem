BEGIN IMMEDIATE;

CREATE TABLE index_jobs (
    id TEXT PRIMARY KEY NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('memory', 'episode_entry')),
    entity_id TEXT NOT NULL,
    index_kind TEXT NOT NULL CHECK (index_kind = 'embedding'),
    generation INTEGER NOT NULL CHECK (generation > 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'running')),
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    available_at INTEGER NOT NULL,
    lease_owner TEXT,
    lease_token TEXT,
    lease_until INTEGER,
    last_error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (entity_type, entity_id, index_kind),
    CHECK (
        (state = 'pending' AND lease_owner IS NULL AND lease_token IS NULL AND lease_until IS NULL) OR
        (state = 'running' AND lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_until IS NOT NULL)
    )
);

CREATE INDEX index_jobs_claim_idx
    ON index_jobs (state, available_at, lease_until, updated_at, id);

INSERT INTO index_jobs (
    id, entity_type, entity_id, index_kind, generation, state, attempts,
    available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
)
SELECT
    'memory:' || id || ':embedding',
    'memory',
    id,
    'embedding',
    1,
    'pending',
    0,
    updated_at,
    NULL,
    NULL,
    NULL,
    NULL,
    created_at,
    updated_at
FROM memories
WHERE status = 'active';

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
JOIN episodes AS e ON e.id = ee.episode_id;

CREATE TRIGGER memories_index_job_insert AFTER INSERT ON memories
WHEN new.status = 'active'
BEGIN
    INSERT INTO index_jobs (
        id, entity_type, entity_id, index_kind, generation, state, attempts,
        available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
    ) VALUES (
        'memory:' || new.id || ':embedding',
        'memory', new.id, 'embedding', 1, 'pending', 0,
        new.updated_at, NULL, NULL, NULL, NULL, new.created_at, new.updated_at
    )
    ON CONFLICT(entity_type, entity_id, index_kind) DO UPDATE SET
        generation = index_jobs.generation + 1,
        state = 'pending',
        attempts = 0,
        available_at = excluded.available_at,
        lease_owner = NULL,
        lease_token = NULL,
        lease_until = NULL,
        last_error = NULL,
        updated_at = excluded.updated_at;
END;

CREATE TRIGGER memories_index_job_active_update AFTER UPDATE OF text, status ON memories
WHEN new.status = 'active'
BEGIN
    INSERT INTO index_jobs (
        id, entity_type, entity_id, index_kind, generation, state, attempts,
        available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
    ) VALUES (
        'memory:' || new.id || ':embedding',
        'memory', new.id, 'embedding', 1, 'pending', 0,
        new.updated_at, NULL, NULL, NULL, NULL, new.created_at, new.updated_at
    )
    ON CONFLICT(entity_type, entity_id, index_kind) DO UPDATE SET
        generation = index_jobs.generation + 1,
        state = 'pending',
        attempts = 0,
        available_at = excluded.available_at,
        lease_owner = NULL,
        lease_token = NULL,
        lease_until = NULL,
        last_error = NULL,
        updated_at = excluded.updated_at;
END;

CREATE TRIGGER memories_index_job_inactive_update AFTER UPDATE OF status ON memories
WHEN new.status != 'active'
BEGIN
    DELETE FROM index_jobs
    WHERE entity_type = 'memory' AND entity_id = new.id AND index_kind = 'embedding';
END;

CREATE TRIGGER memories_index_job_delete AFTER DELETE ON memories
BEGIN
    DELETE FROM index_jobs
    WHERE entity_type = 'memory' AND entity_id = old.id AND index_kind = 'embedding';
END;

CREATE TRIGGER episode_entries_index_job_insert AFTER INSERT ON episode_entries
BEGIN
    INSERT INTO index_jobs (
        id, entity_type, entity_id, index_kind, generation, state, attempts,
        available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
    ) VALUES (
        'episode_entry:' || new.id || ':embedding',
        'episode_entry', new.id, 'embedding', 1, 'pending', 0,
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000),
        NULL, NULL, NULL, NULL,
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000),
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000)
    )
    ON CONFLICT(entity_type, entity_id, index_kind) DO UPDATE SET
        generation = index_jobs.generation + 1,
        state = 'pending',
        attempts = 0,
        available_at = excluded.available_at,
        lease_owner = NULL,
        lease_token = NULL,
        lease_until = NULL,
        last_error = NULL,
        updated_at = excluded.updated_at;
END;

CREATE TRIGGER episode_entries_index_job_update AFTER UPDATE OF text ON episode_entries
BEGIN
    INSERT INTO index_jobs (
        id, entity_type, entity_id, index_kind, generation, state, attempts,
        available_at, lease_owner, lease_token, lease_until, last_error, created_at, updated_at
    ) VALUES (
        'episode_entry:' || new.id || ':embedding',
        'episode_entry', new.id, 'embedding', 1, 'pending', 0,
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000),
        NULL, NULL, NULL, NULL,
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000),
        COALESCE(new.occurred_at, CAST(strftime('%s', 'now') AS INTEGER) * 1000)
    )
    ON CONFLICT(entity_type, entity_id, index_kind) DO UPDATE SET
        generation = index_jobs.generation + 1,
        state = 'pending',
        attempts = 0,
        available_at = excluded.available_at,
        lease_owner = NULL,
        lease_token = NULL,
        lease_until = NULL,
        last_error = NULL,
        updated_at = excluded.updated_at;
END;

CREATE TRIGGER episode_entries_index_job_delete AFTER DELETE ON episode_entries
BEGIN
    DELETE FROM index_jobs
    WHERE entity_type = 'episode_entry' AND entity_id = old.id AND index_kind = 'embedding';
END;

PRAGMA user_version = 3;
COMMIT;
