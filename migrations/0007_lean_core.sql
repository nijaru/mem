DROP TRIGGER IF EXISTS memories_fts_insert;
DROP TRIGGER IF EXISTS memories_fts_delete;
DROP TRIGGER IF EXISTS memories_fts_update;
DROP TRIGGER IF EXISTS episode_entries_fts_insert;
DROP TRIGGER IF EXISTS episode_entries_fts_delete;
DROP TRIGGER IF EXISTS episode_entries_fts_update;
DROP TRIGGER IF EXISTS memories_index_job_insert;
DROP TRIGGER IF EXISTS memories_index_job_active_update;
DROP TRIGGER IF EXISTS memories_index_job_inactive_update;
DROP TRIGGER IF EXISTS memories_index_job_delete;
DROP TRIGGER IF EXISTS episode_entries_index_job_insert;
DROP TRIGGER IF EXISTS episode_entries_index_job_update;
DROP TRIGGER IF EXISTS episode_entries_index_job_delete;
DROP TRIGGER IF EXISTS index_jobs_embedding_requires_result_before_delete;
DROP TRIGGER IF EXISTS memories_embedding_invalidate_active_update;
DROP TRIGGER IF EXISTS memories_embedding_invalidate_inactive_update;
DROP TRIGGER IF EXISTS memories_embedding_invalidate_delete;
DROP TRIGGER IF EXISTS episode_entries_embedding_invalidate_update;
DROP TRIGGER IF EXISTS episode_entries_embedding_invalidate_delete;

DROP TABLE IF EXISTS memories_fts;
DROP TABLE IF EXISTS episode_entries_fts;

ALTER TABLE memories RENAME TO memories_v6;
ALTER TABLE memory_sources RENAME TO memory_sources_v6;
ALTER TABLE memory_relations RENAME TO memory_relations_v6;
ALTER TABLE workspace_state RENAME TO workspace_state_v6_old;
ALTER TABLE embeddings RENAME TO embeddings_v6;

CREATE TABLE memories (
    id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    actor TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted')),
    superseded_by TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    deleted_at INTEGER
);

INSERT INTO memories (
    id, kind, text, actor, source_type, source_ref, status,
    superseded_by, created_at, updated_at, deleted_at
)
SELECT
    m.id,
    m.kind,
    m.text,
    m.actor,
    COALESCE((
        SELECT s.source_type
        FROM memory_sources_v6 AS s
        WHERE s.memory_id = m.id
        ORDER BY s.created_at, s.id
        LIMIT 1
    ), 'legacy'),
    (
        SELECT s.locator
        FROM memory_sources_v6 AS s
        WHERE s.memory_id = m.id
        ORDER BY s.created_at, s.id
        LIMIT 1
    ),
    m.status,
    (
        SELECT r.to_memory_id
        FROM memory_relations_v6 AS r
        WHERE r.from_memory_id = m.id
          AND r.relation_type = 'superseded_by'
        ORDER BY r.created_at, r.to_memory_id
        LIMIT 1
    ),
    m.created_at,
    m.updated_at,
    m.deleted_at
FROM memories_v6 AS m;

CREATE INDEX memories_status_updated_idx
    ON memories (status, updated_at DESC, id);

CREATE TABLE workspace_state (
    workspace TEXT PRIMARY KEY NOT NULL,
    session TEXT,
    goal TEXT,
    task TEXT,
    checkpoint TEXT,
    updated_at INTEGER NOT NULL
);

INSERT INTO workspace_state (workspace, session, goal, task, checkpoint, updated_at)
SELECT workspace_id, last_session_id, active_goal, active_task_ref, checkpoint, updated_at
FROM workspace_state_v6_old;

CREATE TABLE embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    source_updated_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (memory_id, model)
);

INSERT INTO embeddings (
    memory_id, model, dimensions, vector, source_updated_at, updated_at
)
SELECT e.entity_id, e.model, e.dimensions, e.vector, m.updated_at, e.updated_at
FROM embeddings_v6 AS e
JOIN memories AS m ON m.id = e.entity_id
WHERE e.entity_type = 'memory'
  AND m.status = 'active';

CREATE INDEX embeddings_model_memory_idx
    ON embeddings (model, memory_id);

DROP TABLE IF EXISTS index_jobs;
DROP TABLE IF EXISTS episode_entries;
DROP TABLE IF EXISTS episodes;
DROP TABLE memory_sources_v6;
DROP TABLE memory_relations_v6;
DROP TABLE workspace_state_v6_old;
DROP TABLE embeddings_v6;
DROP TABLE memories_v6;

CREATE VIRTUAL TABLE memories_fts USING fts5(
    text,
    content = 'memories',
    content_rowid = 'rowid',
    tokenize = 'unicode61'
);

INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');

CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER memories_fts_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER memories_embedding_invalidate
AFTER UPDATE OF text, status ON memories
BEGIN
    DELETE FROM embeddings WHERE memory_id = new.id;
END;

PRAGMA user_version = 7;
