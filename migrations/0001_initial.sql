BEGIN IMMEDIATE;

CREATE TABLE memories (
    id TEXT PRIMARY KEY NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('global', 'project')),
    project_id TEXT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    deleted_at INTEGER,
    CHECK (
        (scope = 'global' AND project_id IS NULL) OR
        (scope = 'project' AND project_id IS NOT NULL)
    )
);

CREATE INDEX memories_scope_status_idx
    ON memories (project_id, status, updated_at DESC);

CREATE TABLE memory_sources (
    id TEXT PRIMARY KEY NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    locator TEXT,
    excerpt TEXT,
    created_at INTEGER NOT NULL
);

CREATE INDEX memory_sources_memory_idx
    ON memory_sources (memory_id, created_at);

CREATE TABLE memory_relations (
    from_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (from_memory_id, to_memory_id, relation_type)
);

CREATE TABLE episodes (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    started_at INTEGER,
    ended_at INTEGER,
    summary TEXT,
    metadata_json TEXT
);

CREATE INDEX episodes_project_time_idx
    ON episodes (project_id, started_at DESC);

CREATE TABLE workspace_state (
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    last_session_id TEXT,
    active_goal TEXT,
    active_task_ref TEXT,
    checkpoint TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (project_id, workspace_id)
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    text,
    content = 'memories',
    content_rowid = 'rowid',
    tokenize = 'unicode61'
);

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

PRAGMA user_version = 1;
COMMIT;
