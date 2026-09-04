BEGIN IMMEDIATE;

CREATE TABLE memories (
    id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    actor TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted')),
    superseded_by TEXT REFERENCES memories(id),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    deleted_at INTEGER
);

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

CREATE TABLE embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    source_updated_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (memory_id, model)
);

CREATE INDEX embeddings_model_memory_idx
    ON embeddings (model, memory_id);

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

CREATE TRIGGER memories_embedding_invalidate
AFTER UPDATE OF text, status ON memories
BEGIN
    DELETE FROM embeddings WHERE memory_id = new.id;
END;

PRAGMA user_version = 7;
COMMIT;
