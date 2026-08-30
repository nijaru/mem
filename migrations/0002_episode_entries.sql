BEGIN IMMEDIATE;

ALTER TABLE episodes ADD COLUMN workspace_id TEXT;

CREATE UNIQUE INDEX episodes_source_idx
    ON episodes (source_type, source_ref);

CREATE INDEX episodes_project_workspace_time_idx
    ON episodes (project_id, workspace_id, started_at DESC);

CREATE TABLE episode_entries (
    id TEXT PRIMARY KEY NOT NULL,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    source_ref TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    role TEXT,
    text TEXT NOT NULL,
    occurred_at INTEGER,
    metadata_json TEXT,
    UNIQUE (episode_id, source_ref)
);

CREATE INDEX episode_entries_episode_order_idx
    ON episode_entries (episode_id, ordinal, id);

CREATE VIRTUAL TABLE episode_entries_fts USING fts5(
    text,
    content = 'episode_entries',
    content_rowid = 'rowid',
    tokenize = 'unicode61'
);

CREATE TRIGGER episode_entries_fts_insert AFTER INSERT ON episode_entries BEGIN
    INSERT INTO episode_entries_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER episode_entries_fts_delete AFTER DELETE ON episode_entries BEGIN
    INSERT INTO episode_entries_fts(episode_entries_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER episode_entries_fts_update AFTER UPDATE OF text ON episode_entries BEGIN
    INSERT INTO episode_entries_fts(episode_entries_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO episode_entries_fts(rowid, text) VALUES (new.rowid, new.text);
END;

PRAGMA user_version = 2;
COMMIT;
