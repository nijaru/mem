BEGIN IMMEDIATE;

CREATE TABLE workspace_state_v6 (
    workspace_id TEXT PRIMARY KEY NOT NULL,
    last_session_id TEXT,
    active_goal TEXT,
    active_task_ref TEXT,
    checkpoint TEXT,
    updated_at INTEGER NOT NULL
);

INSERT INTO workspace_state_v6 (
    workspace_id, last_session_id, active_goal, active_task_ref, checkpoint, updated_at
)
SELECT
    ws.workspace_id, ws.last_session_id, ws.active_goal, ws.active_task_ref,
    ws.checkpoint, ws.updated_at
FROM workspace_state AS ws
WHERE ws.rowid = (
    SELECT newer.rowid
    FROM workspace_state AS newer
    WHERE newer.workspace_id = ws.workspace_id
    ORDER BY newer.updated_at DESC, newer.rowid DESC
    LIMIT 1
);

DROP TABLE workspace_state;
ALTER TABLE workspace_state_v6 RENAME TO workspace_state;

PRAGMA user_version = 6;
COMMIT;
