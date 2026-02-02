

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY, 
    user_id TEXT NOT NULL DEFAULT 'default_user',
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, 
    session_id TEXT NOT NULL, 
    sender TEXT NOT NULL CHECK(sender IN ('user', 'bot')),
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    thinking_process TEXT,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    data_json TEXT,
    embedding TEXT,
    importance REAL DEFAULT 0.5,
    tier TEXT DEFAULT 'short',
    event_time DATETIME,
    expires_at DATETIME,
    source_session_id TEXT,
    source_message_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at DATETIME
);

CREATE TABLE IF NOT EXISTS memory_entities (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    head TEXT NOT NULL,
    relation TEXT NOT NULL,
    tail TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_meta (
    user_id TEXT PRIMARY KEY,
    last_reflect_at DATETIME
);

CREATE TABLE IF NOT EXISTS memory_pending (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    data_json TEXT,
    importance REAL DEFAULT 0.5,
    tier TEXT DEFAULT 'short',
    event_time DATETIME,
    expires_at DATETIME,
    source_session_id TEXT,
    source_message_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_pending_user_id ON memory_pending(user_id);
