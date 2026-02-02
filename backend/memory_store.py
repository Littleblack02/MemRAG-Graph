import json
import logging
import sqlite3
import uuid
import re
import threading
from datetime import datetime, timedelta
from typing import List

import numpy as np


class MemoryStore:
    def __init__(
        self,
        db_path: str,
        logger: logging.Logger | None = None,
        embed_model_name: str | None = None,
    ):
        self.db_path = db_path
        self.logger = logger or logging.getLogger("RAG")
        self.embed_model_name = embed_model_name
        self._embedder = None
        self._lock = threading.Lock()
        self._ctx = threading.local()  # hold optional external connection
        self._ensure_tables()
        self._ensure_chat_user_id()

    def _get_external_conn(self):
        return getattr(self._ctx, "conn", None)

    def set_external_connection(self, conn):
        """Provide an external SQLite connection for a request; caller must clear it."""
        self._ctx.conn = conn

    def clear_external_connection(self):
        self._ctx.conn = None

    def _connect(self):
        external = self._get_external_conn()
        if external:
            return external
        # Increase timeout and enable WAL to reduce 'database is locked' under concurrent access
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 15000")
            conn.execute("PRAGMA journal_mode = WAL")
        except Exception:
            pass
        return conn

    def _ensure_chat_user_id(self):
        conn = self._connect()
        try:
            cols = [row["name"] for row in conn.execute("PRAGMA table_info(chat_sessions)")]
            if "user_id" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT DEFAULT 'default_user'")
                conn.commit()
        finally:
            conn.close()

    def _ensure_tables(self):
        conn = self._connect()
        try:
            conn.executescript(
                """
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
                """
            )
            # Ensure embedding column exists for existing DBs
            cols = [row["name"] for row in conn.execute("PRAGMA table_info(memories)")]
            if "embedding" not in cols:
                conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            conn.commit()
        finally:
            conn.close()

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            if not self.embed_model_name:
                return None
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embed_model_name)
            return self._embedder
        except Exception as e:
            self.logger.warning(f"Memory embedder init failed: {e}")
            self._embedder = None
            return None

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        if not embedder:
            return []
        vecs = embedder.encode(texts)
        # Normalize for cosine similarity
        vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
        return vecs.tolist()

    def _parse_time(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

    def _normalize_importance(self, value) -> float:
        try:
            v = float(value)
        except Exception:
            v = 0.5
        return max(0.0, min(1.0, v))

    def _normalize_tier(self, tier: str | None) -> str:
        if tier in ("core", "stable", "short"):
            return tier
        return "short"

    def add_memories(
        self,
        user_id: str,
        memories: list[dict],
        source_session_id: str | None = None,
        source_message_id: str | None = None,
    ) -> int:
        if not memories:
            return 0
        with self._lock:
            conn = self._connect()
            using_external = conn is self._get_external_conn()
            added = 0
            try:
                cur = conn.cursor()
                for mem in memories:
                    mem_type = mem.get("type") or mem.get("memory_type") or "fact"
                    content = (mem.get("content") or "").strip()
                    if not content:
                        continue
                    importance = self._normalize_importance(mem.get("importance", 0.5))
                    tier = self._normalize_tier(mem.get("tier"))
                    event_time = mem.get("event_time")
                    expires_at = mem.get("expires_at") or event_time
                    data_json = mem.get("data") or {}
                    entities = mem.get("entities") or []
                    embedding = None
                    emb_list = self._embed_texts([content]) if self._get_embedder() else []
                    if emb_list:
                        embedding = json.dumps(emb_list[0])

                    row = cur.execute(
                        "SELECT id, importance, tier FROM memories WHERE user_id=? AND memory_type=? AND content=?",
                        (user_id, mem_type, content),
                    ).fetchone()
                    if row:
                        mem_id = row["id"]
                        new_importance = max(importance, row["importance"] or 0.0)
                        new_tier = tier
                        cur.execute(
                            """
                            UPDATE memories
                            SET importance=?, tier=?, data_json=?, event_time=?, expires_at=?,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=?
                            """,
                            (
                                new_importance,
                                new_tier,
                                json.dumps(data_json, ensure_ascii=False),
                                event_time,
                                expires_at,
                                mem_id,
                            ),
                        )
                    else:
                        mem_id = str(uuid.uuid4())
                        cur.execute(
                            """
                            INSERT INTO memories
                            (id, user_id, memory_type, content, data_json, embedding, importance, tier, event_time, expires_at,
                             source_session_id, source_message_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                mem_id,
                                user_id,
                                mem_type,
                                content,
                                json.dumps(data_json, ensure_ascii=False),
                                embedding,
                                importance,
                                tier,
                                event_time,
                                expires_at,
                                source_session_id,
                                source_message_id,
                            ),
                        )
                    for ent in entities:
                        head = (ent.get("head") or "").strip()
                        rel = (ent.get("relation") or "").strip()
                        tail = (ent.get("tail") or "").strip()
                        conf = self._normalize_importance(ent.get("confidence", 0.5))
                        if not head or not rel or not tail:
                            continue
                        cur.execute(
                            """
                            INSERT INTO memory_entities (id, memory_id, head, relation, tail, confidence)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (str(uuid.uuid4()), mem_id, head, rel, tail, conf),
                        )
                    added += 1
                if not using_external:
                    conn.commit()
            finally:
                if not using_external:
                    conn.close()
            return added

    def add_pending_memories(
        self,
        user_id: str,
        memories: list[dict],
        source_session_id: str | None = None,
        source_message_id: str | None = None,
    ) -> list[dict]:
        if not memories:
            return []
        with self._lock:
            conn = self._connect()
            using_external = conn is self._get_external_conn()
            pending = []
            try:
                cur = conn.cursor()
                for mem in memories:
                    mem_type = mem.get("type") or mem.get("memory_type") or "fact"
                    content = (mem.get("content") or "").strip()
                    if not content:
                        continue
                    mem_id = str(uuid.uuid4())
                    data_json = mem.get("data") or {}
                    data_json["entities"] = mem.get("entities", [])
                    cur.execute(
                        """
                        INSERT INTO memory_pending
                        (id, user_id, memory_type, content, data_json, importance, tier, event_time, expires_at,
                         source_session_id, source_message_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            mem_id,
                            user_id,
                            mem_type,
                            content,
                            json.dumps(data_json, ensure_ascii=False),
                            self._normalize_importance(mem.get("importance", 0.5)),
                            self._normalize_tier(mem.get("tier")),
                            mem.get("event_time"),
                            mem.get("expires_at") or mem.get("event_time"),
                            source_session_id,
                            source_message_id,
                        ),
                    )
                    pending.append({"id": mem_id, "type": mem_type, "content": content})
                if not using_external:
                    conn.commit()
            finally:
                if not using_external:
                    conn.close()
            return pending

    def list_pending(self, user_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, memory_type, content, importance, tier, event_time, expires_at, created_at
                FROM memory_pending WHERE user_id=?
                ORDER BY created_at ASC
                """,
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def confirm_pending(self, user_id: str, pending_id: str, confirm: bool) -> bool:
        with self._lock:
            conn = self._connect()
            using_external = conn is self._get_external_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM memory_pending WHERE id=? AND user_id=?",
                    (pending_id, user_id),
                ).fetchone()
                if not row:
                    return False
                if confirm:
                    data = {}
                    if row["data_json"]:
                        try:
                            data = json.loads(row["data_json"])
                        except Exception:
                            data = {}
                    mem = {
                        "type": row["memory_type"],
                        "content": row["content"],
                        "importance": row["importance"],
                        "tier": row["tier"],
                        "event_time": row["event_time"],
                        "expires_at": row["expires_at"],
                        "entities": data.get("entities", []),
                        "data": {k: v for k, v in data.items() if k != "entities"},
                    }
                    self.add_memories(
                        user_id=user_id,
                        memories=[mem],
                        source_session_id=row["source_session_id"],
                        source_message_id=row["source_message_id"],
                    )
                conn.execute("DELETE FROM memory_pending WHERE id=?", (pending_id,))
                if not using_external:
                    conn.commit()
                return True
            finally:
                if not using_external:
                    conn.close()

    def get_relevant(self, user_id: str, query: str, limit: int = 8) -> list[str]:
        conn = self._connect()
        using_external = conn is self._get_external_conn()
        now = datetime.utcnow()
        now_iso = now.isoformat()
        try:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id=?
                AND (expires_at IS NULL OR expires_at > ?)
                """,
                (user_id, now_iso),
            ).fetchall()
            q_tokens = {t for t in re.findall(r"[A-Za-z0-9]+", (query or "").lower()) if len(t) > 2}
            q_emb = None
            q_norm = 1.0
            if self._get_embedder():
                emb_list = self._embed_texts([query]) if query else []
                if emb_list:
                    q_emb = np.array(emb_list[0], dtype=np.float32)
                    q_norm = np.linalg.norm(q_emb) + 1e-9
            scored = []
            for row in rows:
                content = row["content"]
                tokens = set(re.findall(r"[A-Za-z0-9]+", (content or "").lower()))
                overlap = len(q_tokens & tokens)
                importance = float(row["importance"] or 0.0)
                tier = row["tier"] or "short"
                tier_boost = 0.6 if tier == "core" else 0.3 if tier == "stable" else 0.0
                updated_at = self._parse_time(row["updated_at"]) or now
                days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
                recency = max(0.0, 1.0 - (days / 30.0))
                score = overlap + importance * 2.0 + tier_boost + recency * 0.5
                emb = row["embedding"]
                if q_emb is not None and emb:
                    try:
                        vec = np.array(json.loads(emb), dtype=np.float32)
                        v_norm = np.linalg.norm(vec) + 1e-9
                        sim = float(np.dot(q_emb, vec) / (q_norm * v_norm))
                        score += sim * 2.0
                    except Exception:
                        pass
                if row["memory_type"] == "preference":
                    score += 2.0
                scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)

            picked = []
            for _, row in scored:
                if len(picked) >= limit:
                    break
                label = row["memory_type"]
                content = row["content"]
                picked.append(f"{label}: {content}")

            if picked:
                ids = [row["id"] for _, row in scored[: len(picked)]]
                conn.executemany(
                    "UPDATE memories SET last_accessed_at=CURRENT_TIMESTAMP WHERE id=?",
                    [(i,) for i in ids],
                )
                if not using_external:
                    conn.commit()
            return picked
        finally:
            if not using_external:
                conn.close()

    def get_recent_memories(self, user_id: str, limit: int = 60) -> list[dict]:
        conn = self._connect()
        using_external = conn is self._get_external_conn()
        try:
            rows = conn.execute(
                """
                SELECT memory_type, content, importance, tier, event_time, expires_at, updated_at
                FROM memories
                WHERE user_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if not using_external:
                conn.close()

    def get_last_reflect_at(self, user_id: str) -> datetime | None:
        conn = self._connect()
        using_external = conn is self._get_external_conn()
        try:
            row = conn.execute(
                "SELECT last_reflect_at FROM memory_meta WHERE user_id=?",
                (user_id,),
            ).fetchone()
            if not row or not row["last_reflect_at"]:
                return None
            return self._parse_time(row["last_reflect_at"])
        finally:
            if not using_external:
                conn.close()

    def set_last_reflect_at(self, user_id: str, when: datetime):
        with self._lock:
            conn = self._connect()
            using_external = conn is self._get_external_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO memory_meta (user_id, last_reflect_at)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET last_reflect_at=excluded.last_reflect_at
                    """,
                    (user_id, when.isoformat()),
                )
                if not using_external:
                    conn.commit()
                return True
            except sqlite3.OperationalError as e:
                self.logger.warning(f"set_last_reflect_at skipped due to lock: {e}")
                return False
            except Exception as e:
                self.logger.warning(f"set_last_reflect_at failed: {e}")
                return False
            finally:
                if not using_external:
                    conn.close()

    def forget(self, user_id: str):
        conn = self._connect()
        using_external = conn is self._get_external_conn()
        try:
            now = datetime.utcnow()
            now_iso = now.isoformat()
            # Delete expired events
            conn.execute(
                "DELETE FROM memories WHERE user_id=? AND expires_at IS NOT NULL AND expires_at <= ?",
                (user_id, now_iso),
            )
            # Demote stale stable memories
            stale_stable = (now - timedelta(days=90)).isoformat()
            conn.execute(
                """
                UPDATE memories SET tier='short'
                WHERE user_id=? AND tier='stable' AND importance < 0.5 AND updated_at <= ?
                """,
                (user_id, stale_stable),
            )
            # Delete low-importance short memories
            stale_short = (now - timedelta(days=30)).isoformat()
            conn.execute(
                """
                DELETE FROM memories
                WHERE user_id=? AND tier='short' AND importance < 0.4 AND updated_at <= ?
                """,
                (user_id, stale_short),
            )
            if not using_external:
                conn.commit()
            return True
        except sqlite3.OperationalError as e:
            self.logger.warning(f"forget skipped due to lock: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"forget failed: {e}")
            return False
        finally:
            if not using_external:
                conn.close()
