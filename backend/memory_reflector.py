import json
import logging
from datetime import datetime, timedelta


class MemoryReflector:
    def __init__(
        self,
        memory_store,
        generator,
        reflect_hours: int = 24,
        logger: logging.Logger | None = None,
    ):
        self.memory_store = memory_store
        self.generator = generator
        self.reflect_hours = reflect_hours
        self.logger = logger or logging.getLogger("RAG")

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {}
            return {}

    def _reflect(self, user_id: str, now: datetime):
        memories = self.memory_store.get_recent_memories(user_id, limit=80)
        if not memories:
            return
        # Keep prompt size controlled
        context = "\n".join(
            [
                f"- ({m['memory_type']}, {m['tier']}, {m['importance']}) {m['content']}"
                for m in memories[:40]
            ]
        )
        prompt = (
            "You are a reflection assistant. Derive high-level, stable insights about the user.\n"
            "Do not repeat raw facts; synthesize abstract patterns or long-term themes.\n"
            "Output JSON only. No extra text.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "insights": [\n'
            "    {\n"
            '      "content": "insight sentence",\n'
            '      "importance": 0.0-1.0,\n'
            '      "tier": "core|stable"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Memory context:\n{context}\n"
            "JSON:\n"
        )
        raw = self.generator.generate(prompt)
        data = self._parse_json(raw)
        insights = data.get("insights", [])
        if not isinstance(insights, list):
            return
        to_add = []
        for ins in insights:
            if not isinstance(ins, dict):
                continue
            content = (ins.get("content") or "").strip()
            if not content:
                continue
            to_add.append(
                {
                    "type": "insight",
                    "content": content,
                    "importance": ins.get("importance", 0.6),
                    "tier": ins.get("tier", "stable"),
                    "event_time": None,
                    "expires_at": None,
                    "entities": [],
                    "data": {},
                }
            )
        if to_add:
            self.memory_store.add_memories(user_id, to_add)

    def run_if_due(self, user_id: str):
        now = datetime.utcnow()
        last = self.memory_store.get_last_reflect_at(user_id)
        if last and (now - last) < timedelta(hours=self.reflect_hours):
            return
        try:
            self._reflect(user_id, now)
        except Exception as e:
            self.logger.warning(f"Memory reflection failed: {e}")
        finally:
            self.memory_store.set_last_reflect_at(user_id, now)
