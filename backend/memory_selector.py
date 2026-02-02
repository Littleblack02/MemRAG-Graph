import json
import logging
from datetime import datetime

from openai_generator import OpenAIGenerator


class MemorySelector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("RAG")
        self.model_name = model_name
        self.generator = OpenAIGenerator(model_name=model_name)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            # Attempt to extract the first JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {}
            return {}

    def extract(self, user_text: str, assistant_text: str, now: datetime) -> list[dict]:
        now_str = now.strftime("%Y-%m-%d")
        prompt = (
            "You are a memory extraction assistant. Extract ONLY durable, user-specific memories.\n"
            "Store the following kinds of memories:\n"
            "1) Persistent info: future events with dates, commitments, deadlines.\n"
            "2) Structured entities and relations explicitly stated.\n"
            "3) User preferences (e.g., concise answers).\n"
            "4) Personal info explicitly shared (e.g., email, phone, address).\n\n"
            "Rules:\n"
            "- If information is not durable or not user-specific, do NOT store it.\n"
            "- Resolve relative dates using today's date.\n"
            "- Output JSON only. No extra text.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "memories": [\n'
            "    {\n"
            '      "type": "event|preference|fact|entity_relation|personal",\n'
            '      "content": "short natural-language memory",\n'
            '      "importance": 0.0-1.0,\n'
            '      "tier": "core|stable|short",\n'
            '      "event_time": "YYYY-MM-DD or null",\n'
            '      "expires_at": "YYYY-MM-DD or null",\n'
            '      "entities": [{"head":"","relation":"","tail":"","confidence":0.0-1.0}]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Today: {now_str}\n"
            f"User message: {user_text}\n"
            f"Assistant reply: {assistant_text}\n"
            "JSON:\n"
        )
        try:
            raw = self.generator.generate(prompt)
            data = self._parse_json(raw)
            memories = data.get("memories", [])
            if not isinstance(memories, list):
                return []
            cleaned = []
            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                mem_type = mem.get("type") or "fact"
                content = (mem.get("content") or "").strip()
                if not content:
                    continue
                event_time = mem.get("event_time")
                expires_at = mem.get("expires_at") or event_time
                cleaned.append(
                    {
                        "type": mem_type,
                        "content": content,
                        "importance": mem.get("importance", 0.5),
                        "tier": mem.get("tier", "short"),
                        "event_time": event_time,
                        "expires_at": expires_at,
                        "entities": mem.get("entities", []),
                        "data": {},
                    }
                )
            return cleaned
        except Exception as e:
            self.logger.warning(f"Memory extraction failed: {e}")
            return []
