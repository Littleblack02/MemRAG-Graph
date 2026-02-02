import json
import logging

from openai_generator import OpenAIGenerator


class MemoryVerifier:
    def __init__(self, model_name: str, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("RAG")
        self.model_name = model_name
        self.generator = OpenAIGenerator(model_name=model_name)

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

    def verify(self, user_text: str, assistant_text: str, memory: dict) -> bool:
        mem_json = json.dumps(memory, ensure_ascii=False)
        prompt = (
            "You are a memory validation assistant.\n"
            "Decide if the proposed memory is durable, user-specific, and explicitly supported by the conversation.\n"
            "Reject if it is speculative, ephemeral, or not clearly stated.\n"
            "Output JSON only.\n\n"
            "JSON schema:\n"
            '{ "accept": true|false, "reason": "short reason" }\n\n'
            f"User message: {user_text}\n"
            f"Assistant reply: {assistant_text}\n"
            f"Proposed memory: {mem_json}\n"
            "JSON:\n"
        )
        try:
            raw = self.generator.generate(prompt)
            data = self._parse_json(raw)
            return bool(data.get("accept", False))
        except Exception as e:
            self.logger.warning(f"Memory verification failed: {e}")
            return False
