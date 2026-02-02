from abc import ABC, abstractmethod
from typing import List, Tuple
import logging
import os
import requests

logger = logging.getLogger('RAG')


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[Tuple[str, str, float]], k: int) -> List[Tuple[str, str, float]]:
        """Rerank documents and return top-k in (doc_id, doc_text, score) format."""
        raise NotImplementedError


class QwenReranker(BaseReranker):
    """
    Reranker that calls a Qwen rerank API endpoint (DashScope compatible-mode).
    Expected response format (flexible):
      - {"results":[{"index":0,"relevance_score":0.98}, ...]}
      - {"output":{"results":[{"index":0,"relevance_score":0.98}, ...]}}
    """
    def __init__(self,
                 api_key: str | None = None,
                 api_url: str | None = None,
                 model_name: str | None = None,
                 timeout_s: int = 30):
        self.api_key = api_key or os.getenv("RAG_RERANK_API_KEY")
        self.api_url = api_url or os.getenv("RAG_RERANK_API_URL")
        self.model_name = model_name or os.getenv("RAG_RERANK_MODEL", "qwen3-rerank")
        self.timeout_s = timeout_s
        if not self.api_url:
            raise ValueError("RAG_RERANK_API_URL not set")
        if not self.api_key:
            raise ValueError("RAG_RERANK_API_KEY not set")

        api_url = self.api_url.rstrip("/")
        if api_url.endswith("/v1"):
            api_url = f"{api_url}/rerank"
        self.api_url = api_url

    def rerank(self, query: str, docs: List[Tuple[str, str, float]], k: int) -> List[Tuple[str, str, float]]:
        if not docs:
            return []

        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": [doc_text for _, doc_text, _ in docs],
            },
            "parameters": {"top_n": k},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.debug(f"Qwen rerank request: model={self.model_name}, docs={len(docs)}")
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        results = None
        if isinstance(data, dict):
            if "output" in data and isinstance(data["output"], dict):
                results = data["output"].get("results")
            elif "results" in data:
                results = data["results"]

        if not results:
            logger.warning("Rerank API response missing results; returning original ordering.")
            return docs[:k]

        reranked = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", 0.0)
            if idx is None or idx >= len(docs):
                continue
            doc_id, doc_text, _ = docs[idx]
            reranked.append((doc_id, doc_text, float(score)))

        if not reranked:
            return docs[:k]
        return reranked[:k]
