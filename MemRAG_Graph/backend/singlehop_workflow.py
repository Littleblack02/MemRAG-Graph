from typing import List, Dict, Any, Tuple, Optional
from retriever_base import BaseRetriever
import logging
import re

logger = logging.getLogger('RAG')

class SingleHopWorkflow:
    def __init__(self,
                 retriever: BaseRetriever,
                 generator,
                 reranker = None,
                 memory_context: Optional[List[str]] = None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.memory_context = memory_context or []

    def _post_filter(
        self,
        query: str,
        docs: List[Tuple[str, str, float]],
        max_docs: int = 4,
        min_ratio: float = 0.6,
        min_overlap: int = 2,
    ) -> List[Tuple[str, str, float]]:
        if not docs:
            return []
        # Dedupe by doc_id, keep best score
        best = {}
        for doc_id, text, score in docs:
            prev = best.get(doc_id)
            if prev is None or score > prev[1]:
                best[doc_id] = (text, score)
        items = [(doc_id, text, score) for doc_id, (text, score) in best.items()]
        items.sort(key=lambda x: x[2], reverse=True)

        max_score = items[0][2]
        q_tokens = {t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2}
        filtered = []
        for doc_id, text, score in items:
            if score < max_score * min_ratio:
                continue
            doc_tokens = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
            if len(q_tokens & doc_tokens) < min_overlap:
                continue
            filtered.append((doc_id, text, score))
        return filtered[:max_docs]

    def _compress_doc(self, query: str, text: str, max_sentences: int = 2) -> str:
        q_tokens = {t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2}
        sents = re.split(r"(?<=[.!?])\s+", text)
        scored = []
        for s in sents:
            if not s.strip():
                continue
            stoks = set(re.findall(r"[A-Za-z0-9]+", s.lower()))
            scored.append((len(q_tokens & stoks), s.strip()))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for score, s in scored[:max_sentences] if score > 0]
        return " ".join(top)


    def answer_from_docs(
        self,
        query: str,
        retrieved_docs: List[str],
        max_doc_chars: int = 800,
        max_mem_chars: int = 400,
    ) -> str:
        """
        Builds the prompt string from the documents for the LLM.
        Generates an answer based on the query and retrieved documents.

        Args:
            query (str): The user's query.
            retrieved_docs (List[str]): List of retrieved document texts.
            max_doc_chars (int): Max characters per doc snippet in prompt.

        Returns:
            str : the answer
        """
        docs = [doc for doc in retrieved_docs if doc]
        mems = [m for m in (self.memory_context or []) if m]
        if not docs and not mems:
            return "I don't know."
        evidence_snippets = "\n".join(
            [f"[{i+1}] {doc[:max_doc_chars]}" for i, doc in enumerate(docs)]
        )
        memory_snippets = "\n".join(
            [f"[M{i+1}] {mem[:max_mem_chars]}" for i, mem in enumerate(mems)]
        )
        
        prompt = (
            "You are a strict extractive QA assistant.\n"
            "Use Evidence and Memory below. Prefer Evidence for factual claims.\n"
            "Use Memory for user preferences and persistent user-specific facts.\n"
            "If Evidence is sufficient, do not use Memory.\n"
            "If both are insufficient, respond exactly: I don't know.\n"
            "Ignore evidence about other entities even if the names are similar.\n"
            "Every sentence must cite sources like [1], [2] or [M1].\n\n"
            "Evidence:\n"
            f"{evidence_snippets}\n\n"
        )
        if memory_snippets:
            prompt += f"Memory:\n{memory_snippets}\n\n"
        prompt += (
            "Question:\n"
            f"{query}\n\n"
            "Answer (with citations):"
        )

        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generator.generate(prompt)
        if response.startswith("Answer:"):
            response = response[len("Answer:"):]
        return response
    

    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the single-hop workflow.

        Args:
            question (str): The input question.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the final answer (str)
            and a list of intermediate steps (List[Dict]) for the UI.
        """
        steps_log = []

        # If not multi-hop, run standard single-turn RAG
        candidate_k = 40
        retrieved_docs = self.retriever.retrieve(question, k=candidate_k)
        if self.reranker:
            try:
                retrieved_docs = self.reranker.rerank(question, retrieved_docs, k=10)
            except Exception as e:
                logger.warning(f"Rerank failed, fallback to original ranking: {e}")
        retrieved_docs = self._post_filter(question, retrieved_docs, max_docs=4)
        steps_log.append({
            "step": 1,
            "description": "Standard RAG Retrieval (Single-Hop)",
            "type" : "single_hop_retrieval",
            "query": question,
            "retrieved_docs": retrieved_docs
        })

        evidence_docs = [self._compress_doc(question, doc_text) for (_, doc_text, _) in retrieved_docs]
        evidence_docs = [doc for doc in evidence_docs if doc]
        final_answer = self.answer_from_docs(question, evidence_docs)
        steps_log.append({
            "step": 2,
            "description": "Standard RAG Generation (Single-Hop)",
            "type" : "single_hop_generation",
            "query": question,
            "result": final_answer
        })

        return final_answer, steps_log
