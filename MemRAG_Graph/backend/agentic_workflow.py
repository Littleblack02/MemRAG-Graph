from typing import List, Dict, Any, Tuple, Optional
from retriever_base import BaseRetriever
from decomposition_checker import DecompositionChecker
import logging
import re
import os
import numpy as np
from openai_generator import OpenAIGenerator

logger = logging.getLogger('RAG')

class AgenticWorkflow:
    def __init__(self,
                 retriever: BaseRetriever,
                 generator,
                 reranker = None,
                 need_reformulate : bool = False,
                 session_history : List = [],
                 memory_context: Optional[List[str]] = None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.need_reformulate = need_reformulate
        self.session_history = session_history
        self.memory_context = memory_context or []
        self.decomp_checker = DecompositionChecker()
        self.MAX_DECOMPOSITION_STEPS = 10
        self.qe_model_name = os.getenv("RAG_QE_MODEL", "qwen3-30b-a3b-instruct-2507")

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

    def _expand_queries(self, query: str) -> List[str]:
        """
        Generate query variants with a small model, keep those with cosine similarity >=0.8 to the original.
        Requires dense_retriever to be available for embeddings; otherwise returns [query].
        """
        variants = [query]
        try:
            if not hasattr(self.retriever, "dense_retriever"):
                return variants
            qe_gen = OpenAIGenerator(model_name=self.qe_model_name)
            prompt = (
                "Rewrite the question into up to 3 equivalent queries.\n"
                "One per line, no numbering, no explanations, do not answer.\n"
                f"Question: {query}\n"
                "Rewrites:"
            )
            rewrites_raw = qe_gen.generate(prompt)
            rewrites = [r.strip("- ").strip() for r in rewrites_raw.split("\n") if r.strip()]
            # embed
            model = self.retriever.dense_retriever.dense_model
            emb_all = model.encode([query] + rewrites)
            base = emb_all[0]
            sims = [float(np.dot(base, e) / (np.linalg.norm(base) * np.linalg.norm(e) + 1e-9)) for e in emb_all]
            for r, sim in zip(rewrites, sims[1:]):
                if sim >= 0.9 and r not in variants:
                    variants.append(r)
        except Exception as e:
            logger.warning(f"Query expansion failed; using original query only: {e}")
        return variants

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
        # simple sanity check: if response too long or lacks any bracketed citation, fallback
        if len(response) > 512 or "[" not in response:
            return "I don't know."
        return response
    

    def reformulate_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates a follow-up query into a standalone query using conversation history.
        Uses the generator LLM to perform coreference resolution and context expansion.

        Example:
            History: [{"sender": "user", "content": "Where was Barack Obama born?"},
                    {"sender": "bot", "content": "Barack Obama was born in Honolulu, Hawaii."}]
            Current query: "What about his wife?"
            Output: "Where was Barack Obama's wife born?"

        Args:
            generator: The selected generator
            query (str): The current user query.
            history (List[Dict[str, str]]): Prior conversation turns.

        Returns:
            str: A contextually reformulated standalone query.
        """
        try:
            # Build conversation transcript
            conversation_lines = []
            for turn in history:
                role = "User" if turn["sender"] == "user" else "Assistant"
                conversation_lines.append(f"{role}: {turn['content']}")
            conversation_text = "\n".join(conversation_lines)
            
            # Construct prompt for query reformulation
            reformulation_prompt = (
                "You are a precise query rewriting assistant. Rewrite the final user query as a standalone question.\n"
                "Rules:\n"
                "- Resolve coreferences using the conversation history only.\n"
                "- Do NOT change intent, add facts, or answer the question.\n"
                "- Output ONLY the rewritten question, no prefix, no explanation.\n\n"
                "Conversation History:\n"
                f"{conversation_text}\n\n"
                "Final User Query:\n"
                f"{query}\n\n"
                "Standalone Question:"
            )
            logger.debug(f"Reformulation prompt:\n {reformulation_prompt}")
            # Use the generator to reformulate
            reformulated = self.generator.generate(reformulation_prompt).strip()

            # Fallback if LLM returns empty or malformed output
            if not reformulated or len(reformulated) < 3:
                logger.warning("Query reformulation returned empty or short output. Falling back to original query.")
                return query
            
            logger.debug(f"Reformulated query: {reformulated}")
            return reformulated

        except Exception as e:
            logger.error(f"Error during query reformulation: {e}. Falling back to original query.")
            return query


    def decompose_query(self, question: str) -> List[str]:
        """
        Decomposes a multi-hop question into sub-questions using the LLM.
        """
        decomposition_prompt = (
            "Decompose the following question into a sequence of simple, answerable sub-questions only if it is complex. "
            "If the question is simple, just return the original question."
            "Each sub-question must build logically on the previous one and use concrete entities. Do NOT answer鈥攋ust list sub-questions line by line.\n\n"
            
            #f"{few_shot}"
            f"Complex Question: {question}\n"
            "Sub-questions:\n"
        )
        
        logger.debug(f'decompose_query prompt :\n {decomposition_prompt}')
        decomposition_response = self.generator.generate(decomposition_prompt)
        logger.debug(f'decomposition_response: \n {decomposition_response}')

        # Enhanced parsing with multiple formats
        lines = decomposition_response.split('\n')
        sub_questions = lines

        # Remove empty strings and limit to max steps
        sub_questions = [sq for sq in sub_questions if sq.strip()][:self.MAX_DECOMPOSITION_STEPS]
        return sub_questions



    def synthesize_answer(self, question: str, sub_answers: List[str]) -> str:
        """
        Synthesizes the final answer from the original question and the answers to sub-questions.
        """
        context_for_synthesis = "\n".join([f"Sub-answer {i+1}: {ans}" for i, ans in enumerate(sub_answers)])
        synthesis_prompt = (
            "Answer the Original Question using only the provided sub-answers."
            "Be concise, relevant, and do not add extra information."
            "If the Original Question is simple, give a short direct answer."
            "Do NOT use external knowledge or speculate. If sub-answers are insufficient, say 'I don't know.' "
            "Preserve any evidence citations like [1], [2] from sub-answers.\n\n"
            
            f"Original Question: {question}\n\n"
            "Sub-answers:\n"
            f"{context_for_synthesis}\n\n"
            
            "Final Answer:"
        )
        final_answer = self.generator.generate(synthesis_prompt)
        # A simple heuristic might be to take the last non-empty line.
        lines = final_answer.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        final_answer = final_answer.strip()
        # Post-process to remove potential prefixes like "The final answer is..."
        if final_answer.startswith("Final Answer:"):
            final_answer = final_answer[len("Final Answer:"):]
        return final_answer


    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the agentic workflow.

        Args:
            question (str): The input question.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the final answer (str)
            and a list of intermediate steps (List[Dict]) for the UI.
        """
        steps_log = []
        sub_questions = []
        original_question = question
        step_cnt = 1

        if self.need_reformulate:
            new_query = self.reformulate_query(question, self.session_history)
            if new_query != question:
                steps_log.append({
                    "step": step_cnt,
                    "type" : "query_reformulation",
                    "description": f"Reformulated query to: *'{new_query}'*"
                })
                step_cnt += 1
                question = new_query

        is_multi_hop = self.decomp_checker.identify_multi_hop_pattern(question)
        logger.debug(f'question = {question}, is_multi_hop = {is_multi_hop}')

        steps_log.append({
            "step": step_cnt,
            "description": f"Initial Analysis: Question is {'multi-hop' if is_multi_hop else 'single-hop'}.",
            "type" : "identify",
            "query": question,
            "result": None
        })
        step_cnt += 1

        if is_multi_hop:
            sub_questions = self.decompose_query(question)
            steps_log.append({
                "step": step_cnt,
                "description": f"Query decomposition: The question was broken down into {len(sub_questions)} sub-questions.",
                "type" : "decomposition",
                "sub_questions": sub_questions,
                "result": None
            })
            step_cnt += 1
            if len(sub_questions) < 2:
                # If decomposition failed, fallback to single-hop
                is_multi_hop = False
                steps_log.append({
                    "step": step_cnt,
                    "description": "Decomposition yielded less than 2 sub-questions, falling back to single-hop processing.",
                    "type" : "fallback",
                    "result": None
                })
                step_cnt += 1

        if is_multi_hop:
            sub_answers = []
            for i, sub_q in enumerate(sub_questions):
                # Retrieve relevant documents for the sub-question (with expansion)
                combined = {}
                for qv in self._expand_queries(sub_q):
                    retrieved = self.retriever.retrieve(qv, k = 10)
                    for doc_id, doc_text, score in retrieved:
                        if doc_id not in combined or score > combined[doc_id][1]:
                            combined[doc_id] = (doc_text, score)
                retrieved_docs = [(doc_id, dt, sc) for doc_id, (dt, sc) in combined.items()]
                retrieved_docs.sort(key=lambda x: x[2], reverse=True)
                retrieved_docs = retrieved_docs[:10]
                if self.reranker:
                    try:
                        retrieved_docs = self.reranker.rerank(sub_q, retrieved_docs, k=10)
                    except Exception as e:
                        logger.warning(f"Rerank failed, fallback to original ranking: {e}")
                retrieved_docs = self._post_filter(sub_q, retrieved_docs, max_docs=4)
                steps_log.append({
                    "step": step_cnt,
                    "description": f"Retrieval for Sub-question {i+1} : *{sub_q}*",
                    "type" : "multi_hop_sub_retrieval",
                    "query": sub_q,
                    "retrieved_docs": retrieved_docs
                })
                step_cnt += 1

                # Generate an answer for the sub-question using retrieved docs
                evidence_docs = [self._compress_doc(sub_q, doc_text) for (_, doc_text, _) in retrieved_docs]
                evidence_docs = [doc for doc in evidence_docs if doc]
                sub_answer = self.answer_from_docs(sub_q, evidence_docs)
                sub_answers.append(sub_answer)

                steps_log.append({
                    "step": step_cnt,
                    "description": f"Generated Answer for Sub-question {i+1}",
                    "type" : "multi_hop_sub_generation",
                    "query": sub_q,
                    "result": sub_answer
                })
                step_cnt += 1

            final_answer = self.synthesize_answer(original_question, sub_answers)
            steps_log.append({
                "step": step_cnt,
                "description": "Synthesis: Final answer generated from sub-answers.",
                "type" : "multi_hop_synthesize_answer",
                "result": final_answer
            })
            step_cnt += 1

        else:
            # If not multi-hop, run standard single-turn RAG
            combined = {}
            for qv in self._expand_queries(question):
                retrieved = self.retriever.retrieve(qv, k = 10)
                for doc_id, doc_text, score in retrieved:
                    if doc_id not in combined or score > combined[doc_id][1]:
                        combined[doc_id] = (doc_text, score)
            retrieved_docs = [(doc_id, dt, sc) for doc_id, (dt, sc) in combined.items()]
            retrieved_docs.sort(key=lambda x: x[2], reverse=True)
            retrieved_docs = retrieved_docs[:10]
            if self.reranker:
                try:
                    retrieved_docs = self.reranker.rerank(question, retrieved_docs, k=10)
                except Exception as e:
                    logger.warning(f"Rerank failed, fallback to original ranking: {e}")
            retrieved_docs = self._post_filter(question, retrieved_docs, max_docs=4)
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Retrieval (Single-Hop, with query expansion)",
                "type" : "single_hop_retrieval",
                "query": question,
                "retrieved_docs": retrieved_docs
            })
            step_cnt += 1

            evidence_docs = [self._compress_doc(question, doc_text) for (_, doc_text, _) in retrieved_docs]
            evidence_docs = [doc for doc in evidence_docs if doc]
            final_answer = self.answer_from_docs(question, evidence_docs)
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Generation (Single-Hop)",
                "type" : "single_hop_generation",
                "query": question,
                "result": final_answer
            })

        return final_answer, steps_log
