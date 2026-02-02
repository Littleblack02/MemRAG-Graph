import os
import re
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from langgraph.graph import StateGraph, END

from retriever_base import BaseRetriever
from agentic_workflow import AgenticWorkflow
from decomposition_checker import DecompositionChecker
from huggingface_generator import HuggingfaceGenerator
from openai_generator import OpenAIGenerator
from reranker import QwenReranker
from memory_store import MemoryStore
from memory_selector import MemorySelector
from memory_verifier import MemoryVerifier
from memory_reflector import MemoryReflector


class GraphState(TypedDict, total=False):
    query: str
    standalone_query: str
    active_query: str
    session_history: List[Dict[str, str]]
    user_id: str
    memory_context: List[str]
    retrieved_docs: List[Tuple[str, str, float]]
    filtered_docs: List[Tuple[str, str, float]]
    evidence_docs: List[str]
    is_multi_hop: bool
    sub_questions: List[str]
    sub_idx: int
    sub_answers: List[str]
    final_answer: str
    thinking_process: List[Dict[str, Any]]
    memory_confirmation: List[Dict[str, Any]]
    has_next: bool


class RAGLangGraphAgent:
    def __init__(
        self,
        retriever_type: str,
        collection_path: str,
        model_name: str,
        enable_rerank: bool = True,
        enable_memory: bool = True,
        memory_top_k: int = 8,
        memory_model: str = "qwen2.5-1.5b-instruct",
        memory_verify_model: str = "qwen2.5-0.5b-instruct",
        memory_embed_model: str = "BAAI/bge-small-en-v1.5",
        memory_reflect_hours: int = 24,
        storage_dir: str | None = None,
        cache_dir: str | None = None,
    ):
        self.retriever = BaseRetriever.create_retriever(
            retriever_type=retriever_type,
            collection_path=collection_path,
            cache_dir=cache_dir or os.getenv("RAG_CACHE_DIR", "./cache"),
        )
        self.generator = self._init_generator(model_name)
        self.reranker = None
        if enable_rerank:
            try:
                self.reranker = QwenReranker()
            except Exception:
                self.reranker = None

        self.workflow = AgenticWorkflow(
            retriever=self.retriever,
            generator=self.generator,
            reranker=self.reranker,
            need_reformulate=False,
            session_history=[],
            memory_context=[],
        )
        self.decomp_checker = DecompositionChecker()

        self.enable_memory = enable_memory
        self.memory_top_k = memory_top_k
        self.memory_store = None
        self.memory_selector = None
        self.memory_verifier = None
        self.memory_reflector = None
        if enable_memory:
            db_path = os.path.join(storage_dir or os.getenv("RAG_STORAGE_DIR", "./storage"), "chat_history.db")
            self.memory_store = MemoryStore(db_path, embed_model_name=memory_embed_model)
            self.memory_selector = MemorySelector(model_name=memory_model)
            self.memory_verifier = MemoryVerifier(model_name=memory_verify_model)
            self.memory_reflector = MemoryReflector(
                memory_store=self.memory_store,
                generator=self.memory_selector.generator,
                reflect_hours=memory_reflect_hours,
            )

        self.max_history_turns = int(os.getenv("RAG_SHORT_TERM_TURNS", "6"))
        self.graph = self._build_graph()

    def _init_generator(self, model_name: str):
        if model_name.startswith("Qwen/"):
            return HuggingfaceGenerator(model_name=model_name)
        return OpenAIGenerator(model_name=model_name)

    def _truncate_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not history or self.max_history_turns <= 0:
            return history
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            return history[-max_messages:]
        return history

    def _is_high_risk_memory(self, mem: dict) -> bool:
        mem_type = (mem.get("type") or mem.get("memory_type") or "").lower()
        if mem_type in ("event", "personal"):
            return True
        if mem.get("event_time"):
            return True
        content = (mem.get("content") or "").lower()
        patterns = [
            r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
            r"\b(address|street|st\.|avenue|ave\.|road|rd\.|zip|postal)\b",
            r"\b(id|passport|ssn|social security)\b",
        ]
        return any(re.search(p, content) for p in patterns)

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("decide_multihop", self._decide_multihop)
        graph.add_node("decompose", self._decompose)
        graph.add_node("init_sub", self._init_sub)

        graph.add_node("retrieve_docs_single", self._retrieve_docs_single)
        graph.add_node("filter_docs_single", self._filter_docs_single)
        graph.add_node("retrieve_memory_single", self._retrieve_memory_single)
        graph.add_node("answer_single", self._answer_single)

        graph.add_node("retrieve_docs_sub", self._retrieve_docs_sub)
        graph.add_node("filter_docs_sub", self._filter_docs_sub)
        graph.add_node("retrieve_memory_sub", self._retrieve_memory_sub)
        graph.add_node("answer_sub", self._answer_sub)
        graph.add_node("sub_next", self._sub_next)
        graph.add_node("synthesize", self._synthesize)

        graph.add_node("memory_update", self._memory_update)
        graph.add_node("reflect_if_due", self._reflect_if_due)

        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "decide_multihop")
        graph.add_conditional_edges(
            "decide_multihop",
            lambda s: "multi" if s.get("is_multi_hop") else "single",
            {"multi": "decompose", "single": "retrieve_docs_single"},
        )

        # Single-hop path
        graph.add_edge("retrieve_docs_single", "filter_docs_single")
        graph.add_edge("filter_docs_single", "retrieve_memory_single")
        graph.add_edge("retrieve_memory_single", "answer_single")
        graph.add_edge("answer_single", "memory_update")
        graph.add_edge("memory_update", "reflect_if_due")
        graph.add_edge("reflect_if_due", END)

        # Multi-hop path
        graph.add_edge("decompose", "init_sub")
        graph.add_edge("init_sub", "retrieve_docs_sub")
        graph.add_edge("retrieve_docs_sub", "filter_docs_sub")
        graph.add_edge("filter_docs_sub", "retrieve_memory_sub")
        graph.add_edge("retrieve_memory_sub", "answer_sub")
        graph.add_edge("answer_sub", "sub_next")
        graph.add_conditional_edges(
            "sub_next",
            lambda s: "loop" if s.get("has_next") else "synthesize",
            {"loop": "retrieve_docs_sub", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", "memory_update")
        graph.add_edge("memory_update", "reflect_if_due")
        graph.add_edge("reflect_if_due", END)

        return graph.compile()

    def _rewrite_query(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]
        history = self._truncate_history(state.get("session_history") or [])
        need_reformulate = len(history) >= 2
        standalone = query
        if need_reformulate:
            standalone = self.workflow.reformulate_query(query, history)
        return {
            "standalone_query": standalone,
            "active_query": standalone,
            "session_history": history,
        }

    def _decide_multihop(self, state: GraphState) -> Dict[str, Any]:
        is_multi = self.decomp_checker.identify_multi_hop_pattern(state["standalone_query"])
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "identify",
                "description": f"Question is {'multi-hop' if is_multi else 'single-hop'}",
                "query": state["standalone_query"],
            }
        )
        return {"is_multi_hop": is_multi, "thinking_process": steps}

    def _decompose(self, state: GraphState) -> Dict[str, Any]:
        sub_questions = self.workflow.decompose_query(state["standalone_query"])
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "decomposition",
                "description": f"Decomposed into {len(sub_questions)} sub-questions",
                "sub_questions": sub_questions,
            }
        )
        return {"sub_questions": sub_questions, "sub_idx": 0, "sub_answers": [], "thinking_process": steps}

    def _init_sub(self, state: GraphState) -> Dict[str, Any]:
        sub_questions = state.get("sub_questions") or []
        if sub_questions:
            return {"active_query": sub_questions[0], "sub_idx": 0}
        return {"active_query": state["standalone_query"], "sub_idx": 0}

    def _retrieve_docs(self, state: GraphState) -> Dict[str, Any]:
        query = state["active_query"]
        combined = {}
        for qv in self.workflow._expand_queries(query):
            retrieved = self.retriever.retrieve(qv, k=10)
            for doc_id, doc_text, score in retrieved:
                if doc_id not in combined or score > combined[doc_id][1]:
                    combined[doc_id] = (doc_text, score)
        retrieved_docs = [(doc_id, dt, sc) for doc_id, (dt, sc) in combined.items()]
        retrieved_docs.sort(key=lambda x: x[2], reverse=True)
        retrieved_docs = retrieved_docs[:10]
        if self.reranker:
            try:
                retrieved_docs = self.reranker.rerank(query, retrieved_docs, k=10)
            except Exception:
                pass
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "retrieval",
                "description": "Retrieved documents",
                "query": query,
                "retrieved_docs": retrieved_docs,
            }
        )
        return {"retrieved_docs": retrieved_docs, "thinking_process": steps}

    def _retrieve_docs_single(self, state: GraphState) -> Dict[str, Any]:
        return self._retrieve_docs(state)

    def _retrieve_docs_sub(self, state: GraphState) -> Dict[str, Any]:
        return self._retrieve_docs(state)

    def _filter_docs(self, state: GraphState) -> Dict[str, Any]:
        query = state["active_query"]
        filtered = self.workflow._post_filter(query, state.get("retrieved_docs") or [], max_docs=4)
        evidence_docs = [self.workflow._compress_doc(query, doc_text) for (_, doc_text, _) in filtered]
        evidence_docs = [d for d in evidence_docs if d]
        return {"filtered_docs": filtered, "evidence_docs": evidence_docs}

    def _filter_docs_single(self, state: GraphState) -> Dict[str, Any]:
        return self._filter_docs(state)

    def _filter_docs_sub(self, state: GraphState) -> Dict[str, Any]:
        return self._filter_docs(state)

    def _retrieve_memory(self, state: GraphState) -> Dict[str, Any]:
        if not self.enable_memory or not self.memory_store:
            return {"memory_context": []}
        user_id = state.get("user_id", "default_user")
        mems = self.memory_store.get_relevant(user_id, state["active_query"], limit=self.memory_top_k)
        return {"memory_context": mems}

    def _retrieve_memory_single(self, state: GraphState) -> Dict[str, Any]:
        return self._retrieve_memory(state)

    def _retrieve_memory_sub(self, state: GraphState) -> Dict[str, Any]:
        return self._retrieve_memory(state)

    def _answer_single(self, state: GraphState) -> Dict[str, Any]:
        self.workflow.memory_context = state.get("memory_context") or []
        answer = self.workflow.answer_from_docs(state["active_query"], state.get("evidence_docs") or [])
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "generation",
                "description": "Single-hop generation",
                "query": state["active_query"],
                "result": answer,
            }
        )
        return {"final_answer": answer, "thinking_process": steps}

    def _answer_sub(self, state: GraphState) -> Dict[str, Any]:
        self.workflow.memory_context = state.get("memory_context") or []
        answer = self.workflow.answer_from_docs(state["active_query"], state.get("evidence_docs") or [])
        sub_answers = list(state.get("sub_answers") or [])
        sub_answers.append(answer)
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "sub_generation",
                "description": "Sub-question generation",
                "query": state["active_query"],
                "result": answer,
            }
        )
        return {"sub_answers": sub_answers, "thinking_process": steps}

    def _sub_next(self, state: GraphState) -> Dict[str, Any]:
        sub_questions = state.get("sub_questions") or []
        idx = state.get("sub_idx", 0)
        if idx + 1 < len(sub_questions):
            next_idx = idx + 1
            return {"sub_idx": next_idx, "active_query": sub_questions[next_idx], "has_next": True}
        return {"has_next": False}

    def _synthesize(self, state: GraphState) -> Dict[str, Any]:
        final = self.workflow.synthesize_answer(state["standalone_query"], state.get("sub_answers") or [])
        steps = list(state.get("thinking_process", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "type": "synthesis",
                "description": "Synthesize final answer",
                "result": final,
            }
        )
        return {"final_answer": final, "thinking_process": steps}

    def _memory_update(self, state: GraphState) -> Dict[str, Any]:
        if not (self.enable_memory and self.memory_store and self.memory_selector):
            return {"memory_confirmation": []}
        user_id = state.get("user_id", "default_user")
        extracted = self.memory_selector.extract(state["query"], state.get("final_answer", ""), datetime.utcnow())
        verified = []
        if extracted and self.memory_verifier:
            for mem in extracted:
                if self.memory_verifier.verify(state["query"], state.get("final_answer", ""), mem):
                    verified.append(mem)
        else:
            verified = extracted
        high_risk = [m for m in verified if self._is_high_risk_memory(m)]
        safe = [m for m in verified if m not in high_risk]
        memory_confirmation = []
        if safe:
            self.memory_store.add_memories(user_id=user_id, memories=safe)
        if high_risk:
            pending = self.memory_store.add_pending_memories(user_id=user_id, memories=high_risk)
            for p in pending:
                memory_confirmation.append(
                    {
                        "id": p["id"],
                        "type": p["type"],
                        "content": p["content"],
                        "prompt": "是否允许保存为长期记忆？",
                    }
                )
        return {"memory_confirmation": memory_confirmation}

    def _reflect_if_due(self, state: GraphState) -> Dict[str, Any]:
        if self.enable_memory and self.memory_reflector and self.memory_store:
            user_id = state.get("user_id", "default_user")
            try:
                self.memory_reflector.run_if_due(user_id)
            except Exception as e:
                # reflection is best-effort; don't fail the request
                return {"memory_reflect_error": str(e)}
            try:
                self.memory_store.forget(user_id)
            except Exception as e:
                return {"memory_reflect_error": str(e)}
        return {}

    def run(
        self,
        query: str,
        session_history: Optional[List[Dict[str, str]]] = None,
        user_id: str = "default_user",
        external_conn=None,
    ) -> Dict[str, Any]:
        state = {
            "query": query,
            "session_history": session_history or [],
            "user_id": user_id,
            "thinking_process": [],
        }
        # If an external DB connection is provided, let memory_store reuse it for this run
        if self.enable_memory and self.memory_store and external_conn:
            self.memory_store.set_external_connection(external_conn)
        try:
            result = self.graph.invoke(state)
            return {
                "answer": result.get("final_answer", ""),
                "thinking_process": result.get("thinking_process", []),
                "memory_confirmation": result.get("memory_confirmation", []),
            }
        finally:
            if self.enable_memory and self.memory_store and external_conn:
                self.memory_store.clear_external_connection()
