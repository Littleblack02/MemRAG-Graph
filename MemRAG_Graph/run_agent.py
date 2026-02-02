import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.langgraph_agent import RAGLangGraphAgent
agent = RAGLangGraphAgent(
    retriever_type="hybrid",
    collection_path="izhx/COMP5423-25Fall-HQ-small",
    model_name="qwen3-max",
    enable_rerank=True,
    enable_memory=True,
    memory_model=os.getenv("RAG_MEMORY_MODEL", "qwen3-vl-8b-instruct"),
    memory_verify_model=os.getenv("RAG_MEMORY_VERIFY_MODEL", "qwen3-30b-a3b-instruct-2507"),
)

res = agent.run(
    query="who is obama?",
    session_history=[],
    user_id="u_001",
)
print(res["answer"])
print(res.get("memory_confirmation"))
