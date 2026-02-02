# MemRAG Graph

A multi-hop Retrieval-Augmented Generation (RAG) system with hybrid search, optional reranking, LangGraph-based reasoning, and a durable user memory layer. Ships with a Flask API, evaluation scripts, and switchable generators (HuggingFace local or OpenAI/DashScope-compatible APIs).

## Features
- **Hybrid retrieval**: BM25 (sparse) + BGE (dense) with weighted fusion; optional Qwen reranker API.
- **Multi-/single-hop routing**: Automatic detection, decomposition, sub-question answering, and synthesis via LangGraph (`backend/langgraph_agent.py`).
- **Memory system**: SQLite-backed short/long-term memory with extraction, verification, pending approval, reflection, and forgetting (`backend/memory_*`).
- **Pluggable generation**: Local HuggingFace models (`Qwen/Qwen2.5-*`) or remote models via OpenAI-compatible endpoints.
- **API ready**: Flask server exposes health, chat sessions, messaging, and memory confirmation endpoints.
- **Evaluation**: HotpotQA-style QA and retrieval evaluation scripts under `evaluate/`.

## Directory Structure
- `run_agent.py` -- minimal single-run example.
- `backend/` -- retrievers, generators, LangGraph agent, memory, Flask server, env specs.
- `evaluate/` -- evaluation scripts and sample JSONL datasets.
- `.env.example` -- template for environment variables (copy to `.env`).
- `LICENSE` -- MIT License.

## Setup
1) Install dependencies (Conda recommended)
```bash
conda env create -f backend/environment.yml
conda activate agentic_RAG
python -m spacy download en_core_web_sm  # required by DecompositionChecker
```
Or with pip:
```bash
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

2) Configure environment variables
Copy `.env.example` to `.env` and fill in at least:
- `RAG_OPENAI_API_KEY`, `RAG_OPENAI_API_URL` (OpenAI-compatible generation)
- `RAG_RERANK_API_KEY`, `RAG_RERANK_API_URL`, `RAG_RERANK_MODEL` (if rerank is enabled)
- `RAG_DEFAULT_MODEL` (e.g., `qwen3-max` or a local HF model name)
- `RAG_MEMORY_MODEL`, `RAG_MEMORY_VERIFY_MODEL`, `RAG_MEMORY_EMBED_MODEL`
- `RAG_STORAGE_DIR`, `RAG_CACHE_DIR` (defaults: `backend/storage`, `backend/cache`)
- Optional: `RAG_ENABLE_MEMORY`, `RAG_ENABLE_RERANK`, `RAG_SHORT_TERM_TURNS`, `RAG_MEMORY_REFLECT_HOURS`, `RAG_MEMORY_TOP_K`, `RAG_QE_MODEL`

3) Prepare retrieval corpus
Default `collection_path` is `izhx/COMP5423-25Fall-HQ-small` (see `run_agent.py` and `backend/server.py`). Replace it if you use another corpus.

## Run
- Quick demo (no API):
```bash
python run_agent.py
```
- Start Flask API:
```bash
python backend/server.py
```
SQLite will initialize at `RAG_STORAGE_DIR/chat_history.db` using `backend/db_init.sql`, and RAG modules load asynchronously.

## API Overview (Flask)
- `GET /api/health` -- readiness and path info
- `GET /api/chats/list?user_id=` -- list chat sessions
- `POST /api/chats/new` -- create a chat session
- `DELETE /api/chat/<chat_id>` -- delete a session
- `GET /api/chat/<chat_id>/messages` -- fetch messages + thinking traces
- `POST /api/chat/<chat_id>/messages` -- send a message and get RAG answer
- `GET /api/memory/pending` -- list pending (high-risk) memories
- `POST /api/memory/confirm` -- confirm or reject a pending memory

Example message call:
```bash
curl -X POST http://localhost:5000/api/chat/<chat_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "who is obama?", "model_name": "qwen3-max", "user_id": "u_001"}'
```

## Memory & Safety
- High-risk or time-sensitive memories go to `memory_pending` and require explicit confirmation.
- Reflection runs periodically (`RAG_MEMORY_REFLECT_HOURS`) to distill stable insights; `MemoryStore.forget` prunes expired/low-importance memories.

## Evaluation
- `evaluate/eval_hotpotqa.py` -- multi-hop QA evaluation
- `evaluate/eval_retrieval.py` -- retrieval quality evaluation
- Sample data: `evaluate/test_predict.jsonl`, `evaluate/validation.jsonl`

## Troubleshooting
- First run may be slow while BM25/FAISS/Word2Vec indexes cache in `RAG_CACHE_DIR`.
- GPU vs CPU: HuggingFace generator auto-selects CUDA; otherwise uses CPU (`float32`, `low_cpu_mem_usage`).
- spaCy model missing: `python -m spacy download en_core_web_sm`.

## License
Released under the MIT License (see `LICENSE`).

