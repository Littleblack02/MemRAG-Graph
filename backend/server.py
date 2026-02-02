from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import time
import uuid
from datetime import datetime
import logging
import os
import traceback
import threading
import re
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env at startup
load_dotenv(find_dotenv())

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]')
    file_handler.setFormatter(file_format)

    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# Configuration
RETRIEVER_TYPE = "hybrid"
DEFAULT_MODEL = os.getenv("RAG_DEFAULT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MEMORY_MODEL = os.getenv("RAG_MEMORY_MODEL", "qwen3-vl-8b-instruct")
MEMORY_VERIFY_MODEL = os.getenv("RAG_MEMORY_VERIFY_MODEL", "qwen3-30b-a3b-instruct-2507")
MEMORY_EMBED_MODEL = os.getenv("RAG_MEMORY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MEMORY_REFLECT_HOURS = int(os.getenv("RAG_MEMORY_REFLECT_HOURS", "24"))
MEMORY_TOP_K = int(os.getenv("RAG_MEMORY_TOP_K", "8"))
ENABLE_MEMORY = os.getenv("RAG_ENABLE_MEMORY", "1") == "1"
RAG_STORAGE_DIR = os.getenv('RAG_STORAGE_DIR', './storage')
RAG_CACHE_DIR=os.getenv('RAG_CACHE_DIR', './cache')
RAG_BACKEND_PORT = os.getenv('RAG_BACKEND_PORT', '5000')
RAG_BACKEND_HOST = os.getenv('RAG_BACKEND_HOST', '0.0.0.0')
os.makedirs(RAG_STORAGE_DIR, exist_ok = True)
os.makedirs(RAG_CACHE_DIR, exist_ok = True)
DATABASE_PATH = os.path.join(RAG_STORAGE_DIR, 'chat_history.db')
LOGGER_PATH = os.path.join(RAG_STORAGE_DIR, 'rag.log')

# Logger
APPNAME = "RAG"
logger = setup_logger(APPNAME, LOGGER_PATH, level=logging.DEBUG)
logger.info("== RAG Server Starting ==")
logger.info(f"Using storage directory: {RAG_STORAGE_DIR}")
logger.info(f"Using cache directory: {RAG_CACHE_DIR}")
logger.info(f"Using address: {RAG_BACKEND_HOST}:{RAG_BACKEND_PORT}")
logger.info(f"Using database: {DATABASE_PATH}")
logger.info(f"Using logger: {LOGGER_PATH}")


rag_agent = None
RAG_Initialized = False
init_error = None
init_lock = threading.Lock()  # Thread-safe lock for initialization
memory_store = None
memory_selector = None
memory_reflector = None
memory_verifier = None
memory_lock = threading.Lock()

# Initialize RAG modules
def initialize_rag_modules():
    """Function to initialize RAG modules in background thread"""
    global rag_agent, RAG_Initialized, init_error, memory_store, memory_selector, memory_reflector, memory_verifier

    with init_lock:  # Ensure only one initialization happens
        if RAG_Initialized:
            return
        try:
            logger.info("import RAG modules...")
            now = time.time()
            from langgraph_agent import RAGLangGraphAgent
            logger.info(f"RAG modules loaded. ({(time.time() - now):.2f} seconds)")

            logger.info("Initialize RAG modules...")
            now = time.time()
            rag_agent = RAGLangGraphAgent(
                retriever_type=RETRIEVER_TYPE,
                collection_path="izhx/COMP5423-25Fall-HQ-small",
                model_name=DEFAULT_MODEL,
                enable_rerank=os.getenv("RAG_ENABLE_RERANK", "1") == "1",
                enable_memory=ENABLE_MEMORY,
                memory_top_k=MEMORY_TOP_K,
                memory_model=MEMORY_MODEL,
                memory_verify_model=MEMORY_VERIFY_MODEL,
                memory_embed_model=MEMORY_EMBED_MODEL,
                memory_reflect_hours=MEMORY_REFLECT_HOURS,
                storage_dir=RAG_STORAGE_DIR,
                cache_dir=RAG_CACHE_DIR,
            )
            # Expose memory modules for other endpoints
            memory_store = rag_agent.memory_store
            memory_selector = rag_agent.memory_selector
            memory_reflector = rag_agent.memory_reflector
            memory_verifier = rag_agent.memory_verifier
            RAG_Initialized = True
            logger.info(f"RAG modules initialized. ({(time.time() - now):.2f} seconds)")
        except Exception as e:
            logger.error(f"Error init rag modules: {e}")
            logger.error(traceback.format_exc())
            init_error = str(e)
            return


def initialize_memory_modules():
    """Initialize memory modules lazily."""
    global memory_store, memory_selector, memory_reflector, memory_verifier
    if not ENABLE_MEMORY:
        return
    with memory_lock:
        if memory_store and memory_selector and memory_reflector and memory_verifier:
            return
        try:
            # Reuse agent's memory modules if available
            if rag_agent and getattr(rag_agent, "memory_store", None):
                memory_store = rag_agent.memory_store
                memory_selector = rag_agent.memory_selector
                memory_reflector = rag_agent.memory_reflector
                memory_verifier = rag_agent.memory_verifier
                logger.info("Memory modules reused from LangGraph agent.")
                return
            from memory_store import MemoryStore
            from memory_selector import MemorySelector
            from memory_reflector import MemoryReflector
            from memory_verifier import MemoryVerifier
            memory_store = MemoryStore(
                DATABASE_PATH,
                logger=logger,
                embed_model_name=MEMORY_EMBED_MODEL,
            )
            memory_selector = MemorySelector(model_name=MEMORY_MODEL, logger=logger)
            memory_verifier = MemoryVerifier(model_name=MEMORY_VERIFY_MODEL, logger=logger)
            memory_reflector = MemoryReflector(
                memory_store=memory_store,
                generator=memory_selector.generator,
                reflect_hours=MEMORY_REFLECT_HOURS,
                logger=logger,
            )
            logger.info("Memory modules initialized.")
        except Exception as e:
            logger.error(f"Error init memory modules: {e}")
            logger.error(traceback.format_exc())


def _is_high_risk_memory(mem: dict) -> bool:
    mem_type = (mem.get("type") or mem.get("memory_type") or "").lower()
    if mem_type in ("event", "personal"):
        return True
    if mem.get("event_time"):
        return True
    content = (mem.get("content") or "").lower()
    patterns = [
        r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b",  # phone
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # date
        r"\b\d{4}-\d{2}-\d{2}\b",  # date
        r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",  # email
        r"\b(address|street|st\.|avenue|ave\.|road|rd\.|zip|postal)\b",
        r"\b(id|passport|ssn|social security)\b",
    ]
    return any(re.search(p, content) for p in patterns)


logger.info("Starting Flask app...")
app = Flask(__name__)
CORS(app)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # This allows us to fetch rows as dictionaries
    return conn


def init_db():
    """Initializes the database with the required tables."""
    with app.app_context():
        db = get_db_connection()
        with app.open_resource('db_init.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        # Ensure user_id column exists for existing databases
        cols = [row["name"] for row in db.execute("PRAGMA table_info(chat_sessions)")]
        if "user_id" not in cols:
            db.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT DEFAULT 'default_user'")
        db.commit()
        db.close()


@app.route('/api/health')
def api_health():
    return jsonify({
        "ok": True,
        "storage" : RAG_STORAGE_DIR,
        "cache" : RAG_CACHE_DIR,
        "logger" : LOGGER_PATH,
        "db" : DATABASE_PATH,
        "ready" : RAG_Initialized,
        "init_error" : init_error,
    })


@app.route('/api/chats/list', methods=['GET'])
def get_chat_history():
    """
    Retrieves a list of all chat sessions for the sidebar.
    Returns a list of dictionaries containing id and title.
    """
    try:
        conn = get_db_connection()
        user_id = request.args.get('user_id')
        if user_id:
            chats = conn.execute('''
                SELECT id, title, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,)).fetchall()
        else:
            chats = conn.execute('''
                SELECT id, title, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
            ''').fetchall()
        conn.close()

        # Convert Row objects to dictionaries
        chat_list = [dict(chat) for chat in chats]
        return jsonify(chat_list), 200
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@app.route('/api/chats/new', methods=['POST'])
def create_new_chat():
    """
    Creates a new chat session using a UUID.
    """
    try:
        payload = request.get_json(silent=True) or {}
        user_id = payload.get('user_id') or request.headers.get('X-User-Id') or 'default_user'
        conn = get_db_connection()
        # Generate a new UUID for the session
        new_chat_id = str(uuid.uuid4())
        # Insert a new session with the generated UUID
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO chat_sessions (id, title, user_id)
            VALUES (?, ?, ?)
        ''', (new_chat_id, 'New Chat', user_id)) # Explicitly set the ID and default title
        conn.commit()
        conn.close()

        # Return the ID of the new chat
        return jsonify({'id': new_chat_id, 'title': 'New Chat'}), 201
    except Exception as e:
        logger.error(f"Error creating new chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to create new chat'}), 500


@app.route('/api/chat/<string:chat_id>', methods=['DELETE'])
def delete_chat(chat_id : str):
    """
    Deletes a specific chat session and its messages.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the chat session exists
        session_check = cur.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404

        # Delete the session (messages will be deleted automatically due to CASCADE)
        cur.execute('DELETE FROM chat_sessions WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()

        # If the deleted chat was the current one, the frontend might want to clear the chat panel
        # Returning a success message is sufficient here
        return jsonify({'message': 'Chat deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        logger.error(traceback.format_exc())
        conn.rollback() # Rollback in case of error
        return jsonify({'error': 'Failed to delete chat'}), 500


@app.route('/api/message/<string:message_id>', methods=['DELETE'])
def delete_message(message_id : str):
    """
    Deletes a specific message.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the session (messages will be deleted automatically due to CASCADE)
        cur.execute('DELETE FROM messages WHERE id = ?', (message_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'message deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}")
        logger.error(traceback.format_exc())
        conn.rollback() # Rollback in case of error
        return jsonify({'error': 'Failed to delete chat'}), 500


@app.route('/api/chat/<string:chat_id>/messages', methods=['GET'])
def get_messages(chat_id : str):
    """
    Retrieves all messages for a specific chat session identified by UUID.
    """
    try:
        conn = get_db_connection()
        # validate the session
        session_check = conn.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404
        
        messages = conn.execute('''
            SELECT id, sender, content, timestamp, thinking_process
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (chat_id,)).fetchall() # Use the UUID string as the parameter
        conn.close()

        message_list = [dict(msg) for msg in messages]
        return jsonify(message_list), 200
    except Exception as e:
        logger.error(f"Error fetching messages for chat {chat_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to retrieve messages for this chat'}), 500


@app.route('/api/memory/pending', methods=['GET'])
def get_pending_memories():
    if not ENABLE_MEMORY:
        return jsonify([]), 200
    try:
        initialize_memory_modules()
        user_id = request.args.get('user_id') or request.headers.get('X-User-Id') or 'default_user'
        if not memory_store:
            return jsonify([]), 200
        pending = memory_store.list_pending(user_id)
        return jsonify(pending), 200
    except Exception as e:
        logger.error(f"Error fetching pending memories: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to retrieve pending memories'}), 500


@app.route('/api/memory/confirm', methods=['POST'])
def confirm_memory():
    if not ENABLE_MEMORY:
        return jsonify({'ok': False}), 200
    try:
        initialize_memory_modules()
        payload = request.get_json(silent=True) or {}
        user_id = payload.get('user_id') or request.headers.get('X-User-Id') or 'default_user'
        memory_id = payload.get('memory_id')
        confirm = bool(payload.get('confirm', False))
        if not memory_id:
            return jsonify({'error': 'memory_id required'}), 400
        if not memory_store:
            return jsonify({'ok': False}), 500
        ok = memory_store.confirm_pending(user_id, memory_id, confirm)
        return jsonify({'ok': ok}), 200
    except Exception as e:
        logger.error(f"Error confirming memory: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to confirm memory'}), 500


@app.route('/api/chat/<string:chat_id>/messages', methods=['POST'])
def send_message(chat_id : str):
    """
    Handles a new user message for a specific chat UUID, calls the RAG system,
    stores both messages, and returns the bot's response.
    Expects JSON: { "message": "user's query" }
    """
    if not RAG_Initialized:
        return jsonify({'error': 'RAG module not initialized'}), 500
    try:
        user_data = request.get_json()
        user_message = user_data.get('message', '').strip()
        model_name = user_data.get('model_name', DEFAULT_MODEL).strip()
        logger.debug(f"user send new query using model: {model_name}")

        if not user_message:
            return jsonify({'error': 'Message content is required'}), 400

        # Validate that the chat_id exists before proceeding
        conn = get_db_connection()
        session_check = conn.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404

        # Resolve user_id for cross-session memory
        user_row = conn.execute('SELECT user_id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        user_id = None
        if user_row:
            user_id = user_row['user_id']
        if not user_id:
            user_id = user_data.get('user_id') or request.headers.get('X-User-Id') or 'default_user'
            conn.execute('UPDATE chat_sessions SET user_id = ? WHERE id = ?', (user_id, chat_id))
        
        # Collect history dislogues for Multi-turn
        history_dialogues = conn.execute('''
            SELECT sender, content 
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (chat_id,)).fetchall()
        history_dialogues = [dict(msg) for msg in history_dialogues]
        history_dialogues
        logger.debug(f'Collected history dialogues: {history_dialogues}')

        user_message_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content)
            VALUES (?, ?, 'user', ?)
        ''', (user_message_id, chat_id, user_message)) # Use generated UUID and provided chat_id

        now = time.time()
        logger.info(f"processing message for [{user_message_id}]:[{user_message[:50]}...]")
        rag_result = rag_agent.run(
            query=user_message,
            session_history=history_dialogues,
            user_id=user_id,
            external_conn=conn,
        )
        bot_response = rag_result.get("answer", "")
        thinking_process = rag_result.get("thinking_process", []) # [str]
        logger.info(f"Query {user_message_id} took {(time.time() - now):.2f} seconds.")
        logger.debug(f"Bot response: {bot_response}")
        logger.debug(f"Thinking process: {thinking_process}")

 
        bot_message_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content, thinking_process)
            VALUES (?, ?, 'bot', ?, ?)
        ''', (bot_message_id, chat_id, bot_response, json.dumps(thinking_process)))

        existing_messages = conn.execute('''
            SELECT id FROM messages WHERE session_id = ? LIMIT 2
        ''', (chat_id,)).fetchall()
        if len(existing_messages) == 2: 
             title = (user_message[:20] + '..') if len(user_message) > 20 else user_message
             cur.execute('UPDATE chat_sessions SET title = ? WHERE id = ?', (title, chat_id))

        conn.commit()
        conn.close()

        memory_confirmation = rag_result.get("memory_confirmation", [])

        return jsonify({
            'user_message_id' : user_message_id,
            'id': bot_message_id, # Return the ID of the bot's message just inserted
            'sender': 'bot',
            'content': bot_response,
            'thinking_process': thinking_process,
            'memory_confirmation': memory_confirmation
        }), 200

    except Exception as e:
        logger.error(f"Error processing message for chat {chat_id}: {e}")
        logger.error(traceback.format_exc())
        # Consider rolling back the transaction if both messages should be atomic
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': 'Failed to process message'}), 500


if __name__ == '__main__':
    # Initialize the database when the script is run directly
    init_db()
    # Initialize RAG module asynchronizely
    init_thread = threading.Thread(target=initialize_rag_modules, daemon=True)
    init_thread.start()
    # Run the Flask app
    app.run(debug=True, host=RAG_BACKEND_HOST, port=int(RAG_BACKEND_PORT), use_reloader=False)
