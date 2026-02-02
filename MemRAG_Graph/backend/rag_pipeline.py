from typing import List, Dict, Any, Tuple, Optional
import logging

from agentic_workflow import AgenticWorkflow
from singlehop_workflow import SingleHopWorkflow
from retriever_base import BaseRetriever
from huggingface_generator import HuggingfaceGenerator
from openai_generator import OpenAIGenerator
from reranker import QwenReranker
import os

logger = logging.getLogger('RAG')

class RAGPipeline:
    """
    The main RAG Pipeline orchestrator.
    Integrates retrieval and generation modules to produce a final answer,
    along with supporting information like retrieved documents and thinking process.
    Designed to support single-turn and future multi-turn interactions.
    """
    def __init__(self, retriever: BaseRetriever, enable_agentic_workflow : bool = True):
        """
        Initializes the RAG Pipeline.

        Args:
            retriever (BaseRetriever): The retrieval module instance.
            generator (QwenGenerator): The generation module instance.
        """
        self.retriever = retriever
        self.enable_agentic_workflow = enable_agentic_workflow
        self.generator_map = {} # generator map
        self.reranker = None
        self.max_history_turns = int(os.getenv("RAG_SHORT_TERM_TURNS", "6"))

        rerank_api_url = os.getenv("RAG_RERANK_API_URL")
        rerank_api_key = os.getenv("RAG_RERANK_API_KEY")
        rerank_enabled = os.getenv("RAG_ENABLE_RERANK", "1") == "1"
        if rerank_enabled and rerank_api_url and rerank_api_key:
            try:
                self.reranker = QwenReranker()
                logger.info("Qwen reranker enabled.")
            except Exception as e:
                logger.warning(f"Failed to init reranker, fallback to no rerank: {e}")


    def init_generator(self, model_name : str):
        """
        Initialize a model and add it into the generator map
        
        Args:
            model_name (str) : The huggingface model or the openai API model
        """
        if model_name in self.generator_map:
            logger.debug(f"{model_name} already been loaded, skip")
            return self.generator_map[model_name]
        logger.info("start init new generator: {model_name}")
        if model_name == "Qwen/Qwen2.5-0.5B-Instruct":
            self.generator_map[model_name] = HuggingfaceGenerator(model_name=model_name)
        else:
            self.generator_map[model_name] = OpenAIGenerator(model_name=model_name)
        logger.info("new generator: {model_name} loaded.")
        return self.generator_map[model_name]


    def run(
        self,
        query: str,
        session_history: Optional[List[Dict[str, str]]] = None,
        model_name : str = "Qwen/Qwen2.5-0.5B-Instruct",
        memory_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a single query.

        Args:
            query (str): The user's query.
            session_history (Optional[List[Dict[str, str]]]): For future multi-turn support.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, retrieved docs, and thinking process.
        """
        # check generator
        self.init_generator(model_name)
        generator = self.generator_map[model_name]

        thinking_process = []

        need_reformulate = False
        if session_history:
            if self.max_history_turns > 0:
                max_messages = self.max_history_turns * 2
                if len(session_history) > max_messages:
                    session_history = session_history[-max_messages:]
            if len(session_history) < 2:
                logger.debug("Session history too short; using original query.")
            else:
                logger.debug("Reformulating query based on session history...")
                need_reformulate = True
        else:
            logger.debug("No session history provided; using original query.")

        workflow = None
        if self.enable_agentic_workflow:
            logger.debug("Generating answer with Agentic Workflow...")
            workflow = AgenticWorkflow(
                self.retriever,
                generator,
                self.reranker,
                need_reformulate,
                session_history,
                memory_context,
            )
        else:
            workflow = SingleHopWorkflow(self.retriever, generator, self.reranker, memory_context)
        
        final_answer, intermediate_steps = workflow.run(query)

        for step in intermediate_steps:
            thinking_process_item = {}
            step_no = step['step']
            step_description = step['description']
            step_type = step['type']
            thinking_process_item['type'] = step['type']
            thinking_process_item['step'] = step_no
            thinking_process_item['description'] = f"[{step_no}] {step_description}"
            if step_type == 'multi_hop_sub_generation':
                thinking_process_item['result'] = step['result']
            if 'retrieved_docs' in step:
                thinking_process_item['retrieved_docs'] = [{'id' : id, 'text': text, 'score': score} for id, text, score in step['retrieved_docs']]
            thinking_process.append(thinking_process_item)
                
        response_data = {
            "answer": final_answer,
            "thinking_process":thinking_process
        }

        logger.debug("RAG pipeline completed successfully.")
        return response_data
