from typing import List, Tuple, Dict, Optional
import logging
import os
from abc import ABC, abstractmethod
from typing import List
import re

logger = logging.getLogger('RAG')

# chunk parameters
TARGET_TOKENS = 200
MIN_TOKENS = 100
OVERLAP_TOKENS = 100

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Provides common functionality and factory method.
    """
    def __init__(
        self,
        collection_path: str,
        cache_dir: str = os.getenv('RAG_CACHE_DIR', './cache')
    ):
        self.collection_path = collection_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # map chunk_id -> original doc_id for eval/metrics alignment
        self.chunk_to_orig = {}
        
        # Load collection
        self._load_collection(collection_path)

    def _split_markdown_blocks(self, text: str) -> List[str]:
        """Split by markdown headings to keep sections together."""
        parts = re.split(r"(?=^#{1,6}\\s+)", text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, section: str) -> List[str]:
        """
        Lightweight sentence splitter to avoid nltk data dependency.
        Splits on ., ?, ! while keeping delimiters.
        """
        # Normalize whitespace
        section = re.sub(r'\s+', ' ', section).strip()
        if not section:
            return []
        # Split on sentence end markers
        parts = re.split(r'(?<=[.!?])\s+', section)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_text(self, text: str, base_id: str) -> List[tuple[str, str]]:
        """
        Chunk text by sentences with overlap, respecting markdown sections.
        Returns list of (chunk_id, chunk_text).
        """
        chunks = []
        sections = self._split_markdown_blocks(text)
        for sec_idx, section in enumerate(sections):
            sents = self._split_sentences(section)
            sent_tokens = [sent.split() for sent in sents]
            i = 0
            while i < len(sent_tokens):
                buf, count = [], 0
                start_i = i
                while i < len(sent_tokens) and count < TARGET_TOKENS:
                    buf.extend(sent_tokens[i])
                    count += len(sent_tokens[i])
                    i += 1
                if count < MIN_TOKENS and i < len(sent_tokens):
                    # pull one more sentence to reach minimal length
                    buf.extend(sent_tokens[i])
                    count += len(sent_tokens[i])
                    i += 1
                chunk_id = f"{base_id}_sec{sec_idx}_chunk{start_i}"
                chunks.append((chunk_id, " ".join(buf)))
                # overlap window
                if i < len(sent_tokens):
                    # move back to create overlap of roughly OVERLAP_TOKENS
                    back_tokens = 0
                    j = i - 1
                    while j >= 0 and back_tokens < OVERLAP_TOKENS:
                        back_tokens += len(sent_tokens[j])
                        j -= 1
                    i = max(j + 1, start_i + 1)
        return chunks if chunks else [(base_id, text)]

    def _load_collection(self, path: str):
        """Loads the document collection from HuggingFace or a file."""
        from datasets import load_dataset
        logger.info(f"Loading collection from {path}...")
        hotpot_dataset = load_dataset(path)
        collection_dataset = hotpot_dataset["collection"]
        doc_texts, doc_ids = [], []
        for ex in collection_dataset:
            text = ex["text"]
            orig_id = ex["id"]
            for cid, ctext in self._chunk_text(text, orig_id):
                doc_ids.append(cid)
                doc_texts.append(ctext)
                self.chunk_to_orig[cid] = orig_id
        self.doc_texts = doc_texts
        self.doc_ids = doc_ids
        self.id_to_text = {id_: text for id_, text in zip(self.doc_ids, self.doc_texts)}
        logger.info(f"Loaded {len(self.doc_texts)} documents after chunking.")


    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Abstract method to retrieve documents.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (doc_id, doc_text, score)
        """
        pass


    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs):
        """
        Factory method to create retriever instances.
        
        Args:
            retriever_type: One of 'sparse', 'static_embedding', 'dense', 'hybrid'
            **kwargs: Arguments for the specific retriever
            
        Returns:
            An instance of the specified retriever
        """
        if retriever_type.lower() == 'sparse':
            from sparse_retriever import SparseRetriever
            return SparseRetriever(**kwargs)
        elif retriever_type.lower() == 'static_embedding':
            from static_embedding_retriever import StaticEmbeddingRetriever
            return StaticEmbeddingRetriever(**kwargs)
        elif retriever_type.lower() == 'dense':
            from dense_retriever import DenseRetriever
            return DenseRetriever(**kwargs)
        elif retriever_type.lower() == 'hybrid':
            from hybrid_retriever import HybridRetriever
            return HybridRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
