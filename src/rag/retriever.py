"""
FAQ Retriever for RAG-based question answering.

Handles:
- Loading FAISS indices from disk
- Retrieving relevant FAQ entries based on user queries
- Formatting retrieved context for LLM consumption
"""

import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)


class FAQRetriever:
    """
    Retriever class for FAQ-based RAG.
    
    This class handles:
    1. Loading pre-built FAISS indices
    2. Encoding user queries
    3. Retrieving top-k relevant FAQ entries
    4. Formatting context for LLM consumption
    
    Attributes:
        index_path: Path to the FAISS index
        embedding_model_id: The embedding model being used
        top_k: Number of documents to retrieve
        vector_store: The loaded FAISS vector store
    """
    
    def __init__(
        self,
        index_path: Path | str,
        embedding_model_id: str = "BAAI/bge-m3",
        top_k: int = 3,
        device: str = "auto",
    ):
        """
        Initialize the FAQ retriever.
        
        Args:
            index_path: Path to the FAISS index directory
            embedding_model_id: Hugging Face model ID for embeddings
            top_k: Number of documents to retrieve per query
            device: Device to run embeddings on
        """
        self.index_path = Path(index_path)
        self.embedding_model_id = embedding_model_id
        self.top_k = top_k
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Initializing embeddings with {embedding_model_id} on {device}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_id,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        self.vector_store: FAISS | None = None
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a placeholder if not found."""
        if self.index_path.exists() and (self.index_path / "index.faiss").exists():
            logger.info(f"Loading FAISS index from {self.index_path}")
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_placeholder_index()
        else:
            logger.warning(f"FAISS index not found at {self.index_path}")
            self._create_placeholder_index()
    
    def _create_placeholder_index(self) -> None:
        """Create a placeholder index when no real index exists."""
        logger.info("Creating placeholder FAISS index")
        
        # Create minimal placeholder documents
        placeholder_docs = [
            Document(
                page_content="Welcome to CTBC Bank customer service. How may I help you today?",
                metadata={
                    "id": "placeholder",
                    "category": "General",
                    "question_zh": "歡迎使用中國信託客服",
                    "answer_zh": "請問有什麼可以幫助您的？",
                }
            )
        ]
        
        self.vector_store = FAISS.from_documents(placeholder_docs, self.embeddings)
        
        # Save the placeholder index
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_path))
        logger.info(f"Placeholder index saved to {self.index_path}")
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant FAQ documents for a query.
        
        Args:
            query: The user's query (in English)
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            logger.warning("No vector store available")
            return []
        
        k = top_k or self.top_k
        
        logger.debug(f"Retrieving top {k} documents for query: {query[:50]}...")
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve relevant FAQ documents with similarity scores.
        
        Args:
            query: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            logger.warning("No vector store available")
            return []
        
        k = top_k or self.top_k
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def format_context(
        self,
        documents: list[Document],
        include_metadata: bool = True,
    ) -> str:
        """
        Format retrieved documents into a context string for LLM.
        
        Args:
            documents: List of retrieved documents
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant FAQ information found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            parts = [f"[FAQ Entry {i}]"]
            
            if include_metadata:
                metadata = doc.metadata
                if metadata.get("category"):
                    parts.append(f"Category: {metadata['category']}")
            
            # Prefer English content if available, otherwise use Chinese
            if doc.metadata.get("question_en"):
                parts.append(f"Q: {doc.metadata['question_en']}")
            elif doc.metadata.get("question_zh"):
                parts.append(f"Q (Chinese): {doc.metadata['question_zh']}")
            
            if doc.metadata.get("answer_en"):
                parts.append(f"A: {doc.metadata['answer_en']}")
            elif doc.metadata.get("answer_zh"):
                parts.append(f"A (Chinese): {doc.metadata['answer_zh']}")
            
            context_parts.append("\n".join(parts))
        
        return "\n\n".join(context_parts)
    
    def get_relevant_context(self, query: str) -> str:
        """
        Convenience method to retrieve and format context in one call.
        
        Args:
            query: The user's query
            
        Returns:
            Formatted context string ready for LLM
        """
        docs = self.retrieve(query)
        return self.format_context(docs)
    
    def reload_index(self) -> bool:
        """
        Reload the FAISS index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._load_or_create_index()
            return True
        except Exception as e:
            logger.error(f"Failed to reload index: {e}")
            return False

