"""
FAISS Index Builder for CTBC FAQ data.

Handles:
- Loading processed FAQ data from disk
- Encoding FAQ entries using embedding models
- Building and persisting FAISS vector indices
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)


class FAQIndexBuilder:
    """
    Builder class for creating FAISS indices from FAQ data.
    
    This class handles the process of:
    1. Loading FAQ data from JSONL files
    2. Converting FAQ entries to LangChain Documents
    3. Encoding documents using embedding models
    4. Building and saving FAISS indices
    
    Attributes:
        embedding_model_id: The Hugging Face embedding model ID
        embeddings: The initialized embedding model
    """
    
    def __init__(
        self,
        embedding_model_id: str = "BAAI/bge-m3",
        device: str = "auto",
    ):
        """
        Initialize the FAQ index builder.
        
        Args:
            embedding_model_id: Hugging Face model ID for embeddings
            device: Device to run embeddings on ("auto", "cuda", "cpu", "mps")
        """
        self.embedding_model_id = embedding_model_id
        
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
    
    def load_faq_data(self, data_path: Path) -> list[dict[str, Any]]:
        """
        Load FAQ data from JSONL files.
        
        Args:
            data_path: Path to directory containing FAQ JSONL files
            
        Returns:
            List of FAQ entries as dictionaries
        """
        faq_entries: list[dict[str, Any]] = []
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"FAQ data path does not exist: {data_path}")
            return faq_entries
        
        # Load from JSONL files
        for jsonl_file in data_path.glob("*.jsonl"):
            logger.info(f"Loading FAQ data from {jsonl_file}")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            faq_entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line: {e}")
        
        # Also load from JSON files (array format)
        for json_file in data_path.glob("*.json"):
            logger.info(f"Loading FAQ data from {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        faq_entries.extend(data)
                    else:
                        faq_entries.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse {json_file}: {e}")
        
        logger.info(f"Loaded {len(faq_entries)} FAQ entries")
        return faq_entries
    
    def _create_documents(self, faq_entries: list[dict[str, Any]]) -> list[Document]:
        """
        Convert FAQ entries to LangChain Documents.
        
        Args:
            faq_entries: List of FAQ dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents: list[Document] = []
        
        for entry in faq_entries:
            # Build document content
            # Combine question and answer for better retrieval
            question_zh = entry.get("question_zh", "")
            answer_zh = entry.get("answer_zh", "")
            question_en = entry.get("question_en", "")
            answer_en = entry.get("answer_en", "")
            
            # Create content that includes both Chinese and English if available
            content_parts = []
            if question_zh:
                content_parts.append(f"問題: {question_zh}")
            if answer_zh:
                content_parts.append(f"答案: {answer_zh}")
            if question_en:
                content_parts.append(f"Question: {question_en}")
            if answer_en:
                content_parts.append(f"Answer: {answer_en}")
            
            content = "\n".join(content_parts)
            
            if not content.strip():
                continue
            
            # Create metadata
            metadata = {
                "id": entry.get("id", ""),
                "category": entry.get("category", ""),
                "source_url": entry.get("source_url", ""),
                "question_zh": question_zh,
                "answer_zh": answer_zh,
                "question_en": question_en,
                "answer_en": answer_en,
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Created {len(documents)} documents from FAQ entries")
        return documents
    
    def build_index(
        self,
        data_path: Path,
        output_path: Path,
    ) -> FAISS:
        """
        Build a FAISS index from FAQ data.
        
        Args:
            data_path: Path to FAQ data directory
            output_path: Path to save the FAISS index
            
        Returns:
            The built FAISS vector store
        """
        # Load FAQ data
        faq_entries = self.load_faq_data(data_path)
        
        if not faq_entries:
            logger.warning("No FAQ entries found, creating empty index with placeholder")
            # Create a minimal placeholder document
            faq_entries = [{
                "id": "placeholder",
                "category": "General",
                "question_zh": "歡迎使用中國信託客服",
                "answer_zh": "請問有什麼可以幫助您的？",
                "question_en": "Welcome to CTBC customer service",
                "answer_en": "How may I help you today?",
            }]
        
        # Convert to documents
        documents = self._create_documents(faq_entries)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save index
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving FAISS index to {output_path}")
        vector_store.save_local(str(output_path))
        
        logger.info("FAISS index built and saved successfully")
        return vector_store
    
    def add_documents(
        self,
        index_path: Path,
        new_documents: list[Document],
    ) -> FAISS:
        """
        Add new documents to an existing FAISS index.
        
        Args:
            index_path: Path to existing FAISS index
            new_documents: List of new documents to add
            
        Returns:
            Updated FAISS vector store
        """
        index_path = Path(index_path)
        
        # Load existing index
        vector_store = FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        
        # Add new documents
        vector_store.add_documents(new_documents)
        
        # Save updated index
        vector_store.save_local(str(index_path))
        
        logger.info(f"Added {len(new_documents)} documents to index")
        return vector_store

