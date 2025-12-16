"""
RAG (Retrieval-Augmented Generation) module.

Provides components for:
- Building and persisting FAISS vector indices
- Retrieving relevant FAQ documents based on user queries
"""

from src.rag.index_builder import FAQIndexBuilder
from src.rag.retriever import FAQRetriever

__all__ = ["FAQIndexBuilder", "FAQRetriever"]

