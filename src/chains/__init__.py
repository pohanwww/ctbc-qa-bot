"""
LangChain chains and runnables for CTBC QA Bot.

Contains the main chatbot chain that orchestrates RAG retrieval
and LLM generation for customer service interactions.
"""

from src.chains.chatbot import CTBCChatbot

__all__ = ["CTBCChatbot"]

