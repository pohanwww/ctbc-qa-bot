"""
LLM loading and management module.

Provides abstracted interfaces for loading language models,
supporting both base models and LoRA-adapted models.
"""

from src.llm.loader import load_llm, LLMWrapper

__all__ = ["load_llm", "LLMWrapper"]

