"""
Configuration management for CTBC QA Bot.

Handles loading configuration from environment variables and .env files,
with sensible defaults for all settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/app/config.py to project root
    return Path(__file__).parent.parent.parent


@dataclass
class ModelConfig:
    """Configuration for LLM and embedding models."""

    # Base LLM model
    model_id: str = field(default_factory=lambda: os.getenv("HF_MODEL_ID", "Qwen/Qwen3-4B"))

    # Embedding model for RAG
    embedding_model_id: str = field(
        default_factory=lambda: os.getenv("HF_EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    )

    # Optional LoRA adapter path
    lora_adapter_path: str | None = field(
        default_factory=lambda: os.getenv("LORA_ADAPTER_PATH") or None
    )

    # Hugging Face token for gated models
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN") or None)

    # Device configuration
    device: Literal["auto", "cuda", "cpu", "mps"] = field(
        default_factory=lambda: os.getenv("DEVICE", "auto")  # type: ignore
    )


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval."""

    # Path to FAISS index
    faiss_index_path: Path = field(
        default_factory=lambda: _get_project_root()
        / os.getenv("FAISS_INDEX_PATH", "artifacts/faiss")
    )

    # Number of documents to retrieve
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "3")))

    # Path to processed FAQ data
    faq_data_path: Path = field(
        default_factory=lambda: _get_project_root()
        / os.getenv("CTBC_FAQ_PATH", "data/processed/ctbc_faq")
    )


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # Maximum tokens to generate
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "512")))

    # Temperature for generation
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))


@dataclass
class DataConfig:
    """Configuration for data paths."""

    # Raw data directories
    raw_data_path: Path = field(default_factory=lambda: _get_project_root() / "data/raw")

    # Processed data directories
    processed_data_path: Path = field(
        default_factory=lambda: _get_project_root() / "data/processed"
    )

    # Fine-tuning data path
    finetune_data_path: Path = field(
        default_factory=lambda: _get_project_root()
        / os.getenv("FINETUNE_DATA_PATH", "data/processed/finetune")
    )

    # Artifacts directory
    artifacts_path: Path = field(default_factory=lambda: _get_project_root() / "artifacts")


@dataclass
class AppConfig:
    """Main application configuration combining all config sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment."""
        return cls()

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.finetune_data_path,
            self.data.artifacts_path,
            self.rag.faiss_index_path,
            self.rag.faq_data_path,
            self.data.raw_data_path / "banking_conversation_corpus",
            self.data.raw_data_path / "bitext_customer_support",
            self.data.raw_data_path / "ctbc_faq_raw",
            self.data.artifacts_path / "models",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance (lazy loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def get_project_root() -> Path:
    """Get the project root directory."""
    return _get_project_root()
