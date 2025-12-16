"""
FAISS Index Builder Script.

Builds the FAISS vector index from processed CTBC FAQ data.

Usage:
    python -m scripts.build_faiss_index
    
    Or with options:
    python -m scripts.build_faiss_index --data-path data/processed/ctbc_faq --output-path artifacts/faiss
"""

import argparse
import logging
from pathlib import Path

from src.app.config import get_config
from src.rag.index_builder import FAQIndexBuilder


logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for building FAISS index."""
    parser = argparse.ArgumentParser(description="Build FAISS index from FAQ data")
    
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to FAQ data directory (default: from config)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path to save FAISS index (default: from config)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-m3",
        help="Hugging Face embedding model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device for embedding computation",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load config for defaults
    config = get_config()
    
    # Determine paths
    data_path = args.data_path or config.rag.faq_data_path
    output_path = args.output_path or config.rag.faiss_index_path
    
    logger.info(f"Building FAISS index")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Embedding model: {args.embedding_model}")
    logger.info(f"  Device: {args.device}")
    
    # Check if data exists
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please run the FAQ scraper first:")
        logger.info("  python -m scripts.scrape_ctbc_faq")
        return
    
    # Build index
    builder = FAQIndexBuilder(
        embedding_model_id=args.embedding_model,
        device=args.device,
    )
    
    try:
        vector_store = builder.build_index(data_path, output_path)
        logger.info("FAISS index built successfully!")
        
        # Print some stats
        if hasattr(vector_store, 'index'):
            logger.info(f"Index contains {vector_store.index.ntotal} vectors")
            
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise


if __name__ == "__main__":
    main()

