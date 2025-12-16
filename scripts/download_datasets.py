"""
Dataset Download Script.

Downloads and prepares training datasets:
1. Bitext Gen AI Chatbot Customer Support Dataset (from Hugging Face)
2. Banking Conversation Corpus (instructions for manual download)

Usage:
    python -m scripts.download_datasets

    Or with options:
    python -m scripts.download_datasets --output-dir data/raw --bitext-only
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


def download_bitext_dataset(output_dir: Path) -> Path:
    """
    Download the Bitext Customer Support Dataset from Hugging Face.

    Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
    License: Apache 2.0

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Path to the saved dataset file
    """
    logger.info("Downloading Bitext Customer Support Dataset...")

    output_path = output_dir / "bitext_customer_support"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
        )

        # Save as JSONL
        output_file = output_path / "bitext_customer_support.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(dataset)} examples to {output_file}")

        # Also save dataset info
        info_file = output_path / "dataset_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": "Bitext Customer Support Dataset",
                    "source": "huggingface",
                    "dataset_id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                    "license": "Apache 2.0",
                    "num_examples": len(dataset),
                    "fields": ["instruction", "category", "intent", "response"],
                },
                f,
                indent=2,
            )

        return output_file

    except Exception as e:
        logger.error(f"Failed to download Bitext dataset: {e}")
        raise


def create_banking_corpus_instructions(output_dir: Path) -> Path:
    """
    Create instructions for downloading the Banking Conversation Corpus.

    The Banking Conversation Corpus may need to be downloaded manually
    from Kaggle or other sources.

    Args:
        output_dir: Directory to save instructions

    Returns:
        Path to the instructions file
    """
    logger.info("Creating instructions for Banking Conversation Corpus...")

    output_path = output_dir / "banking_conversation_corpus"
    output_path.mkdir(parents=True, exist_ok=True)

    instructions = """# Banking Conversation Corpus

## Overview
This directory should contain banking conversation data for fine-tuning.

## Recommended Datasets

### 1. Banking77 (Hugging Face)
A dataset of banking customer service queries with 77 intents.

```python
from datasets import load_dataset
dataset = load_dataset("banking77")
```

### 2. Customer Support on Twitter (Kaggle)
Real customer support conversations from various companies.

Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

### 3. Taskmaster Banking Dataset
Multi-turn conversations about banking tasks.

Download from: https://github.com/google-research-datasets/Taskmaster

## Expected Format

Place your data files in this directory in one of these formats:

### Option A: JSONL with Q&A pairs
```json
{"question": "How do I reset my password?", "answer": "You can reset your password by..."}
```

### Option B: JSONL with messages array
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Option C: JSONL with conversation array
```json
{"conversation": ["user message", "assistant response", "user message", ...]}
```

## Processing

After adding data files, run the data processor:

```bash
python -m training.data_processor
```

This will convert all datasets to the unified training format.
"""

    readme_file = output_path / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(instructions)

    # Create a sample file to show expected format
    sample_file = output_path / "sample_format.jsonl"
    samples = [
        {
            "question": "How do I check my account balance?",
            "answer": "You can check your account balance through our mobile app, online banking, or by calling our customer service hotline.",
        },
        {
            "question": "What are the fees for international transfers?",
            "answer": "International transfer fees vary depending on the destination country and transfer amount. Please check our fee schedule or contact customer service for specific details.",
        },
    ]

    with open(sample_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Created instructions at {readme_file}")
    logger.info(f"Created sample format at {sample_file}")

    return readme_file


def download_banking77(output_dir: Path) -> Path | None:
    """
    Download the Banking77 dataset from Hugging Face.

    This is a simpler alternative to the full Banking Conversation Corpus.

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Path to saved file, or None if download fails
    """
    logger.info("Downloading Banking77 dataset...")

    output_path = output_dir / "banking_conversation_corpus"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("banking77", split="train")

        # Convert to Q&A format (Banking77 is classification, we'll create synthetic answers)
        output_file = output_path / "banking77.jsonl"

        # Get label names
        label_names = dataset.features["label"].names

        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                query = example["text"]
                intent = label_names[example["label"]]

                # Create a simple response based on intent
                response = f"I understand you're asking about {intent.replace('_', ' ')}. Let me help you with that. "

                # Add some context based on common intents
                if "balance" in intent:
                    response += (
                        "You can check your balance through our mobile app, online banking, or ATM."
                    )
                elif "transfer" in intent:
                    response += (
                        "I can help you with your transfer. Please provide the recipient details."
                    )
                elif "card" in intent:
                    response += "For card-related inquiries, please have your card number ready."
                elif "payment" in intent:
                    response += "I'll assist you with your payment query."
                else:
                    response += "Please provide more details so I can assist you better."

                f.write(
                    json.dumps(
                        {
                            "question": query,
                            "answer": response,
                            "intent": intent,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(f"Saved {len(dataset)} examples to {output_file}")
        return output_file

    except Exception as e:
        logger.warning(f"Failed to download Banking77: {e}")
        return None


def main() -> None:
    """Main entry point for dataset download."""
    parser = argparse.ArgumentParser(description="Download training datasets")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--bitext-only",
        action="store_true",
        help="Only download Bitext dataset",
    )
    parser.add_argument(
        "--include-banking77",
        action="store_true",
        help="Also download Banking77 dataset",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download datasets
    try:
        # Always download Bitext
        download_bitext_dataset(args.output_dir)

        if not args.bitext_only:
            # Create instructions for banking corpus
            create_banking_corpus_instructions(args.output_dir)

            # Optionally download Banking77
            if args.include_banking77:
                download_banking77(args.output_dir)

        logger.info("Dataset download complete!")
        logger.info(f"Data saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


if __name__ == "__main__":
    main()
