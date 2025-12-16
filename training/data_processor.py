"""
Data Processor for Fine-tuning Datasets.

Handles loading, processing, and unifying training datasets:
- Banking Conversation Corpus
- Bitext Gen AI Chatbot Customer Support Dataset

Outputs a unified JSONL format suitable for instruction fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, load_dataset


logger = logging.getLogger(__name__)


# Standard system message for all fine-tuning examples
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful, professional, and friendly customer service agent for a bank. "
    "Assist customers with their questions about banking products, services, and policies. "
    "Be polite, patient, and provide clear, accurate information. "
    "If you don't know something, say so and offer to help find the answer."
)


class DataProcessor:
    """
    Processor for converting various customer service datasets to a unified format.
    
    The unified format is:
    {
        "system": str,  # System message
        "messages": [   # Conversation turns
            {"role": "user", "content": str},
            {"role": "assistant", "content": str},
            ...
        ],
        "source": str   # Dataset source identifier
    }
    """
    
    def __init__(
        self,
        raw_data_path: Path,
        output_path: Path,
        max_turns: int = 10,
        max_length: int = 2048,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
    ):
        """
        Initialize the data processor.
        
        Args:
            raw_data_path: Path to raw dataset files
            output_path: Path to write processed data
            max_turns: Maximum conversation turns to keep
            max_length: Maximum total character length per example
            system_message: System message to use for all examples
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.max_turns = max_turns
        self.max_length = max_length
        self.system_message = system_message
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def process_bitext_dataset(self) -> Iterator[dict[str, Any]]:
        """
        Process the Bitext Gen AI Chatbot Customer Support Dataset.
        
        This dataset is available on Hugging Face:
        https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
        
        Yields:
            Processed conversation examples in unified format
        """
        logger.info("Processing Bitext Customer Support Dataset...")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset(
                "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                split="train"
            )
            
            for example in dataset:
                # Extract fields from Bitext dataset
                instruction = example.get("instruction", "")
                response = example.get("response", "")
                category = example.get("category", "")
                intent = example.get("intent", "")
                
                if not instruction or not response:
                    continue
                
                # Build unified format
                processed = {
                    "system": self.system_message,
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response},
                    ],
                    "source": "bitext_customer_support",
                    "metadata": {
                        "category": category,
                        "intent": intent,
                    }
                }
                
                yield processed
                
        except Exception as e:
            logger.error(f"Failed to process Bitext dataset: {e}")
            logger.info("Attempting to load from local file...")
            
            # Try loading from local file
            local_path = self.raw_data_path / "bitext_customer_support"
            if local_path.exists():
                yield from self._load_local_bitext(local_path)
    
    def _load_local_bitext(self, data_path: Path) -> Iterator[dict[str, Any]]:
        """Load Bitext dataset from local files."""
        for file_path in data_path.glob("*.json*"):
            logger.info(f"Loading local Bitext data from {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            example = json.loads(line)
                            yield self._convert_bitext_example(example)
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for example in data:
                            yield self._convert_bitext_example(example)
    
    def _convert_bitext_example(self, example: dict) -> dict[str, Any]:
        """Convert a single Bitext example to unified format."""
        return {
            "system": self.system_message,
            "messages": [
                {"role": "user", "content": example.get("instruction", "")},
                {"role": "assistant", "content": example.get("response", "")},
            ],
            "source": "bitext_customer_support",
            "metadata": {
                "category": example.get("category", ""),
                "intent": example.get("intent", ""),
            }
        }
    
    def process_banking_corpus(self) -> Iterator[dict[str, Any]]:
        """
        Process the Banking Conversation Corpus.
        
        Note: This dataset may need to be downloaded separately from Kaggle
        or another source. The structure varies, so this method handles
        common formats.
        
        Yields:
            Processed conversation examples in unified format
        """
        logger.info("Processing Banking Conversation Corpus...")
        
        corpus_path = self.raw_data_path / "banking_conversation_corpus"
        
        if not corpus_path.exists():
            logger.warning(f"Banking corpus not found at {corpus_path}")
            logger.info("Please download the dataset and place it in the correct location.")
            return
        
        # Process all JSON/JSONL files in the directory
        for file_path in corpus_path.glob("*.json*"):
            logger.info(f"Processing {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            example = json.loads(line)
                            processed = self._convert_banking_example(example)
                            if processed:
                                yield processed
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for example in data:
                            processed = self._convert_banking_example(example)
                            if processed:
                                yield processed
    
    def _convert_banking_example(self, example: dict) -> dict[str, Any] | None:
        """
        Convert a banking corpus example to unified format.
        
        Handles multiple possible formats:
        - {"question": str, "answer": str}
        - {"messages": [{"role": str, "content": str}, ...]}
        - {"conversation": [...]}
        """
        messages = []
        
        # Format 1: Simple Q&A
        if "question" in example and "answer" in example:
            messages = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        
        # Format 2: Messages array
        elif "messages" in example:
            for msg in example["messages"][:self.max_turns * 2]:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                if role in ("user", "human", "customer"):
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "bot", "agent"):
                    messages.append({"role": "assistant", "content": content})
        
        # Format 3: Conversation array
        elif "conversation" in example:
            for i, turn in enumerate(example["conversation"][:self.max_turns * 2]):
                if isinstance(turn, str):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": turn})
                elif isinstance(turn, dict):
                    content = turn.get("text", turn.get("content", ""))
                    role = turn.get("role", "user" if i % 2 == 0 else "assistant")
                    if role in ("user", "human", "customer"):
                        role = "user"
                    else:
                        role = "assistant"
                    messages.append({"role": role, "content": content})
        
        if not messages:
            return None
        
        # Check total length
        total_length = sum(len(m["content"]) for m in messages)
        if total_length > self.max_length:
            # Truncate from the beginning, keeping recent turns
            while total_length > self.max_length and len(messages) > 2:
                removed = messages.pop(0)
                total_length -= len(removed["content"])
        
        return {
            "system": self.system_message,
            "messages": messages,
            "source": "banking_conversation_corpus",
            "metadata": example.get("metadata", {}),
        }
    
    def process_all(self) -> Path:
        """
        Process all datasets and write to unified output file.
        
        Returns:
            Path to the output JSONL file
        """
        output_file = self.output_path / "unified_training_data.jsonl"
        
        logger.info(f"Processing all datasets to {output_file}")
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            # Process Bitext dataset
            for example in self.process_bitext_dataset():
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
            
            # Process Banking corpus
            for example in self.process_banking_corpus():
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"Wrote {count} examples to {output_file}")
        return output_file
    
    def create_train_val_split(
        self,
        input_file: Path,
        val_ratio: float = 0.1,
    ) -> tuple[Path, Path]:
        """
        Split the unified dataset into train and validation sets.
        
        Args:
            input_file: Path to the unified JSONL file
            val_ratio: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_path, val_path)
        """
        import random
        
        # Load all examples
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        
        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - val_ratio))
        
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Write train set
        train_file = self.output_path / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        # Write validation set
        val_file = self.output_path / "val.jsonl"
        with open(val_file, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        logger.info(f"Train set: {len(train_examples)} examples -> {train_file}")
        logger.info(f"Val set: {len(val_examples)} examples -> {val_file}")
        
        return train_file, val_file


def main() -> None:
    """Main entry point for data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process training datasets")
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/finetune"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Process datasets
    processor = DataProcessor(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
    )
    
    unified_file = processor.process_all()
    processor.create_train_val_split(unified_file, args.val_ratio)


if __name__ == "__main__":
    main()

