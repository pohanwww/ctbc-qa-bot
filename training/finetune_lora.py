"""
LoRA/QLoRA Fine-tuning Script for CTBC QA Bot.

This script handles fine-tuning the base LLM using:
- PEFT (Parameter-Efficient Fine-Tuning) with LoRA
- Optional 4-bit quantization (QLoRA) for memory efficiency
- Hugging Face Transformers and TRL for training

Usage:
    python -m training.finetune_lora --config training/configs/lora_config.yaml

    Or with command-line overrides:
    python -m training.finetune_lora --model-id Qwen/Qwen2.5-3B-Instruct --epochs 3
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # Model settings
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_token: str | None = None
    trust_remote_code: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training settings
    output_dir: str = "artifacts/models/lora_adapter"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Data settings
    train_data_path: str = "data/processed/finetune/train.jsonl"
    val_data_path: str = "data/processed/finetune/val.jsonl"

    # Logging settings
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Misc
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # W&B logging (optional)
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


def load_training_data(
    train_path: str,
    val_path: str | None = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Load training and validation datasets from JSONL files.

    Args:
        train_path: Path to training data JSONL
        val_path: Optional path to validation data JSONL

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading training data from {train_path}")

    train_data = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))

    train_dataset = Dataset.from_list(train_data)
    logger.info(f"Loaded {len(train_dataset)} training examples")

    val_dataset = None
    if val_path and Path(val_path).exists():
        logger.info(f"Loading validation data from {val_path}")
        val_data = []
        with open(val_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    val_data.append(json.loads(line))
        val_dataset = Dataset.from_list(val_data)
        logger.info(f"Loaded {len(val_dataset)} validation examples")

    return train_dataset, val_dataset


def format_conversation(example: dict, tokenizer: Any) -> dict:
    """
    Format a conversation example for training.

    Converts our unified format to the model's chat template.

    Args:
        example: Training example with "system" and "messages" fields
        tokenizer: The model's tokenizer

    Returns:
        Dictionary with "text" field containing formatted conversation
    """
    system = example.get("system", "")
    messages = example.get("messages", [])

    # Build conversation in Qwen format
    conversation_parts = []

    # Add system message
    if system:
        conversation_parts.append(f"<|im_start|>system\n{system}<|im_end|>")

    # Add conversation turns
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            conversation_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            conversation_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    text = "\n".join(conversation_parts)

    return {"text": text}


def setup_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    """
    Set up the model and tokenizer for training.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model_id}")

    # Configure quantization
    quantization_config = None
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=config.hf_token,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
        torch_dtype=torch.float16,  # Force float16 for T4 compatibility
    )

    # Prepare for k-bit training if quantized
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model: Any, config: TrainingConfig) -> Any:
    """
    Apply LoRA configuration to the model.

    Args:
        model: The base model
        config: Training configuration

    Returns:
        Model with LoRA applied
    """
    logger.info("Applying LoRA configuration")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"Total params: {total_params:,} || "
        f"Trainable%: {100 * trainable_params / total_params:.2f}%"
    )

    return model


def train(config: TrainingConfig) -> None:
    """
    Run the fine-tuning process.

    Args:
        config: Training configuration
    """
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Apply LoRA
    model = setup_lora(model, config)

    # Load data
    train_dataset, val_dataset = load_training_data(
        config.train_data_path,
        config.val_data_path,
    )

    # Format datasets
    def format_fn(example):
        return format_conversation(example, tokenizer)

    train_dataset = train_dataset.map(format_fn)
    if val_dataset:
        val_dataset = val_dataset.map(format_fn)

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        report_to="wandb" if config.wandb_project else "none",
        run_name=config.wandb_run_name,
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Initialize W&B if configured
    if config.wandb_project:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    final_output_dir = Path(config.output_dir) / "final"
    logger.info(f"Saving final model to {final_output_dir}")
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    logger.info("Training complete!")


def merge_lora_weights(
    base_model_id: str,
    lora_adapter_path: str,
    output_path: str,
    hf_token: str | None = None,
) -> None:
    """
    Merge LoRA weights into the base model.

    This creates a full model that can be loaded without PEFT.

    Args:
        base_model_id: Hugging Face model ID for base model
        lora_adapter_path: Path to LoRA adapter
        output_path: Path to save merged model
        hf_token: Optional HF token
    """
    logger.info(f"Merging LoRA weights from {lora_adapter_path}")

    from peft import PeftModel

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Merge weights
    logger.info("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")


def main() -> None:
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for CTBC QA Bot")

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights into base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to LoRA adapter (for merging)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle merge mode
    if args.merge:
        if not args.lora_path or not args.output_dir:
            parser.error("--merge requires --lora-path and --output-dir")

        config = TrainingConfig()
        if args.model_id:
            config.model_id = args.model_id

        merge_lora_weights(
            base_model_id=config.model_id,
            lora_adapter_path=args.lora_path,
            output_path=args.output_dir,
        )
        return

    # Load configuration
    if args.config and args.config.exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Apply command-line overrides
    if args.model_id:
        config.model_id = args.model_id
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.use_4bit:
        config.use_4bit = True

    # Run training
    train(config)


if __name__ == "__main__":
    main()
