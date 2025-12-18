"""
LLM Loader Module.

Handles loading of language models from Hugging Face, with support for:
- Base models (e.g., Qwen2.5-3B-Instruct)
- LoRA adapters (loaded on top of base models)
- Different device configurations (CUDA, CPU, MPS)

The module is designed to be easily swappable for different backends
(e.g., vLLM, Ollama) in the future.
"""

import logging
from typing import Any, Literal

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Wrapper class for language models.

    Provides a unified interface for different LLM backends,
    making it easy to swap implementations in the future.

    Attributes:
        model: The underlying model (HuggingFacePipeline or custom)
        model_id: The model identifier
        device: The device the model is running on
    """

    def __init__(
        self,
        model: HuggingFacePipeline,
        model_id: str,
        device: str,
        tokenizer: Any = None,
    ):
        """
        Initialize the LLM wrapper.

        Args:
            model: The LangChain-compatible LLM pipeline
            model_id: The Hugging Face model ID
            device: The device being used
            tokenizer: The tokenizer (optional, for direct access)
        """
        self.model = model
        self.model_id = model_id
        self.device = device
        self.tokenizer = tokenizer

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the model with a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        return self.model.invoke(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Alias for invoke()."""
        return self.invoke(prompt, **kwargs)


def _get_device(device: Literal["auto", "cuda", "cpu", "mps"]) -> str:
    """
    Determine the appropriate device for model loading.

    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps")

    Returns:
        The resolved device string
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def _get_torch_dtype(device: str) -> torch.dtype:
    """
    Get the appropriate torch dtype for the device.

    Args:
        device: The target device

    Returns:
        The appropriate torch dtype
    """
    if device == "cuda" or device == "mps":
        return torch.float16
    else:
        return torch.float32


def load_llm(
    model_id: str = "Qwen/Qwen3-4B",
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
    lora_adapter_path: str | None = None,
    hf_token: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    trust_remote_code: bool = True,
) -> LLMWrapper:
    """
    Load a language model from Hugging Face.

    This function handles:
    - Loading the base model
    - Optionally loading LoRA adapters
    - Configuring quantization (4-bit or 8-bit)
    - Setting up the generation pipeline

    Args:
        model_id: Hugging Face model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
        device: Device to load the model on
        lora_adapter_path: Optional path to LoRA adapter weights
        hf_token: Hugging Face token for gated models
        load_in_4bit: Whether to use 4-bit quantization
        load_in_8bit: Whether to use 8-bit quantization
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        trust_remote_code: Whether to trust remote code in model

    Returns:
        LLMWrapper instance containing the loaded model

    Raises:
        ValueError: If both 4-bit and 8-bit quantization are requested
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot use both 4-bit and 8-bit quantization")

    resolved_device = _get_device(device)
    torch_dtype = _get_torch_dtype(resolved_device)

    logger.info(f"Loading model {model_id} on device {resolved_device}")

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif resolved_device != "cpu":
        model_kwargs["device_map"] = resolved_device

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    # Load LoRA adapter if specified
    if lora_adapter_path:
        logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
        try:
            from peft import PeftModel

            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, lora_adapter_path)

            # Note: We keep using the base model's tokenizer because:
            # 1. LoRA adapters don't change the vocabulary
            # 2. The model's embedding layer size matches the base tokenizer
            # 3. Using adapter's tokenizer could cause mismatches if vocab sizes differ
            # The tokenizer saved with the adapter is just a copy for convenience
            logger.info("Using base model tokenizer (LoRA doesn't change vocabulary)")

            logger.info("LoRA adapter loaded successfully")
        except ImportError as e:
            raise ImportError(
                "PEFT library is required to load LoRA adapters. Install with: uv pip install peft"
            ) from e

    # Create text generation pipeline
    # Ensure temperature is valid and handle edge cases to prevent probability tensor errors
    safe_temperature = max(0.1, min(temperature, 1.5)) if temperature > 0 else 0.0
    do_sample = safe_temperature > 0

    # Build pipeline kwargs
    pipeline_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        pipeline_kwargs.update(
            {
                "temperature": safe_temperature,
                "do_sample": True,
                "top_p": 0.95,  # Nucleus sampling to avoid extreme probabilities
                "top_k": 50,  # Limit vocabulary to top-k tokens
                "repetition_penalty": 1.1,
            }
        )
    else:
        pipeline_kwargs["do_sample"] = False

    pipe = pipeline("text-generation", **pipeline_kwargs)

    # Wrap in LangChain pipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    logger.info(f"Model {model_id} loaded successfully on {resolved_device}")

    return LLMWrapper(
        model=llm,
        model_id=model_id,
        device=resolved_device,
        tokenizer=tokenizer,
    )


def load_llm_for_api(
    model_id: str,
    api_url: str,
    api_key: str | None = None,
) -> LLMWrapper:
    """
    Load an LLM from a remote API endpoint.

    This is a placeholder for future implementation to support
    remote inference backends like vLLM, Triton, or cloud APIs.

    Args:
        model_id: Model identifier
        api_url: The API endpoint URL
        api_key: Optional API key

    Returns:
        LLMWrapper instance

    Raises:
        NotImplementedError: This is a placeholder for future implementation
    """
    # TODO: Implement remote API support
    # Options to consider:
    # - vLLM OpenAI-compatible API
    # - Triton Inference Server
    # - Cloud provider endpoints (AWS SageMaker, GCP Vertex AI, etc.)
    raise NotImplementedError(
        "Remote API support is not yet implemented. Use load_llm() for local model loading."
    )
