# CTBC QA Bot

CLI-based English customer service chatbot for CTBC Bank (中國信託商業銀行), powered by RAG and a small open-source LLM.

## Features

- **RAG-powered responses**: Knowledge about CTBC Bank products and services comes from scraped FAQ data
- **Small LLM**: Uses Qwen3-4B, optimized for single GPU deployment (16GB VRAM)
- **Fine-tuning ready**: LoRA/QLoRA training pipeline for customer service style adaptation
- **Multi-turn conversation**: Maintains conversation history within sessions
- **CLI interface**: Simple terminal-based chat interaction

## Tech Stack

- **LLM**: Qwen3-4B (via Hugging Face Transformers)
- **Embeddings**: BAAI/bge-m3 (multilingual, supports Chinese-English cross-lingual retrieval)
- **Vector Store**: FAISS (faiss-cpu)
- **Orchestration**: LangChain
- **Package Manager**: uv

## Project Structure

```
ctbc-qa-bot/
├── src/
│   ├── app/
│   │   ├── cli_chat.py        # Main CLI entrypoint
│   │   └── config.py          # Configuration management
│   ├── llm/
│   │   └── loader.py          # LLM loading (base + LoRA)
│   ├── rag/
│   │   ├── index_builder.py   # FAISS index construction
│   │   └── retriever.py       # RAG retrieval
│   └── chains/
│       └── chatbot.py         # LangChain chat chain
├── training/
│   ├── data_processor.py      # Dataset normalization
│   ├── finetune_lora.py       # LoRA training script
│   └── configs/
│       └── lora_config.yaml   # Training configuration
├── scripts/
│   ├── download_datasets.py   # Download training data
│   ├── scrape_ctbc_faq.py     # Scrape CTBC FAQ pages
│   └── build_faiss_index.py   # Build vector index
├── data/
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Processed data for RAG/training
├── artifacts/
│   ├── faiss/                 # FAISS index files
│   └── models/                # Fine-tuned model weights
├── pyproject.toml
└── .env.example
```

## Quick Start

### 1. Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ctbc-qa-bot.git
cd ctbc-qa-bot

# Create virtual environment and install dependencies
uv sync

# Copy environment configuration
cp .env.example .env
# Edit .env as needed (optional)
```

### 3. Prepare the Knowledge Base

```bash
# Option A: Scrape CTBC FAQ pages (requires internet)
uv run python -m scripts.scrape_ctbc_faq

# Option B: Use sample data only (for quick testing)
uv run python -m scripts.scrape_ctbc_faq --sample-only

# Build FAISS index
uv run python -m scripts.build_faiss_index
```

### 4. Run the Chatbot

```bash
uv run python -m src.app.cli_chat
```

Example interaction:
```
╭──────────────────────────────────────────────────────────────────────────────╮
│                            CTBC QA Bot                                        │
│ Welcome to CTBC Bank Customer Service                                         │
│ ...                                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

You: How do I check my account balance?

Bot: You can check your CTBC Bank account balance through several methods:
1. **Mobile App**: Download the CTBC Mobile Banking app from App Store or Google Play
2. **Online Banking**: Log in to your internet banking account at www.ctbcbank.com
3. **ATM**: Visit any CTBC ATM with your debit card
4. **Customer Service**: Call 0800-024-365 (toll-free in Taiwan)

Is there anything else I can help you with?

You: exit
```

## Fine-tuning (Optional)

The project includes a LoRA/QLoRA training pipeline for adapting the model to customer service style.

### 1. Download Training Datasets

```bash
# Download Bitext Customer Support dataset (from Hugging Face)
uv run python -m scripts.download_datasets

# For Banking77 dataset
uv run python -m scripts.download_datasets --include-banking77
```

### 2. Process Datasets

```bash
uv run python -m training.data_processor
```

### 3. Run Fine-tuning

```bash
# Using default config (requires GPU with ~16GB VRAM)
uv run python -m training.finetune_lora --config training/configs/lora_config.yaml

# Or with command-line overrides
uv run python -m training.finetune_lora \
    --model-id Qwen/Qwen3-4B \
    --epochs 3 \
    --batch-size 4 \
    --use-4bit
```

### 4. Use Fine-tuned Model

Set the `LORA_ADAPTER_PATH` in your `.env`:

```bash
LORA_ADAPTER_PATH=artifacts/models/lora_adapter/final
```

Or merge LoRA weights into the base model:

```bash
uv run python -m training.finetune_lora \
    --merge \
    --lora-path artifacts/models/lora_adapter/final \
    --output-dir artifacts/models/merged
```

## Configuration

All configuration options can be set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_ID` | `Qwen/Qwen3-4B` | Base LLM model |
| `HF_EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | Embedding model for RAG |
| `LORA_ADAPTER_PATH` | (empty) | Path to LoRA adapter |
| `DEVICE` | `auto` | Device for inference (auto/cuda/cpu/mps) |
| `RAG_TOP_K` | `3` | Number of FAQ entries to retrieve |
| `MAX_NEW_TOKENS` | `512` | Maximum response length |
| `TEMPERATURE` | `0.7` | Generation temperature |

## Hardware Requirements

### Inference
- **Minimum**: 16GB RAM (CPU inference, slow)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, T4)
- **Optimal**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 4090, A10)

### Fine-tuning
- **Minimum**: NVIDIA GPU with 16GB VRAM (e.g., T4, RTX 4080)
- **Recommended**: NVIDIA GPU with 24GB+ VRAM (e.g., A10, RTX 4090)

With QLoRA (4-bit quantization), fine-tuning is possible on 16GB VRAM GPUs.

## Cloud Deployment

The codebase is designed to be easily deployable on cloud GPU VMs:

1. **AWS EC2**: Use `g4dn.xlarge` (1× T4 16GB) or `g5.xlarge` (1× A10G 24GB)
2. **GCP**: Use `n1-standard-4` with 1× T4 GPU
3. **Azure**: Use `NC4as_T4_v3` (1× T4 16GB)

Future enhancements may include:
- FastAPI HTTP service layer
- vLLM/TGI backend for optimized inference
- Docker containerization

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

This chatbot is for demonstration purposes. For official CTBC Bank services and information, please:
- Visit: https://www.ctbcbank.com
- Call: 0800-024-365 (Taiwan toll-free)
- International: +886-2-2745-8080

The chatbot does not provide financial, legal, or tax advice. Always consult with qualified professionals for such matters.

