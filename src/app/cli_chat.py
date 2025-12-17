"""
CLI Chat Interface for CTBC QA Bot.

Provides a REPL interface with three response modes:
- Base: Pure Qwen3 response
- RAG: Qwen3 + RAG retrieval
- Fine-tuned RAG: Fine-tuned Qwen3 + RAG
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.app.config import AppConfig, get_config, get_project_root
from src.chains.chatbot import CTBCChatbot
from src.llm.loader import load_llm
from src.rag.retriever import FAQRetriever

console = Console()


def print_welcome_message(compare_mode: bool) -> None:
    """Print the welcome message and instructions."""
    mode_text = (
        "**Compare Mode**: Showing all 3 response types"
        if compare_mode
        else "**Single Mode**: RAG responses only"
    )

    welcome_text = f"""
# Welcome to CTBC Bank Customer Service

I'm your AI assistant, ready to help you with questions about CTBC Bank.

{mode_text}

**Commands:**
- Type your question and press Enter
- `exit` / `quit` - End conversation
- `clear` - Clear conversation history
- `compare` - Toggle compare mode (show all 3 responses)
- `help` - Show this message
    """
    console.print(Panel(Markdown(welcome_text), title="🏦 CTBC QA Bot", border_style="blue"))


def print_help() -> None:
    """Print help information."""
    help_text = """
**Available Commands:**
- `exit` / `quit` - End the conversation
- `clear` - Clear conversation history
- `compare` - Toggle compare mode (all 3 responses)
- `help` - Show this help message

**Response Modes (in compare mode):**
- 🔵 **Base**: Pure Qwen3 (no FAQ knowledge)
- 🟢 **RAG**: Qwen3 + FAQ retrieval
- 🟣 **Fine-tuned + RAG**: Fine-tuned Qwen3 + FAQ retrieval
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))


def print_comparison_responses(responses: dict[str, str]) -> None:
    """Print responses from all three modes in a comparison format."""

    # Ensure all required keys exist
    if "base" not in responses:
        responses["base"] = "[Error: Base response not available]"
    if "rag" not in responses:
        responses["rag"] = "[Error: RAG response not available]"
    if "finetuned_rag" not in responses:
        responses["finetuned_rag"] = "[Error: Fine-tuned RAG response not available]"

    # Base response
    console.print(
        Panel(
            responses["base"],
            title="🔵 Base Qwen3 (No RAG)",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # RAG response
    console.print(
        Panel(
            responses["rag"],
            title="🟢 Qwen3 + RAG",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Fine-tuned RAG response
    finetuned_response = responses["finetuned_rag"]
    if finetuned_response == "[Fine-tuned model not loaded]":
        console.print(
            Panel(
                "[dim]Fine-tuned model not available. Run fine-tuning first.[/dim]",
                title="🟣 Fine-tuned Qwen3 + RAG",
                border_style="magenta",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                finetuned_response,
                title="🟣 Fine-tuned Qwen3 + RAG",
                border_style="magenta",
                padding=(1, 2),
            )
        )


def initialize_chatbot(
    config: AppConfig,
    load_finetuned: bool = False,
    lora_path: str | None = None,
) -> CTBCChatbot:
    """
    Initialize the chatbot with all required components.

    Args:
        config: Application configuration
        load_finetuned: Whether to load the fine-tuned model
        lora_path: Custom path to LoRA adapter

    Returns:
        Initialized CTBCChatbot instance
    """
    console.print("[yellow]Initializing chatbot components...[/yellow]")

    # Load base LLM
    console.print("  [dim]Loading base language model...[/dim]")
    base_llm = load_llm(
        model_id=config.model.model_id,
        device=config.model.device,
        hf_token=config.model.hf_token,
        max_new_tokens=config.inference.max_new_tokens,
    )

    # Load fine-tuned LLM if requested
    finetuned_llm = None
    if load_finetuned:
        adapter_path = lora_path or config.model.lora_adapter_path
        # If no path specified, try default location
        if not adapter_path:
            project_root = get_project_root()
            default_path = project_root / "artifacts" / "models" / "lora_adapter" / "final"
            if default_path.exists():
                adapter_path = str(default_path)
        if adapter_path and Path(adapter_path).exists():
            console.print(f"  [dim]Loading fine-tuned model from {adapter_path}...[/dim]")
            try:
                finetuned_llm = load_llm(
                    model_id=config.model.model_id,
                    device=config.model.device,
                    lora_adapter_path=adapter_path,
                    hf_token=config.model.hf_token,
                    max_new_tokens=config.inference.max_new_tokens,
                )
                console.print("  [green]✓ Fine-tuned model loaded[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Failed to load fine-tuned model: {e}[/red]")
        else:
            console.print("  [yellow]⚠ Fine-tuned model path not found, skipping[/yellow]")

    # Load RAG retriever
    console.print("  [dim]Loading RAG retriever...[/dim]")
    retriever = FAQRetriever(
        index_path=config.rag.faiss_index_path,
        embedding_model_id=config.model.embedding_model_id,
        top_k=config.rag.top_k,
    )

    # Create chatbot
    console.print("  [dim]Building LCEL chains...[/dim]")
    chatbot = CTBCChatbot(
        llm=base_llm,
        retriever=retriever,
        finetuned_llm=finetuned_llm,
    )

    console.print("[green]✓ Chatbot ready![/green]\n")
    return chatbot


def run_chat_loop(
    chatbot: CTBCChatbot, compare_mode: bool = False, default_mode: str = "rag"
) -> NoReturn:
    """
    Run the main chat loop.

    Args:
        chatbot: Initialized CTBCChatbot instance
        compare_mode: Whether to show all three response modes
    """
    print_welcome_message(compare_mode)

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            user_input = user_input.strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("exit", "quit"):
                console.print("\n[yellow]Thank you for using CTBC QA Bot. Goodbye![/yellow]")
                sys.exit(0)

            if user_input.lower() == "clear":
                chatbot.clear_history()
                console.print("[green]Conversation history cleared.[/green]")
                continue

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "compare":
                compare_mode = not compare_mode
                status = "enabled" if compare_mode else "disabled"
                console.print(f"[yellow]Compare mode {status}[/yellow]")
                continue

            # Get response(s) from chatbot
            if compare_mode:
                with console.status("[bold green]Generating all 3 responses...", spinner="dots"):
                    try:
                        responses = chatbot.chat_all(user_input)
                        print_comparison_responses(responses)
                    except Exception as e:
                        console.print(f"\n[red]Error generating responses: {e}[/red]")
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
                        continue
            else:
                with console.status("[bold green]Thinking...", spinner="dots"):
                    response = chatbot.chat(user_input, mode=default_mode)
                console.print(f"\n[bold green]Bot[/bold green]: {response}")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Please try again or type 'exit' to quit.[/dim]")
            continue


def main() -> None:
    """Main entry point for the CLI chat application."""
    parser = argparse.ArgumentParser(description="CTBC QA Bot CLI")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Start in compare mode (show all 3 response types)",
    )
    parser.add_argument(
        "--load-finetuned",
        action="store_true",
        help="Load fine-tuned model if available",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to LoRA adapter (overrides config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "rag", "finetuned_rag"],
        default="rag",
        help="Response mode: 'base' (no RAG), 'rag' (base + RAG), 'finetuned_rag' (fine-tuned + RAG)",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        config = get_config()

        # Ensure directories exist
        config.ensure_directories()

        # Determine if fine-tuned model is needed
        needs_finetuned = args.load_finetuned or args.compare or args.mode == "finetuned_rag"

        # Initialize chatbot
        chatbot = initialize_chatbot(
            config,
            load_finetuned=needs_finetuned,
            lora_path=args.lora_path,
        )

        # Run chat loop
        run_chat_loop(chatbot, compare_mode=args.compare, default_mode=args.mode)

    except Exception as e:
        console.print(f"[red]Failed to initialize chatbot: {e}[/red]")
        console.print("[dim]Please check your configuration and try again.[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
