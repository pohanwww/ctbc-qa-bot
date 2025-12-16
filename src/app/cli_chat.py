"""
CLI Chat Interface for CTBC QA Bot.

Provides a simple REPL (Read-Eval-Print Loop) interface for interacting
with the CTBC customer service chatbot.
"""

import sys
from typing import NoReturn

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.app.config import AppConfig, get_config
from src.chains.chatbot import CTBCChatbot
from src.llm.loader import load_llm
from src.rag.retriever import FAQRetriever

console = Console()


def print_welcome_message() -> None:
    """Print the welcome message and instructions."""
    welcome_text = """
# Welcome to CTBC Bank Customer Service

I'm your AI assistant, ready to help you with questions about CTBC Bank's products and services.

**Commands:**
- Type your question and press Enter to chat
- Type `exit` or `quit` to end the conversation
- Type `clear` to clear conversation history
- Type `help` to see this message again

**Note:** My knowledge about CTBC Bank comes from the official FAQ. For specific account inquiries or transactions, please contact CTBC Bank directly.
    """
    console.print(Panel(Markdown(welcome_text), title="CTBC QA Bot", border_style="blue"))


def print_help() -> None:
    """Print help information."""
    help_text = """
**Available Commands:**
- `exit` / `quit` - End the conversation
- `clear` - Clear conversation history and start fresh
- `help` - Show this help message

**Tips:**
- Ask questions in English for best results
- Be specific about what you want to know
- If I don't have the information, I'll suggest contacting CTBC Bank directly
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))


def initialize_chatbot(config: AppConfig) -> CTBCChatbot:
    """
    Initialize the chatbot with all required components.

    Args:
        config: Application configuration

    Returns:
        Initialized CTBCChatbot instance
    """
    console.print("[yellow]Initializing chatbot components...[/yellow]")

    # Load LLM
    console.print("  [dim]Loading language model...[/dim]")
    llm = load_llm(
        model_id=config.model.model_id,
        device=config.model.device,
        lora_adapter_path=config.model.lora_adapter_path,
        hf_token=config.model.hf_token,
    )

    # Load RAG retriever
    console.print("  [dim]Loading RAG retriever...[/dim]")
    retriever = FAQRetriever(
        index_path=config.rag.faiss_index_path,
        embedding_model_id=config.model.embedding_model_id,
        top_k=config.rag.top_k,
    )

    # Create chatbot
    console.print("  [dim]Setting up chat chain...[/dim]")
    chatbot = CTBCChatbot(
        llm=llm,
        retriever=retriever,
        max_new_tokens=config.inference.max_new_tokens,
        temperature=config.inference.temperature,
    )

    console.print("[green]✓ Chatbot ready![/green]\n")
    return chatbot


def run_chat_loop(chatbot: CTBCChatbot) -> NoReturn:
    """
    Run the main chat loop.

    Args:
        chatbot: Initialized CTBCChatbot instance
    """
    print_welcome_message()

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

            # Get response from chatbot
            with console.status("[bold green]Thinking...", spinner="dots"):
                response = chatbot.chat(user_input)

            # Print response
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
    try:
        # Load configuration
        config = get_config()

        # Ensure directories exist
        config.ensure_directories()

        # Initialize and run chatbot
        chatbot = initialize_chatbot(config)
        run_chat_loop(chatbot)

    except Exception as e:
        console.print(f"[red]Failed to initialize chatbot: {e}[/red]")
        console.print("[dim]Please check your configuration and try again.[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
