import logging

import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # Load .env so DARTMOUTH_CHAT_API_KEY etc. are available
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from simplecoder.agent import Agent
from simplecoder.permissions import Permission


logging.basicConfig(level=logging.ERROR)
console = Console()


def _ask_permission(permission: Permission, path: str, detail: str) -> bool:
    """Prompt user for permission. For SHELL, always ask yes/no."""
    if permission == Permission.SHELL:
        console.print(f"\n[bold yellow]Run shell command?[/bold yellow] {path}")
        ans = Prompt.ask("Allow? [y/n]", default="n")
        return ans.strip().lower() in ("y", "yes")
    return True  # allow other permissions (read, write, etc.)


def main():
    """Entry point for the simple coder agent."""
    cli()


@click.command()
@click.option(
    "--model",
    default="vertex_ai.gemini-2.5-pro",
    help="LLM model to use (see models.txt)"
)
@click.option(
    "--max-iterations",
    default=10,
    type=int,
    help="Maximum number of ReAct iterations"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Run in interactive mode"
)
@click.option(
    "--use-planning",
    is_flag=True,
    help="Enable planning and task decomposition"
)
@click.option(
    "--use-rag",
    is_flag=True,
    help="Enable RAG"
)
@click.option(
    "--rag-embedder",
    default="google_genai.gemini-embedding-001",
    help="Embedding model for RAG (see models.txt)"
)
@click.option(
    "--rag-index-pattern",
    default="**/*.py",
    help="File pattern for RAG"
)
@click.option(
    "--use-reflection",
    is_flag=True,
    help="Enable reflection when importance exceeds threshold"
)
@click.option(
    "--dangerous",
    is_flag=True,
    help="Enable use_shell tool (run shell commands). Commands are validated; use with caution."
)
@click.argument(
    "task",
    required=False
)
def cli(
    model: str,
    max_iterations: int,
    verbose: bool,
    interactive: bool,
    use_planning: bool,
    use_rag: bool,
    rag_embedder: str,
    rag_index_pattern: str,
    use_reflection: bool,
    dangerous: bool,
    task: str | None
) -> None:
    """A simple coding agent."""
    if dangerous:
        console.print(
            Panel(
                "[bold red]DANGEROUS MODE[/bold red]\n\n"
                "Shell commands are enabled. You will be prompted before each command.\n"
                "Only run commands you understand and trust.",
                border_style="red",
                title="[bold]Warning[/bold]",
            )
        )

    agent = Agent(
        model=model,
        max_iterations=max_iterations,
        verbose=verbose,
        use_planning=use_planning,
        use_rag=use_rag,
        rag_embedder=rag_embedder,
        rag_index_pattern=rag_index_pattern,
        use_reflection=use_reflection,
        dangerous=dangerous,
        permission_callback=_ask_permission,
    )

    if task:
        response = agent.run(task)
        while agent._resumable:
            console.print(response)
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]", default="continue")
            if user_input.strip().lower() in ("continue", "yes"):
                response = agent.run("continue", reset=False)
            else:
                response = agent.run(user_input, reset=False)
        console.print(Panel(Markdown(response), title="[bold green]Agent Response[/bold green]", border_style="green"))
        return

    if interactive:
        intro = (
            "[bold cyan]SimpleCoder Agent[/bold cyan]\n\n"
            "Type your requests and I'll help you code.\n"
            "Type 'exit', 'quit', or 'q' to quit.\n"
            "After max iterations, type 'continue' or 'yes' to resume."
        )
        if dangerous:
            intro += "\n\n[bold red]Dangerous mode:[/bold red] You will be prompted before each shell command."
        console.print(Panel(intro, border_style="cyan"))
        console.print()

        while True:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.strip().lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            response = agent.run(user_input, reset=False)
            if agent._resumable:
                console.print(response)
            else:
                console.print()
                console.print(Panel(
                    Markdown(response),
                    title="[bold green]Agent[/bold green]",
                    border_style="green"
                ))


if __name__ == "__main__":
    main()
