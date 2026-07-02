import sys
from contextlib import contextmanager
from typing import Generator
import click
from rich.console import Console


class TUIConsole:
    """Encapsulates the Rich console to handle Graceful Degradation (Edge Case 3)."""

    def __init__(self) -> None:
        self.is_interactive = sys.stdout.isatty()
        # If not interactive, Rich automatically strips ANSI codes when force_terminal=False.
        self.console = Console(
            force_terminal=self.is_interactive, force_interactive=self.is_interactive
        )

    def print(self, message: str, nl: bool = True) -> None:
        """Print a message to the console."""
        self.console.print(message, end="\n" if nl else "")


tui_console = TUIConsole()


@contextmanager
def AnalysisSpinner(
    message: str = "Analysis Block Active: Populating tmp-ContextDB...",
) -> Generator[None, None, None]:
    """Provides a spinner that gracefully degrades to static logging in non-TTY environments."""
    if tui_console.is_interactive:
        with tui_console.console.status(
            f"[bold cyan]⚙️  {message}[/bold cyan]", spinner="dots"
        ):
            yield
    else:
        click.echo(f"   ⚙️  {message}")
        yield
        click.echo("   ✅ Analysis Block complete.")


def stream_supervisor_thought(model: str, message: str) -> None:
    """Format and stream supervisor thoughts during Triage and Network phases."""
    tui_console.print(f"   [bold blue]👤 Supervisor ({model})[/bold blue]: {message}")


def log_event(message: str) -> None:
    """Generic log event with rich formatting."""
    tui_console.print(message)
