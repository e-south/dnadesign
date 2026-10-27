"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/app.py

Root OPS CLI application with lazily imported subcommand modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence

import click
import typer
from typer.main import get_command

from .common import pop_pending_stderr_messages, reset_pending_stderr_messages
from .dispatch import LazyGroup

app = typer.Typer(
    cls=LazyGroup,
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    help=(
        "Cross-tool orchestration commands for deterministic batch plans. "
        "Start with `uv run ops catalog list --simple` to browse routes from the terminal."
    ),
)


@app.callback()
def root_callback() -> None:
    """Root OPS CLI callback."""


def _write_stderr(message: str, *, stderr_fd: int = 2) -> None:
    encoded = message.encode("utf-8", errors="replace")
    try:
        if stderr_fd != 2:
            # Yield between line writes so duplicate-fd stderr output is fully
            # visible to subprocess capture before the console process exits.
            for chunk in encoded.splitlines(keepends=True) or (encoded,):
                os.write(stderr_fd, chunk)
                time.sleep(0.001)
            return
        os.write(stderr_fd, encoded)
    except (AttributeError, OSError):  # pragma: no cover
        stream = getattr(sys, "__stderr__", sys.stderr)
        stream.write(message)
        stream.flush()


def _render_usage_error(exc: click.UsageError, *, prog_name: str) -> str:
    if exc.ctx is not None:
        usage = exc.ctx.get_usage().rstrip()
        command_path = exc.ctx.command_path or prog_name
    else:
        usage = f"Usage: {prog_name} [OPTIONS] COMMAND [ARGS]..."
        command_path = prog_name
    return f"{usage}\nTry '{command_path} --help' for help.\n\nError: {exc.format_message()}\n"


def _render_click_error(exc: click.ClickException) -> str:
    return f"Error: {exc.format_message()}\n"


def main(argv: Sequence[str] | None = None, *, stderr_fd: int = 2) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = get_command(app)
    reset_pending_stderr_messages()
    try:
        result = command.main(args=args, prog_name="ops", standalone_mode=False)
    except click.UsageError as exc:
        pending = "".join(pop_pending_stderr_messages())
        if pending:
            _write_stderr(pending, stderr_fd=stderr_fd)
        _write_stderr(_render_usage_error(exc, prog_name="ops"), stderr_fd=stderr_fd)
        return int(exc.exit_code or 2)
    except click.ClickException as exc:
        pending = "".join(pop_pending_stderr_messages())
        if pending:
            _write_stderr(pending, stderr_fd=stderr_fd)
        _write_stderr(_render_click_error(exc), stderr_fd=stderr_fd)
        return int(exc.exit_code or 2)
    except click.exceptions.Exit as exc:
        pending = "".join(pop_pending_stderr_messages())
        if pending:
            _write_stderr(pending, stderr_fd=stderr_fd)
        return int(exc.exit_code or 0)
    except Exception as exc:  # pragma: no cover
        pending = "".join(pop_pending_stderr_messages())
        if pending:
            _write_stderr(pending, stderr_fd=stderr_fd)
        _write_stderr(f"OPS internal error: {exc}\n", stderr_fd=stderr_fd)
        return 1
    pending = "".join(pop_pending_stderr_messages())
    if pending:
        _write_stderr(pending, stderr_fd=stderr_fd)
    return int(result or 0)


def __getattr__(name: str):
    return getattr(app, name)


__all__ = ["app", "main"]
