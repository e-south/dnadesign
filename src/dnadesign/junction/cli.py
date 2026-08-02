"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/cli.py

Thin command-line adapter for the junction public API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from dnadesign.junction.api import build, plan, preflight, verify
from dnadesign.junction.errors import (
    JunctionBundleError,
    JunctionConfigError,
    JunctionDesignError,
    JunctionError,
)

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Plan three-way-junction oligos from complete exact-target requests.",
)


def _format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"text", "json"}:
        raise typer.BadParameter("Output format must be text or json.")
    return normalized


def _emit(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"status: {payload.get('status', 'planned')}")
    if plan_id := payload.get("plan_id"):
        typer.echo(f"plan_id: {plan_id}")
    if path := payload.get("path") or payload.get("bundle"):
        typer.echo(f"path: {path}")


def _run(operation: Any, *, output_format: str) -> None:
    normalized_format = _format(output_format)
    try:
        result = operation()
    except JunctionError as exc:
        error_codes = {
            JunctionConfigError: "config_error",
            JunctionDesignError: "design_error",
            JunctionBundleError: "bundle_error",
        }
        code = next(
            (error_code for error_type, error_code in error_codes.items() if isinstance(exc, error_type)),
            "junction_error",
        )
        if normalized_format == "json":
            typer.echo(
                json.dumps(
                    {
                        "status": "error",
                        "error": {
                            "code": code,
                            "message": str(exc),
                            "retryable": False,
                        },
                    },
                    sort_keys=True,
                ),
                err=True,
            )
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = result.to_mapping()
    _emit(payload, output_format=normalized_format)


@app.command("preflight")
def preflight_command(
    request: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    """Validate the request and run the design without writing files."""

    _run(lambda: preflight(request), output_format=output_format)


@app.command("plan")
def plan_command(
    request: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    output_format: str = typer.Option(
        "json",
        "--format",
        help="Output format: json for the complete plan, or text for a summary.",
    ),
) -> None:
    """Print the complete plan as JSON, or a short text summary."""

    _run(lambda: plan(request), output_format=output_format)


@app.command("build")
def build_command(
    request: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    destination: Path = typer.Option(..., "--output", help="New bundle directory; it must not already exist."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    """Design, verify, and publish a bundle in a new directory."""

    _run(lambda: build(request, destination=destination), output_format=output_format)


@app.command("verify")
def verify_command(
    bundle: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    """Recompute and verify an existing bundle."""

    _run(lambda: verify(bundle), output_format=output_format)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
