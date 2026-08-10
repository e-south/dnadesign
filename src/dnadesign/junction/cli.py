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

from dnadesign.junction.api import (
    build,
    load_sequence_records,
    plan,
    preflight,
    request_from_sequences,
    sequence_record,
    verify,
)
from dnadesign.junction.contracts.request import canonical_request_bytes, load_request
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


def _call(operation: Any, *, output_format: str) -> Any:
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
    return result


def _run(operation: Any, *, output_format: str) -> None:
    normalized_format = _format(output_format)
    result = _call(operation, output_format=normalized_format)
    payload = result.to_mapping()
    _emit(payload, output_format=normalized_format)


@app.command("request")
def request_command(
    base_request: Path = typer.Option(
        ...,
        "--base-request",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Existing request whose targets are replaced while its design policy is reused.",
    ),
    sequence: str | None = typer.Option(None, "--sequence", help="One raw DNA sequence."),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Text DNA or FASTA; auto detection uses the file suffix.",
    ),
    input_format: str = typer.Option("auto", "--input-format", help="Input file format: auto, text, or fasta."),
    target_id: str | None = typer.Option(None, "--target-id", help="Target ID for raw or text input."),
    assembly_group_id: str = typer.Option(
        "assembly-01",
        "--assembly-group",
        help="Targets in this group are designed and checked together.",
    ),
    primer_binding_length: int = typer.Option(
        ...,
        "--primer-binding-length",
        min=1,
        help="Number of terminal target bases used by each recovery primer.",
    ),
    recovery_mode: str = typer.Option(
        "target_specific",
        "--recovery-mode",
        help="Recovery-primer mode: target_specific or universal.",
    ),
    forward_extension: str = typer.Option(
        "",
        "--forward-5-prime-extension",
        help="Optional exact 5-prime sequence prepended to each forward primer.",
    ),
    reverse_extension: str = typer.Option(
        "",
        "--reverse-5-prime-extension",
        help="Optional exact 5-prime sequence prepended to each reverse primer.",
    ),
) -> None:
    """Write canonical request JSON to stdout from raw, text, or FASTA sequences."""

    def prepare_request():
        if (sequence is None) == (input_path is None):
            raise JunctionConfigError("Provide exactly one of --sequence or --input")
        if sequence is not None:
            records = (sequence_record(sequence, target_id=target_id or "target-01"),)
        else:
            assert input_path is not None
            records = load_sequence_records(input_path, source_format=input_format, target_id=target_id)
        base = load_request(base_request)
        return request_from_sequences(
            records,
            planning=base.planning,
            order_policy=base.order_policy,
            seed=base.seed,
            primer_binding_length=primer_binding_length,
            assembly_group_id=assembly_group_id,
            recovery_mode=recovery_mode,
            forward_five_prime_extension=forward_extension,
            reverse_five_prime_extension=reverse_extension,
        )

    request = _call(prepare_request, output_format="json")
    typer.echo(canonical_request_bytes(request).decode("utf-8"), nl=False)


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
