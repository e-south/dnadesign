"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/io.py

Retron MSD CLI text and JSON output helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any

import typer

from .messages import next_step_for_error


def format_option(output_format: str) -> str:
    format_norm = str(output_format or "").strip().lower()
    if format_norm not in {"text", "json"}:
        raise typer.BadParameter("Output format must be text or json.")
    return format_norm


def emit(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"status: {payload.get('status')}")
    if payload.get("reference") is not None:
        reference = payload["reference"]
        typer.echo(f"msd_design_id: {reference.get('msd_design_id')}")
        typer.echo(f"construct_id: {reference.get('construct_id')}")
    if payload.get("catalog_path") is not None:
        typer.echo(f"catalog_path: {payload['catalog_path']}")
        typer.echo(f"record_count: {payload.get('record_count')}")
    elif payload.get("record_count") is not None:
        typer.echo(f"record_count: {payload.get('record_count')}")
    if payload.get("output_dir") is not None:
        typer.echo(f"output_dir: {payload['output_dir']}")
    if payload.get("references_dir") is not None:
        typer.echo(f"references_dir: {payload['references_dir']}")
    if payload.get("index_path") is not None:
        typer.echo(f"index_path: {payload['index_path']}")
    if payload.get("manifest_path") is not None:
        typer.echo(f"manifest_path: {payload['manifest_path']}")
    if payload.get("readme_path") is not None:
        typer.echo(f"readme_path: {payload['readme_path']}")
    if payload.get("sequence_manifest_path") is not None:
        typer.echo(f"sequence_manifest_path: {payload['sequence_manifest_path']}")
    if payload.get("sequence_index_path") is not None:
        typer.echo(f"sequence_index_path: {payload['sequence_index_path']}")
    if payload.get("variants_dir") is not None:
        typer.echo(f"variants_dir: {payload['variants_dir']}")
    if payload.get("composition_configs_dir") is not None:
        typer.echo(f"composition_configs_dir: {payload['composition_configs_dir']}")
    if payload.get("finder_open") is not None:
        typer.echo(f"finder_open: {payload['finder_open']}")
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            typer.echo(f"warning: {warning}")
    if payload.get("next_step") is not None:
        typer.echo(f"next_step: {payload['next_step']}")


def exit_with_error(exc: Exception, *, output_format: str) -> None:
    next_step = next_step_for_error(exc)
    if output_format == "json":
        emit(
            {
                "status": "error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "next_step": next_step,
            },
            output_format=output_format,
        )
    else:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"next_step: {next_step}", err=True)
    raise typer.Exit(code=1) from exc


__all__ = ["emit", "exit_with_error", "format_option"]
