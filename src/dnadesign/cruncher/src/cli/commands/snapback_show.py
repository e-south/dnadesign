"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/snapback_show.py

Read-only show commands for Snapback bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from dnadesign.cruncher.cli.commands.snapback_presenters import console
from dnadesign.cruncher.cli.commands.snapback_services import released_show_payload, snapback_show_payload


def released_show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a released-product snapback output root."),
    json_output: bool = typer.Option(False, "--json", help="Print the released show payload as JSON."),
) -> None:
    try:
        payload = released_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Released-product bundle -> {payload['spec_name']}")
    console.print(f"Kind -> {payload['kind']}")
    console.print(f"Status -> {payload['status']}")
    console.print(f"Manifest -> {payload['manifest_path']}")
    console.print(f"Status file -> {payload['status_path']}")
    console.print(f"Report JSON -> {payload['report_json']}")
    console.print(f"Projection JSON -> {payload['projection_json']}")


def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a snapback design or solve output root."),
    json_output: bool = typer.Option(False, "--json", help="Print the show payload as JSON."),
) -> None:
    try:
        payload = snapback_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Snapback bundle -> {payload['spec_name']}")
    console.print(f"Kind -> {payload['kind']}")
    console.print(f"Status -> {payload['status']}")
    if payload["kind"] == "explicit":
        console.print(f"Manifest -> {payload['manifest_path']}")
        console.print(f"Status file -> {payload['status_path']}")
        console.print(f"Report JSON -> {payload['report_json']}")
        console.print(f"Report Markdown -> {payload['report_md']}")
    else:
        console.print(f"Solve manifest -> {payload['solve_manifest']}")
        console.print(f"Solve status -> {payload['solve_status']}")
        console.print(f"Solve report -> {payload['solve_report']}")


__all__ = ["released_show_cmd", "show_cmd"]
