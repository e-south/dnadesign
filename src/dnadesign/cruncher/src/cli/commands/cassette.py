"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/cassette.py

CLI entrypoints for the dual-context cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Validate and materialize dual-context hairpin cassette specs.")
console = Console()


def validate_cassette_spec(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_workflow import validate_cassette_spec as _validate_cassette_spec

    return _validate_cassette_spec(*args, **kwargs)


def run_cassette_design(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_workflow import run_cassette_design as _run_cassette_design

    return _run_cassette_design(*args, **kwargs)


def cassette_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_workflow import cassette_show_payload as _cassette_show_payload

    return _cassette_show_payload(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"Cassette spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    if report.candidate is not None:
        console.print(f"Cassette -> {report.candidate.cassette_sequence}")
        console.print(
            "Nicks -> "
            f"{report.candidate.left_nick.nickase}@{report.candidate.left_nick.nick_coordinate}, "
            f"{report.candidate.right_nick.nickase}@{report.candidate.right_nick.nick_coordinate}"
        )
        console.print(
            "Bounded segment -> "
            f"{report.candidate.bounded_segment.start}..{report.candidate.bounded_segment.end} "
            f"(length={report.candidate.bounded_segment.length})"
        )
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


@app.command("validate", help="Validate a cassette spec and emit a deterministic planning report.")
def validate_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/cassettes/<name>.cassette.yaml.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        report = validate_cassette_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("design", help="Write cassette run artifacts for a validated spec.")
def design_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/cassettes/<name>.cassette.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic run directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        run_dir, report = run_cassette_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Cassette outputs -> {run_dir}")
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("show", help="Show paths and summary for a cassette run directory.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a cassette run directory under outputs/cassettes/."),
) -> None:
    try:
        payload = cassette_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Cassette run -> {payload['spec_name']}")
    console.print(f"Status -> {payload['status']}: {payload['status_message']}")
    console.print(f"Report JSON -> {payload['report_json']}")
    console.print(f"Report MD -> {payload['report_md']}")
    if payload.get("render_contract"):
        console.print(f"Render contract -> {payload['render_contract']}")
