"""
Explicit preserved-site Snapback CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from dnadesign.cruncher.app.snapback_cli_requests import build_snapback_target_search_invocation
from dnadesign.cruncher.cli.commands.snapback_presenters import (
    console,
    print_report,
    print_solve_report,
    print_target_search_report,
)
from dnadesign.cruncher.cli.commands.snapback_services import (
    run_snapback_design,
    run_snapback_solve,
    run_snapback_target_search,
    validate_snapback_spec,
)


def validate_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.yaml.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        report = validate_snapback_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


def design_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing workspace output root if it already exists.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the run payload as JSON."),
) -> None:
    try:
        validation_report = validate_snapback_spec(spec)
        if validation_report.status == "invalid_catalog":
            if json_output:
                typer.echo(validation_report.model_dump_json(indent=2))
            else:
                print_report(validation_report)
            raise typer.Exit(code=1)
        run_dir, report = run_snapback_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(
            json.dumps(
                {"run_dir": str(run_dir), "status": report.status, "spec_name": report.spec_name},
                indent=2,
            )
        )
    else:
        console.print(f"Outputs -> {run_dir}")
        print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


def solve_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.solve.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing workspace solve output root.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the solve report as JSON."),
) -> None:
    try:
        run_dir, report = run_snapback_solve(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Snapback solve outputs -> {run_dir}")
        print_solve_report(report)
    if report.status not in {"satisfied", "search_truncated"}:
        raise typer.Exit(code=1)


def target_search_cmd(
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Primary builtin nickase preset. Defaults to neb_nicking_v1 when no catalog source is provided.",
    ),
    additional_preset: list[str] = typer.Option(
        [],
        "--additional-preset",
        help="Additional builtin nickase preset ids (repeatable).",
    ),
    additional_path: list[Path] = typer.Option(
        [],
        "--additional-path",
        help="Additional workspace-relative nickase catalog overlays (repeatable).",
    ),
    workspace_root: Path = typer.Option(
        Path("."),
        "--workspace-root",
        help="Workspace root for resolving --additional-path values.",
    ),
    nick_boundary: int = typer.Option(0, "--nick-boundary", help="Requested nick_boundary_from_left."),
    paired_bp: int = typer.Option(3, "--paired-bp", help="Requested retained homology length."),
    cap_nt: int = typer.Option(3, "--cap-nt", help="Requested effective cap nt. Must equal 3."),
    max_results: int = typer.Option(8, "--max-results", help="Maximum exact or near hits to return."),
    normalize_to_top_strand_nick: bool = typer.Option(
        True,
        "--normalize-to-top-strand-nick/--allow-complement-nick",
        help="Restrict search to orientations that normalize to a top-strand nick.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the target-search report as JSON."),
) -> None:
    try:
        invocation = build_snapback_target_search_invocation(
            preset=preset,
            additional_preset=additional_preset,
            additional_path=additional_path,
            workspace_root=workspace_root,
            nick_boundary=nick_boundary,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
            max_results=max_results,
            normalize_to_top_strand_nick=normalize_to_top_strand_nick,
        )
        report = run_snapback_target_search(
            catalog=invocation.catalog,
            workspace_root=invocation.workspace_root,
            target=invocation.target,
            normalize_to_top_strand_nick=invocation.normalize_to_top_strand_nick,
            max_results=invocation.max_results,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_target_search_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


__all__ = ["design_cmd", "solve_cmd", "target_search_cmd", "validate_cmd"]
