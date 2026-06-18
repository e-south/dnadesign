"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/scar_nick.py

CLI entrypoints for terminal scar-nick processing design.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Validate, design, and inspect terminal Type IIS scar plus top/bottom nick processing candidates.",
)
console = Console()


def validate_scar_nick_spec(*args, **kwargs):
    from dnadesign.cruncher.app.scar_nick_workflow import validate_scar_nick_spec as _validate_scar_nick_spec

    return _validate_scar_nick_spec(*args, **kwargs)


def run_scar_nick_design(*args, **kwargs):
    from dnadesign.cruncher.app.scar_nick_workflow import run_scar_nick_design as _run_scar_nick_design

    return _run_scar_nick_design(*args, **kwargs)


def scar_nick_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.scar_nick_workflow import scar_nick_show_payload as _scar_nick_show_payload

    return _scar_nick_show_payload(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"Scar-nick spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    console.print(f"Accepted candidates -> {len(report.candidates)}")
    if report.release_placement is not None:
        release = report.release_placement
        console.print(
            "Release -> "
            f"{release.variant_id} site=[{release.recognition_site_start},{release.recognition_site_end}) "
            f"cuts={release.top_cut_boundary}/{release.bottom_cut_boundary}"
        )
    if report.candidates:
        console.print("Candidates:")
        for candidate in report.candidates[:8]:
            console.print(
                f"  - rank {candidate.rank}: {candidate.left_base}/{candidate.right_base} "
                f"profile={candidate.profile_s3s2s1s0} nick={candidate.nickase_site}"
            )
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


@app.command("validate", help="Validate a scar-nick spec and rank feasible terminal junction candidates.")
def validate_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/scar_nick/<name>.scar_nick.yaml.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        report = validate_scar_nick_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("design", help="Write the deterministic scar-nick design bundle for a validated spec.")
def design_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/scar_nick/<name>.scar_nick.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing deterministic scar-nick run directory.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        run_dir, report = run_scar_nick_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"Scar-nick outputs -> {run_dir}")
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("show", help="Show paths and summary for a scar-nick run directory.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to an outputs/scar_nick/<name> run directory."),
) -> None:
    try:
        payload = scar_nick_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Scar-nick run -> {payload['spec_name']}")
    console.print(f"Run dir -> {payload['run_dir']}")
    console.print(f"Status -> {payload['status']}: {payload['status_message']}")
    console.print(f"Candidates -> {payload['candidate_count']}")
    console.print(f"Manifest -> {payload['manifest_path']}")
    console.print(f"Status file -> {payload['status_path']}")
    console.print(f"Report JSON -> {payload['report_json']}")
    console.print(f"Report MD -> {payload['report_md']}")
    console.print(f"Candidate profiles -> {payload['candidate_profiles']}")
    console.print(f"Nickase geometry audit -> {payload['nickase_geometry_audit']}")
    console.print(f"Candidate table -> {payload['candidate_table']}")
    console.print(f"Candidate pair-call table -> {payload['candidate_pair_call_table']}")
    console.print(f"Nickase geometry audit table -> {payload['nickase_geometry_audit_table']}")
    if payload.get("views_manifest") is not None:
        console.print(f"Views manifest -> {payload['views_manifest']}")
    if payload.get("terminal_nick_visual_contract") is not None:
        console.print(f"Terminal nick visual -> {payload['terminal_nick_visual_contract']}")
    if payload.get("scar_nick_terminal_nick_visual_contracts") is not None:
        console.print(f"Visual contracts JSONL -> {payload['scar_nick_terminal_nick_visual_contracts']}")
    if payload.get("baserender_job") is not None:
        console.print(f"BaseRender job -> {payload['baserender_job']}")
    console.print(f"Spec snapshot -> {payload['spec_snapshot']}")
    console.print(f"Nickase catalog -> {payload['nickase_catalog']}")
    console.print(f"Release catalog -> {payload['release_catalog']}")


__all__ = ["app"]
