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
from itertools import zip_longest
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, design, solve, inspect, and catalog dual-context hairpin cassette workflows.",
)
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect or export cassette nickase preset catalogs.")
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


def solve_cassette_spec(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_solve_workflow import solve_cassette_spec as _solve_cassette_spec

    return _solve_cassette_spec(*args, **kwargs)


def init_cassette_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_workspace_service import init_cassette_workspace as _init_cassette_workspace

    return _init_cassette_workspace(*args, **kwargs)


def run_cassette_solve(*args, **kwargs):
    from dnadesign.cruncher.app.cassette_solve_workflow import run_cassette_solve as _run_cassette_solve

    return _run_cassette_solve(*args, **kwargs)


def read_builtin_nickase_catalog_preset_text(*args, **kwargs):
    from dnadesign.cruncher.cassette.catalog import (
        read_builtin_nickase_catalog_preset_text as _read_builtin_nickase_catalog_preset_text,
    )

    return _read_builtin_nickase_catalog_preset_text(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"Cassette spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    console.print(f"Mode -> schema v{report.metadata.spec_schema_version} / {report.metadata.coordinate_semantics}")
    for warning in report.metadata.warnings:
        console.print(f"Warning -> {warning}")
    if report.candidate is not None:
        console.print(f"Cassette -> {report.candidate.cassette_sequence}")
        console.print(
            "Nicks -> "
            f"{report.candidate.intended_left_nick.variant_id}@{report.candidate.intended_left_nick.boundary}, "
            f"{report.candidate.intended_right_nick.variant_id}@{report.candidate.intended_right_nick.boundary}"
        )
        console.print(
            "Bounded nicked segment -> "
            f"{report.candidate.bounded_nicked_segment.start_boundary}.."
            f"{report.candidate.bounded_nicked_segment.end_boundary} "
            f"(length={report.candidate.bounded_nicked_segment.length_nt})"
        )
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _print_solve_report(report) -> None:
    console.print(f"Cassette solve spec -> {report.spec_path}")
    console.print(f"Status -> {report.status}")
    if report.solve_id:
        console.print(f"Solve id -> {report.solve_id}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    if report.baserender_hits_contract_path:
        console.print(f"Baserender contract -> {report.baserender_hits_contract_path}")
    if report.metadata.catalog_preset:
        console.print(f"Preset -> {report.metadata.catalog_preset}")
    for path in report.metadata.catalog_additional_paths:
        console.print(f"Catalog overlay -> {path}")
    if report.selection_summary is not None:
        selection = report.selection_summary
        console.print(
            "Selection -> "
            f"{selection.policy} ({selection.distance_metric}, defaulted={selection.selection_policy_defaulted})"
        )
        console.print(
            "Selected -> "
            f"{selection.selected_hit_count}/{selection.max_hits} "
            f"ids={', '.join(selection.selected_hit_ids) if selection.selected_hit_ids else 'none'}"
        )
        console.print(
            "Pool -> "
            f"kept={selection.accepted_pool_size}/{selection.pool_size}, "
            f"admitted={selection.accepted_pool_admitted_count}, "
            f"rejected={selection.accepted_pool_rejected_count}, "
            f"truncated={selection.accepted_pool_truncated}"
        )
        if selection.selection_pool_non_exhaustive_reason:
            console.print(f"Selection bounds -> {selection.selection_pool_non_exhaustive_reason}")
        if selection.policy_underfilled:
            console.print(
                "Selection filter -> "
                f"{selection.policy_underfilled_reason} "
                f"(missing={selection.policy_limited_hit_count})"
            )
        if selection.pairwise_distance_summary.min is not None:
            console.print(
                "Pairwise distance -> "
                f"min={selection.pairwise_distance_summary.min:.1f}, "
                f"mean={selection.pairwise_distance_summary.mean:.1f}, "
                f"max={selection.pairwise_distance_summary.max:.1f}"
            )
    for code, warning in zip_longest(report.metadata.warning_codes, report.metadata.warnings, fillvalue=None):
        if code and warning:
            console.print(f"Warning -> {code}: {warning}")
        elif warning:
            console.print(f"Warning -> {warning}")
        elif code:
            console.print(f"Warning -> {code}")
    console.print(
        "Search -> "
        f"nodes={report.metadata.visited_search_node_count}, "
        f"enumerated={report.metadata.enumerated_candidate_count}, "
        f"accepted={report.metadata.accepted_candidate_count}, "
        f"variant_pairs={report.metadata.considered_variant_pair_count}"
    )
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.hit_id} "
                f"{hit.left_variant_id}@{hit.left_nick_boundary} -> "
                f"{hit.right_variant_id}@{hit.right_nick_boundary} "
                f"score={hit.score}"
            )
            if hit.materialized_run_dir:
                line += f" run={hit.materialized_run_dir}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


@app.command("init-workspace", help="Scaffold an isolated cassette workspace with pressure-test solve profiles.")
def init_workspace_cmd(
    output: Path = typer.Option(
        ...,
        "--output",
        help="New cassette workspace root to create. Existing symlink roots are rejected.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing scaffold created by this command.",
    ),
) -> None:
    try:
        result = init_cassette_workspace(output, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Cassette workspace scaffold -> {result.workspace_root}")
    console.print(f"README -> {result.readme_path}")
    console.print(f"Manifest -> {result.manifest_path}")
    for name, path in result.solve_specs.items():
        console.print(f"Spec -> {name}: {path}")


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
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"Cassette outputs -> {run_dir}")
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("solve", help="Search for concrete cassette sequences that satisfy a cassette solve spec.")
def solve_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/cassettes/<name>.cassette.solve.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic solve run directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full solve report as JSON."),
) -> None:
    try:
        run_dir, report = run_cassette_solve(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        if run_dir is not None:
            console.print(f"Cassette solve outputs -> {run_dir}")
        _print_solve_report(report)
    if report.status != "solved":
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


@catalog_app.command("init-neb", help="Write the built-in neb_nicking_v1 cassette nickase preset to disk.")
def catalog_init_neb_cmd(
    output: Path = typer.Option(
        ...,
        "--output",
        help="Destination YAML path for the exported built-in preset.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing output file.",
    ),
) -> None:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not force_overwrite:
        console.print(f"Error: Output path already exists: {output_path}. Use --force-overwrite to replace it.")
        raise typer.Exit(code=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(read_builtin_nickase_catalog_preset_text("neb_nicking_v1"), encoding="utf-8")
    console.print(f"Cassette preset -> {output_path}")


app.add_typer(
    catalog_app,
    name="catalog",
    help="Inspect or export cassette nickase preset catalogs.",
)
