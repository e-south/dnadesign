"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/snapback.py

CLI entrypoints for explicit v2 and co-design solve v3 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, design, solve, and inspect single-nick snapback workflows.",
)
console = Console()


def validate_snapback_spec(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec as _validate_snapback_spec

    return _validate_snapback_spec(*args, **kwargs)


def run_snapback_design(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import run_snapback_design as _run_snapback_design

    return _run_snapback_design(*args, **kwargs)


def run_snapback_solve(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve as _run_snapback_solve

    return _run_snapback_solve(*args, **kwargs)


def run_snapback_target_search(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_target_search_workflow import (
        run_snapback_target_search as _run_snapback_target_search,
    )

    return _run_snapback_target_search(*args, **kwargs)


def validate_released_snapback_spec(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_workflow import (
        validate_released_snapback_spec as _validate_released_snapback_spec,
    )

    return _validate_released_snapback_spec(*args, **kwargs)


def run_released_snapback_design(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_workflow import (
        run_released_snapback_design as _run_released_snapback_design,
    )

    return _run_released_snapback_design(*args, **kwargs)


def run_released_snapback_target_search(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_target_search_workflow import (
        run_released_snapback_target_search as _run_released_snapback_target_search,
    )

    return _run_released_snapback_target_search(*args, **kwargs)


def run_released_snapback_solve(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_solve_workflow import (
        run_released_snapback_solve as _run_released_snapback_solve,
    )

    return _run_released_snapback_solve(*args, **kwargs)


def snapback_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import snapback_show_payload as _snapback_show_payload

    return _snapback_show_payload(*args, **kwargs)


def released_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_show import released_show_payload as _released_show_payload

    return _released_show_payload(*args, **kwargs)


def init_snapback_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import init_snapback_workspace as _init_snapback_workspace

    return _init_snapback_workspace(*args, **kwargs)


def snapback_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import snapback_workspace_path as _snapback_workspace_path

    return _snapback_workspace_path(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"Snapback spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    if report.candidate is not None:
        console.print(
            "Intended nick -> "
            f"{report.candidate.intended_nick.variant_id}@{report.candidate.nick_boundary} "
            f"({report.candidate.intended_site.orientation})"
        )
        console.print(f"Released prefix nt -> {report.candidate.released_prefix_nt}")
        console.print(f"Cap nt -> {report.candidate.cap_nt}")
        console.print(f"Added nt -> {report.candidate.added_nt}")
        console.print(f"Terminal ligatable duplex bp -> {report.candidate.terminal_ligatable_duplex_bp}")
        console.print(f"Max uninterrupted duplex bp -> {report.candidate.max_uninterrupted_duplex_bp}")
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _print_solve_report(report) -> None:
    console.print(f"Snapback solve spec -> {report.spec_path}")
    console.print(f"Status -> {report.status}")
    if report.solve_id:
        console.print(f"Solve id -> {report.solve_id}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    resolved = report.metadata.resolved_search_space
    console.print(
        "Resolved compact search -> "
        f"boundary={resolved.nick_boundary_window.min}..{resolved.nick_boundary_window.max}, "
        f"paired_bp={resolved.retained_homology_length.min}..{resolved.retained_homology_length.max}, "
        f"terminal_duplex={resolved.terminal_ligatable_duplex_bp.min}..{resolved.terminal_ligatable_duplex_bp.max}, "
        f"max_uninterrupted={resolved.max_uninterrupted_duplex_bp}"
    )
    console.print(
        "Search -> "
        f"nodes={report.metadata.visited_search_node_count}, "
        f"enumerated={report.metadata.enumerated_candidate_count}, "
        f"accepted={report.metadata.accepted_candidate_count}, "
        f"frontier_rows={report.metadata.frontier_row_count}, "
        f"materialized={report.metadata.materialized_hit_count}"
    )
    if report.metadata.first_satisfied_frontier is not None:
        frontier = report.metadata.first_satisfied_frontier
        console.print(
            "First satisfied frontier -> "
            f"boundary={frontier.nick_boundary_from_left}, "
            f"paired_bp={frontier.paired_bp}, "
            f"cap_ext_nt={frontier.cap_extension_nt}, "
            f"accepted={frontier.accepted_candidate_count}"
        )
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        console.print(f"Warning -> {code}: {warning}")
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.hit_id} "
                f"{hit.variant_id}@{hit.nick_boundary} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"site_mutations={hit.site_mutation_count} cap={hit.cap_sequence} added_nt={hit.added_nt}"
            )
            if hit.materialized_run_dir is not None:
                line += f" bundle={hit.materialized_run_dir}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _print_target_search_report(report) -> None:
    console.print("Snapback target search")
    console.print(f"Status -> {report.status}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Catalog -> {report.metadata.catalog_source}")
    console.print(f"Orientations evaluated -> {report.metadata.evaluated_orientation_count}")
    console.print(f"Exact hits -> {report.metadata.exact_hit_count}")
    console.print(f"Near hits -> {report.metadata.near_hit_count}")
    if report.exact_hits:
        console.print("Exact hits:")
        for hit in report.exact_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} "
                f"extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} "
                f"outside_site={outside_site}"
            )
    if report.near_hits:
        console.print("Near hits:")
        for hit in report.near_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} "
                f"extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} "
                f"outside_site={outside_site}"
            )
    if report.feasibility:
        console.print("Feasibility:")
        for row in report.feasibility[:8]:
            blockers = ",".join(row.exact_boundary_blockers) if row.exact_boundary_blockers else "-"
            console.print(
                "  - "
                f"{row.variant_id} {row.orientation} "
                f"exact={row.exact_boundary_hit_possible} "
                f"target_site_start={row.site_start_at_target_boundary} "
                "earliest_boundary="
                f"{row.earliest_feasible_boundary if row.earliest_feasible_boundary is not None else '-'} "
                f"blockers={blockers}"
            )


def _print_released_report(report) -> None:
    console.print(f"Released-product snapback spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    if report.candidate is not None and report.projection is not None:
        console.print(
            "Active bottom -> "
            f"input_nt={report.candidate.active_bottom_input_length_nt} "
            f"bottom_nt={report.candidate.active_bottom_length_nt} "
            f"nick={report.candidate.nick_boundary_from_left} "
            f"paired_bp={report.candidate.paired_bp} cap_nt={report.candidate.cap_nt}"
        )
        console.print(
            "Release cuts -> "
            f"top={report.projection.release_top_cut_precursor} "
            f"bottom={report.projection.release_bottom_cut_precursor}"
        )
        console.print(
            "Site survival -> "
            f"nickase={report.projection.nickase_site_survives_post_release} "
            f"release={report.projection.release_site_survives_post_release}"
        )
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _print_released_target_search_report(report) -> None:
    console.print("Released-product snapback target search")
    console.print(f"Status -> {report.status}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    console.print(f"Pairs evaluated -> {report.metadata.evaluated_pair_count}")
    console.print(
        "Hits -> "
        f"exact={report.metadata.post_truncation_exact_hit_count}/"
        f"{report.metadata.pre_truncation_exact_hit_count}, "
        f"near={report.metadata.post_truncation_near_hit_count}/"
        f"{report.metadata.pre_truncation_near_hit_count}"
    )
    if report.metadata.blocker_counts:
        console.print("Blockers:")
        for code, count in sorted(report.metadata.blocker_counts.items()):
            console.print(f"  - {code}: {count}")
    if report.exact_hits:
        console.print("Exact hits:")
        for hit in report.exact_hits:
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"active_bottom_input_nt={hit.active_bottom_input_length_nt} "
                f"precursor_nt={hit.precursor_length_nt} "
                f"tail_nt={hit.sacrificial_downstream_tail_nt}"
            )
    if report.near_hits:
        console.print("Near hits:")
        for hit in report.near_hits:
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"active_bottom_input_nt={hit.active_bottom_input_length_nt} "
                f"precursor_nt={hit.precursor_length_nt} "
                f"tail_nt={hit.sacrificial_downstream_tail_nt}"
            )


def _print_released_solve_report(report) -> None:
    console.print("Released-product snapback solve")
    console.print(f"Status -> {report.status}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    console.print(f"Pairs evaluated -> {report.metadata.evaluated_pair_count}")
    console.print(
        "Available hits -> "
        f"exact={report.metadata.available_exact_hit_count}, "
        f"near={report.metadata.available_near_hit_count}, "
        f"selected={report.metadata.selected_hit_kind or '-'}"
    )
    console.print(
        "Materialized -> "
        f"{report.metadata.materialized_hit_count}/{report.metadata.requested_materialize_top_k} "
        f"render_format={report.metadata.render_format} "
        f"emit_renders={report.metadata.emit_renders}"
    )
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"kind={hit.hit_kind} bundle={hit.materialized_run_dir}"
            )
            if hit.rendered_plot_path is not None:
                line += f" plot={hit.rendered_plot_path}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _echo_scaffold_line(label: str, value: str | Path) -> None:
    typer.echo(f"{label} -> {value}")


@app.command("init-workspace", help="Scaffold a snapback workspace with v2 explicit and v3 co-design solve examples.")
def init_workspace_cmd(
    workspace: str | None = typer.Argument(
        None,
        metavar="WORKSPACE",
        help="Workspace directory name. Creates <root>/WORKSPACE unless --output is used.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Parent Cruncher workspaces directory for WORKSPACE. Defaults to the standard Cruncher workspaces root.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Explicit snapback workspace root to create. Overrides WORKSPACE/--root.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing scaffold created by this command.",
    ),
) -> None:
    try:
        if output is not None and workspace is not None:
            raise typer.BadParameter("Use either WORKSPACE [--root] or --output, not both.")
        if output is not None and root is not None:
            raise typer.BadParameter("Use either --output or --root, not both.")
        if output is None and workspace is None:
            raise typer.BadParameter("Provide WORKSPACE or --output.")
        target_root = output if output is not None else snapback_workspace_path(workspace, root=root)
        result = init_snapback_workspace(target_root, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    _echo_scaffold_line("Snapback workspace scaffold", result.workspace_root)
    _echo_scaffold_line("README", result.readme_path)
    _echo_scaffold_line("Runbook", result.runbook_path)
    _echo_scaffold_line("Runbook config", result.runbook_config_path)
    _echo_scaffold_line("Example spec", result.example_spec_path)
    _echo_scaffold_line("Example solve spec", result.example_solve_spec_path)
    _echo_scaffold_line("Catalog", result.catalog_path)


@app.command("validate", help="Validate a v2 explicit snapback spec and emit a deterministic report.")
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
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("design", help="Materialize one v2 explicit snapback design bundle.")
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
                _print_report(validation_report)
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
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command(
    "solve",
    help=(
        "Search for concrete snapback designs that satisfy a v3 co-design solve spec. "
        "Omitted boundary and size windows resolve to compact-first defaults."
    ),
)
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
        _print_solve_report(report)
    if report.status not in {"satisfied", "search_truncated"}:
        raise typer.Exit(code=1)


@app.command(
    "target-search",
    help=(
        "Search the nickase catalog for shortest preserved-site snapback hits at a requested geometry. "
        "This mode is target-first and does not assume an authored canonical top strand."
    ),
)
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
        from dnadesign.cruncher.snapback.models import CatalogSources
        from dnadesign.cruncher.snapback.target_models import SnapbackTargetGeometry

        resolved_workspace_root = workspace_root.expanduser().resolve()
        if preset is None and not additional_preset and not additional_path:
            preset = "neb_nicking_v1"
            additional_preset = ["thermo_nicking_v1"]
        catalog = CatalogSources(
            preset=preset,
            additional_presets=additional_preset,
            additional_paths=additional_path,
        )
        target = SnapbackTargetGeometry(
            nick_boundary_from_left=nick_boundary,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
            require_site_sequence_preserved=True,
        )
        report = run_snapback_target_search(
            catalog=catalog,
            workspace_root=resolved_workspace_root,
            target=target,
            normalize_to_top_strand_nick=normalize_to_top_strand_nick,
            max_results=max_results,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_target_search_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


@app.command(
    "released-design",
    help="Materialize one explicit released-product snapback bundle from a two-stage precursor spec.",
)
def released_design_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.released.snapback.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing released-product workspace output root.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the released-product report as JSON."),
) -> None:
    try:
        validation_report = validate_released_snapback_spec(spec)
        if validation_report.status == "invalid_catalog":
            if json_output:
                typer.echo(validation_report.model_dump_json(indent=2))
            else:
                _print_released_report(validation_report)
            raise typer.Exit(code=1)
        run_dir, report = run_released_snapback_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Released-product outputs -> {run_dir}")
        _print_released_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command(
    "released-target-search",
    help=(
        "Search paired nickase plus release-enzyme combinations for exposed-bottom snapback targets. "
        "This lane evaluates the final geometry on the exposed post-release bottom strand."
    ),
)
def released_target_search_cmd(
    nick_preset: str | None = typer.Option(
        None,
        "--nick-preset",
        help="Primary builtin nickase preset. Explicit nickase sources are required for hermetic runs.",
    ),
    nick_additional_preset: list[str] = typer.Option(
        [],
        "--nick-additional-preset",
        help="Additional builtin nickase preset ids (repeatable).",
    ),
    nick_additional_path: list[Path] = typer.Option(
        [],
        "--nick-additional-path",
        help="Additional workspace-relative nickase catalog overlays (repeatable).",
    ),
    release_preset: str | None = typer.Option(
        None,
        "--release-preset",
        help="Primary builtin release-enzyme preset. Explicit release-enzyme sources are required for hermetic runs.",
    ),
    release_additional_preset: list[str] = typer.Option(
        [],
        "--release-additional-preset",
        help="Additional builtin release-enzyme preset ids (repeatable).",
    ),
    release_additional_path: list[Path] = typer.Option(
        [],
        "--release-additional-path",
        help="Additional workspace-relative release-enzyme catalog overlays (repeatable).",
    ),
    workspace_root: Path = typer.Option(
        Path("."),
        "--workspace-root",
        help="Workspace root for resolving additional catalog paths.",
    ),
    nick_boundary: int = typer.Option(0, "--nick-boundary", help="Requested nick_boundary_from_left."),
    paired_bp: int = typer.Option(3, "--paired-bp", help="Requested retained homology length."),
    cap_nt: int = typer.Option(3, "--cap-nt", help="Requested effective cap nt. Must equal 3."),
    max_results: int = typer.Option(8, "--max-results", help="Maximum exact or near hits to return."),
    near_boundary_search_limit: int = typer.Option(
        8,
        "--near-boundary-search-limit",
        help=(
            "How many boundary offsets on either side of the target to probe per pair "
            "when the exact target is unavailable."
        ),
    ),
    allow_demo_hits: bool = typer.Option(
        False,
        "--allow-demo-hits",
        help="Allow hits from entries explicitly marked demo_only in the resolved catalogs.",
    ),
    allow_frequent_cutter_nickases: bool = typer.Option(
        False,
        "--allow-frequent-cutter-nickases",
        help="Allow nickases flagged FREQUENT_CUTTER in the released-product lane.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the released target-search report as JSON."),
) -> None:
    try:
        from dnadesign.cruncher.snapback.models import CatalogSources
        from dnadesign.cruncher.snapback.released_models import (
            ReleaseCatalogSources,
            ReleasedFinalTargetGeometry,
            ReleasedTargetSearchConfig,
            SingleNickReleasedTargetSearchRequest,
        )

        resolved_workspace_root = workspace_root.expanduser().resolve()
        if nick_preset is None and not nick_additional_preset and not nick_additional_path:
            raise ValueError(
                "released-target-search requires at least one explicit nickase source "
                "(--nick-preset, --nick-additional-preset, or --nick-additional-path)."
            )
        if release_preset is None and not release_additional_preset and not release_additional_path:
            raise ValueError(
                "released-target-search requires at least one explicit release-enzyme source "
                "(--release-preset, --release-additional-preset, or --release-additional-path)."
            )
        request = SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(
                nick_boundary_from_left=nick_boundary,
                paired_bp=paired_bp,
                cap_nt=cap_nt,
            ),
            nick_sources=CatalogSources(
                preset=nick_preset,
                additional_presets=nick_additional_preset,
                additional_paths=nick_additional_path,
            ),
            release_sources=ReleaseCatalogSources(
                preset=release_preset,
                additional_presets=release_additional_preset,
                additional_paths=release_additional_path,
            ),
            search=ReleasedTargetSearchConfig(
                max_results=max_results,
                near_boundary_search_limit=near_boundary_search_limit,
                allow_demo_hits=allow_demo_hits,
                disallowed_nickase_warning_codes=[] if allow_frequent_cutter_nickases else ["FREQUENT_CUTTER"],
            ),
        )
        report = run_released_snapback_target_search(
            request=request,
            workspace_root=resolved_workspace_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_released_target_search_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


@app.command(
    "released-solve",
    help=(
        "Search the released-product dual-enzyme catalog space, materialize the top exposed-bottom hits, "
        "and optionally render one plot per hit."
    ),
)
def released_solve_cmd(
    nick_preset: str | None = typer.Option(
        None,
        "--nick-preset",
        help="Primary builtin nickase preset. Explicit nickase sources are required for hermetic runs.",
    ),
    nick_additional_preset: list[str] = typer.Option(
        [],
        "--nick-additional-preset",
        help="Additional builtin nickase preset ids (repeatable).",
    ),
    nick_additional_path: list[Path] = typer.Option(
        [],
        "--nick-additional-path",
        help="Additional workspace-relative nickase catalog overlays (repeatable).",
    ),
    release_preset: str | None = typer.Option(
        None,
        "--release-preset",
        help="Primary builtin release-enzyme preset. Explicit release-enzyme sources are required for hermetic runs.",
    ),
    release_additional_preset: list[str] = typer.Option(
        [],
        "--release-additional-preset",
        help="Additional builtin release-enzyme preset ids (repeatable).",
    ),
    release_additional_path: list[Path] = typer.Option(
        [],
        "--release-additional-path",
        help="Additional workspace-relative release-enzyme catalog overlays (repeatable).",
    ),
    workspace_root: Path = typer.Option(
        Path("."),
        "--workspace-root",
        help="Workspace root for resolving additional catalog paths.",
    ),
    nick_boundary: int = typer.Option(0, "--nick-boundary", help="Requested nick_boundary_from_left."),
    paired_bp: int = typer.Option(3, "--paired-bp", help="Requested retained homology length."),
    cap_nt: int = typer.Option(3, "--cap-nt", help="Requested effective cap nt. Must equal 3."),
    max_results: int = typer.Option(8, "--max-results", help="Maximum hits to retain from the search report."),
    near_boundary_search_limit: int = typer.Option(
        8,
        "--near-boundary-search-limit",
        help="How many boundary offsets on either side of the target to probe per pair when exact hits are absent.",
    ),
    materialize_top_k: int = typer.Option(
        8,
        "--materialize-top-k",
        help="Maximum number of ranked hits to materialize as released-product hit bundles.",
    ),
    run_dir: Path = typer.Option(
        Path("outputs/released_solve"),
        "--run-dir",
        help="Workspace-relative output root for the released solve bundle.",
    ),
    render_format: str = typer.Option(
        "pdf",
        "--render-format",
        help="Per-hit render format when plots are emitted.",
    ),
    emit_renders: bool = typer.Option(
        False,
        "--emit-renders",
        help="Render the per-hit origin-anchored exposed-bottom plots after materialization.",
    ),
    allow_frequent_cutter_nickases: bool = typer.Option(
        False,
        "--allow-frequent-cutter-nickases",
        help="Allow nickases flagged FREQUENT_CUTTER in the released-product lane.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing released solve output root.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the released solve report as JSON."),
) -> None:
    try:
        from dnadesign.cruncher.snapback.models import CatalogSources
        from dnadesign.cruncher.snapback.released_models import (
            ReleaseCatalogSources,
            ReleasedFinalTargetGeometry,
            ReleasedSolveOutputConfig,
            ReleasedTargetSearchConfig,
            SingleNickReleasedTargetSearchRequest,
        )

        resolved_workspace_root = workspace_root.expanduser().resolve()
        if nick_preset is None and not nick_additional_preset and not nick_additional_path:
            raise ValueError(
                "released-solve requires at least one explicit nickase source "
                "(--nick-preset, --nick-additional-preset, or --nick-additional-path)."
            )
        if release_preset is None and not release_additional_preset and not release_additional_path:
            raise ValueError(
                "released-solve requires at least one explicit release-enzyme source "
                "(--release-preset, --release-additional-preset, or --release-additional-path)."
            )
        request = SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(
                nick_boundary_from_left=nick_boundary,
                paired_bp=paired_bp,
                cap_nt=cap_nt,
            ),
            nick_sources=CatalogSources(
                preset=nick_preset,
                additional_presets=nick_additional_preset,
                additional_paths=nick_additional_path,
            ),
            release_sources=ReleaseCatalogSources(
                preset=release_preset,
                additional_presets=release_additional_preset,
                additional_paths=release_additional_path,
            ),
            search=ReleasedTargetSearchConfig(
                max_results=max(max_results, materialize_top_k),
                near_boundary_search_limit=near_boundary_search_limit,
                disallowed_nickase_warning_codes=[] if allow_frequent_cutter_nickases else ["FREQUENT_CUTTER"],
            ),
        )
        output = ReleasedSolveOutputConfig(
            run_dir=run_dir,
            materialize_top_k=materialize_top_k,
            render_format=render_format,
            emit_renders=emit_renders,
        )
        resolved_run_dir, report = run_released_snapback_solve(
            request=request,
            output=output,
            workspace_root=resolved_workspace_root,
            force_overwrite=force_overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Released-product solve outputs -> {resolved_run_dir}")
        _print_released_solve_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


@app.command(
    "released-show",
    help="Read a released-product snapback bundle and print a path-oriented summary with drift checks.",
)
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


@app.command("show", help="Read a snapback design or solve workspace output root and print a path-oriented summary.")
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
