"""
Released-product Snapback CLI commands.
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.app.snapback_cli_requests import (
    build_released_solve_invocation,
    build_released_target_search_invocation,
)
from dnadesign.cruncher.cli.commands.snapback_presenters import (
    console,
    print_released_report,
    print_released_solve_report,
    print_released_target_search_report,
)
from dnadesign.cruncher.cli.commands.snapback_services import (
    run_released_snapback_design,
    run_released_snapback_solve,
    run_released_snapback_target_search,
    validate_released_snapback_spec,
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
                print_released_report(validation_report)
            raise typer.Exit(code=1)
        run_dir, report = run_released_snapback_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Released-product outputs -> {run_dir}")
        print_released_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


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
    allow_top_active_routes: bool = typer.Option(
        False,
        "--allow-top-active-routes",
        help=(
            "Opt into broader retained-active audits where the active post-release strand may come from the top strand."
        ),
    ),
    allow_precut_footprint_outside_active_product: bool = typer.Option(
        False,
        "--allow-precut-footprint-outside-active-product",
        help=(
            "Preserve full vendor nickase footprints during retained-active audits and allow pre-cut "
            "site bases to sit outside the final active product."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the released target-search report as JSON."),
) -> None:
    try:
        invocation = build_released_target_search_invocation(
            nick_preset=nick_preset,
            nick_additional_preset=nick_additional_preset,
            nick_additional_path=nick_additional_path,
            release_preset=release_preset,
            release_additional_preset=release_additional_preset,
            release_additional_path=release_additional_path,
            workspace_root=workspace_root,
            nick_boundary=nick_boundary,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
            max_results=max_results,
            near_boundary_search_limit=near_boundary_search_limit,
            allow_demo_hits=allow_demo_hits,
            allow_frequent_cutter_nickases=allow_frequent_cutter_nickases,
            allow_top_active_routes=allow_top_active_routes,
            allow_precut_footprint_outside_active_product=allow_precut_footprint_outside_active_product,
        )
        report = run_released_snapback_target_search(
            request=invocation.request,
            workspace_root=invocation.workspace_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_released_target_search_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


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
        help="Render the per-hit origin-anchored retained-active-strand plots after materialization.",
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
    allow_top_active_routes: bool = typer.Option(
        False,
        "--allow-top-active-routes",
        help=(
            "Opt into broader retained-active audits where the active post-release strand may come from the top strand."
        ),
    ),
    allow_precut_footprint_outside_active_product: bool = typer.Option(
        False,
        "--allow-precut-footprint-outside-active-product",
        help=(
            "Preserve full vendor nickase footprints during retained-active audits and allow pre-cut "
            "site bases to sit outside the final active product."
        ),
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing released solve output root.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the released solve report as JSON."),
) -> None:
    try:
        invocation = build_released_solve_invocation(
            nick_preset=nick_preset,
            nick_additional_preset=nick_additional_preset,
            nick_additional_path=nick_additional_path,
            release_preset=release_preset,
            release_additional_preset=release_additional_preset,
            release_additional_path=release_additional_path,
            workspace_root=workspace_root,
            nick_boundary=nick_boundary,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
            max_results=max_results,
            near_boundary_search_limit=near_boundary_search_limit,
            materialize_top_k=materialize_top_k,
            run_dir=run_dir,
            render_format=render_format,
            emit_renders=emit_renders,
            allow_demo_hits=allow_demo_hits,
            allow_frequent_cutter_nickases=allow_frequent_cutter_nickases,
            allow_top_active_routes=allow_top_active_routes,
            allow_precut_footprint_outside_active_product=allow_precut_footprint_outside_active_product,
        )
        resolved_run_dir, report = run_released_snapback_solve(
            request=invocation.request,
            output=invocation.output,
            workspace_root=invocation.workspace_root,
            force_overwrite=force_overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Released-product solve outputs -> {resolved_run_dir}")
        print_released_solve_report(report)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


__all__ = [
    "released_design_cmd",
    "released_solve_cmd",
    "released_target_search_cmd",
]
