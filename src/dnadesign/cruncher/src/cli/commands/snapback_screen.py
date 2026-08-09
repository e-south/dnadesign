"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/snapback_screen.py

CLI command for the released-product Snapback screen objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.cli.commands.snapback_presenters import console, print_snapback_screen_report
from dnadesign.cruncher.cli.commands.snapback_services import (
    build_snapback_screen_request,
    parse_retained_product_strands,
    run_snapback_screen,
)


def screen_cmd(
    workspace_root: Path = typer.Option(
        Path("."),
        "--workspace-root",
        help="Workspace root for resolving catalog overlays and recording report context.",
    ),
    target_origin: int = typer.Option(
        0,
        "--target-origin",
        help="Logical origin boundary in the retained-product snapback frame.",
    ),
    stem_bp: int = typer.Option(3, "--stem-bp", help="Requested logical stem base pairs."),
    cap_nt: int = typer.Option(3, "--cap-nt", help="Requested effective cap nt. Must equal 3."),
    nick_preset: str | None = typer.Option(
        "neb_nicking_v1",
        "--nick-preset",
        help="Primary builtin nickase preset.",
    ),
    nick_additional_preset: list[str] = typer.Option(
        ["thermo_nicking_v1"],
        "--nick-additional-preset",
        help="Additional builtin nickase preset ids (repeatable).",
    ),
    nick_additional_path: list[Path] = typer.Option(
        [],
        "--nick-additional-path",
        help="Additional workspace-relative nickase catalog overlays (repeatable).",
    ),
    release_preset: str | None = typer.Option(
        "type_iis_release_v1",
        "--release-preset",
        help="Primary builtin release-enzyme preset.",
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
    release_variant_id: list[str] = typer.Option(
        ["BspQI"],
        "--release-variant-id",
        help="Release-enzyme variant id to include in the screen (repeatable). Defaults to BspQI.",
    ),
    allow_retained_strands: str = typer.Option(
        "top,bottom",
        "--allow-retained-strands",
        help="Comma-separated retained active product strands to evaluate: top, bottom, or top,bottom.",
    ),
    use_vendor_footprints: bool = typer.Option(
        True,
        "--use-vendor-footprints/--no-use-vendor-footprints",
        help="Allow oriented vendor nickase footprint bases to sit outside the final active product.",
    ),
    max_results: int = typer.Option(16, "--max-results", help="Maximum exact or near hits to return."),
    near_boundary_search_limit: int = typer.Option(
        8,
        "--near-boundary-search-limit",
        help="How many boundary offsets on either side of the target to probe when exact hits are absent.",
    ),
    allow_demo_hits: bool = typer.Option(
        False,
        "--allow-demo-hits",
        help="Allow hits from entries explicitly marked demo_only in the resolved catalogs.",
    ),
    allow_frequent_cutter_nickases: bool = typer.Option(
        False,
        "--allow-frequent-cutter-nickases",
        help="Allow nickases flagged FREQUENT_CUTTER in the screen.",
    ),
    emit_mechanism_ledger: bool = typer.Option(
        True,
        "--emit-mechanism-ledger/--no-mechanism-ledger",
        help="Print the mechanism ledger in text output. JSON output always includes it.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the screen report as JSON."),
) -> None:
    try:
        retained_product_strands = parse_retained_product_strands(allow_retained_strands)
        request = build_snapback_screen_request(
            target_origin=target_origin,
            stem_bp=stem_bp,
            cap_nt=cap_nt,
            nick_preset=nick_preset,
            nick_additional_presets=nick_additional_preset,
            nick_additional_paths=nick_additional_path,
            release_preset=release_preset,
            release_additional_presets=release_additional_preset,
            release_additional_paths=release_additional_path,
            release_variant_ids=release_variant_id,
            retained_product_strands=retained_product_strands,
            use_vendor_footprints=use_vendor_footprints,
            max_results=max_results,
            near_boundary_search_limit=near_boundary_search_limit,
            allow_demo_hits=allow_demo_hits,
            allow_frequent_cutter_nickases=allow_frequent_cutter_nickases,
        )
        report = run_snapback_screen(
            request=request,
            workspace_root=workspace_root.expanduser().resolve(),
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_snapback_screen_report(report, emit_mechanism_ledger=emit_mechanism_ledger)
    if report.status == "no_hits":
        raise typer.Exit(code=1)


__all__ = ["screen_cmd"]
