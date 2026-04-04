"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/yiu.py

CLI entrypoints for the payload-centric YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, render, and inspect payload-centric YIU workflows.",
)
console = Console()


def validate_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.validate import validate_yiu_spec as _validate_yiu_spec

    return _validate_yiu_spec(*args, **kwargs)


def render_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.render import render_yiu_spec as _render_yiu_spec

    return _render_yiu_spec(*args, **kwargs)


def show_yiu_bundle(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.show import show_yiu_bundle as _show_yiu_bundle

    return _show_yiu_bundle(*args, **kwargs)


def init_yiu_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import init_yiu_workspace as _init_yiu_workspace

    return _init_yiu_workspace(*args, **kwargs)


def yiu_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import yiu_workspace_path as _yiu_workspace_path

    return _yiu_workspace_path(*args, **kwargs)


def _mismatch_summary_text(mismatch_sites: list[dict[str, object]]) -> str:
    return ", ".join(
        f"idx={site['payload_index']} off={site['junction_offset']} "
        f"{site['mutated_strand']} {site['native_base']}->{site['mutated_base']} "
        f"(opp={site['opposing_base']})"
        for site in mismatch_sites
    )


def _print_payload_summary(
    *,
    payload_label: str | None,
    input_kind: str,
    payload_length: int,
    junction: dict[str, object],
    mismatch_sites: list[dict[str, object]],
    pwm_mode: str,
    pwm_effective: bool,
    worst_loss: float,
    total_loss: float,
) -> None:
    if payload_label:
        console.print(f"Payload label -> {payload_label}")
    console.print(f"Input kind -> {input_kind}")
    console.print(f"Payload length -> {payload_length}")
    console.print(f"Junction window -> start={junction['start']} end={junction['end']} mode={junction['mode']}")
    console.print(f"Mismatch count -> {len(mismatch_sites)}")
    if mismatch_sites:
        console.print(f"Mismatch sites -> {_mismatch_summary_text(mismatch_sites)}")
    console.print(f"PWM -> mode={pwm_mode} effective={pwm_effective}")
    if pwm_effective:
        console.print(f"PWM losses -> worst={worst_loss:.6f} total={total_loss:.6f}")


def _print_split_row_debug(rows: list[dict[str, object]]) -> None:
    for row in rows:
        console.print(
            "Split row -> "
            f"{row['fragment_side']} "
            f"selected_sticky_end={row['selected_sticky_end_sequence_5to3']} "
            f"canonical_sticky_end={row['canonical_sticky_end_sequence_5to3']} "
            f"ghost_excised_context={row['ghost_excised_context'] is not None}"
        )


@app.command("init-workspace", help="Scaffold a payload-centric YIU workspace.")
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
        help="Explicit YIU workspace root to create. Overrides WORKSPACE/--root.",
    ),
    force_overwrite: bool = typer.Option(False, "--force-overwrite", help="Replace an existing workspace root."),
) -> None:
    try:
        if output is not None and workspace is not None:
            raise typer.BadParameter("Use either WORKSPACE [--root] or --output, not both.")
        if output is not None and root is not None:
            raise typer.BadParameter("Use either --output or --root, not both.")
        if output is None and workspace is None:
            raise typer.BadParameter("Provide WORKSPACE or --output.")
        target_root = output if output is not None else yiu_workspace_path(workspace, root=root)
        result = init_yiu_workspace(target_root, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"YIU workspace scaffold -> {result.workspace_root}")
    console.print(f"Runbook -> {result.runbook_path}")
    console.print(f"Runbook doc -> {result.runbook_doc_path}")
    console.print(f"Spec -> {result.spec_path}")


@app.command("validate", help="Validate a payload-centric YIU spec and print the normalized payload summary.")
def validate_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    json_output: bool = typer.Option(False, "--json", help="Print the validation report as JSON."),
) -> None:
    try:
        report = validate_yiu_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return
    report_payload = report.model_dump(mode="json")
    console.print(f"Spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    _print_payload_summary(
        payload_label=report_payload.get("payload_label"),
        input_kind=report.input_kind,
        payload_length=report.payload_length,
        junction=report_payload["junction"],
        mismatch_sites=report_payload["mismatches"],
        pwm_mode=report_payload["pwm_mode"],
        pwm_effective=report_payload["pwm_effective"],
        worst_loss=report_payload["worst_loss"],
        total_loss=report_payload["total_loss"],
    )
    console.print("Bundle write -> no")
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("render", help="Validate a payload-centric YIU spec, publish its bundle, and optionally render its views.")
def render_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic bundle directory."
    ),
    emit_renders: bool = typer.Option(
        False,
        "--emit-renders",
        help="Immediately render every published BaseRender job after the bundle is written.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the render report as JSON."),
) -> None:
    try:
        bundle_dir, report = render_yiu_spec(spec, force_overwrite=force_overwrite, emit_renders=emit_renders)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    manifest_payload = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    published_plot_artifact_path = manifest_payload.get("published_plot_artifact_path")
    if isinstance(published_plot_artifact_path, str) and published_plot_artifact_path.strip():
        from dnadesign.cruncher.yiu.integrity import resolve_workspace_root

        workspace_root = resolve_workspace_root(bundle_dir)
        published_plot_artifact_path = (
            None if workspace_root is None else str((workspace_root / published_plot_artifact_path).resolve())
        )
    else:
        published_plot_artifact_path = None
    payload = {
        "bundle_dir": str(bundle_dir),
        "outputs_root": str(bundle_dir.parent.resolve()),
        "composite_render_artifact_path": str((bundle_dir / "payload_views.pdf").resolve()),
        "published_plot_artifact_path": published_plot_artifact_path,
        "bundle_manifest_path": str((bundle_dir / "bundle_manifest.json").resolve()),
        "normalized_payload_path": str((bundle_dir / "normalized_payload.json").resolve()),
        "visual_inventory_path": str((bundle_dir / "visual_inventory.json").resolve()),
        "report": report.model_dump(mode="json"),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"YIU bundle -> {bundle_dir}")
    console.print(f"Spec -> {report.spec_name}")
    _print_payload_summary(
        payload_label=report.payload_label,
        input_kind=report.input_kind,
        payload_length=report.payload_length,
        junction=report.junction.model_dump(mode="json"),
        mismatch_sites=[entry.model_dump(mode="json") for entry in report.mismatches],
        pwm_mode=report.pwm_mode,
        pwm_effective=report.pwm_effective,
        worst_loss=report.worst_loss,
        total_loss=report.total_loss,
    )
    console.print(f"Bundle write -> {bundle_dir}")
    console.print(f"Bundle manifest -> {payload['bundle_manifest_path']}")
    console.print(f"Normalized payload -> {payload['normalized_payload_path']}")
    console.print(f"Visual inventory -> {payload['visual_inventory_path']}")
    if emit_renders:
        console.print(f"Composite render target -> {payload['composite_render_artifact_path']}")
    if payload["published_plot_artifact_path"] is not None:
        console.print(f"Published plot -> {payload['published_plot_artifact_path']}")
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("show", help="Show payload-centric summary for one YIU bundle.")
def show_cmd(
    bundle: Path = typer.Option(..., "--bundle", help="Path to a published YIU payload bundle."),
    json_output: bool = typer.Option(False, "--json", help="Print the normalized bundle summary as JSON."),
    verbose: bool = typer.Option(False, "--verbose", help="Include split-row debug details in the output."),
) -> None:
    try:
        payload = show_yiu_bundle(bundle, verbose=verbose)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Bundle -> {payload['bundle_dir']}")
    console.print(f"Bundle contract -> {payload['bundle_contract']}")
    console.print(f"Provenance -> {json.dumps(payload['provenance'], sort_keys=True)}")
    _print_payload_summary(
        payload_label=payload.get("payload_label"),
        input_kind=payload["input_kind"],
        payload_length=payload["payload_length"],
        junction=payload["junction"],
        mismatch_sites=payload.get("mismatches", []),
        pwm_mode=payload["pwm_mode"],
        pwm_effective=payload["pwm_effective"],
        worst_loss=payload["worst_loss"],
        total_loss=payload["total_loss"],
    )
    motif_context = payload.get("motif_context", {})
    fallback_reason = motif_context.get("fallback_reason")
    if payload["pwm_mode"] != "none" and not payload["pwm_effective"] and fallback_reason:
        console.print(f"PWM fallback reason -> {fallback_reason}")
    if verbose:
        _print_split_row_debug(payload.get("split_row_debug", []))
    console.print(f"Views -> {', '.join(payload['view_ids'])}")
    console.print(f"Render status -> {payload['render_status']}")
    console.print(f"Available renders -> {len(payload['available_renders'])}")
    console.print(f"Integrity -> {payload['integrity']['status']}")
    if payload.get("composite_render_artifact_path") is not None:
        console.print(f"Composite render -> {payload['composite_render_artifact_path']}")
    if payload.get("published_plot_artifact_path") is not None:
        console.print(f"Published plot -> {payload['published_plot_artifact_path']}")
    console.print(f"Bundle manifest -> {payload['bundle_manifest_path']}")
    console.print(f"Normalized payload -> {payload['normalized_payload_path']}")
    console.print(f"Visual inventory -> {payload['visual_inventory_path']}")
