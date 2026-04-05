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
from typing import Any, Mapping

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


def render_yiu_spec_outcome(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.render import render_yiu_spec_outcome as _render_yiu_spec_outcome

    return _render_yiu_spec_outcome(*args, **kwargs)


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


def _row_payload(row: object) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")
    raise TypeError(f"unsupported split-row debug payload: {type(row)!r}")


def _print_split_row_debug(rows: list[object]) -> None:
    for row in rows:
        payload = _row_payload(row)
        console.print(
            "Split row -> "
            f"{payload['fragment_side']} "
            f"selected_sticky_end={payload['selected_sticky_end_sequence_5to3']} "
            f"canonical_sticky_end={payload['canonical_sticky_end_sequence_5to3']} "
            f"ghost_excised_context={payload['ghost_excised_context'] is not None}"
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
        outcome = render_yiu_spec_outcome(spec, force_overwrite=force_overwrite, emit_renders=emit_renders)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    payload = outcome.model_dump(mode="json")
    report = outcome.report
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"YIU bundle -> {outcome.bundle_dir}")
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
    console.print(f"Bundle write -> {outcome.bundle_dir}")
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
        outcome = show_yiu_bundle(bundle, verbose=verbose)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(outcome.model_dump(mode="json", exclude_unset=True), indent=2))
        return
    console.print(f"Bundle -> {outcome.bundle_dir}")
    console.print(f"Bundle contract -> {outcome.bundle_contract}")
    console.print(f"Provenance -> {json.dumps(outcome.provenance, sort_keys=True)}")
    _print_payload_summary(
        payload_label=outcome.payload_label,
        input_kind=outcome.input_kind,
        payload_length=outcome.payload_length,
        junction=outcome.junction.model_dump(mode="json"),
        mismatch_sites=[entry.model_dump(mode="json") for entry in outcome.mismatches],
        pwm_mode=outcome.pwm_mode,
        pwm_effective=outcome.pwm_effective,
        worst_loss=outcome.worst_loss,
        total_loss=outcome.total_loss,
    )
    fallback_reason = outcome.motif_context.fallback_reason
    if outcome.pwm_mode != "none" and not outcome.pwm_effective and fallback_reason:
        console.print(f"PWM fallback reason -> {fallback_reason}")
    if verbose:
        _print_split_row_debug(outcome.split_row_debug)
    console.print(f"Views -> {', '.join(outcome.view_ids)}")
    console.print(f"Render status -> {outcome.render_status}")
    console.print(f"Available renders -> {len(outcome.available_renders)}")
    console.print(f"Integrity -> {outcome.integrity.status}")
    if outcome.composite_render_artifact_path is not None:
        console.print(f"Composite render -> {outcome.composite_render_artifact_path}")
    if outcome.published_plot_artifact_path is not None:
        console.print(f"Published plot -> {outcome.published_plot_artifact_path}")
    console.print(f"Bundle manifest -> {outcome.bundle_manifest_path}")
    console.print(f"Normalized payload -> {outcome.normalized_payload_path}")
    console.print(f"Visual inventory -> {outcome.visual_inventory_path}")
