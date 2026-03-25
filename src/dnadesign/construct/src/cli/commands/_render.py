"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/_render.py

Shared CLI rendering helpers for construct command surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...config import JobConfig
from ...runtime import PreflightResult, RunResult
from ._format import echo_json


def validate_result_payload(
    *,
    config_path: Path,
    loaded: JobConfig,
    preflight: PreflightResult | None,
) -> dict[str, object]:
    return {
        "status": "ok",
        "config_path": config_path,
        "job": {
            "id": loaded.job.id,
            "input_dataset": loaded.job.input.source.dataset,
            "output_dataset": loaded.job.output.target.dataset,
        },
        "runtime_preflight": preflight,
    }


def echo_validate_result(
    *,
    config_path: Path,
    loaded: JobConfig,
    preflight: PreflightResult | None,
    output_format: str = "text",
) -> None:
    if output_format == "json":
        echo_json(validate_result_payload(config_path=config_path, loaded=loaded, preflight=preflight))
        return
    typer.echo(f"Config OK: {config_path}")
    typer.echo(f"job_id: {loaded.job.id}")
    typer.echo(f"input_dataset: {loaded.job.input.source.dataset}")
    typer.echo(f"output_dataset: {loaded.job.output.target.dataset}")
    if preflight is None:
        return
    typer.echo(f"input_root: {preflight.input_root}")
    typer.echo(f"output_root: {preflight.output_root}")
    typer.echo(f"template_id: {preflight.template_id}")
    typer.echo(f"template_kind: {preflight.template_kind}")
    typer.echo(f"template_source: {preflight.template_source}")
    if preflight.template_dataset is not None:
        typer.echo(f"template_dataset: {preflight.template_dataset}")
    if preflight.template_field is not None:
        typer.echo(f"template_field: {preflight.template_field}")
    if preflight.template_record_id is not None:
        typer.echo(f"template_record_id: {preflight.template_record_id}")
    typer.echo(f"template_length: {preflight.template_length}")
    typer.echo(f"template_circular: {str(preflight.template_circular).lower()}")
    typer.echo(f"template_sha256: {preflight.template_sha256}")
    typer.echo(f"realize_mode: {preflight.realize_mode}")
    typer.echo(f"focal_part: {preflight.focal_part or ''}")
    typer.echo(f"window_semantics: {preflight.window_semantics or ''}")
    typer.echo(f"window_reference: {preflight.window_reference or ''}")
    typer.echo(f"window_direction: {preflight.window_direction or ''}")
    typer.echo(f"window_size_bp: {preflight.window_size_bp if preflight.window_size_bp is not None else ''}")
    typer.echo(
        f"window_upstream_bp: {preflight.window_upstream_bp if preflight.window_upstream_bp is not None else ''}"
    )
    typer.echo(
        f"window_downstream_bp: {preflight.window_downstream_bp if preflight.window_downstream_bp is not None else ''}"
    )
    typer.echo(f"window_offset_bp: {preflight.window_offset_bp if preflight.window_offset_bp is not None else ''}")
    typer.echo(f"spec_id: {preflight.spec_id}")
    typer.echo(f"output_on_conflict: {preflight.output_on_conflict}")
    typer.echo(f"existing_output_collisions: {preflight.existing_output_collisions}")
    for placement in preflight.placements:
        typer.echo(
            "placement: "
            f"part={placement.part_name} "
            f"role={placement.part_role} "
            f"sequence_source={placement.sequence_source} "
            f"sequence_field={placement.sequence_field or ''} "
            f"kind={placement.placement_kind} "
            f"template_start={placement.template_start} "
            f"template_end={placement.template_end} "
            f"template_span_bp={placement.template_span_bp} "
            f"orientation={placement.orientation} "
            f"locator_kind={placement.locator_kind} "
            f"locator_upstream_sequence={placement.locator_upstream_sequence or ''} "
            f"locator_downstream_sequence={placement.locator_downstream_sequence or ''} "
            f"guard_mode={placement.guard_mode} "
            f"guard_require_unique_forward_matches={str(placement.guard_require_unique_forward_matches).lower()} "
            "guard_replaced_span_bp="
            f"{placement.guard_replaced_span_bp if placement.guard_replaced_span_bp is not None else ''} "
            f"template_sequence={placement.template_sequence} "
            f"guard_replaced_sequence={placement.guard_replaced_sequence or ''} "
            f"guard_upstream_sequence={placement.guard_upstream_sequence or ''} "
            f"observed_guard_upstream_sequence={placement.observed_guard_upstream_sequence or ''} "
            f"guard_downstream_sequence={placement.guard_downstream_sequence or ''} "
            f"observed_guard_downstream_sequence={placement.observed_guard_downstream_sequence or ''}"
        )
    typer.echo(f"rows_total: {preflight.records_total}")
    for row in preflight.planned_rows:
        typer.echo(
            "row: "
            f"input_id={row.input_id} "
            f"output_id={row.output_id} "
            f"input_length={row.input_length} "
            f"focal_part_length={row.focal_part_length if row.focal_part_length is not None else ''} "
            f"full_construct_length={row.full_construct_length} "
            f"output_length={row.output_length}"
        )


def run_result_payload(result: RunResult) -> dict[str, object]:
    return {
        "status": "ok",
        "run": result,
    }


def echo_run_result(result: RunResult, *, output_format: str = "text") -> None:
    if output_format == "json":
        echo_json(run_result_payload(result))
        return
    if result.dry_run:
        typer.echo(
            "Config validated (dry run): "
            f"job={result.job_id} rows={result.records_total} "
            f"output_collisions={result.records_skipped_existing}"
        )
        typer.echo(f"output_root: {result.output_root}")
        typer.echo(f"output_dataset: {result.output_dataset}")
        typer.echo(f"spec_id: {result.spec_id}")
        return

    typer.echo(
        "Construct run complete: "
        f"job={result.job_id} "
        f"rows_planned={result.records_total} "
        f"rows_written={result.records_written} "
        f"rows_skipped_existing={result.records_skipped_existing}"
    )
    typer.echo(f"output_root: {result.output_root}")
    typer.echo(f"output_dataset: {result.output_dataset}")
    typer.echo(f"spec_id: {result.spec_id}")
