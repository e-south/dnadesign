"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/yiu_presenter.py

Presentation helpers for the YIU CLI surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console

from dnadesign.cruncher.cli.paths import render_path


def mismatch_summary_text(mismatch_sites: list[dict[str, object]]) -> str:
    return ", ".join(
        f"idx={site['payload_index']} off={site['junction_offset']} "
        f"{site['mutated_strand']} {site['native_base']}->{site['mutated_base']} "
        f"(opp={site['opposing_base']})"
        for site in mismatch_sites
    )


def print_payload_summary(
    console: Console,
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
        console.print(f"Mismatch sites -> {mismatch_summary_text(mismatch_sites)}")
    console.print(f"PWM -> mode={pwm_mode} effective={pwm_effective}")
    if pwm_effective:
        console.print(f"PWM losses -> worst={worst_loss:.6f} total={total_loss:.6f}")


def row_payload(row: object) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")
    raise TypeError(f"unsupported split-row debug payload: {type(row)!r}")


def print_split_row_debug(console: Console, rows: list[object]) -> None:
    for row in rows:
        payload = row_payload(row)
        console.print(
            "Split row -> "
            f"{payload['fragment_side']} "
            f"payload_body_5to3={payload['payload_body_sequence_5to3']} "
            f"display_body={payload['display_payload_body_sequence_5to3']} "
            f"selected_sticky_end={payload['selected_sticky_end_sequence_5to3']} "
            f"canonical_sticky_end={payload['canonical_sticky_end_sequence_5to3']} "
            f"ghost_excised_context={payload['ghost_excised_context'] is not None}"
        )


def build_show_json_payload(outcome: object, *, verbose: bool) -> dict[str, Any]:
    exclude: set[str] = set()
    if not verbose:
        exclude.update({"motif_context", "optimization_decision", "split_row_debug"})
    return outcome.model_dump(mode="json", exclude=exclude, exclude_unset=True)


def print_path_detail(console: Console, label: str, path: str | None, *, base: Path | None = None) -> None:
    console.print(f"{label} -> {render_path(path, base=base)}", soft_wrap=True)


def print_validation_report(console: Console, report: object) -> None:
    console.print(f"Spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    print_payload_summary(
        console,
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
    console.print("Bundle write -> no")


def print_render_outcome(console: Console, outcome: object, *, emit_renders: bool) -> None:
    report = outcome.report
    console.print(f"YIU bundle -> {outcome.bundle_dir}")
    console.print(f"Spec -> {report.spec_name}")
    print_payload_summary(
        console,
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
    console.print(f"Bundle summary -> {outcome.bundle_summary_path}")
    console.print(f"Bundle manifest -> {outcome.bundle_manifest_path}")
    console.print(f"Normalized payload -> {outcome.normalized_payload_path}")
    console.print(f"Visual inventory -> {outcome.visual_inventory_path}")
    if emit_renders:
        console.print(f"Composite render target -> {outcome.composite_render_artifact_path}")
    if outcome.published_plot_artifact_path is not None:
        console.print(f"Published plot -> {outcome.published_plot_artifact_path}")


def print_show_outcome(console: Console, outcome: object, *, verbose: bool) -> None:
    sequence_summary = outcome.bundle_summary.sequence_summary
    split_summary = sequence_summary.split_payload
    bundle_base = Path(outcome.bundle_dir)
    print_path_detail(console, "Bundle", outcome.bundle_dir)
    console.print(f"Bundle contract -> {outcome.bundle_contract}")
    console.print(f"Provenance -> {json.dumps(outcome.provenance, sort_keys=True)}")
    print_payload_summary(
        console,
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
    console.print(f"Payload 5' -> 3' -> {sequence_summary.selected_payload_sequence_5to3}")
    console.print(
        "Split payload 5' -> 3' -> "
        f"left={split_summary.left_payload_body_sequence_5to3} "
        f"sticky={split_summary.selected_sticky_end_sequence_5to3} "
        f"right={split_summary.right_payload_body_sequence_5to3}"
    )
    console.print(f"Reference sticky end 5' -> 3' -> {split_summary.canonical_sticky_end_sequence_5to3}")
    fallback_reason = outcome.motif_context.fallback_reason
    if outcome.pwm_mode != "none" and not outcome.pwm_effective and fallback_reason:
        console.print(f"PWM fallback reason -> {fallback_reason}")
    if verbose:
        print_split_row_debug(console, outcome.split_row_debug)
    console.print(f"Views -> {', '.join(outcome.view_ids)}")
    console.print(f"Render status -> {outcome.render_status}")
    console.print(f"Available renders -> {len(outcome.available_renders)}")
    console.print(f"Integrity -> {outcome.integrity.status}")
    if outcome.composite_render_artifact_path is not None:
        print_path_detail(console, "Composite render", outcome.composite_render_artifact_path, base=bundle_base)
    if outcome.published_plot_artifact_path is not None:
        print_path_detail(console, "Published plot", outcome.published_plot_artifact_path, base=bundle_base)
    print_path_detail(console, "Bundle summary", outcome.bundle_summary_path, base=bundle_base)
    print_path_detail(console, "Bundle manifest", outcome.bundle_manifest_path, base=bundle_base)
    print_path_detail(console, "Normalized payload", outcome.normalized_payload_path, base=bundle_base)
    print_path_detail(console, "Visual inventory", outcome.visual_inventory_path, base=bundle_base)


__all__ = [
    "build_show_json_payload",
    "print_render_outcome",
    "print_show_outcome",
    "print_split_row_debug",
    "print_validation_report",
    "print_payload_summary",
]
