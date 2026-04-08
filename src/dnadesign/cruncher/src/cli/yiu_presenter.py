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
from dnadesign.cruncher.yiu.mismatch_notation import compact_mismatch_notation_text


def mismatch_summary_text(mismatch_sites: list[dict[str, object]]) -> str:
    return compact_mismatch_notation_text(mismatch_sites)


def _ligation_state(ligation: object) -> str:
    explicit_state = getattr(ligation, "state", None)
    if explicit_state:
        return str(explicit_state)
    profile = getattr(ligation, "profile", None)
    awareness_mode = getattr(ligation, "awareness_mode", None)
    applied = bool(getattr(ligation, "applied", False))
    if profile == "none":
        return "legacy"
    if not applied or awareness_mode == "disabled":
        return "inert"
    position_classes = getattr(ligation, "position_classes", [])
    if position_classes and all(position == "middle" for position in position_classes):
        return "edge_blind"
    return "active"


def _ligation_state_note(ligation: object) -> str:
    state = _ligation_state(ligation)
    if state == "legacy":
        return "Legacy ranking is active because ligation_profile=none."
    if state == "inert":
        return "Ligation profile is configured, but ligation-aware scoring is disabled."
    if state == "edge_blind":
        return "Ligation-aware scoring is active, but edge-vs-middle comparison is unavailable in the winning plan."
    return "Ligation-aware scoring is active and edge-vs-middle comparison is available."


def _edge_comparison_available(ligation: object) -> bool:
    explicit_value = getattr(ligation, "edge_comparison_available", None)
    if explicit_value is not None:
        return bool(explicit_value)
    position_classes = getattr(ligation, "position_classes", [])
    return bool(position_classes) and any(position == "edge" for position in position_classes)


def _candidate_position_pool(ligation: object) -> str:
    positions = getattr(ligation, "candidate_positions", []) or []
    return "-" if not positions else ",".join(str(position) for position in positions)


def _bad_pattern_scope(ligation: object) -> str:
    return "tnna_like_only" if getattr(ligation, "bad_pattern_heuristics", False) else "disabled"


def _trace_summary_text(source: object) -> str | None:
    trace = getattr(source, "trace", None)
    if trace is None:
        trace = getattr(source, "trace_sample", None)
    if trace is None:
        return None
    sampled_count = getattr(trace, "sample_count", getattr(trace, "sampled_count", None))
    if sampled_count is None:
        return None
    sample_limit = getattr(trace, "sample_limit", sampled_count)
    note = getattr(trace, "note", "Optimizer trace summary unavailable.")
    truncated = bool(getattr(trace, "truncated", False))
    return f"Trace -> sampled={sampled_count} sample_limit={sample_limit} truncated={truncated} note={note}"


def print_ligation_summary(console: Console, *, ligation: object) -> None:
    state = _ligation_state(ligation)
    chosen_classes = ",".join(ligation.chosen_mismatch_classes) if ligation.chosen_mismatch_classes else "-"
    position_classes = ",".join(ligation.position_classes) if ligation.position_classes else "-"
    selection_mode = getattr(ligation, "selection_mode", "secondary")
    candidate_count_before_filter = getattr(ligation, "candidate_count_before_filter", None)
    candidate_count_after_filter = getattr(ligation, "candidate_count_after_filter", None)
    filtered_candidate_count = getattr(ligation, "filtered_candidate_count", None)
    console.print(
        "Ligation -> "
        f"profile={ligation.profile} "
        f"mode={ligation.awareness_mode} "
        f"selection={selection_mode} "
        f"applied={ligation.applied} "
        f"pool={_candidate_position_pool(ligation)} "
        f"classes={chosen_classes} "
        f"positions={position_classes} "
        f"bad_patterns={_bad_pattern_scope(ligation)}"
    )
    if (
        candidate_count_before_filter is not None
        and candidate_count_after_filter is not None
        and filtered_candidate_count is not None
        and filtered_candidate_count > 0
    ):
        console.print(
            "Ligation filter -> "
            f"before={candidate_count_before_filter} "
            f"after={candidate_count_after_filter} "
            f"filtered={filtered_candidate_count}"
        )
    console.print(f"Ligation state -> state={state} edge_comparison_available={_edge_comparison_available(ligation)}")
    if getattr(ligation, "state_note", None):
        console.print(f"Ligation state note -> {ligation.state_note}")
    else:
        console.print(f"Ligation state note -> {_ligation_state_note(ligation)}")
    console.print(f"Ligation note -> {ligation.decision_note}")


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
        console.print(f"Mismatch edits (PS=payload, AS=complement; 1-based) -> {mismatch_summary_text(mismatch_sites)}")
    console.print(f"PWM -> mode={pwm_mode} effective={pwm_effective}")
    if pwm_effective:
        console.print(f"PWM losses -> worst={worst_loss:.6f} total={total_loss:.6f}")


def print_sequence_view_summary(console: Console, *, label: str, view_summary: object) -> None:
    console.print(
        f"{label} canonical 5' -> 3' -> "
        f"top={view_summary.canonical.top_strand_5to3} "
        f"bottom={view_summary.canonical.bottom_strand_5to3}",
        soft_wrap=True,
    )
    console.print(
        f"{label} mismatch-present 5' -> 3' -> "
        f"top={view_summary.mismatch_present.top_strand_5to3} "
        f"bottom={view_summary.mismatch_present.bottom_strand_5to3}",
        soft_wrap=True,
    )
    if view_summary.changed_rows:
        console.print(f"{label} changed rows -> {', '.join(view_summary.changed_rows)}")


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
        exclude.update(
            {
                "motif_context",
                "optimization_decision",
                "split_row_debug",
                "bundle_manifest_path",
                "normalized_payload_path",
                "visual_inventory_path",
                "provenance",
                "payload_label",
                "input_kind",
                "payload_length",
                "selected_payload_sequence",
                "selected_complement_sequence",
                "junction",
                "mismatches",
                "pwm_mode",
                "pwm_effective",
                "worst_loss",
                "total_loss",
                "view_ids",
            }
        )
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
    print_ligation_summary(console, ligation=report.ligation)
    trace_summary = _trace_summary_text(report)
    if trace_summary is not None:
        console.print(trace_summary)
    console.print("Bundle write -> no")


def print_render_outcome(console: Console, outcome: object, *, emit_renders: bool) -> None:
    report = outcome.report
    print_path_detail(console, "Bundle", outcome.bundle_dir)
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
    print_ligation_summary(console, ligation=report.ligation)
    trace_summary = _trace_summary_text(report)
    if trace_summary is not None:
        console.print(trace_summary)
    if emit_renders:
        bundle_base = Path(outcome.bundle_dir)
        if outcome.composite_render_artifact_path is not None:
            print_path_detail(console, "Composite render", outcome.composite_render_artifact_path, base=bundle_base)
        if outcome.published_plot_artifact_path is not None:
            print_path_detail(console, "Published plot", outcome.published_plot_artifact_path, base=bundle_base)


def print_show_outcome(console: Console, outcome: object, *, verbose: bool) -> None:
    sequence_summary = outcome.bundle_summary.sequence_summary
    bundle_base = Path(outcome.bundle_dir)
    print_path_detail(console, "Bundle", outcome.bundle_dir)
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
    print_ligation_summary(console, ligation=outcome.bundle_summary.ligation)
    trace_summary = _trace_summary_text(outcome.bundle_summary)
    if trace_summary is not None:
        console.print(trace_summary)
    console.print(f"Junction payload 5' -> 3' -> {sequence_summary.junction_payload_sequence_5to3}")
    console.print(
        "Overhang 5' -> 3' -> "
        f"canonical={sequence_summary.overhang_5to3.canonical_sequence_5to3} "
        f"mismatch-present={sequence_summary.overhang_5to3.mismatch_present_sequence_5to3}"
    )
    print_sequence_view_summary(console, label="Payload", view_summary=sequence_summary.views.payload)
    print_sequence_view_summary(console, label="Split left", view_summary=sequence_summary.views.split_left)
    print_sequence_view_summary(console, label="Split right", view_summary=sequence_summary.views.split_right)
    print_sequence_view_summary(console, label="Assembled", view_summary=sequence_summary.views.assembled)
    fallback_reason = outcome.motif_context.fallback_reason
    if outcome.pwm_mode != "none" and not outcome.pwm_effective and fallback_reason:
        console.print(f"PWM fallback reason -> {fallback_reason}")
    if verbose:
        console.print(f"Bundle contract -> {outcome.bundle_contract}")
        console.print(f"Provenance -> {json.dumps(outcome.provenance, sort_keys=True)}")
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
        print_split_row_debug(console, outcome.split_row_debug)


__all__ = [
    "build_show_json_payload",
    "print_render_outcome",
    "print_show_outcome",
    "print_ligation_summary",
    "print_split_row_debug",
    "print_validation_report",
    "print_payload_summary",
]
