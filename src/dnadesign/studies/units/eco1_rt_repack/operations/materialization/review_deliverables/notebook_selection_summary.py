"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_selection_summary.py

Selection-summary rendering helpers for the Eco1 review-deliverables notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    normalize_handoff_readiness,
)


def render_selection_funnel_summary(row: dict[str, Any], *, mo: Any, manifest_path: Path) -> Any:
    """Render selection-readiness counts and policy from the selection manifest."""

    loaded = _load_selection_manifest(manifest_path)
    row_counts = _dict_or_empty(loaded.get("row_counts"))
    gate_counts = _dict_or_empty(loaded.get("gate_counts"))
    funnel_stages = [dict(stage) for stage in list(loaded.get("selection_funnel_stages") or [])]
    selected_ids = [str(value) for value in list(loaded.get("selected_candidate_ids") or [])]
    policy_rows = [
        {"field": "Source manifest", "value": "selection_readiness_manifest.yaml"},
        {"field": "Selection policy", "value": str(loaded.get("selection_policy_id") or "")},
        {"field": "Governing rule", "value": str(loaded.get("governing_rule") or "")},
        {"field": "ESMC policy", "value": str(loaded.get("esmc_policy") or "")},
        {"field": "SAE policy", "value": str(loaded.get("sae_window_policy") or "")},
    ]
    count_rows = _count_rows(row_counts=row_counts, gate_counts=gate_counts)
    selected_rows = [{"candidate_id": candidate_id} for candidate_id in selected_ids]
    title = html.escape(str(row.get("title") or "Panel selection keeps fold checks separate from activity claims"))
    stage_note = (
        "The funnel table lists the steps that actually filter or rank candidates. "
        "Other views explain or audit the selected rows."
    )
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.md(stage_note),
            mo.ui.table(funnel_stages, page_size=10),
            mo.md("ESMC and SAE are review annotations, not panel-selection evidence."),
            mo.ui.table(policy_rows, page_size=8),
            mo.ui.table(count_rows, page_size=16),
            mo.ui.table(selected_rows, page_size=10),
        ],
        gap=0.25,
    )


def render_handoff_readiness(row: dict[str, Any], *, mo: Any, manifest_path: Path) -> Any:
    """Render the RT-only handoff readiness checklist."""

    loaded = _load_selection_manifest(manifest_path)
    readiness = _handoff_readiness(manifest_path=manifest_path, loaded=loaded)
    handoff_path = manifest_path.parent / str(readiness["candidate_handoff_path"])
    status_text = (
        "candidate_handoff.yaml is present; panel selection remains separate from construct subjects."
        if handoff_path.exists()
        else "candidate_handoff.yaml is absent; panel selection remains separate from construct subjects."
    )
    title = html.escape(
        str(row.get("title") or "RT-only handoff remains blocked until candidate_handoff.yaml is materialized")
    )
    checklist_rows = [{"field": str(key), "value": _display_value(value)} for key, value in readiness.items()]
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.md(status_text),
            mo.ui.table(checklist_rows, page_size=8),
        ],
        gap=0.25,
    )


def _load_selection_manifest(manifest_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected selection-readiness manifest mapping at {manifest_path}")
    return loaded


def _dict_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _count_rows(*, row_counts: dict[str, Any], gate_counts: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, count in row_counts.items():
        rows.append({"category": "row_counts", "label": str(label), "count": count})
    for category, counts in gate_counts.items():
        if isinstance(counts, dict):
            for label, count in counts.items():
                rows.append({"category": str(category), "label": str(label), "count": count})
    return rows


def _handoff_readiness(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, object]:
    raw = _dict_or_empty(loaded.get("handoff_readiness"))
    return normalize_handoff_readiness(selection_root=manifest_path.parent, raw=raw)


def _display_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
