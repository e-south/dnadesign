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
    selection_summary = _dict_or_empty(loaded.get("selection_summary"))
    candidate_counts = _dict_or_empty(selection_summary.get("candidate_counts"))
    funnel_stages = [dict(stage) for stage in list(loaded.get("selection_funnel_stages") or [])]
    selected_ids = [str(value) for value in list(loaded.get("selected_candidate_ids") or [])]
    policy_rows = [
        {"field": "Source manifest", "value": "selection_readiness_manifest.yaml"},
        {"field": "Selection policy", "value": str(loaded.get("selection_policy_id") or "")},
        {"field": "Governing rule", "value": str(loaded.get("governing_rule") or "")},
    ]
    panel_summary = _dict_or_empty(selection_summary.get("selected_mutation_overlap"))
    overlap_by_policy = _dict_or_empty(selection_summary.get("selected_mutation_overlap_by_policy"))
    panel_coverage = _dict_or_empty(loaded.get("panel_coverage"))
    selected_panel_size = int(panel_coverage.get("selected_panel_size") or len(selected_ids))
    panel_summary_rows = [
        {"field": "Selected panel", "value": _display_value(selected_panel_size)},
        {"field": "Mutations per sequence", "value": _display_range(panel_summary.get("mutation_count_range"))},
        {
            "field": "Peripheral mutations per sequence",
            "value": _display_range(panel_summary.get("peripheral_mutation_count_range")),
        },
        {
            "field": "Peripheral charge change",
            "value": _display_range(panel_summary.get("peripheral_charge_change_range"), signed=True),
        },
        {
            "field": "Unique mutated positions",
            "value": _display_value(panel_summary.get("unique_mutated_position_count")),
        },
        {
            "field": "Unique substitutions",
            "value": _display_value(panel_summary.get("unique_exact_substitution_count")),
        },
        {
            "field": f"Positions shared by all {selected_panel_size}",
            "value": _display_value(panel_summary.get("shared_mutated_position_count")),
        },
        {
            "field": "Mean position-set Jaccard distance",
            "value": _display_value(panel_summary.get("mean_pairwise_mutated_position_jaccard_distance")),
        },
        {
            "field": "Mean exact-substitution Jaccard distance",
            "value": _display_value(panel_summary.get("mean_pairwise_exact_substitution_jaccard_distance")),
        },
    ]
    policy_overlap_rows = [
        {
            "policy": policy_id,
            "selected sequences": _display_value(_dict_or_empty(summary).get("selected_candidate_count")),
            "shared positions": _display_value(_dict_or_empty(summary).get("shared_mutated_position_count")),
            "mean position-set distance": _display_value(
                _dict_or_empty(summary).get("mean_pairwise_mutated_position_jaccard_distance")
            ),
            "minimum position-set distance": _display_value(
                _dict_or_empty(summary).get("minimum_pairwise_mutated_position_jaccard_distance")
            ),
            "mean exact-substitution distance": _display_value(
                _dict_or_empty(summary).get("mean_pairwise_exact_substitution_jaccard_distance")
            ),
        }
        for policy_id, summary in overlap_by_policy.items()
    ]
    selected_rows = [{"candidate_id": candidate_id} for candidate_id in selected_ids]
    title = html.escape(str(row.get("title") or "Selection flow and panel summary"))
    r13a_match_count = int(candidate_counts.get("wang_r13a_interface_disruption_evidence_match") or 0)
    r13a_note = (
        "No generated sequence matches the tested R13A substitution."
        if r13a_match_count == 0
        else f"{r13a_match_count} generated sequences match the tested R13A substitution."
    )
    stage_note = (
        "Local geometry screens predicted structural disruption. Generation records confirm the fixed/open residue "
        "sets and allowed amino-acid alphabets. Exact F10 and R13 states are reported but do not filter or rank "
        f"rows. {r13a_note} Oligomeric state was not evaluated. The three design groups represent different "
        "interventions, not quality levels. All eight rows form one selected panel."
    )
    distance_note = (
        "Global panel distance is increased by policies that open different residue sets. The policy-stratified "
        "table is the relevant check for mutation-profile collapse within each design group. The pair-first "
        "procedure is deterministic but does not claim a globally optimal three-sequence subset."
    )
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.md(stage_note),
            mo.ui.table(funnel_stages, page_size=10),
            mo.Html("<h4 style='margin:0.45rem 0 0.1rem 0;'>Selected-panel summary</h4>"),
            mo.ui.table(panel_summary_rows, page_size=8),
            mo.md(distance_note),
            mo.ui.table(policy_overlap_rows, page_size=8),
            mo.ui.table(policy_rows, page_size=8),
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


def _handoff_readiness(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, object]:
    raw = _dict_or_empty(loaded.get("handoff_readiness"))
    return normalize_handoff_readiness(selection_root=manifest_path.parent, raw=raw)


def _display_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _display_range(value: object, *, signed: bool = False) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return ""
    formatter = (lambda item: f"{int(item):+d}") if signed else (lambda item: str(int(item)))
    return f"{formatter(value[0])} to {formatter(value[1])}"
