"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/visual_hierarchy.py

Review-section hierarchy for OPAL notebook visual deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ._support import mapping
from .selection_overlap import CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND


@dataclass(frozen=True)
class NotebookVisualGroup:
    key: str
    label: str
    rank: int


VISUAL_GROUPS: tuple[NotebookVisualGroup, ...] = (
    NotebookVisualGroup("decision", "Decision review", 10),
    NotebookVisualGroup("assay", "Assay evidence", 20),
    NotebookVisualGroup("eda", "EDA comparisons", 30),
    NotebookVisualGroup("model", "Model diagnostics", 40),
    NotebookVisualGroup("method", "Method diagnostics", 50),
    NotebookVisualGroup("handoff", "Handoff", 60),
)

_GROUPS_BY_KEY = {group.key: group for group in VISUAL_GROUPS}
_GROUPS_BY_LABEL = {group.label: group for group in VISUAL_GROUPS}
_BASERENDER_SURFACE_KINDS = {"baserender", "campaign_set_baserender"}
_READER_SURFACE_KIND = "reader_evidence"
_SELECTION_BATCH_SURFACE_KIND = "selection_batch"
_DECISION_NAMES = {
    "selected_vec8_summary": 0,
    "score_vs_rank_over_rounds": 10,
}
_DECISION_KINDS = {
    "response_magnitude_feasibility_frontier": 20,
    "response_magnitude_feasibility_constraint_decomposition": 30,
}
_EDA_NAMES = {
    "effect_scaled_vs_logic_fidelity_latest": 0,
    "fold_change_vs_logic_fidelity_latest": 10,
    "sfxi_observed_logic_closeness_over_rounds": 20,
    "sfxi_factorial_effects_latest": 30,
    "score_selected_over_rounds": 40,
    "score_threshold_over_rounds": 50,
}
_MODEL_KINDS = {"feature_importance_heatmap", "feature_importance_bars", "sfxi_support_diagnostics", "sfxi_uncertainty"}
_METHOD_KINDS = {"sfxi_setpoint_sweep", "sfxi_intensity_scaling"}


def annotate_notebook_visual_choices(choices: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach stable review-section metadata to visual choices."""

    rows: list[dict[str, Any]] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping):
            continue
        group, item_rank = notebook_visual_group(choice)
        row = dict(choice)
        row["review_group_key"] = group.key
        row["review_group_label"] = group.label
        row["review_group_rank"] = group.rank
        row["review_item_rank"] = item_rank
        row["_visual_index"] = index
        rows.append(row)
    return rows


def build_notebook_visual_group_options(choices: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return review-section labels with at least one available deliverable."""

    present: dict[str, NotebookVisualGroup] = {}
    for choice in choices:
        group = _choice_group(choice)
        if group is not None:
            present[group.key] = group
    return [group.label for group in sorted(present.values(), key=lambda item: item.rank)]


def filter_notebook_visual_choices_by_group(
    choices: Iterable[Mapping[str, Any]],
    group_label: str | None,
) -> list[dict[str, Any]]:
    """Return choices in the requested review section, sorted by section priority."""

    rows = [dict(choice) for choice in choices if isinstance(choice, Mapping)]
    if not rows or group_label in (None, ""):
        return rows
    group = _group_from_text(group_label)
    filtered = [row for row in rows if str(row.get("review_group_key") or "") == group.key]
    if not filtered:
        available = build_notebook_visual_group_options(rows)
        raise ValueError(f"Review section not available: {group_label}. Available: {available}")
    return sorted(
        filtered,
        key=lambda row: (
            int(row.get("review_item_rank") or 0),
            int(row.get("_visual_index") or 0),
            str(row.get("label") or ""),
        ),
    )


def notebook_visual_group(choice: Mapping[str, Any]) -> tuple[NotebookVisualGroup, int]:
    """Return the review section and intra-section priority for one visual choice."""

    explicit = _explicit_group(choice)
    if explicit is not None:
        return explicit, _explicit_rank(choice)

    surface_kind = str(choice.get("surface_kind") or "").strip()
    name = _choice_name(choice)
    kind = str(choice.get("kind") or mapping(choice.get("manifest")).get("kind") or "").strip()

    if surface_kind in _BASERENDER_SURFACE_KINDS:
        return _GROUPS_BY_KEY["handoff"], 0
    if surface_kind == _SELECTION_BATCH_SURFACE_KIND:
        return _GROUPS_BY_KEY["handoff"], -10
    if surface_kind == _READER_SURFACE_KIND:
        return _GROUPS_BY_KEY["assay"], _reader_rank(choice)
    if surface_kind == CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND:
        return _GROUPS_BY_KEY["eda"], 0
    if surface_kind.startswith("campaign_set_"):
        return _GROUPS_BY_KEY["eda"], 10
    if name in _DECISION_NAMES:
        return _GROUPS_BY_KEY["decision"], _DECISION_NAMES[name]
    if kind in _DECISION_KINDS:
        return _GROUPS_BY_KEY["decision"], _DECISION_KINDS[kind]
    if name in _EDA_NAMES:
        return _GROUPS_BY_KEY["eda"], _EDA_NAMES[name]
    if kind in _MODEL_KINDS:
        return _GROUPS_BY_KEY["model"], 0
    if kind in _METHOD_KINDS:
        return _GROUPS_BY_KEY["method"], 0
    return _GROUPS_BY_KEY["eda"], 100


def _choice_group(choice: Mapping[str, Any]) -> NotebookVisualGroup | None:
    group_key = str(choice.get("review_group_key") or "").strip()
    if group_key in _GROUPS_BY_KEY:
        return _GROUPS_BY_KEY[group_key]
    group_label = str(choice.get("review_group_label") or "").strip()
    if group_label in _GROUPS_BY_LABEL:
        return _GROUPS_BY_LABEL[group_label]
    if isinstance(choice, Mapping):
        return notebook_visual_group(choice)[0]
    return None


def _explicit_group(choice: Mapping[str, Any]) -> NotebookVisualGroup | None:
    for source in (choice, mapping(choice.get("entry")), mapping(choice.get("manifest")).get("params") or {}):
        if not isinstance(source, Mapping):
            continue
        text = str(source.get("review_group") or source.get("deliverable_group") or "").strip()
        if text:
            return _group_from_text(text)
    return None


def _explicit_rank(choice: Mapping[str, Any]) -> int:
    for source in (choice, mapping(choice.get("entry")), mapping(choice.get("manifest")).get("params") or {}):
        if not isinstance(source, Mapping):
            continue
        value = source.get("review_rank") or source.get("deliverable_rank")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"review_rank must be an integer, got {value!r}.") from exc
    return 0


def _group_from_text(value: str | None) -> NotebookVisualGroup:
    text = str(value or "").strip()
    if text in _GROUPS_BY_LABEL:
        return _GROUPS_BY_LABEL[text]
    key = text.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "decision": "decision",
        "decision_review": "decision",
        "primary": "decision",
        "assay": "assay",
        "assay_evidence": "assay",
        "eda": "eda",
        "eda_comparisons": "eda",
        "model": "model",
        "model_diagnostics": "model",
        "method": "method",
        "method_diagnostics": "method",
        "handoff": "handoff",
    }
    group = _GROUPS_BY_KEY.get(aliases.get(key, key))
    if group is None:
        raise ValueError(
            f"Unknown review section: {value!r}. Expected one of {[group.label for group in VISUAL_GROUPS]}."
        )
    return group


def _choice_name(choice: Mapping[str, Any]) -> str:
    name = str(choice.get("name") or "").strip()
    if name:
        return name
    entry_name = str(mapping(choice.get("entry")).get("name") or "").strip()
    if entry_name:
        return entry_name
    return str(mapping(choice.get("manifest")).get("name") or "").strip()


def _reader_rank(choice: Mapping[str, Any]) -> int:
    label = str(choice.get("reader_plot_type_label") or choice.get("label") or "").lower()
    if "time series + snapshot" in label:
        return 0
    if "vec8" in label or "heatmap" in label:
        return 10
    return 20


__all__ = [
    "NotebookVisualGroup",
    "VISUAL_GROUPS",
    "annotate_notebook_visual_choices",
    "build_notebook_visual_group_options",
    "filter_notebook_visual_choices_by_group",
    "notebook_visual_group",
]
