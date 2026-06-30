"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/plot_context.py

Study-owned residue-context spans for WT ESMC mutation-scoring plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_RT_INTERVAL_COLOR = "#111111"
_PROTECTED_COLOR = "#E69F00"
_MOTIF_COLOR = "#D55E00"
_MOTIF_LABELS = {
    "retron_x_naxxh": "NAxxH",
    "catalytic_yadd": "YADD",
    "retron_y_vtg": "VTG",
}


def build_position_context_spans(mask_set_path: Path) -> list[dict[str, object]]:
    """Build generic plotting spans from Eco1 mask rows.

    RT intervals are review annotations. Protected and motif spans reflect the
    active mask rows. The returned objects intentionally use generic plotting
    fields so the Permuter plotter stays study-agnostic.
    """

    mask_set = yaml.safe_load(mask_set_path.read_text(encoding="utf-8"))
    residues = list(mask_set.get("residues", [])) if isinstance(mask_set, dict) else []
    spans: list[dict[str, object]] = []
    spans.extend(_rt_interval_spans(residues))
    spans.extend(
        _boolean_spans(
            residues,
            field="protected",
            label="Mask-protected residues",
            color=_PROTECTED_COLOR,
            alpha=0.16,
            zorder=0.24,
        )
    )
    spans.extend(_motif_spans(residues))
    return spans


def _rt_interval_spans(residues: list[dict[str, Any]]) -> list[dict[str, object]]:
    by_label: dict[str, list[int]] = {}
    for row in residues:
        label = str(row.get("rt_interval_review_label") or "")
        if not label:
            continue
        by_label.setdefault(label, []).append(int(row["canonical_position"]))
    spans: list[dict[str, object]] = []
    for label in sorted(by_label, key=_rt_label_sort_key):
        for start, end in _segments(by_label[label]):
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "legend_label": "RT1-RT7 annotation intervals",
                    "color": _RT_INTERVAL_COLOR,
                    "alpha": 0.075,
                    "zorder": 0.10,
                }
            )
    return spans


def _motif_spans(residues: list[dict[str, Any]]) -> list[dict[str, object]]:
    by_reason: dict[str, list[int]] = {}
    for row in residues:
        if not bool(row.get("motif_protected")):
            continue
        reason = str(row.get("manual_mask_reason") or "motif_protected")
        by_reason.setdefault(reason, []).append(int(row["canonical_position"]))
    spans: list[dict[str, object]] = []
    for reason in sorted(by_reason):
        label = _MOTIF_LABELS.get(reason, reason.replace("_", " "))
        for start, end in _segments(by_reason[reason]):
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "legend_label": "NAxxH/YADD/VTG motif anchors",
                    "color": _MOTIF_COLOR,
                    "alpha": 0.28,
                    "zorder": 0.28,
                }
            )
    return spans


def _boolean_spans(
    residues: list[dict[str, Any]],
    *,
    field: str,
    label: str,
    color: str,
    alpha: float,
    zorder: float,
) -> list[dict[str, object]]:
    positions = [int(row["canonical_position"]) for row in residues if bool(row.get(field))]
    return [
        {
            "start": start,
            "end": end,
            "label": label,
            "legend_label": label,
            "color": color,
            "alpha": alpha,
            "zorder": zorder,
        }
        for start, end in _segments(positions)
    ]


def _segments(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    sorted_positions = sorted(set(positions))
    segments: list[tuple[int, int]] = []
    start = sorted_positions[0]
    previous = start
    for position in sorted_positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        segments.append((start, previous))
        start = previous = position
    segments.append((start, previous))
    return segments


def _rt_label_sort_key(label: str) -> tuple[int, str]:
    number = "".join(character for character in label if character.isdigit())
    return (int(number) if number else 999, label)
