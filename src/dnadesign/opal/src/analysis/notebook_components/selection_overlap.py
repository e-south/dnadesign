"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/selection_overlap.py

Campaign-set selected-candidate overlap visual for OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable, Mapping

CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND = "campaign_set_selection_overlap"


def build_notebook_campaign_set_selection_overlap_choice(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    round_selector: str | int | None = "latest",
) -> dict[str, Any] | None:
    """Return a notebook visual choice for pooled top-k overlap across campaigns."""

    rows = build_notebook_campaign_set_selection_overlap_rows(campaigns, round_selector=round_selector)
    if not rows:
        return None
    summary = _selection_overlap_summary(rows)
    return {
        "id": "pooled_selection_overlap",
        "visual_id": "pooled_selection_overlap",
        "label": "Pooled selection overlap",
        "title": "Pooled selection overlap",
        "surface_kind": CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
        "kind": CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
        "comparison_set_key": "selection_overlap",
        "comparison_set_label": "Selection overlap",
        "evidence_tier_label": "EDA comparisons",
        "review_group": "EDA comparisons",
        "premise": "Show which ranked selections are shared across campaign objectives before pooling a build set.",
        "claim_boundary": "Overlap is a selection-policy diagnostic, not measured biological validation.",
        "rows": rows,
        "summary": summary,
        "row_count": len(rows),
        "campaign_count": summary["campaign_count"],
        "unique_candidate_count": summary["unique_candidate_count"],
        "shared_all_count": summary["shared_all_count"],
    }


def build_notebook_campaign_set_selection_overlap_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    round_selector: str | int | None = "latest",
) -> list[dict[str, Any]]:
    """Read top-k selection CSVs and return one row per campaign-selected candidate."""

    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        if not isinstance(campaign, Mapping):
            continue
        campaign_meta = campaign.get("campaign") if isinstance(campaign.get("campaign"), Mapping) else {}
        workdir = Path(str(campaign_meta.get("workdir") or ""))
        if not str(workdir):
            continue
        selection_path = _selection_csv_path(workdir, round_selector=round_selector)
        if selection_path is None:
            continue
        campaign_slug = str(campaign_meta.get("slug") or selection_path.parts[-5])
        campaign_label = _campaign_label(campaign_meta, fallback=campaign_slug)
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                candidate_id = str(raw.get("id") or "").strip()
                if not candidate_id:
                    continue
                rows.append(
                    {
                        "campaign": campaign_slug,
                        "campaign_label": campaign_label,
                        "round": _int_or_none(raw.get("as_of_round")),
                        "run_id": str(raw.get("run_id") or ""),
                        "id": candidate_id,
                        "short_id": _short_id(candidate_id),
                        "rank": _int_or_none(raw.get("sel__rank_competition")),
                        "score": _float_or_none(raw.get("pred__score_selected")),
                        "score_ref": str(raw.get("pred__score_ref") or ""),
                        "selection_path": str(selection_path),
                        "sequence": str(raw.get("sequence") or ""),
                    }
                )
    return rows


def render_notebook_campaign_set_selection_overlap_image(
    choice: Mapping[str, Any],
    *,
    dpi: int = 180,
) -> dict[str, Any] | None:
    """Render the pooled-selection overlap heatmap as PNG bytes."""

    rows = [dict(row) for row in choice.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    campaigns = _ordered_unique(str(row.get("campaign_label") or row.get("campaign")) for row in rows)
    ids = _ordered_candidate_ids(rows)
    rank_by_cell = {(str(row.get("id")), str(row.get("campaign_label") or row.get("campaign"))): row for row in rows}
    max_rank = max(int(row.get("rank") or 0) for row in rows if row.get("rank") is not None)
    matrix = np.full((len(ids), len(campaigns)), np.nan)
    for row_index, candidate_id in enumerate(ids):
        for col_index, campaign in enumerate(campaigns):
            row = rank_by_cell.get((candidate_id, campaign))
            if row is None or row.get("rank") is None:
                continue
            matrix[row_index, col_index] = float(max_rank + 1 - int(row["rank"]))

    fig_height = max(3.8, 0.34 * len(ids) + 1.5)
    fig_width = max(6.0, 1.65 * len(campaigns) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F3F4F6")
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=1, vmax=max(max_rank, 1))
    counts = _candidate_counts(rows)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{_short_id(candidate_id)} ({counts[candidate_id]})" for candidate_id in ids], fontsize=9)
    ax.set_xticks(range(len(campaigns)))
    ax.set_xticklabels(campaigns, rotation=24, ha="right", fontsize=9)
    ax.set_xlabel("Campaign")
    ax.set_ylabel("Selected candidate")
    ax.set_title("Pooled selection overlap", fontweight="semibold")
    for row_index, candidate_id in enumerate(ids):
        for col_index, campaign in enumerate(campaigns):
            row = rank_by_cell.get((candidate_id, campaign))
            if row is None:
                continue
            ax.text(col_index, row_index, f"r{row.get('rank')}", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_xticks(np.arange(-0.5, len(campaigns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), facecolor="white")
    plt.close(fig)

    summary = _selection_overlap_summary(rows)
    group_text = ", ".join(campaigns)
    alt_text = (
        "Heatmap of pooled top-k selection overlap across campaigns. "
        f"Columns are {group_text}; rows are {summary['unique_candidate_count']} unique selected candidates; "
        f"{summary['shared_all_count']} candidates are selected by every campaign. Cell labels show selection rank."
    )
    caption = (
        "Rows are unique selected candidate IDs and columns are campaigns. "
        "A filled cell means the candidate appears in that campaign's top-k selection; the cell label gives rank. "
        "This supports pooled-build review and does not imply measured response."
    )
    return {
        "image_bytes": buffer.getvalue(),
        "alt_text": alt_text,
        "caption": caption,
        "title": "Pooled selection overlap",
        "summary": summary,
    }


def build_notebook_campaign_set_selection_overlap_card_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact evidence rows for the pooled-selection overlap visual."""

    summary = choice.get("summary") if isinstance(choice.get("summary"), Mapping) else {}
    return [
        {"field": "campaigns", "value": summary.get("campaign_count") or 0},
        {"field": "selected slots", "value": summary.get("slot_count") or 0},
        {"field": "unique candidates", "value": summary.get("unique_candidate_count") or 0},
        {"field": "selected by every campaign", "value": summary.get("shared_all_count") or 0},
        {"field": "max campaign overlap", "value": summary.get("max_overlap") or 0},
        {"field": "claim boundary", "value": choice.get("claim_boundary") or "not recorded"},
    ]


def _selection_csv_path(workdir: Path, *, round_selector: str | int | None) -> Path | None:
    rounds_dir = workdir / "outputs" / "rounds"
    if not rounds_dir.exists():
        return None
    round_value = str(round_selector or "latest")
    if round_value not in {"latest", "all"}:
        path = rounds_dir / f"round_{int(round_value)}" / "selection" / "selection_top_k.csv"
        return path if path.exists() else None
    candidates = []
    for child in rounds_dir.glob("round_*"):
        if not child.is_dir():
            continue
        try:
            round_index = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        path = child / "selection" / "selection_top_k.csv"
        if path.exists():
            candidates.append((round_index, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _selection_overlap_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    row_list = [row for row in rows if isinstance(row, Mapping)]
    campaigns = {str(row.get("campaign") or "") for row in row_list if str(row.get("campaign") or "")}
    counts = _candidate_counts(row_list)
    return {
        "campaign_count": len(campaigns),
        "slot_count": len(row_list),
        "unique_candidate_count": len(counts),
        "shared_all_count": sum(1 for value in counts.values() if value == len(campaigns) and campaigns),
        "max_overlap": max(counts.values()) if counts else 0,
    }


def _candidate_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    by_id: dict[str, set[str]] = {}
    for row in rows:
        candidate_id = str(row.get("id") or "")
        campaign = str(row.get("campaign") or "")
        if not candidate_id or not campaign:
            continue
        by_id.setdefault(candidate_id, set()).add(campaign)
    return {candidate_id: len(campaigns) for candidate_id, campaigns in by_id.items()}


def _ordered_candidate_ids(rows: list[Mapping[str, Any]]) -> list[str]:
    counts = _candidate_counts(rows)
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        candidate_id = str(row.get("id") or "")
        if not candidate_id:
            continue
        first_seen.setdefault(candidate_id, index)
        rank = row.get("rank")
        if rank is not None:
            best_rank[candidate_id] = min(best_rank.get(candidate_id, int(rank)), int(rank))
    return sorted(
        counts,
        key=lambda candidate_id: (-counts[candidate_id], best_rank.get(candidate_id, 10_000), first_seen[candidate_id]),
    )


def _campaign_label(campaign: Mapping[str, Any], *, fallback: str) -> str:
    name = str(campaign.get("name") or fallback).split("|", 1)[0].strip()
    for prefix in ("SECG ", "Stress ethanol/ciprofloxacin "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name or fallback


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def _short_id(value: str) -> str:
    text = str(value or "")
    return text[:8] if len(text) > 8 else text


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND",
    "build_notebook_campaign_set_selection_overlap_card_rows",
    "build_notebook_campaign_set_selection_overlap_choice",
    "build_notebook_campaign_set_selection_overlap_rows",
    "render_notebook_campaign_set_selection_overlap_image",
]
