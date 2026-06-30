"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_sae_features.py

SAE feature-inspector helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME,
    SECTION_ESMC_FEATURE_REVIEW,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)


def load_sae_top_feature_rows(*, manifest_root: Path, deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load per-protein top-feature rows if the table is materialized."""

    top_table_path, _residue_table_path = biohub_sae_paths(manifest_root=manifest_root, deliverables=deliverables)
    if not top_table_path.exists():
        return []
    return pq.read_table(top_table_path).to_pylist()


def load_sae_feature_residue_rows(
    *,
    manifest_root: Path,
    deliverables: list[dict[str, Any]],
    candidate_id: str,
    feature_index: int,
) -> list[dict[str, Any]]:
    """Load residue activations for one selected protein and feature."""

    _top_table_path, residue_table_path = biohub_sae_paths(manifest_root=manifest_root, deliverables=deliverables)
    if not residue_table_path.exists():
        return []
    return pq.read_table(
        residue_table_path,
        filters=[("candidate_id", "==", str(candidate_id)), ("feature_index", "==", int(feature_index))],
        columns=["sequence_position_one_based", "value"],
    ).to_pylist()


def biohub_sae_paths(*, manifest_root: Path, deliverables: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Resolve the top-feature table and sparse residue table paths."""

    top_table_row = next(
        (row for row in deliverables if str(row.get("deliverable_id") or "") == "biohub_esmc_protein_top_sae_features"),
        None,
    )
    if top_table_row is None:
        top_table_path = manifest_root / BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME / "protein_top_sae_features.parquet"
    else:
        top_table_path = resolve_manifest_path(manifest_root, str(top_table_row["path"]))
    return top_table_path, manifest_root.parent / "biohub_esmc_residue_features.parquet"


def sae_protein_lookup(rows: list[dict[str, Any]], *, selected_section: str) -> dict[str, str]:
    """Build protein dropdown labels with WT first."""

    if selected_section != SECTION_ESMC_FEATURE_REVIEW:
        return {}
    candidate_ids = sorted({str(row["candidate_id"]) for row in rows})
    candidate_ids = (["wild_type"] if "wild_type" in candidate_ids else []) + [
        candidate_id for candidate_id in candidate_ids if candidate_id != "wild_type"
    ]
    return {_sae_protein_label(candidate_id): candidate_id for candidate_id in candidate_ids}


def selected_sae_feature_rows(rows: list[dict[str, Any]], *, candidate_id: str) -> list[dict[str, Any]]:
    """Filter top-feature rows for one protein."""

    return [row for row in rows if str(row.get("candidate_id") or "") == str(candidate_id)]


def sae_feature_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build feature dropdown labels for one protein."""

    return {_sae_feature_label(row): row for row in rows}


def render_sae_feature_inspector(
    *,
    mo: Any,
    manifest_root: Path,
    deliverables: list[dict[str, Any]],
    selected_row: dict[str, Any] | None,
    protein_ui: Any,
    feature_ui: Any,
) -> Any:
    """Render residue-level activation for one selected SAE feature."""

    if selected_row is None:
        return mo.md("Select a protein and SAE feature to inspect residue-level activations.")
    residue_rows = load_sae_feature_residue_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
        candidate_id=str(selected_row["candidate_id"]),
        feature_index=int(selected_row["feature_index"]),
    )
    svg = _sae_feature_svg(selected_row, residue_rows)
    metric_rows = _sae_feature_metric_rows(selected_row)
    return mo.vstack(
        [
            mo.hstack([protein_ui, feature_ui], justify="start", align="stretch", wrap=True, gap=1.0, widths="equal"),
            mo.Html(svg),
            mo.ui.table(metric_rows),
        ]
    )


def _sae_protein_label(candidate_id: str) -> str:
    if candidate_id == "wild_type":
        return "WT Ec86 control"
    if candidate_id.startswith("thread_candidate_"):
        return "ProteinMPNN variant " + candidate_id.removeprefix("thread_candidate_")[:12]
    return candidate_id


def _sae_feature_label(row: dict[str, Any]) -> str:
    feature_index = int(row["feature_index"])
    max_rank = row.get("rank_by_max_activation") or "-"
    prevalence_rank = row.get("rank_by_prevalence") or "-"
    description = _concise_feature_description(
        label=str(row.get("label") or ""),
        description=str(row.get("description") or ""),
    )
    return f"F{feature_index} | {description} | peak rank {max_rank} | prevalence rank {prevalence_rank}"


def _sae_feature_svg(selected_row: dict[str, Any], residue_rows: list[dict[str, Any]]) -> str:
    candidate_id = str(selected_row["candidate_id"])
    feature_index = int(selected_row["feature_index"])
    values = {int(row["sequence_position_one_based"]): float(row["value"]) for row in residue_rows}
    position_count = int(selected_row["sequence_residue_count"])
    max_value = max(values.values()) if values else 1.0
    width, height, left, top = 920, 210, 46, 38
    plot_width, plot_height = width - left - 24, 92
    bars = [
        _sae_feature_bar(position, values, max_value, left, top, plot_width, plot_height, position_count)
        for position in range(1, position_count + 1)
    ]
    ticks = [
        _sae_feature_tick(position, left, top, plot_width, plot_height, position_count)
        for position in [1, 50, 100, 150, 200, 250, 300, position_count]
        if position <= position_count
    ]
    description = str(selected_row.get("description") or "")
    description_status = str(selected_row.get("description_status") or "")
    description_text = description or "No source-backed description is available for this exact SAE dictionary."
    title = f"{html.escape(candidate_id)} F{feature_index} residue activation"
    desc = f"{html.escape(description_text)} Peak activation {float(selected_row['activation_max']):.3f}."
    return f"""
    <figure style="margin:0;">
      <svg role="img" aria-labelledby="sae-feature-title sae-feature-desc" viewBox="0 0 {width} {height}"
           style="width:100%; height:auto; display:block; border:1px solid #d8dee4;
                  border-radius:6px; background:white;">
        <title id="sae-feature-title">{title}</title>
        <desc id="sae-feature-desc">{html.escape(desc)}</desc>
        <text x="{left}" y="24" font-size="16" font-weight="600">{title}</text>
        <line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height:.2f}"
              y2="{top + plot_height:.2f}" stroke="#333" stroke-width="1"/>
        {"".join(bars)}
        {"".join(ticks)}
      </svg>
      <figcaption style="font-size:0.9rem; line-height:1.4; margin-top:0.45rem; color:#57606a;">
        <strong>Description status:</strong> {html.escape(description_status)}
        <br/>
        <strong>Description:</strong> {html.escape(description_text)}
      </figcaption>
    </figure>
    """


def _sae_feature_bar(
    position: int,
    values: dict[int, float],
    max_value: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    position_count: int,
) -> str:
    value = values.get(position, 0.0)
    bar_height = 0.0 if max_value <= 0 else (value / max_value) * plot_height
    x = left + ((position - 1) / max(1, position_count)) * plot_width
    y = top + plot_height - bar_height
    width = max(1.0, plot_width / position_count)
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{bar_height:.2f}" fill="#0072b2" fill-opacity="0.82" />'
    )


def _sae_feature_tick(
    position: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    position_count: int,
) -> str:
    x = left + ((position - 1) / max(1, position_count)) * plot_width
    return (
        f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top + plot_height:.2f}" '
        f'y2="{top + plot_height + 5:.2f}" stroke="#333" stroke-width="1" />'
        f'<text x="{x:.2f}" y="{top + plot_height + 20:.2f}" text-anchor="middle" font-size="11">{position}</text>'
    )


def _sae_feature_metric_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"field": "feature_index", "value": str(row["feature_index"])},
        {"field": "rank_by_max_activation", "value": str(row.get("rank_by_max_activation"))},
        {"field": "rank_by_prevalence", "value": str(row.get("rank_by_prevalence"))},
        {"field": "activation_max", "value": f"{float(row['activation_max']):.4f}"},
        {"field": "prevalence_fraction", "value": f"{float(row['prevalence_fraction']):.4f}"},
        {"field": "description_status", "value": str(row.get("description_status") or "")},
    ]


def _concise_feature_description(*, label: str, description: str) -> str:
    source = (description or label).strip()
    if not source:
        return "exact-dictionary description unavailable"
    if source.lower().startswith("summary:"):
        source = source.split(":", 1)[1].strip()
    first_sentence = source.split(". ", 1)[0].strip().rstrip(".")
    return " ".join(first_sentence.split()[:8])
