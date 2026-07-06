"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_sae_features.py

SAE feature-heatmap helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_visuals import (
    render_visual_zoom_controls,
    render_zoom_frame,
    visual_zoom_script,
    zoom_frame_html,
)

_ZERO_COLOR = "#ffffff"
_PALETTE = ("#f7fbff", "#d8eff0", "#8ccfbc", "#2f8f83", "#174a5a")


def is_sae_feature_heatmap_deliverable(row: dict[str, Any] | None) -> bool:
    """Return whether a manifest row should render as the interactive SAE heatmap."""

    if row is None:
        return False
    return (
        str(row.get("artifact_kind") or "") == "sae_feature_heatmap_manifest"
        and str(row.get("status") or "") == "rendered"
    )


def load_sae_feature_heatmap_manifest(
    *,
    manifest_root: Path,
    selected_visual: dict[str, Any] | None,
) -> dict[str, Any]:
    """Load the selected SAE feature-heatmap manifest."""

    if not is_sae_feature_heatmap_deliverable(selected_visual):
        return {}
    path = resolve_manifest_path(manifest_root, str(selected_visual["path"]))
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    payload["_manifest_path"] = path
    payload["_manifest_root"] = path.parent
    return payload


def sae_heatmap_feature_lookup(payload: dict[str, Any]) -> dict[str, int]:
    """Build feature dropdown labels from the heatmap manifest."""

    options: dict[str, int] = {}
    for rank, row in enumerate(payload.get("features") or [], start=1):
        if not isinstance(row, dict) or row.get("feature_index") is None:
            continue
        feature_index = int(row["feature_index"])
        label = str(row.get("label") or "").strip() or f"F{feature_index}"
        options[f"{label} | WT peak order {rank}"] = feature_index
    return options


def render_sae_feature_heatmap(
    *,
    mo: Any,
    heatmap_manifest: dict[str, Any],
    selected_feature_index: int | None,
    feature_ui: Any | None,
) -> Any:
    """Render one selected SAE feature across WT and variant residue positions."""

    if not heatmap_manifest:
        return mo.md("SAE feature heatmap manifest is unavailable.")
    feature_map = {int(row["feature_index"]): dict(row) for row in heatmap_manifest.get("features") or []}
    if selected_feature_index is None and feature_map:
        selected_feature_index = next(iter(feature_map))
    if selected_feature_index not in feature_map:
        return mo.md("Select an SAE feature to render the variant-by-residue heatmap.")
    residue_features_path = _resolve_payload_path(
        heatmap_manifest,
        str(heatmap_manifest.get("residue_features_path") or ""),
    )
    if not residue_features_path.exists():
        return mo.md(f"SAE residue feature table is missing: `{residue_features_path}`")
    candidate_order = [str(value) for value in heatmap_manifest.get("candidate_order") or []]
    wt_sequence = str(heatmap_manifest.get("wt_sequence") or "")
    if not candidate_order or not wt_sequence:
        return mo.md("SAE heatmap manifest is missing candidate order or WT sequence.")
    residue_rows = pq.read_table(
        residue_features_path,
        filters=[("feature_index", "==", int(selected_feature_index)), ("candidate_id", "in", candidate_order)],
        columns=["candidate_id", "sequence_position_one_based", "value"],
    ).to_pylist()
    svg = _heatmap_svg(
        feature_row=feature_map[int(selected_feature_index)],
        candidate_order=candidate_order,
        wt_sequence=wt_sequence,
        residue_rows=residue_rows,
    )
    metric_rows = _feature_metric_rows(feature_map[int(selected_feature_index)], heatmap_manifest)
    panels = [
        render_zoom_frame(
            mo=mo,
            frame_html=svg,
            title="Zoomable visual: selected SAE feature activation across Eco1 RT variants",
            height_css="84vh",
        ),
        mo.accordion(
            {"Selected SAE feature evidence": mo.ui.table(metric_rows, page_size=8)},
            multiple=False,
            lazy=True,
        ),
    ]
    if feature_ui is not None:
        panels.insert(0, feature_ui)
    return mo.vstack(panels, gap=0.35)


def _resolve_payload_path(payload: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(str(payload["_manifest_root"])) / path


def _heatmap_svg(
    *,
    feature_row: dict[str, Any],
    candidate_order: list[str],
    wt_sequence: str,
    residue_rows: list[dict[str, Any]],
) -> str:
    feature_index = int(feature_row["feature_index"])
    feature_label = str(feature_row.get("label") or f"F{feature_index}")
    sequence_length = len(wt_sequence)
    row_count = len(candidate_order)
    cell_width = 4.8
    row_height = 13.6
    left = 176.0
    top = 130.0
    right = 76.0
    bottom = 104.0
    plot_width = cell_width * sequence_length
    plot_height = row_height * row_count
    width = int(left + plot_width + right)
    height = int(top + plot_height + bottom)
    values = {
        (str(row["candidate_id"]), int(row["sequence_position_one_based"])): float(row["value"])
        for row in residue_rows
        if float(row.get("value") or 0.0) > 0.0
    }
    max_value = max(values.values()) if values else 1.0
    title = f"SAE F{feature_index} activation across Eco1 RT variants"
    subtitle = feature_label if feature_label != f"F{feature_index}" else "Selected WT-active SAE feature"
    desc = (
        f"Feature {feature_index} activation heatmap over {row_count} sequences and "
        f"{sequence_length} residue positions. White cells indicate zero recorded sparse activation."
    )
    container_id = (
        "sae-feature-heatmap-"
        + hashlib.sha256(f"{feature_index}:{row_count}:{sequence_length}".encode()).hexdigest()[:12]
    )
    svg_id = f"{container_id}-image"
    zoom_controls = render_visual_zoom_controls(container_id)
    zoom_script = visual_zoom_script(container_id=container_id, image_id=svg_id)
    row_background = []
    row_labels = []
    for row_index, candidate_id in enumerate(candidate_order):
        y = top + row_index * row_height
        if row_index % 2:
            row_background.append(
                f'<rect x="{left:.1f}" y="{y:.1f}" width="{plot_width:.1f}" height="{row_height:.1f}" fill="#f6f8fa"/>'
            )
        row_labels.append(
            f'<text x="{left - 8:.1f}" y="{y + row_height * 0.72:.1f}" text-anchor="end" '
            f'font-size="8.3" fill="#24292f">{html.escape(_candidate_label(candidate_id, row_index))}</text>'
        )
    top_letters = []
    bottom_positions = []
    for position, residue in enumerate(wt_sequence, start=1):
        x = left + (position - 0.5) * cell_width
        top_letters.append(
            f'<text x="{x:.1f}" y="{top - 18:.1f}" text-anchor="middle" font-size="8.7" '
            f'fill="#24292f">{html.escape(residue)}</text>'
        )
        bottom_positions.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 19:.1f}" text-anchor="start" font-size="7.2" '
            f'fill="#57606a" transform="rotate(90 {x:.1f} {top + plot_height + 19:.1f})">{position}</text>'
        )
    heatmap_cells = []
    row_index_by_candidate = {candidate_id: index for index, candidate_id in enumerate(candidate_order)}
    for (candidate_id, position), value in values.items():
        row_index = row_index_by_candidate.get(candidate_id)
        if row_index is None or not 1 <= position <= sequence_length:
            continue
        x = left + (position - 1) * cell_width
        y = top + row_index * row_height
        heatmap_cells.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width:.1f}" height="{row_height:.1f}" '
            f'fill="{_activation_color(value, max_value)}"/>'
        )
    x_grid = [
        f'<line x1="{left + position * cell_width:.1f}" x2="{left + position * cell_width:.1f}" '
        f'y1="{top:.1f}" y2="{top + plot_height:.1f}" stroke="#d8dee4" stroke-width="0.25"/>'
        for position in range(0, sequence_length + 1, 10)
    ]
    axis_labels_svg = (
        f'<text x="{left - 8:.1f}" y="{top - 18:.1f}" text-anchor="end" '
        'font-size="10.4" fill="#57606a">WT residue</text>'
        f'<text x="{left - 8:.1f}" y="{top + plot_height + 21:.1f}" text-anchor="end" '
        'font-size="10.4" fill="#57606a">Position</text>'
    )
    colorbar_width = min(460.0, max(320.0, plot_width * 0.34))
    colorbar_left = left + (plot_width - colorbar_width) / 2.0
    colorbar = _colorbar_svg(left=colorbar_left, y=height - 34, width=colorbar_width, max_value=max_value)
    return zoom_frame_html(
        body_html=f"""
      {zoom_controls}
      <div id="{container_id}" style="overflow:auto; width:100%; height:calc(100vh - 4.4rem);
           border:1px solid #d8dee4; border-radius:6px; background:#ffffff; padding:0.25rem;
           box-sizing:border-box;">
        <svg id="{svg_id}" role="img" aria-labelledby="{svg_id}-title {svg_id}-desc"
             viewBox="0 0 {width} {height}"
             style="display:block; width:100%; max-width:none; height:auto; transform-origin:top left;
                    background:#ffffff;">
          <title id="{svg_id}-title">{html.escape(title)}</title>
          <desc id="{svg_id}-desc">{html.escape(desc)}</desc>
          <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
          <text x="{left + plot_width / 2.0:.1f}" y="30" text-anchor="middle"
                font-size="19" font-weight="650" fill="#24292f">{html.escape(title)}</text>
          <text x="{left + plot_width / 2.0:.1f}" y="53" text-anchor="middle"
                font-size="12.2" fill="#57606a">{html.escape(subtitle)}</text>
          <text x="{left + plot_width / 2.0:.1f}" y="72" text-anchor="middle"
                font-size="11.2" fill="#57606a">
            Rows are WT plus ProteinMPNN variants; columns are Ec86 canonical positions.
          </text>
          {axis_labels_svg}
          <rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="{_ZERO_COLOR}"/>
          {"".join(row_background)}
          {"".join(heatmap_cells)}
          {"".join(x_grid)}
          <rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}"
                fill="none" stroke="#8c959f" stroke-width="0.8"/>
          {"".join(row_labels)}
          {"".join(top_letters)}
          {"".join(bottom_positions)}
          {colorbar}
        </svg>
      </div>
      {zoom_script}
      <figcaption style="font-size:0.9rem; line-height:1.4; margin-top:0.45rem; color:#57606a;">
        Feature activations are exact-dictionary Biohub ESMC SAE values. Missing sparse entries are rendered as zero.
      </figcaption>
    """
    )


def _candidate_label(candidate_id: str, row_index: int) -> str:
    if candidate_id == "wild_type":
        return "WT Ec86"
    return f"V{row_index:03d} {candidate_id.removeprefix('thread_candidate_')[:8]}"


def _activation_color(value: float, max_value: float) -> str:
    if value <= 0.0 or max_value <= 0.0:
        return _ZERO_COLOR
    scaled = min(1.0, max(0.0, math.sqrt(value / max_value)))
    anchors = [0.0, 0.25, 0.5, 0.75, 1.0]
    for index in range(len(anchors) - 1):
        lower = anchors[index]
        upper = anchors[index + 1]
        if lower <= scaled <= upper:
            fraction = (scaled - lower) / max(upper - lower, 1e-9)
            return _interpolate_hex(_PALETTE[index], _PALETTE[index + 1], fraction)
    return _PALETTE[-1]


def _interpolate_hex(start: str, end: str, fraction: float) -> str:
    start_values = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_values = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = [
        round(start_value + (end_value - start_value) * fraction)
        for start_value, end_value in zip(start_values, end_values, strict=True)
    ]
    return "#" + "".join(f"{value:02x}" for value in values)


def _colorbar_svg(*, left: float, y: float, width: float, max_value: float) -> str:
    segments = []
    segment_count = 80
    segment_width = width / segment_count
    for index in range(segment_count):
        value = max_value * (index + 0.5) / segment_count
        x = left + index * segment_width
        segments.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{segment_width + 0.4:.1f}" height="9" '
            f'fill="{_activation_color(value, max_value)}"/>'
        )
    return (
        "".join(segments) + f'<rect x="{left:.1f}" y="{y:.1f}" width="{width:.1f}" height="9" fill="none" '
        'stroke="#8c959f" stroke-width="0.5"/>'
        + f'<text x="{left:.1f}" y="{y + 25:.1f}" font-size="9" fill="#57606a">0</text>'
        + f'<text x="{left + width:.1f}" y="{y + 25:.1f}" text-anchor="end" font-size="9" '
        f'fill="#57606a">max {max_value:.3g}</text>'
        + f'<text x="{left + width + 12:.1f}" y="{y + 9:.1f}" font-size="9.5" fill="#57606a">'
        "SAE activation</text>"
    )


def _feature_metric_rows(row: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {"field": "feature_index", "value": str(row["feature_index"])},
        {"field": "feature_label", "value": str(row.get("label") or "")},
        {"field": "wt_activation_max", "value": f"{float(row.get('wt_activation_max') or 0.0):.4f}"},
        {"field": "wt_activation_sum", "value": f"{float(row.get('wt_activation_sum') or 0.0):.4f}"},
        {"field": "wt_nonzero_residue_count", "value": str(row.get("wt_nonzero_residue_count") or "")},
        {"field": "sequence_rows", "value": str(payload.get("candidate_count") or "")},
        {"field": "sequence_length", "value": str(payload.get("sequence_length") or "")},
        {"field": "feature_selection_policy", "value": str(payload.get("feature_selection_policy") or "")},
    ]
    audit = payload.get("sae_provenance_audit")
    if isinstance(audit, dict):
        rows.extend(
            [
                {
                    "field": "accepted_sae_profiles",
                    "value": f"{audit.get('accepted_profile_count', 0)} of {audit.get('profile_row_count', 0)}",
                },
                {
                    "field": "sequence_length_range",
                    "value": f"{audit.get('sequence_length_min', 0)}-{audit.get('sequence_length_max', 0)}",
                },
                {"field": "unique_sequence_hashes", "value": str(audit.get("unique_sequence_hash_count") or 0)},
                {
                    "field": "unique_raw_logits_hashes",
                    "value": str(audit.get("unique_raw_logits_response_hash_count") or 0),
                },
                {
                    "field": "wt_activation_cosine_range",
                    "value": _cosine_range_label(
                        audit.get("wt_activation_cosine_min"),
                        audit.get("wt_activation_cosine_max"),
                    ),
                },
            ]
        )
    return rows


def _cosine_range_label(minimum: Any, maximum: Any) -> str:
    if minimum is None or maximum is None:
        return "not available"
    return f"{float(minimum):.4f}-{float(maximum):.4f}"
