"""Manifest, table, and HTML writers for MSA visualization sidecars."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import yaml

from dnadesign.aligner.msa.visualization.contracts.models import (
    MsaVisualizationRequest,
    PositionQc,
    ProfileQc,
)

INDEX_SCHEMA_ID = "dnadesign.aligner.msa.visualization.index"
PROFILE_SCHEMA_ID = "dnadesign.aligner.msa.visualization.profile_qc"
SIDECAR_NOTE = "Visualization sidecar only; downstream tools own interpretation."


def write_profile_qc(qc: ProfileQc, request: MsaVisualizationRequest) -> None:
    """Write one per-profile QC manifest."""

    payload = {
        "schema_id": PROFILE_SCHEMA_ID,
        "schema_version": 1,
        "profile_id": qc.profile_id,
        "status": "materialized",
        "created_at": request.created_at,
        "aligned_fasta_path": str(qc.aligned_fasta_path),
        "record_count": qc.record_count,
        "alignment_length": qc.alignment_length,
        "canonical_position_count": qc.canonical_position_count,
        "inserted_column_count": qc.inserted_column_count,
        "target_row_id": request.target_row_id,
        "target_ungapped_sha256": qc.target_ungapped_sha256,
        "sidecar_note": SIDECAR_NOTE,
    }
    qc.profile_qc_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def write_position_qc_csv(path: Path, rows: list[PositionQc]) -> None:
    """Write long-form target-position QC values."""

    fieldnames = [
        "profile_id",
        "canonical_position",
        "alignment_column",
        "target_aa",
        "non_gap_count",
        "gap_count",
        "gap_fraction",
        "plurality_aa",
        "plurality_count",
        "plurality_frequency",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "profile_id": row.profile_id,
                    "canonical_position": row.canonical_position,
                    "alignment_column": row.alignment_column,
                    "target_aa": row.target_aa,
                    "non_gap_count": row.non_gap_count,
                    "gap_count": row.gap_count,
                    "gap_fraction": f"{row.gap_fraction:.6f}",
                    "plurality_aa": row.plurality_aa,
                    "plurality_count": row.plurality_count,
                    "plurality_frequency": f"{row.plurality_frequency:.6f}",
                }
            )


def write_html_report(
    path: Path,
    profile_qcs: list[ProfileQc],
    missing_profile_ids: list[str],
    *,
    has_exemplar_rows: bool,
    has_alignment_overview: bool,
    has_consensus_histogram: bool,
) -> None:
    """Write a lightweight HTML report linking generated sidecars."""

    rows = "\n".join(
        (
            "<tr>"
            f"<td>{html.escape(qc.profile_id)}</td>"
            f"<td>{qc.record_count}</td>"
            f"<td>{qc.alignment_length}</td>"
            f"<td>{qc.canonical_position_count}</td>"
            f"<td>{qc.inserted_column_count}</td>"
            f'<td><a href="{html.escape(qc.profile_svg_path.name)}">SVG</a></td>'
            "</tr>"
        )
        for qc in profile_qcs
    )
    missing = ""
    if missing_profile_ids:
        items = "".join(f"<li>{html.escape(profile_id)}</li>" for profile_id in missing_profile_ids)
        missing = f"<h2>Missing Profiles</h2><ul>{items}</ul>"
    figures = "\n".join(
        (
            "<figure>"
            f'<img src="{html.escape(qc.profile_svg_path.name)}" alt="{html.escape(qc.profile_id)} MSA QC track">'
            f"<figcaption>{html.escape(qc.profile_id)} target-position QC track</figcaption>"
            "</figure>"
        )
        for qc in profile_qcs
    )
    exemplar_figures = ""
    if has_exemplar_rows:
        exemplar_figures = "\n".join(
            (
                "<figure>"
                f'<img src="{html.escape(qc.profile_exemplar_svg_path.name)}" '
                f'alt="{html.escape(qc.profile_id)} exemplar MSA windows">'
                f"<figcaption>{html.escape(qc.profile_id)} selected exemplar rows around annotated "
                "features</figcaption>"
                "</figure>"
            )
            for qc in profile_qcs
        )
    panel_figures = ""
    if has_alignment_overview or has_consensus_histogram:
        panel_figures = "\n".join(
            (
                (
                    "<figure>"
                    f'<img src="{html.escape(qc.profile_alignment_overview_svg_path.name)}" '
                    f'alt="{html.escape(qc.profile_id)} selected-row overview">'
                    f"<figcaption>{html.escape(qc.profile_id)} selected-row whole-alignment overview</figcaption>"
                    "</figure>"
                    if has_alignment_overview
                    else ""
                )
                + (
                    "<figure>"
                    f'<img src="{html.escape(qc.profile_consensus_histogram_svg_path.name)}" '
                    f'alt="{html.escape(qc.profile_id)} plurality histogram">'
                    f"<figcaption>{html.escape(qc.profile_id)} plurality/gap histogram by target position</figcaption>"
                    "</figure>"
                    if has_consensus_histogram
                    else ""
                )
            )
            for qc in profile_qcs
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MSA QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    .note {{ color: #555; max-width: 760px; }}
    figure {{ margin: 24px 0; }}
    img {{ display: block; max-width: 100%; border: 1px solid #ddd; }}
    figcaption {{ color: #555; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>MSA QC</h1>
  <p class="note">{SIDECAR_NOTE}</p>
  <table>
    <thead>
      <tr>
        <th>Profile</th><th>Records</th><th>Aligned length</th>
        <th>Target positions</th><th>Inserted columns</th><th>Track</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {missing}
  <h2>Position Tracks</h2>
  {figures}
  <h2>Exemplar Windows</h2>
  {exemplar_figures}
  <h2>Alignment Panels</h2>
  {panel_figures}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_index_manifest(
    path: Path,
    *,
    request: MsaVisualizationRequest,
    profile_qcs: list[ProfileQc],
    missing_profile_ids: tuple[str, ...],
    position_qc_csv_path: Path,
    html_report_path: Path,
    has_alignment_overview: bool,
    has_consensus_histogram: bool,
) -> None:
    """Write the top-level visualization bundle manifest."""

    payload = {
        "schema_id": INDEX_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized_partial" if missing_profile_ids else "materialized",
        "created_at": request.created_at,
        "alignment_root": str(request.alignment_root),
        "output_root": str(request.output_root),
        "target_row_id": request.target_row_id,
        "target_sequence_hash": request.target_sequence_hash,
        "annotation_tracks_path": str(request.annotation_tracks_yaml) if request.annotation_tracks_yaml else None,
        "exemplar_rows_path": str(request.exemplar_rows_yaml) if request.exemplar_rows_yaml else None,
        "panel_spec_path": str(request.panel_spec_yaml) if request.panel_spec_yaml else None,
        "profile_ids": [qc.profile_id for qc in profile_qcs],
        "requested_profile_ids": list(request.profile_ids),
        "missing_profile_ids": list(missing_profile_ids),
        "position_qc_csv_path": str(position_qc_csv_path),
        "html_report_path": str(html_report_path),
        "profile_qc_paths": {qc.profile_id: str(qc.profile_qc_path) for qc in profile_qcs},
        "profile_svg_paths": {qc.profile_id: str(qc.profile_svg_path) for qc in profile_qcs},
        "profile_exemplar_svg_paths": {
            qc.profile_id: str(qc.profile_exemplar_svg_path)
            for qc in profile_qcs
            if request.exemplar_rows_yaml and request.annotation_tracks_yaml
        },
        "profile_alignment_overview_svg_paths": {
            qc.profile_id: str(qc.profile_alignment_overview_svg_path) for qc in profile_qcs if has_alignment_overview
        },
        "profile_consensus_histogram_svg_paths": {
            qc.profile_id: str(qc.profile_consensus_histogram_svg_path) for qc in profile_qcs if has_consensus_histogram
        },
        "sidecar_note": SIDECAR_NOTE,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
