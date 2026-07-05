"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_activation_pattern.py

WT Biohub ESMC SAE activation-pattern figure rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

from .biohub_esmc_sae_interpretation_shared import (
    INTERPRETATION_LIMIT,
    METHOD_SUMMARY,
    SECTION,
    SOURCE_TABLES,
    evidence_summary,
    feature_axis_labels,
    position_count,
    set_position_ticks,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WT_ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "wt_sae_activation_white_to_dark",
    ["#ffffff", "#d9f0e3", "#4c9f70", "#174a5a"],
)


def write_wt_activation_pattern_panel(
    *,
    panel_root: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    feature_rows: list[dict[str, Any]],
    sae_provenance_audit: dict[str, Any],
) -> dict[str, Any]:
    title = "WT-active SAE features localize by residue"
    selected_features = [int(row["feature_index"]) for row in feature_rows]
    residue_rows = pq.read_table(
        residue_features_path,
        filters=[
            ("candidate_id", "==", "wild_type"),
            ("feature_index", "in", selected_features),
        ],
        columns=["sequence_position_one_based", "feature_index", "value"],
    ).to_pylist()
    sequence_position_count = position_count(residue_features_path)
    matrix = [[0.0 for _ in range(sequence_position_count)] for _ in selected_features]
    feature_to_row = {feature: row_index for row_index, feature in enumerate(selected_features)}
    for row in residue_rows:
        feature_index = int(row["feature_index"])
        position = int(row["sequence_position_one_based"])
        if feature_index in feature_to_row and 1 <= position <= sequence_position_count:
            matrix[feature_to_row[feature_index]][position - 1] = float(row["value"])
    labels = feature_axis_labels(feature_catalog_path, selected_features)
    fig_height = max(4.8, 0.36 * len(selected_features) + 2.2)
    fig, ax = plt.subplots(figsize=(16.0, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=WT_ACTIVATION_CMAP, vmin=0.0)
    ax.set_yticks(range(len(selected_features)), labels, fontsize=max(6, TICK_SIZE - 1))
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Top WT SAE feature", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    set_position_ticks(ax, sequence_position_count)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.18)
    colorbar.set_label("Per-residue activation", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.88, bottom=0.18)
    path = panel_root / "wt_top_sae_feature_activation_pattern.svg"
    alt = (
        f"Heatmap of WT Ec86 residue activations for the {len(selected_features)} strongest "
        "Biohub ESMC SAE features selected from the WT protein-feature table."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_wt_top_sae_feature_activation_pattern",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes(
            {
                "residue_features": residue_features_path,
                "feature_catalog": feature_catalog_path,
                "request_manifest": request_manifest_path,
            }
        ),
        alt_text=alt,
        description=(
            "Shows where the WT sequence activates the strongest peak-ordered Biohub ESMC SAE features. "
            "The panel uses feature indices rather than names unless the exact SAE dictionary has "
            "source-backed feature descriptions."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary=evidence_summary(
            feature_rows,
            request_manifest_path=request_manifest_path,
            sae_provenance_audit=sae_provenance_audit,
        ),
        role="review_only",
    )
