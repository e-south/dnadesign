"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_interpretation.py

Biohub ESMC SAE interpretation review panels for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq

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

from .biohub_esmc_sae_fold_llr import write_sae_fold_llr_comparison_panel
from .biohub_esmc_sae_tables import (
    make_protein_top_feature_table_row,
    write_protein_top_feature_table,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SECTION = "biohub_esmc_sae_interpretation"
TOP_FEATURE_COUNT = 12
SOURCE_NOTEBOOK = (
    "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
    "esmc_sae_feature_interpretation.ipynb"
)
INTERPRETATION_LIMIT = (
    "These are exact-dictionary Biohub ESMC SAE activations. They support semantic review, "
    "not activity, processivity, strand-displacement, or candidate acceptance claims."
)
METHOD_SUMMARY = (
    "Rank top SAE features from the Biohub ESMC protein-feature table by peak activation and prevalence, "
    "inspect where those features activate over residues, and compare candidate feature sums after "
    "normalizing to WT. Feature names remain blank unless a source-backed interpretation exists for the "
    "exact SAE model, layer, sparsity, and codebook."
)
_SOURCE_TABLES = [
    "biohub_esmc_sae_profile.parquet",
    "biohub_esmc_protein_features.parquet",
    "biohub_esmc_residue_features.parquet",
    "biohub_esmc_feature_catalog.parquet",
    "biohub_esmc_request_manifest.yaml",
]


def write_biohub_esmc_sae_interpretation_panels(
    *,
    panel_root: Path,
    profile_path: Path,
    protein_features_path: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    candidate_table_path: Path,
    foldcheck_report_path: Path,
    foldcheck_ranking_path: Path,
    wt_substitution_llr_path: Path,
) -> list[dict[str, Any]]:
    """Render lightweight SAE interpretation panels from existing sparse Biohub rows."""

    required_inputs = (
        profile_path,
        protein_features_path,
        residue_features_path,
        feature_catalog_path,
        request_manifest_path,
    )
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        return [_missing_row(panel_root, missing)]
    feature_rows = _top_wt_feature_rows(protein_features_path, top_n=TOP_FEATURE_COUNT)
    if not feature_rows:
        return [_missing_row(panel_root, [protein_features_path], reason="WT SAE protein feature rows are absent")]
    selected_features = [int(row["feature_index"]) for row in feature_rows]
    panel_root.mkdir(parents=True, exist_ok=True)
    top_feature_table_path = panel_root / "protein_top_sae_features.parquet"
    write_protein_top_feature_table(
        path=top_feature_table_path,
        protein_features_path=protein_features_path,
        feature_catalog_path=feature_catalog_path,
    )
    return [
        make_protein_top_feature_table_row(
            table_path=top_feature_table_path,
            protein_features_path=protein_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            section=SECTION,
            source_tables=_SOURCE_TABLES,
            interpretation_limit=INTERPRETATION_LIMIT,
            method_summary=METHOD_SUMMARY,
            source_notebook=SOURCE_NOTEBOOK,
        ),
        _write_wt_localization_panel(
            panel_root=panel_root,
            residue_features_path=residue_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            feature_rows=feature_rows,
        ),
        _write_candidate_retention_panel(
            panel_root=panel_root,
            protein_features_path=protein_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            foldcheck_ranking_path=foldcheck_ranking_path,
            selected_features=selected_features,
            wt_feature_rows=feature_rows,
        ),
        write_sae_fold_llr_comparison_panel(
            panel_root=panel_root,
            protein_features_path=protein_features_path,
            feature_catalog_path=feature_catalog_path,
            candidate_table_path=candidate_table_path,
            foldcheck_report_path=foldcheck_report_path,
            foldcheck_ranking_path=foldcheck_ranking_path,
            wt_substitution_llr_path=wt_substitution_llr_path,
            request_manifest_path=request_manifest_path,
            feature_rows=feature_rows,
            source_tables=_SOURCE_TABLES,
        ),
    ]


def _write_wt_localization_panel(
    *,
    panel_root: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "WT-active SAE features localize to specific Ec86 regions"
    selected_features = [int(row["feature_index"]) for row in feature_rows]
    residue_rows = pq.read_table(
        residue_features_path,
        filters=[
            ("candidate_id", "==", "wild_type"),
            ("feature_index", "in", selected_features),
        ],
        columns=["sequence_position_one_based", "feature_index", "value"],
    ).to_pylist()
    position_count = _position_count(residue_features_path)
    matrix = [[0.0 for _ in range(position_count)] for _ in selected_features]
    feature_to_row = {feature: row_index for row_index, feature in enumerate(selected_features)}
    for row in residue_rows:
        feature_index = int(row["feature_index"])
        position = int(row["sequence_position_one_based"])
        if feature_index in feature_to_row and 1 <= position <= position_count:
            matrix[feature_to_row[feature_index]][position - 1] = float(row["value"])
    feature_labels = _feature_axis_labels(feature_catalog_path, selected_features)
    fig_height = max(4.8, 0.36 * len(selected_features) + 2.2)
    fig, ax = plt.subplots(figsize=(16.0, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(range(len(selected_features)), feature_labels, fontsize=max(6, TICK_SIZE - 1))
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Top WT SAE feature", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    _set_position_ticks(ax, position_count)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.18)
    colorbar.set_label("Per-residue activation", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.88, bottom=0.18)
    path = panel_root / "wt_top_sae_feature_localization.svg"
    alt = (
        f"Heatmap of WT Ec86 residue activations for the {len(selected_features)} strongest "
        "Biohub ESMC SAE features selected from the WT protein-feature table."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_wt_top_sae_feature_localization",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=_SOURCE_TABLES,
        input_hashes=file_hashes(
            {
                "residue_features": residue_features_path,
                "feature_catalog": feature_catalog_path,
                "request_manifest": request_manifest_path,
            }
        ),
        alt_text=alt,
        description=(
            "Shows where the WT sequence activates the strongest peak-ranked Biohub ESMC SAE features. "
            "The panel uses feature indices rather than names unless the exact SAE dictionary has "
            "source-backed feature descriptions."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary=_evidence_summary(feature_rows),
        role="review_only",
    )


def _write_candidate_retention_panel(
    *,
    panel_root: Path,
    protein_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    foldcheck_ranking_path: Path,
    selected_features: list[int],
    wt_feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "Candidates retain or shift WT-active SAE features"
    feature_rows = pq.read_table(
        protein_features_path,
        filters=[("feature_index", "in", selected_features)],
        columns=["candidate_id", "feature_index", "activation_sum"],
    ).to_pylist()
    candidate_order = _candidate_order(protein_features_path, foldcheck_ranking_path)
    wt_sums = {int(row["feature_index"]): max(float(row["activation_sum"]), 1e-12) for row in wt_feature_rows}
    feature_to_col = {feature: col_index for col_index, feature in enumerate(selected_features)}
    candidate_to_row = {candidate_id: row_index for row_index, candidate_id in enumerate(candidate_order)}
    matrix = [[0.0 for _ in selected_features] for _ in candidate_order]
    for row in feature_rows:
        candidate_id = str(row["candidate_id"])
        feature = int(row["feature_index"])
        if candidate_id in candidate_to_row and feature in feature_to_col:
            ratio = float(row["activation_sum"]) / wt_sums[feature]
            matrix[candidate_to_row[candidate_id]][feature_to_col[feature]] = math.log2(max(ratio, 1e-6))
    feature_labels = _feature_axis_labels(feature_catalog_path, selected_features)
    transposed_matrix = [list(row) for row in zip(*matrix, strict=True)]
    fig_height = max(5.6, 0.38 * len(selected_features) + 2.3)
    fig_width = max(12.5, min(20.0, 0.13 * len(candidate_order) + 5.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    color_limit = _symmetric_color_limit(matrix)
    image = ax.imshow(
        transposed_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    _set_candidate_axis_ticks(ax, candidate_order)
    ax.set_yticks(range(len(selected_features)), feature_labels, fontsize=max(6, TICK_SIZE - 2))
    ax.set_xlabel("Sequence sorted by fold review", fontsize=LABEL_SIZE)
    ax.set_ylabel("WT-active SAE feature", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.18)
    colorbar.set_label("log2(candidate activation sum / WT activation sum)", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.86, bottom=0.24)
    path = panel_root / "candidate_top_sae_feature_retention.svg"
    alt = (
        f"Heatmap of WT-normalized activation sums for {len(selected_features)} WT-active SAE features "
        f"across {len(candidate_order)} Biohub ESMC queries."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_candidate_top_sae_feature_retention",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[*_SOURCE_TABLES, "foldcheck_review/foldcheck_candidate_ranking.parquet"],
        input_hashes=file_hashes(
            {
                "protein_features": protein_features_path,
                "feature_catalog": feature_catalog_path,
                "request_manifest": request_manifest_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
            }
        ),
        alt_text=alt,
        description=(
            "Compares all Biohub ESMC query rows against the WT peak-active feature subset. Columns are "
            "ordered by fold-review metrics when the ranking table is available, with WT first; feature "
            "labels are kept as single-line row labels for readability."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary=_evidence_summary(wt_feature_rows) | {"sequence_rows": len(candidate_order)},
        role="review_only",
    )


def _missing_row(panel_root: Path, missing: list[Path], *, reason: str | None = None) -> dict[str, Any]:
    message = reason or "Missing Biohub ESMC SAE interpretation input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_sae_interpretation",
        section=SECTION,
        artifact_kind="manifest",
        status="skipped_missing_input",
        path=panel_root / "missing_biohub_esmc_sae_interpretation.txt",
        source_tables=_SOURCE_TABLES,
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="Biohub ESMC SAE interpretation visuals were skipped because sparse SAE inputs were missing.",
        description="The SAE interpretation section requires profile, protein-feature, and residue-feature tables.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title="Biohub ESMC SAE interpretation",
        method_summary=METHOD_SUMMARY,
        evidence_summary={"source_notebook": SOURCE_NOTEBOOK},
        role="review_only",
        skip_reason=message,
    )


def _top_wt_feature_rows(path: Path, *, top_n: int) -> list[dict[str, Any]]:
    rows = pq.read_table(
        path,
        filters=[("candidate_id", "==", "wild_type")],
        columns=["feature_index", "nonzero_residue_count", "activation_sum", "activation_mean", "activation_max"],
    ).to_pylist()
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            float(row["activation_max"]),
            int(row["nonzero_residue_count"]),
            float(row["activation_sum"]),
        ),
        reverse=True,
    )[:top_n]


def _position_count(residue_features_path: Path) -> int:
    table = pq.read_table(
        residue_features_path,
        filters=[("candidate_id", "==", "wild_type")],
        columns=["sequence_position_one_based"],
    )
    values = table.column("sequence_position_one_based").to_pylist()
    return max(int(value) for value in values)


def _candidate_order(protein_features_path: Path, foldcheck_ranking_path: Path) -> list[str]:
    candidates = sorted(
        {
            str(row["candidate_id"])
            for row in pq.read_table(protein_features_path, columns=["candidate_id"]).to_pylist()
            if str(row["candidate_id"])
        }
    )
    ordered = ["wild_type"] if "wild_type" in candidates else []
    if foldcheck_ranking_path.exists():
        ranking_rows = pq.read_table(
            foldcheck_ranking_path,
            columns=["candidate_id", "wt_runtime_ca_rmsd", "plddt"],
        ).to_pylist()
        ranked = sorted(
            [row for row in ranking_rows if str(row.get("candidate_id")) in candidates],
            key=lambda row: (float(row.get("wt_runtime_ca_rmsd") or 1e12), -float(row.get("plddt") or 0.0)),
        )
        ordered.extend(str(row["candidate_id"]) for row in ranked if str(row["candidate_id"]) not in ordered)
    ordered.extend(candidate for candidate in candidates if candidate not in ordered)
    return ordered


def _set_position_ticks(ax: Any, position_count: int) -> None:
    step = 10 if position_count <= 360 else max(25, int(math.ceil(position_count / 36.0)))
    positions = list(range(1, position_count + 1, step))
    if positions[-1] != position_count:
        positions.append(position_count)
    ax.set_xticks(
        [position - 1 for position in positions], [str(position) for position in positions], fontsize=TICK_SIZE
    )
    ax.tick_params(axis="x", length=3)


def _set_candidate_ticks(ax: Any, candidate_order: list[str]) -> None:
    _set_candidate_axis_ticks(ax, candidate_order, axis="y")


def _set_candidate_axis_ticks(ax: Any, candidate_order: list[str], *, axis: str = "x") -> None:
    if len(candidate_order) <= 24:
        ticks = range(len(candidate_order))
    else:
        step = max(1, math.ceil(len(candidate_order) / 16))
        ticks = list(range(0, len(candidate_order), step))
        if ticks[-1] != len(candidate_order) - 1:
            ticks.append(len(candidate_order) - 1)
    labels = ["WT" if candidate_order[index] == "wild_type" else str(index) for index in ticks]
    if axis == "y":
        ax.set_yticks(list(ticks), labels, fontsize=TICK_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
        return
    ax.set_xticks(list(ticks), labels, fontsize=TICK_SIZE)
    ax.tick_params(axis="x", labelrotation=0, length=3)


def _feature_axis_labels(feature_catalog_path: Path, selected_features: list[int]) -> list[str]:
    catalog = {
        int(row["feature_index"]): (str(row.get("label") or ""), str(row.get("description") or ""))
        for row in pq.read_table(feature_catalog_path, columns=["feature_index", "label", "description"]).to_pylist()
    }
    return [
        _feature_axis_label(feature_index, *catalog.get(feature_index, ("", ""))) for feature_index in selected_features
    ]


def _feature_axis_label(feature_index: int, label: str, description: str) -> str:
    text = _concise_feature_description(label=label, description=description)
    if not text:
        return f"F{feature_index}"
    return f"F{feature_index} - {text}"


def _concise_feature_description(*, label: str, description: str) -> str:
    source = (description or label).strip()
    if not source:
        return ""
    if source.lower().startswith("summary:"):
        source = source.split(":", 1)[1].strip()
    first_sentence = source.split(". ", 1)[0].strip().rstrip(".")
    for delimiter in (", with", ", strongest", ", focusing", ";", ":"):
        if delimiter in first_sentence:
            candidate = first_sentence.split(delimiter, 1)[0].strip()
            if len(candidate) >= 24:
                first_sentence = candidate
                break
    return _ellipsize(first_sentence, max_chars=58)


def _ellipsize(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _evidence_summary(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_notebook": SOURCE_NOTEBOOK,
        "feature_selection_rule": "top WT features by activation_max, tie-broken by prevalence and activation_sum",
        "selected_feature_count": len(feature_rows),
        "selected_feature_indices": [int(row["feature_index"]) for row in feature_rows],
    }


def _symmetric_color_limit(matrix: list[list[float]]) -> float:
    values = sorted(abs(value) for row in matrix for value in row if math.isfinite(value))
    if not values:
        return 1.0
    percentile_index = min(len(values) - 1, max(0, math.ceil(0.98 * len(values)) - 1))
    return max(0.02, min(0.5, values[percentile_index]))
