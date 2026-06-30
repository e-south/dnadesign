"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_fold_llr.py

Joint SAE, fold-confidence, and ESMC mutation-score review panel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

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

from .biohub_esmc_model_provenance import combined_sae_fold_llr_model_summary
from .constants import SECTION_ESMC_FEATURE_REVIEW

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

SECTION = SECTION_ESMC_FEATURE_REVIEW
DELIVERABLE_ID = "biohub_esmc_sae_fold_llr_comparison"
TITLE = "SAE similarity, ColabFold confidence, and ESMC mutation scores are compared together"
VISIBLE_TITLE = "WT-like SAE activation patterns are compared with fold and LLR side markers"
INTERPRETATION_LIMIT = (
    "This is a review plot. SAE similarity is model-derived, pLDDT is a structure-model confidence "
    "summary, and the ESMC LLR value is a sum of WT masked-marginal single-substitution scores, "
    "not a joint protein likelihood or an activity measurement."
)
METHOD_SUMMARY = (
    "Rows are ordered by decreasing SAE similarity to WT, computed as 1 / (1 + RMSE) over log2 "
    "candidate/WT activation-sum ratios for WT peak-active SAE features. The side markers show "
    "ColabFold pLDDT and the summed WT ESMC masked-marginal single-substitution LLR for each variant."
)
_MUTATION_PATTERN = re.compile(r"^[A-Z](?P<position>\d+)(?P<alt>[A-Z])$")


def write_sae_fold_llr_comparison_panel(
    *,
    panel_root: Path,
    protein_features_path: Path,
    feature_catalog_path: Path,
    candidate_table_path: Path,
    foldcheck_report_path: Path,
    foldcheck_ranking_path: Path,
    wt_substitution_llr_path: Path,
    request_manifest_path: Path,
    feature_rows: list[dict[str, Any]],
    source_tables: list[str],
) -> dict[str, Any]:
    """Render a joint SAE-feature, ColabFold, and ESMC mutation-score panel."""

    required = (
        protein_features_path,
        feature_catalog_path,
        candidate_table_path,
        wt_substitution_llr_path,
        request_manifest_path,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        return _missing_row(panel_root, missing, source_tables=source_tables)
    selected_features = [int(row["feature_index"]) for row in feature_rows]
    if not selected_features:
        return _missing_row(
            panel_root,
            [protein_features_path],
            reason="No WT SAE features selected",
            source_tables=source_tables,
        )
    panel_root.mkdir(parents=True, exist_ok=True)
    plot_data = _build_plot_data(
        protein_features_path=protein_features_path,
        feature_catalog_path=feature_catalog_path,
        candidate_table_path=candidate_table_path,
        foldcheck_report_path=foldcheck_report_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
        wt_substitution_llr_path=wt_substitution_llr_path,
        selected_features=selected_features,
    )
    path = panel_root / "sae_fold_llr_comparison.svg"
    wt_mutation_manifest_path = wt_substitution_llr_path.parent / "wt_mutation_scoring_manifest.yaml"
    _render_panel(path, plot_data)
    return make_deliverable_row(
        deliverable_id=DELIVERABLE_ID,
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            *source_tables,
            "candidate_table.parquet",
            "foldcheck_report.parquet",
            "foldcheck_review/foldcheck_candidate_ranking.parquet",
            "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet",
        ],
        input_hashes=file_hashes(
            {
                "protein_features": protein_features_path,
                "feature_catalog": feature_catalog_path,
                "candidate_table": candidate_table_path,
                "foldcheck_report": foldcheck_report_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
                "wt_substitution_llr": wt_substitution_llr_path,
                "wt_mutation_scoring_manifest": wt_mutation_manifest_path,
                "request_manifest": request_manifest_path,
            }
        ),
        alt_text=(
            "Heatmap of WT-normalized SAE feature activations with top markers for ColabFold pLDDT "
            "and summed ESMC single-substitution LLR values."
        ),
        description=(
            "Compares all Biohub ESMC query rows in one review panel. The WT column is first; variants are "
            "ordered by SAE similarity to WT. Heatmap rows are WT-active SAE features, "
            "heatmap color encodes log2 candidate/WT activation ratio, marker size encodes ColabFold pLDDT, "
            "and marker color encodes summed ESMC LLR."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary=combined_sae_fold_llr_model_summary(
            sae_request_manifest_path=request_manifest_path,
            wt_mutation_scoring_manifest_path=wt_mutation_manifest_path,
        )
        | {
            "sequence_rows": len(plot_data["row_labels"]),
            "selected_feature_count": len(selected_features),
            "llr_scoring_rule": "sum_variant_single_substitution_llrs",
            "row_sort": "wild_type_first_then_descending_sae_similarity",
        },
        role="manuscript_facing",
    )


def _build_plot_data(
    *,
    protein_features_path: Path,
    feature_catalog_path: Path,
    candidate_table_path: Path,
    foldcheck_report_path: Path,
    foldcheck_ranking_path: Path,
    wt_substitution_llr_path: Path,
    selected_features: list[int],
) -> dict[str, Any]:
    feature_rows = pq.read_table(
        protein_features_path,
        filters=[("feature_index", "in", selected_features)],
        columns=["candidate_id", "feature_index", "activation_sum"],
    ).to_pylist()
    feature_labels = _feature_labels(feature_catalog_path, selected_features)
    candidate_ids = _candidate_ids(feature_rows)
    wt_sums = _wt_feature_sums(feature_rows, selected_features)
    vectors = _candidate_feature_vectors(feature_rows, candidate_ids, selected_features, wt_sums)
    similarity = {candidate_id: _sae_similarity(vectors[candidate_id]) for candidate_id in candidate_ids}
    ordered_ids = ["wild_type"] + sorted(
        [candidate_id for candidate_id in candidate_ids if candidate_id != "wild_type"],
        key=lambda candidate_id: (-similarity[candidate_id], candidate_id),
    )
    plddt = _plddt_by_candidate(
        foldcheck_report_path=foldcheck_report_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
    )
    llr = _llr_sum_by_candidate(candidate_table_path, wt_substitution_llr_path)
    return {
        "ordered_ids": ordered_ids,
        "row_labels": _row_labels(ordered_ids),
        "feature_labels": feature_labels,
        "matrix": [vectors[candidate_id] for candidate_id in ordered_ids],
        "similarity": [similarity[candidate_id] for candidate_id in ordered_ids],
        "plddt": [plddt.get(candidate_id) for candidate_id in ordered_ids],
        "llr": [llr.get(candidate_id) for candidate_id in ordered_ids],
    }


def _render_panel(path: Path, data: dict[str, Any]) -> None:
    row_count = len(data["row_labels"])
    feature_count = len(data["feature_labels"])
    fig_height = max(7.4, 0.42 * feature_count + 3.6)
    fig_width = max(12.5, min(20.0, 0.13 * row_count + 5.6))
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(VISIBLE_TITLE, fontsize=TITLE_SIZE, y=0.985)
    grid = GridSpec(2, 1, height_ratios=[1.25, max(4.2, 0.45 * feature_count)], hspace=0.32, figure=fig)
    metric_ax = fig.add_subplot(grid[0, 0])
    heatmap_ax = fig.add_subplot(grid[1, 0], sharex=metric_ax)
    matrix = data["matrix"]
    transposed_matrix = [list(row) for row in zip(*matrix, strict=True)]
    limit = _symmetric_color_limit(matrix)
    image = heatmap_ax.imshow(
        transposed_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    heatmap_ax.set_yticks(range(feature_count), data["feature_labels"], fontsize=max(6, TICK_SIZE - 2))
    tick_positions, tick_labels = _readable_column_ticks(data["row_labels"])
    heatmap_ax.set_xticks(tick_positions, tick_labels, fontsize=max(6, TICK_SIZE - 2))
    heatmap_ax.set_xlabel("ProteinMPNN variant ordered by SAE similarity", fontsize=LABEL_SIZE)
    heatmap_ax.set_ylabel("SAE feature", fontsize=LABEL_SIZE)
    heatmap_ax.set_title("WT-normalized SAE feature activation", fontsize=LABEL_SIZE, pad=12, loc="center")
    llr_scatter = _render_metric_axis(metric_ax, data)
    colorbar = fig.colorbar(image, ax=heatmap_ax, orientation="horizontal", fraction=0.04, pad=0.16)
    colorbar.set_label("log2(feature activation sum / WT)", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    llr_colorbar = fig.colorbar(llr_scatter, ax=metric_ax, orientation="horizontal", fraction=0.12, pad=0.08)
    llr_colorbar.set_label("LLR sum, scaled within panel", fontsize=LEGEND_SIZE)
    llr_colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.34, right=0.93, top=0.88, bottom=0.22)
    save_accessible_svg(
        fig,
        path,
        title=TITLE,
        description=_panel_accessibility_description(data),
    )


def _panel_accessibility_description(data: dict[str, Any]) -> str:
    """Return concise SVG metadata; full feature text stays in feature tables."""

    feature_count = len(data["feature_labels"])
    row_count = len(data["row_labels"])
    return (
        f"Heatmap comparing {feature_count} WT-active SAE features across {row_count} sequences. "
        "Columns are WT and ProteinMPNN variants ordered by SAE similarity to WT. "
        "Color shows WT-normalized feature activation. Top markers show ColabFold pLDDT "
        "and summed WT masked-marginal single-substitution LLR. Full feature descriptions "
        "are available in the feature inspector."
    )


def _render_metric_axis(ax: Any, data: dict[str, Any]) -> Any:
    columns = range(len(data["row_labels"]))
    llr_values = [value for value in data["llr"] if value is not None]
    llr_limit = max([abs(value) for value in llr_values] + [1.0])
    for column_index, plddt_value in zip(columns, data["plddt"], strict=True):
        size = 18.0 if plddt_value is None else 18.0 + max(0.0, min(1.0, (plddt_value - 75.0) / 20.0)) * 125.0
        ax.scatter(
            [column_index],
            [1],
            s=size,
            color="#4c78a8",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
        )
    llr_scatter = None
    for column_index, llr_value in zip(columns, data["llr"], strict=True):
        color_value = 0.0 if llr_value is None else max(-1.0, min(1.0, llr_value / llr_limit))
        llr_scatter = ax.scatter(
            [column_index],
            [0],
            s=42,
            c=[color_value],
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            marker="D",
            edgecolor="white",
        )
    _add_plddt_size_legend(ax)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1], ["LLR sum", "pLDDT"], fontsize=max(6, TICK_SIZE - 2))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return llr_scatter


def _add_plddt_size_legend(ax: Any) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4c78a8",
            markeredgecolor="white",
            markersize=marker_size,
            label=label,
        )
        for marker_size, label in ((4.5, "pLDDT 75"), (7.5, "pLDDT 85"), (10.5, "pLDDT 95"))
    ]
    ax.figure.legend(
        handles=handles,
        frameon=False,
        fontsize=max(6, LEGEND_SIZE - 1),
        loc="upper right",
        bbox_to_anchor=(0.93, 0.94),
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def _readable_column_ticks(row_labels: list[str]) -> tuple[list[int], list[str]]:
    """Return x-axis labels that stay readable for all-variant Eco1 panels."""

    row_count = len(row_labels)
    if row_count <= 30:
        return list(range(row_count)), row_labels
    positions = [0]
    positions.extend(range(10, row_count, 10))
    if row_count - 1 not in positions:
        positions.append(row_count - 1)
    labels = [row_labels[position] if position == 0 else f"rank {position}" for position in positions]
    return positions, labels


def _candidate_ids(feature_rows: list[dict[str, Any]]) -> list[str]:
    ids = sorted({str(row["candidate_id"]) for row in feature_rows})
    return (["wild_type"] if "wild_type" in ids else []) + [
        candidate_id for candidate_id in ids if candidate_id != "wild_type"
    ]


def _wt_feature_sums(feature_rows: list[dict[str, Any]], selected_features: list[int]) -> dict[int, float]:
    wt_sums = {feature: 0.0 for feature in selected_features}
    for row in feature_rows:
        if str(row["candidate_id"]) == "wild_type":
            wt_sums[int(row["feature_index"])] = float(row["activation_sum"])
    return {feature: max(value, 1e-12) for feature, value in wt_sums.items()}


def _candidate_feature_vectors(
    feature_rows: list[dict[str, Any]],
    candidate_ids: list[str],
    selected_features: list[int],
    wt_sums: dict[int, float],
) -> dict[str, list[float]]:
    sums = {(str(row["candidate_id"]), int(row["feature_index"])): float(row["activation_sum"]) for row in feature_rows}
    vectors: dict[str, list[float]] = {}
    for candidate_id in candidate_ids:
        vectors[candidate_id] = [
            math.log2(max(sums.get((candidate_id, feature), 0.0), 1e-12) / wt_sums[feature])
            for feature in selected_features
        ]
    return vectors


def _sae_similarity(vector: list[float]) -> float:
    rmse = math.sqrt(sum(value * value for value in vector) / max(1, len(vector)))
    return 1.0 / (1.0 + rmse)


def _feature_labels(feature_catalog_path: Path, selected_features: list[int]) -> list[str]:
    catalog = {
        int(row["feature_index"]): (str(row.get("label") or ""), str(row.get("description") or ""))
        for row in pq.read_table(feature_catalog_path, columns=["feature_index", "label", "description"]).to_pylist()
    }
    labels: list[str] = []
    for feature in selected_features:
        label, description = catalog.get(feature, ("", ""))
        text = _concise_feature_description(label=label, description=description)
        if text:
            labels.append(f"F{feature} - {text}")
        else:
            labels.append(f"F{feature}")
    return labels


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


def _plddt_by_candidate(*, foldcheck_report_path: Path, foldcheck_ranking_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for path in (foldcheck_report_path, foldcheck_ranking_path):
        if not path.exists():
            continue
        rows = pq.read_table(path, columns=["candidate_id", "plddt"]).to_pylist()
        values.update({str(row["candidate_id"]): float(row["plddt"]) for row in rows if row.get("plddt") is not None})
    return values


def _llr_sum_by_candidate(candidate_table_path: Path, wt_substitution_llr_path: Path) -> dict[str, float]:
    llr_by_substitution = {
        (int(row["canonical_position"]), str(row["alt_aa"])): float(row["llr"])
        for row in pq.read_table(
            wt_substitution_llr_path,
            columns=["canonical_position", "alt_aa", "llr"],
        ).to_pylist()
    }
    values = {"wild_type": 0.0}
    candidate_rows = pq.read_table(candidate_table_path, columns=["candidate_id", "canonical_mutations"]).to_pylist()
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        total = 0.0
        for mutation in row.get("canonical_mutations") or []:
            match = _MUTATION_PATTERN.match(str(mutation))
            if not match:
                raise ValueError(f"Malformed canonical mutation for {candidate_id}: {mutation!r}")
            key = (int(match.group("position")), match.group("alt"))
            if key not in llr_by_substitution:
                raise ValueError(f"Missing ESMC LLR for {candidate_id} mutation {mutation!r}")
            total += llr_by_substitution[key]
        values[candidate_id] = total
    return values


def _row_labels(ordered_ids: list[str]) -> list[str]:
    labels = ["WT Ec86"] if ordered_ids and ordered_ids[0] == "wild_type" else []
    candidate_ids = [candidate_id for candidate_id in ordered_ids if candidate_id != "wild_type"]
    for index, candidate_id in enumerate(candidate_ids, start=1):
        labels.append(f"V{index:03d} {_short_candidate_id(candidate_id)}")
    return labels


def _short_candidate_id(candidate_id: str) -> str:
    return str(candidate_id).removeprefix("thread_candidate_")[:8]


def _symmetric_color_limit(matrix: list[list[float]]) -> float:
    values = sorted(abs(value) for row in matrix for value in row if math.isfinite(value))
    if not values:
        return 1.0
    percentile_index = min(len(values) - 1, max(0, math.ceil(0.98 * len(values)) - 1))
    return max(0.02, min(0.5, values[percentile_index]))


def _missing_row(
    panel_root: Path,
    missing: list[Path],
    *,
    source_tables: list[str],
    reason: str | None = None,
) -> dict[str, Any]:
    message = reason or "Missing SAE/fold/LLR comparison input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id=DELIVERABLE_ID,
        section=SECTION,
        artifact_kind="svg",
        status="skipped_missing_input",
        path=panel_root / "missing_sae_fold_llr_comparison.txt",
        source_tables=source_tables,
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="SAE/fold/LLR comparison plot was skipped because required inputs were missing.",
        description=(
            "The plot requires SAE protein features, candidate mutations, fold metrics, and WT mutation LLR rows."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"source_notebook": "Biohub ESMC SAE feature interpretation notebook"},
        role="review_only",
        skip_reason=message,
    )
