"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_umap.py

Biohub ESMC SAE delta-UMAP review visual for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from .constants import SECTION_ESMC_FEATURE_REVIEW

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DELIVERABLE_ID = "biohub_esmc_sae_delta_umap"
FILE_NAME = "sae_delta_umap_vs_wt_llr.svg"
EMBEDDING_METHOD_ID = "umap_delta_activation_sum_cosine_v1"
TITLE = "SAE deltas keep variants near WT"
INTERPRETATION_LIMIT = (
    "This plot embeds Biohub ESMC SAE activation-sum deltas relative to WT. It is not evidence of "
    "discrete biological clusters, protein activity, or whole-protein likelihood."
)
METHOD_SUMMARY = (
    "Build one sparse SAE activation-sum vector per WT or candidate sequence, subtract the WT vector, "
    "embed the delta vectors with UMAP using cosine distance, and color candidate points by the existing "
    "additive WT-context ESMC LLR. WT is plotted as an explicit zero-delta, zero-LLR control."
)
SOURCE_TABLES = [
    "biohub_esmc_sae_profile.parquet",
    "biohub_esmc_protein_features.parquet",
    "biohub_esmc_request_manifest.yaml",
    "biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
]
_LLR_CMAP = LinearSegmentedColormap.from_list(
    "eco1_esmc_llr_diverging",
    [OKABE_ITO["vermillion"], "#f6f8fa", OKABE_ITO["blue"]],
)


def write_sae_delta_umap_panel(
    *,
    panel_root: Path,
    profile_path: Path,
    protein_features_path: Path,
    request_manifest_path: Path,
    candidate_preference_table_path: Path,
) -> dict[str, Any]:
    """Render a SAE-delta UMAP plot joined to the existing additive ESMC LLR table."""

    required = (profile_path, protein_features_path, request_manifest_path, candidate_preference_table_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        return _missing_row(panel_root, missing)
    panel_root.mkdir(parents=True, exist_ok=True)
    model_points = _model_points(
        protein_features_path=protein_features_path,
        candidate_preference_table_path=candidate_preference_table_path,
    )
    if len(model_points.rows) < 3:
        return _missing_row(panel_root, [protein_features_path], reason="SAE UMAP requires at least three sequences")
    embedding, embedding_backend = _embed_delta_matrix(model_points.delta_matrix)
    plot_path = panel_root / FILE_NAME
    _render_sae_delta_umap(plot_path, rows=model_points.rows, embedding=embedding, embedding_backend=embedding_backend)
    evidence_summary = _evidence_summary(model_points=model_points, embedding_backend=embedding_backend)
    projection_label = _projection_label(embedding_backend)
    return make_deliverable_row(
        deliverable_id=DELIVERABLE_ID,
        section=SECTION_ESMC_FEATURE_REVIEW,
        artifact_kind="svg",
        status="rendered",
        path=plot_path,
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes(
            {
                "profile": profile_path,
                "protein_features": protein_features_path,
                "request_manifest": request_manifest_path,
                "variant_llr_scores": candidate_preference_table_path,
            }
        ),
        alt_text=(
            f"{projection_label} scatter plot of Biohub ESMC SAE activation-sum deltas relative to WT. Points are "
            "colored by additive ESMC LLR versus WT and WT is shown as a zero-delta control."
        ),
        description=(
            "Shows whether the synthetic variants separate in SAE activation space after subtracting the "
            "WT activation vector. Color uses the existing additive WT-context ESMC LLR so the view can be "
            "read beside the candidate-preference ranking without calling the values whole-protein likelihood. "
            f"The two-dimensional view uses {projection_label.lower()}."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary=evidence_summary,
        role="review_only",
    )


class _ModelPoints:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        delta_matrix: np.ndarray,
        feature_count: int,
        raw_wt_cosines: list[float],
    ) -> None:
        self.rows = rows
        self.delta_matrix = delta_matrix
        self.feature_count = feature_count
        self.raw_wt_cosines = raw_wt_cosines


def _model_points(*, protein_features_path: Path, candidate_preference_table_path: Path) -> _ModelPoints:
    vectors, feature_indices = _activation_sum_vectors(protein_features_path)
    wt_vector = vectors.get("wild_type")
    if wt_vector is None:
        raise ValueError("Biohub ESMC SAE protein-feature rows must include candidate_id='wild_type'")
    llr_rows = _llr_rows(candidate_preference_table_path)
    candidate_ids = _candidate_order(vectors=vectors, llr_rows=llr_rows)
    matrix_rows: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    raw_wt_cosines: list[float] = []
    for index, candidate_id in enumerate(candidate_ids):
        raw_vector = vectors[candidate_id]
        delta_vector = [raw_vector.get(feature, 0.0) - wt_vector.get(feature, 0.0) for feature in feature_indices]
        matrix_rows.append(delta_vector)
        llr_row = llr_rows.get(candidate_id, {})
        llr_total = 0.0 if candidate_id == "wild_type" else _optional_float(llr_row.get("llr_total"))
        mutation_count = 0 if candidate_id == "wild_type" else _optional_int(llr_row.get("mutation_count"))
        raw_wt_cosine = 1.0 if candidate_id == "wild_type" else _cosine_similarity(wt_vector, raw_vector)
        if candidate_id != "wild_type":
            raw_wt_cosines.append(raw_wt_cosine)
        rows.append(
            {
                "candidate_id": candidate_id,
                "display_label": "WT control" if candidate_id == "wild_type" else f"V{index:03d}",
                "llr_total": llr_total,
                "mutation_count": mutation_count,
                "delta_norm": math.sqrt(sum(value * value for value in delta_vector)),
                "raw_wt_cosine": raw_wt_cosine,
                "is_wt": candidate_id == "wild_type",
            }
        )
    return _ModelPoints(
        rows=rows,
        delta_matrix=np.asarray(matrix_rows, dtype=np.float64),
        feature_count=len(feature_indices),
        raw_wt_cosines=raw_wt_cosines,
    )


def _activation_sum_vectors(path: Path) -> tuple[dict[str, dict[int, float]], list[int]]:
    rows = pq.read_table(path, columns=["candidate_id", "feature_index", "activation_sum"]).to_pylist()
    vectors: dict[str, dict[int, float]] = defaultdict(dict)
    feature_indices: set[int] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        feature_index = int(row["feature_index"])
        vectors[candidate_id][feature_index] = float(row.get("activation_sum") or 0.0)
        feature_indices.add(feature_index)
    if len(vectors) < 3:
        raise ValueError("SAE delta UMAP requires at least three sequence vectors")
    if "wild_type" not in vectors:
        raise ValueError("SAE delta UMAP requires a wild_type activation vector")
    return dict(vectors), sorted(feature_indices)


def _llr_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(path, columns=["candidate_id", "llr_total", "mutation_count"]).to_pylist()
    return {str(row["candidate_id"]): dict(row) for row in rows if str(row.get("candidate_id") or "")}


def _candidate_order(*, vectors: dict[str, dict[int, float]], llr_rows: dict[str, dict[str, Any]]) -> list[str]:
    ids = set(vectors)
    ordered = ["wild_type"]
    ranked_ids = [
        candidate_id for candidate_id, _ in sorted(llr_rows.items(), key=lambda item: -float(item[1]["llr_total"]))
    ]
    ordered.extend(candidate_id for candidate_id in ranked_ids if candidate_id in ids and candidate_id != "wild_type")
    ordered.extend(candidate_id for candidate_id in sorted(ids) if candidate_id not in ordered)
    return ordered


def _embed_delta_matrix(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    if matrix.shape[0] <= 3:
        return _linear_embedding(matrix), "linear_small_candidate_set"
    if not np.any(matrix):
        return _linear_embedding(matrix), "linear_zero_delta"
    from umap import UMAP

    n_neighbors = min(15, max(2, matrix.shape[0] - 1))
    embedding = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=17,
        init="random",
        n_jobs=1,
    ).fit_transform(matrix)
    return np.asarray(embedding, dtype=np.float64), "umap-learn"


def _linear_embedding(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    u_matrix, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    first = u_matrix[:, 0] * singular_values[0]
    second = u_matrix[:, 1] * singular_values[1] if singular_values.size > 1 else np.zeros(centered.shape[0])
    return np.column_stack([first, second]).astype(np.float64)


def _render_sae_delta_umap(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    embedding: np.ndarray,
    embedding_backend: str,
) -> None:
    llr_values = [float(row["llr_total"]) for row in rows if row.get("llr_total") is not None]
    lower = min(min(llr_values), -0.1)
    upper = max(max(llr_values), 0.1)
    norm = TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    fig, ax = plt.subplots(figsize=(8.4, 6.9))
    variant_indices = [index for index, row in enumerate(rows) if not row["is_wt"]]
    sizes = [52.0 + 2.0 * float(rows[index].get("mutation_count") or 0) for index in variant_indices]
    colors = [float(rows[index]["llr_total"]) for index in variant_indices]
    scatter = ax.scatter(
        embedding[variant_indices, 0],
        embedding[variant_indices, 1],
        c=colors,
        cmap=_LLR_CMAP,
        norm=norm,
        s=sizes,
        alpha=0.88,
        edgecolors="#24292f",
        linewidths=0.55,
        label="Synthetic variant",
    )
    wt_index = next(index for index, row in enumerate(rows) if row["is_wt"])
    ax.scatter(
        [embedding[wt_index, 0]],
        [embedding[wt_index, 1]],
        marker="*",
        s=210,
        color=OKABE_ITO["black"],
        edgecolors="#ffffff",
        linewidths=0.8,
        label="WT control",
        zorder=4,
    )
    ax.annotate(
        "WT control",
        xy=(embedding[wt_index, 0], embedding[wt_index, 1]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=TICK_SIZE,
        color="#24292f",
    )
    ax.set_title(TITLE, fontsize=TITLE_SIZE, pad=12)
    axis_prefix = "UMAP" if embedding_backend == "umap-learn" else "Linear projection"
    ax.set_xlabel(f"{axis_prefix} 1", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"{axis_prefix} 2", fontsize=LABEL_SIZE)
    style_open_axes(ax, grid=True)
    ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_SIZE)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Additive ESMC LLR versus WT", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.11, right=0.94, top=0.9, bottom=0.12)
    save_accessible_svg(
        fig,
        path,
        title=TITLE,
        description=(
            f"{axis_prefix} plot of SAE activation-sum deltas relative to wild type. WT is a black star at "
            "the zero-delta control point; synthetic variants are circles colored by additive ESMC LLR."
        ),
    )


def _projection_label(embedding_backend: str) -> str:
    return "UMAP" if embedding_backend == "umap-learn" else "Linear projection"


def _evidence_summary(*, model_points: _ModelPoints, embedding_backend: str) -> dict[str, Any]:
    variant_llr_count = sum(1 for row in model_points.rows if not row["is_wt"] and row.get("llr_total") is not None)
    delta_norms = [float(row["delta_norm"]) for row in model_points.rows]
    return {
        "embedding_method_id": EMBEDDING_METHOD_ID,
        "embedding_backend": embedding_backend,
        "sae_vector_basis": "protein_feature_activation_sum_delta_against_wt",
        "candidate_count": len(model_points.rows),
        "synthetic_variant_count": len(model_points.rows) - 1,
        "feature_count": model_points.feature_count,
        "variant_llr_joined_candidate_count": variant_llr_count,
        "wt_control_llr_total": 0.0,
        "wt_control_likelihood_ratio": 1.0,
        "delta_norm_min": min(delta_norms),
        "delta_norm_median": median(delta_norms),
        "delta_norm_max": max(delta_norms),
        "raw_wt_activation_cosine_min": min(model_points.raw_wt_cosines) if model_points.raw_wt_cosines else None,
        "raw_wt_activation_cosine_max": max(model_points.raw_wt_cosines) if model_points.raw_wt_cosines else None,
    }


def _cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    feature_indices = set(left) | set(right)
    numerator = sum(left.get(feature_index, 0.0) * right.get(feature_index, 0.0) for feature_index in feature_indices)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _missing_row(panel_root: Path, missing: list[Path], *, reason: str | None = None) -> dict[str, Any]:
    message = reason or "Missing SAE delta-UMAP input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id=DELIVERABLE_ID,
        section=SECTION_ESMC_FEATURE_REVIEW,
        artifact_kind="svg",
        status="skipped_missing_input",
        path=panel_root / "missing_sae_delta_umap.txt",
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="SAE delta-UMAP plot was skipped because required inputs were missing.",
        description="The plot requires SAE protein-feature rows, Biohub request provenance, and candidate LLR rows.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title=TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"embedding_method_id": EMBEDDING_METHOD_ID},
        role="review_only",
        skip_reason=message,
    )
