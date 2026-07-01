"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sequence_preference_model_stability.py

Biohub ESMC candidate-preference model-stability deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .biohub_esmc_sequence_preference_plot import render_model_stability_plot
from .constants import SECTION_ESMC_FEATURE_REVIEW

SECTION = SECTION_ESMC_FEATURE_REVIEW
MODEL_STABILITY_TITLE = "300M and 6B ESMC additive LLR comparison"
MODEL_STABILITY_TABLE_FILE_NAME = "esmc_candidate_preference_model_stability.parquet"
MODEL_STABILITY_PLOT_FILE_NAME = "esmc_candidate_preference_model_stability.svg"
MODEL_STABILITY_TABLE_DELIVERABLE_ID = "biohub_esmc_candidate_preference_model_stability_table"
MODEL_STABILITY_PLOT_DELIVERABLE_ID = "biohub_esmc_candidate_preference_model_stability"
INTERPRETATION_LIMIT = (
    "This plot sums WT-context masked-marginal single-substitution LLR values for each candidate. "
    "It is not a whole-protein pseudo-likelihood, not a joint likelihood, and not an activity measurement."
)
METHOD_SUMMARY = (
    "Each ProteinMPNN candidate is reduced to its canonical substitutions. For each substitution, the "
    "materializer reads the WT-only Biohub ESMC masked-marginal table and adds log P(alternate residue) "
    "- log P(WT residue) from the corresponding single-position masked context. The table and plot expose "
    "the total and per-mutation additive LLR so this model-derived preference signal can be reviewed "
    "separately from SAE feature activations and ColabFold metrics."
)
_MODEL_STABILITY_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("left_model", pa.string()),
        ("right_model", pa.string()),
        ("left_llr_total", pa.float64()),
        ("right_llr_total", pa.float64()),
        ("left_rank", pa.int32()),
        ("right_rank", pa.int32()),
        ("rank_delta_right_minus_left", pa.int32()),
        ("sign_change", pa.bool_()),
        ("mutation_count", pa.int32()),
        ("review_class", pa.string()),
    ]
)


def write_biohub_esmc_model_stability_deliverables(
    *,
    panel_root: Path,
    left_table_path: Path,
    right_table_path: Path,
    left_label: str = "300M",
    right_label: str = "6B",
) -> list[dict[str, Any]]:
    """Write a two-model candidate-preference stability table and plot."""

    missing = [path for path in (left_table_path, right_table_path) if not path.exists()]
    if missing:
        return [_missing_model_stability_row(panel_root, missing)]
    rows = build_model_stability_rows(left_table_path=left_table_path, right_table_path=right_table_path)
    if not rows:
        return [
            _missing_model_stability_row(
                panel_root,
                [left_table_path, right_table_path],
                reason="No shared candidates",
            )
        ]
    panel_root.mkdir(parents=True, exist_ok=True)
    table_path = panel_root / MODEL_STABILITY_TABLE_FILE_NAME
    plot_path = panel_root / MODEL_STABILITY_PLOT_FILE_NAME
    write_model_stability_table(table_path, rows=rows)
    render_model_stability_plot(
        plot_path,
        rows,
        title=MODEL_STABILITY_TITLE,
        left_label=left_label,
        right_label=right_label,
    )
    evidence = _model_stability_evidence(rows)
    source_tables = [
        "review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
        "review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
    ]
    input_hashes = file_hashes({"left_candidate_llr": left_table_path, "right_candidate_llr": right_table_path})
    return [
        make_deliverable_row(
            deliverable_id=MODEL_STABILITY_TABLE_DELIVERABLE_ID,
            section=SECTION,
            artifact_kind="parquet",
            status="materialized",
            path=table_path,
            source_tables=source_tables,
            input_hashes=input_hashes,
            alt_text="Table comparing 300M and 6B additive ESMC LLR ranks for each shared candidate.",
            description=(
                "One row per shared ProteinMPNN candidate with additive WT-context LLR totals and rank deltas "
                "between the 300M and 6B ESMC scoring lanes."
            ),
            interpretation_limit=INTERPRETATION_LIMIT,
            title="Biohub ESMC candidate-preference model-stability table",
            method_summary=METHOD_SUMMARY,
            evidence_summary=evidence,
            role="review_only",
            render_mode="table",
        ),
        make_deliverable_row(
            deliverable_id=MODEL_STABILITY_PLOT_DELIVERABLE_ID,
            section=SECTION,
            artifact_kind="svg",
            status="rendered",
            path=plot_path,
            source_tables=source_tables,
            input_hashes=input_hashes | file_hashes({"model_stability_table": table_path}),
            alt_text="Scatter plot comparing 300M and 6B additive ESMC LLR totals for shared candidates.",
            description=(
                "Compares whether candidate additive LLR rankings are stable after rescoring the same WT "
                "single-substitution contexts with the 6B ESMC model."
            ),
            interpretation_limit=INTERPRETATION_LIMIT,
            title=MODEL_STABILITY_TITLE,
            method_summary=METHOD_SUMMARY,
            evidence_summary=evidence,
            role="manuscript_facing",
            render_mode="standard_visual",
        ),
    ]


def build_model_stability_rows(*, left_table_path: Path, right_table_path: Path) -> list[dict[str, object]]:
    """Join two candidate-preference tables and compute rank-stability fields."""

    left_rows = _candidate_llr_rows(left_table_path)
    right_rows = _candidate_llr_rows(right_table_path)
    shared = sorted(set(left_rows) & set(right_rows))
    left_ranks = _rank_by_llr(left_rows)
    right_ranks = _rank_by_llr(right_rows)
    rows: list[dict[str, object]] = []
    for candidate_id in shared:
        left = left_rows[candidate_id]
        right = right_rows[candidate_id]
        left_llr = float(left["llr_total"])
        right_llr = float(right["llr_total"])
        left_rank = left_ranks[candidate_id]
        right_rank = right_ranks[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "left_model": str(left.get("model") or ""),
                "right_model": str(right.get("model") or ""),
                "left_llr_total": left_llr,
                "right_llr_total": right_llr,
                "left_rank": left_rank,
                "right_rank": right_rank,
                "rank_delta_right_minus_left": right_rank - left_rank,
                "sign_change": (left_llr < 0 <= right_llr) or (right_llr < 0 <= left_llr),
                "mutation_count": int(right.get("mutation_count") or left.get("mutation_count") or 0),
                "review_class": str(right.get("review_class") or left.get("review_class") or ""),
            }
        )
    return sorted(rows, key=lambda row: (int(row["right_rank"]), str(row["candidate_id"])))


def write_model_stability_table(path: Path, *, rows: list[dict[str, object]]) -> None:
    """Write model-stability rows with a schema marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_MODEL_STABILITY_SCHEMA)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_id"] = b"eco1_rt.biohub_esmc.candidate_preference_model_stability"
    metadata[b"schema_version"] = b"1"
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _candidate_llr_rows(path: Path) -> dict[str, dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    return {str(row["candidate_id"]): dict(row) for row in rows if row.get("candidate_id")}


def _rank_by_llr(rows: dict[str, dict[str, object]]) -> dict[str, int]:
    ordered = sorted(rows.values(), key=lambda row: (-float(row["llr_total"]), str(row["candidate_id"])))
    return {str(row["candidate_id"]): index for index, row in enumerate(ordered, start=1)}


def _model_stability_evidence(rows: list[dict[str, object]]) -> dict[str, object]:
    left_ranks = [float(row["left_rank"]) for row in rows]
    right_ranks = [float(row["right_rank"]) for row in rows]
    top_n = min(10, len(rows))
    top_left = {str(row["candidate_id"]) for row in sorted(rows, key=lambda row: int(row["left_rank"]))[:top_n]}
    top_right = {str(row["candidate_id"]) for row in sorted(rows, key=lambda row: int(row["right_rank"]))[:top_n]}
    return {
        "candidate_count": len(rows),
        "left_model": str(rows[0].get("left_model") or "") if rows else "",
        "right_model": str(rows[0].get("right_model") or "") if rows else "",
        "spearman_rank_correlation": _pearson(left_ranks, right_ranks),
        "top_n": top_n,
        "top_n_overlap_count": len(top_left & top_right),
        "sign_change_count": sum(1 for row in rows if bool(row.get("sign_change"))),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0.0 or y_denom == 0.0:
        return None
    return numerator / (x_denom * y_denom)


def _missing_model_stability_row(
    panel_root: Path,
    missing: list[Path],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    message = reason or "Missing Biohub ESMC model-stability input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id=MODEL_STABILITY_PLOT_DELIVERABLE_ID,
        section=SECTION,
        artifact_kind="svg",
        status="skipped_missing_input",
        path=panel_root / "missing_biohub_esmc_model_stability.txt",
        source_tables=[
            "review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
            "review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        ],
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="Biohub ESMC model-stability plot was skipped because required inputs were missing.",
        description="The plot requires both the default and 6B additive candidate LLR tables.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title=MODEL_STABILITY_TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"scoring_method_id": "candidate_preference_model_stability_v1"},
        role="review_only",
        skip_reason=message,
    )
