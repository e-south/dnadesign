"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sequence_preference.py

Standalone Biohub ESMC candidate-preference deliverables for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
from matplotlib.patches import Patch  # noqa: E402

SECTION = SECTION_ESMC_FEATURE_REVIEW
VARIANT_LLR_FILE_NAME = "biohub_esmc_variant_llr_scores.parquet"
PREFERENCE_PLOT_FILE_NAME = "esmc_candidate_preference_vs_wt.svg"
SEQUENCE_SCORING_MANIFEST_FILE_NAME = "biohub_esmc_sequence_scoring_manifest.yaml"
VARIANT_LLR_SCHEMA_ID = "eco1_rt.biohub_esmc.variant_llr_scores"
SEQUENCE_SCORING_MANIFEST_SCHEMA_ID = "eco1_rt.biohub_esmc.sequence_scoring_manifest"
SCORING_METHOD_ID = "esmc_additive_wt_single_substitution_llr_v1"
PLOT_DELIVERABLE_ID = "biohub_esmc_candidate_preference_vs_wt"
TABLE_DELIVERABLE_ID = "biohub_esmc_variant_llr_scores"
MANIFEST_DELIVERABLE_ID = "biohub_esmc_sequence_scoring_manifest"
TITLE = "Candidate ESMC additive LLR vs WT"
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
_MUTATION_PATTERN = re.compile(r"^[A-Z](?P<position>\d+)(?P<alt>[A-Z])$")
_VARIANT_LLR_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("model", pa.string()),
        ("scoring_method_id", pa.string()),
        ("source_scoring_method_id", pa.string()),
        ("wt_mutation_scoring_request_hash", pa.string()),
        ("mutation_count", pa.int32()),
        ("mutations_scored_count", pa.int32()),
        ("llr_total", pa.float64()),
        ("llr_per_mutation", pa.float64()),
        ("review_class", pa.string()),
        ("plddt", pa.float64()),
        ("wt_runtime_ca_rmsd", pa.float64()),
        ("status", pa.string()),
        ("failure_reason", pa.string()),
    ]
)


def write_biohub_esmc_sequence_preference_deliverables(
    *,
    panel_root: Path,
    candidate_table_path: Path,
    wt_substitution_llr_path: Path,
    wt_mutation_scoring_manifest_path: Path,
    foldcheck_ranking_path: Path,
) -> list[dict[str, Any]]:
    """Write a standalone candidate-preference table and plot from WT ESMC LLR rows."""

    required = (candidate_table_path, wt_substitution_llr_path, wt_mutation_scoring_manifest_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        return [_missing_row(panel_root, missing)]
    panel_root.mkdir(parents=True, exist_ok=True)
    score_rows = build_variant_llr_score_rows(
        candidate_table_path=candidate_table_path,
        wt_substitution_llr_path=wt_substitution_llr_path,
        wt_mutation_scoring_manifest_path=wt_mutation_scoring_manifest_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
    )
    if not score_rows:
        return [_missing_row(panel_root, [candidate_table_path], reason="No candidate rows were available")]
    table_path = panel_root / VARIANT_LLR_FILE_NAME
    write_variant_llr_score_table(table_path, rows=score_rows)
    plot_path = panel_root / PREFERENCE_PLOT_FILE_NAME
    _render_candidate_preference_plot(plot_path, score_rows)
    lane_manifest_path = panel_root / SEQUENCE_SCORING_MANIFEST_FILE_NAME
    upstream_manifest = _load_manifest(wt_mutation_scoring_manifest_path)
    write_sequence_scoring_manifest(
        lane_manifest_path,
        rows=score_rows,
        upstream_manifest=upstream_manifest,
        table_path=table_path,
        plot_path=plot_path,
    )
    source_tables = [
        "candidate_table.parquet",
        "foldcheck_review/foldcheck_candidate_ranking.parquet",
        "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet",
        "biohub_esmc/mutation_scoring/wt_mutation_scoring_manifest.yaml",
    ]
    input_hashes = file_hashes(
        {
            "candidate_table": candidate_table_path,
            "foldcheck_candidate_ranking": foldcheck_ranking_path,
            "wt_substitution_llr": wt_substitution_llr_path,
            "wt_mutation_scoring_manifest": wt_mutation_scoring_manifest_path,
        }
    )
    evidence = _evidence_summary(score_rows)
    return [
        make_deliverable_row(
            deliverable_id=MANIFEST_DELIVERABLE_ID,
            section=SECTION,
            artifact_kind="yaml",
            status="materialized",
            path=lane_manifest_path,
            source_tables=source_tables,
            input_hashes=input_hashes | file_hashes({"variant_llr_scores": table_path, "preference_plot": plot_path}),
            alt_text="YAML provenance manifest for the Biohub ESMC candidate-preference review lane.",
            description=(
                "Records the source WT ESMC masked-marginal run, derived scoring method, model/API "
                "metadata, request counts, and interpretation limits for the candidate-preference table and plot."
            ),
            interpretation_limit=INTERPRETATION_LIMIT,
            title="Biohub ESMC candidate-preference scoring manifest",
            method_summary=METHOD_SUMMARY,
            evidence_summary=evidence,
            role="audit",
            render_mode="manifest",
        ),
        make_deliverable_row(
            deliverable_id=TABLE_DELIVERABLE_ID,
            section=SECTION,
            artifact_kind="parquet",
            status="materialized",
            path=table_path,
            source_tables=source_tables,
            input_hashes=input_hashes,
            alt_text="Table of additive ESMC LLR scores for each ProteinMPNN candidate.",
            description=(
                "One row per ProteinMPNN candidate. The row carries the summed WT-context "
                "single-substitution ESMC LLR and the same score normalized by mutation count."
            ),
            interpretation_limit=INTERPRETATION_LIMIT,
            title="Biohub ESMC additive candidate LLR table",
            method_summary=METHOD_SUMMARY,
            evidence_summary=evidence,
            role="review_only",
            render_mode="table",
        ),
        make_deliverable_row(
            deliverable_id=PLOT_DELIVERABLE_ID,
            section=SECTION,
            artifact_kind="svg",
            status="rendered",
            path=plot_path,
            source_tables=source_tables,
            input_hashes=input_hashes | file_hashes({"variant_llr_scores": table_path}),
            alt_text=(
                "Ranked bar plot of additive WT-context ESMC single-substitution LLR sums for "
                "ProteinMPNN candidates, with WT at zero."
            ),
            description=(
                "Shows whether each candidate's substitutions are assigned a higher or lower summed ESMC "
                "masked-marginal score than the WT residues at those positions. Bars are ordered by total "
                "additive LLR; colors show fold-review class when available."
            ),
            interpretation_limit=INTERPRETATION_LIMIT,
            title=TITLE,
            method_summary=METHOD_SUMMARY,
            evidence_summary=evidence,
            role="manuscript_facing",
            render_mode="wide_visual" if len(score_rows) > 42 else "standard_visual",
        ),
    ]


def build_variant_llr_score_rows(
    *,
    candidate_table_path: Path,
    wt_substitution_llr_path: Path,
    wt_mutation_scoring_manifest_path: Path | None,
    foldcheck_ranking_path: Path | None,
) -> list[dict[str, object]]:
    """Build one additive WT-context ESMC LLR row per candidate."""

    manifest = _load_manifest(wt_mutation_scoring_manifest_path)
    model = str(manifest.get("model") or "")
    source_scoring_method_id = str(manifest.get("scoring_method_id") or "")
    wt_request_hash = str(manifest.get("biohub_request_hash") or "")
    llr_by_substitution = _llr_lookup(wt_substitution_llr_path)
    fold_metrics = _fold_metrics(foldcheck_ranking_path)
    rows: list[dict[str, object]] = []
    for candidate in pq.read_table(candidate_table_path).to_pylist():
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate_table rows must carry candidate_id")
        mutations = [str(value) for value in candidate.get("canonical_mutations") or []]
        total = 0.0
        for mutation in mutations:
            match = _MUTATION_PATTERN.match(mutation)
            if not match:
                raise ValueError(f"Malformed canonical mutation for {candidate_id}: {mutation!r}")
            key = (int(match.group("position")), str(match.group("alt")))
            if key not in llr_by_substitution:
                raise ValueError(f"Missing ESMC LLR for {candidate_id} mutation {mutation!r}")
            total += llr_by_substitution[key]
        mutation_count = int(candidate.get("mutation_count") or len(mutations))
        metrics = fold_metrics.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": str(candidate.get("sequence_hash") or ""),
                "model": model,
                "scoring_method_id": SCORING_METHOD_ID,
                "source_scoring_method_id": source_scoring_method_id,
                "wt_mutation_scoring_request_hash": wt_request_hash,
                "mutation_count": mutation_count,
                "mutations_scored_count": len(mutations),
                "llr_total": float(total),
                "llr_per_mutation": None if len(mutations) == 0 else float(total) / len(mutations),
                "review_class": str(metrics.get("review_class") or ""),
                "plddt": _optional_float(metrics.get("plddt")),
                "wt_runtime_ca_rmsd": _optional_float(metrics.get("wt_runtime_ca_rmsd")),
                "status": "accepted",
                "failure_reason": "",
            }
        )
    return sorted(rows, key=lambda row: (-float(row["llr_total"]), str(row["candidate_id"])))


def write_variant_llr_score_table(path: Path, *, rows: list[dict[str, object]]) -> None:
    """Write additive candidate LLR rows with a schema marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_VARIANT_LLR_SCHEMA)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_id"] = VARIANT_LLR_SCHEMA_ID.encode("utf-8")
    metadata[b"schema_version"] = b"1"
    metadata[b"scoring_method_id"] = SCORING_METHOD_ID.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def write_sequence_scoring_manifest(
    path: Path,
    *,
    rows: list[dict[str, object]],
    upstream_manifest: dict[str, object],
    table_path: Path,
    plot_path: Path,
) -> None:
    """Write lane-local provenance for the derived ESMC candidate-preference score."""

    payload = _sequence_scoring_manifest(
        rows=rows,
        upstream_manifest=upstream_manifest,
        table_path=table_path,
        plot_path=plot_path,
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sequence_scoring_manifest(
    *,
    rows: list[dict[str, object]],
    upstream_manifest: dict[str, object],
    table_path: Path,
    plot_path: Path,
) -> dict[str, object]:
    accepted_count = sum(1 for row in rows if str(row.get("status") or "") == "accepted")
    mutation_counts = [int(row["mutation_count"]) for row in rows]
    upstream_endpoint_flow = upstream_manifest.get("endpoint_flow")
    endpoint_flow = (
        [str(value) for value in upstream_endpoint_flow]
        if isinstance(upstream_endpoint_flow, list) and upstream_endpoint_flow
        else ["POST /api/v1/encode", "POST /api/v1/logits"]
    )
    return {
        "schema_id": SEQUENCE_SCORING_MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "materialization_mode": "derived_from_wt_single_substitution_grid",
        "scoring_method_id": SCORING_METHOD_ID,
        "source_scoring_method_id": str(upstream_manifest.get("scoring_method_id") or ""),
        "model": str(upstream_manifest.get("model") or ""),
        "biohub_api_base_url": str(upstream_manifest.get("biohub_api_base_url") or "https://biohub.ai"),
        "biohub_api_version": str(upstream_manifest.get("biohub_api_version") or "v1"),
        "endpoint_flow": endpoint_flow,
        "authorization": "<redacted>",
        "source_request_hash": str(upstream_manifest.get("source_request_hash") or ""),
        "biohub_request_hash": str(upstream_manifest.get("biohub_request_hash") or ""),
        "candidate_count": len(rows),
        "accepted_candidate_count": accepted_count,
        "errored_candidate_count": len(rows) - accepted_count,
        "additional_biohub_request_count": 0,
        "upstream_position_count": int(upstream_manifest.get("position_count") or 0),
        "upstream_accepted_position_count": int(upstream_manifest.get("accepted_position_count") or 0),
        "upstream_errored_position_count": int(upstream_manifest.get("errored_position_count") or 0),
        "mutation_count_min": min(mutation_counts) if mutation_counts else None,
        "mutation_count_median": median(mutation_counts) if mutation_counts else None,
        "mutation_count_max": max(mutation_counts) if mutation_counts else None,
        "whole_protein_pseudolikelihood_status": "not_materialized_request_heavy",
        "interpretation_limit": INTERPRETATION_LIMIT,
        "method_summary": METHOD_SUMMARY,
        "method_references": upstream_manifest.get("method_references") or [],
        "artifact_hashes": file_hashes({"variant_llr_scores": table_path, "candidate_preference_plot": plot_path}),
    }


def _render_candidate_preference_plot(path: Path, rows: list[dict[str, object]]) -> None:
    ordered = list(rows)
    values = [float(row["llr_total"]) for row in ordered]
    y_positions = list(range(len(ordered)))
    labels = [_rank_label(index, str(row["candidate_id"])) for index, row in enumerate(ordered, start=1)]
    fig_width = max(9.0, min(16.0, 0.06 * len(ordered) + 9.0))
    fig_height = max(5.2, min(18.0, 0.22 * len(ordered) + 2.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = [_review_class_color(str(row.get("review_class") or "")) for row in ordered]
    ax.barh(y_positions, values, color=colors, edgecolor="#ffffff", linewidth=0.45)
    ax.axvline(0.0, color="#24292f", linewidth=1.0)
    ax.set_yticks(y_positions, labels, fontsize=max(6, min(TICK_SIZE, 9)))
    ax.invert_yaxis()
    ax.set_xlabel("WT-context single-substitution LLR sum", fontsize=LABEL_SIZE)
    ax.set_ylabel("ProteinMPNN candidate rank by ESMC additive LLR", fontsize=LABEL_SIZE)
    ax.set_title(TITLE, fontsize=TITLE_SIZE, pad=10)
    style_open_axes(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.legend(
        handles=_legend_handles(ordered),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=min(4, max(1, len({str(row.get("review_class") or "") for row in ordered}))),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.22)
    save_accessible_svg(
        fig,
        path,
        title=TITLE,
        description=(
            f"Ranked bar plot of additive WT-context ESMC LLR sums for {len(ordered)} ProteinMPNN "
            "candidate sequences. Positive values indicate that the ESMC masked-marginal grid assigns "
            "higher probability to the candidate substitutions than to the WT residues at those positions."
        ),
    )


def _llr_lookup(path: Path) -> dict[tuple[int, str], float]:
    rows = pq.read_table(path, columns=["canonical_position", "alt_aa", "llr"]).to_pylist()
    lookup: dict[tuple[int, str], float] = {}
    for row in rows:
        lookup[(int(row["canonical_position"]), str(row["alt_aa"]))] = float(row["llr"])
    if not lookup:
        raise ValueError("WT substitution LLR table must contain at least one row")
    return lookup


def _fold_metrics(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    rows = pq.read_table(path).to_pylist()
    return {str(row.get("candidate_id") or ""): dict(row) for row in rows}


def _load_manifest(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return dict(loaded)


def _evidence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["llr_total"]) for row in rows]
    per_mutation = [float(row["llr_per_mutation"]) for row in rows if row.get("llr_per_mutation") is not None]
    model = str(rows[0].get("model") or "") if rows else ""
    return {
        "candidate_count": len(rows),
        "model": model,
        "scoring_method_id": SCORING_METHOD_ID,
        "llr_total_min": min(values) if values else None,
        "llr_total_median": median(values) if values else None,
        "llr_total_max": max(values) if values else None,
        "llr_per_mutation_median": median(per_mutation) if per_mutation else None,
        "whole_protein_pseudolikelihood_status": "not_materialized_request_heavy",
    }


def _rank_label(index: int, candidate_id: str) -> str:
    return f"V{index:03d} {candidate_id.removeprefix('thread_candidate_')[:8]}"


def _review_class_color(review_class: str) -> str:
    palette = {
        "strong_fold_preserved": OKABE_ITO["blue"],
        "review_band": OKABE_ITO["orange"],
        "fold_watch": OKABE_ITO["purple"],
        "": OKABE_ITO["gray"],
    }
    return palette.get(review_class, OKABE_ITO["green"])


def _legend_handles(rows: list[dict[str, object]]) -> list[Patch]:
    classes = sorted({str(row.get("review_class") or "") for row in rows})
    if "" in classes:
        classes.remove("")
        classes.append("")
    return [
        Patch(
            facecolor=_review_class_color(review_class),
            edgecolor="none",
            label=_review_class_label(review_class),
        )
        for review_class in classes
    ]


def _review_class_label(review_class: str) -> str:
    labels = {
        "strong_fold_preserved": "CA RMSD <= 1.25 A; pLDDT >= 91.5",
        "good_fold_preserved": "CA RMSD <= 2.0 A; pLDDT >= 90",
        "review_band": "Intermediate fold-review band",
        "low_confidence": "pLDDT < 90",
        "structural_outlier": "CA RMSD > 5.0 A",
        "metric_missing": "Fold metric missing",
        "": "Fold class unavailable",
    }
    return labels.get(review_class, review_class.replace("_", " "))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _missing_row(panel_root: Path, missing: list[Path], *, reason: str | None = None) -> dict[str, Any]:
    message = reason or "Missing Biohub ESMC sequence-preference input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id=PLOT_DELIVERABLE_ID,
        section=SECTION,
        artifact_kind="svg",
        status="skipped_missing_input",
        path=panel_root / "missing_biohub_esmc_sequence_preference.txt",
        source_tables=[
            "candidate_table.parquet",
            "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet",
            "biohub_esmc/mutation_scoring/wt_mutation_scoring_manifest.yaml",
        ],
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="Biohub ESMC candidate-preference plot was skipped because required inputs were missing.",
        description="The plot requires candidate mutations and the WT ESMC masked-marginal substitution table.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title=TITLE,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"scoring_method_id": SCORING_METHOD_ID},
        role="review_only",
        skip_reason=message,
    )
