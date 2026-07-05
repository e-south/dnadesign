"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_interpretation_shared.py

Shared helpers for Biohub ESMC SAE interpretation deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    TICK_SIZE,
)

from .biohub_esmc_model_provenance import sae_request_manifest_summary
from .biohub_esmc_sae_tables import (
    FEATURE_PREVALENCE_THRESHOLD,
    thresholded_feature_stats,
    thresholded_stats,
)
from .constants import SECTION_ESMC_FEATURE_REVIEW

SECTION = SECTION_ESMC_FEATURE_REVIEW
TOP_FEATURE_COUNT = 12
RETIRED_OUTPUT_NAMES = (
    "candidate_top_sae_feature_activation_ratio.svg",
    "sae_fold_llr_comparison.svg",
    "missing_sae_fold_llr_comparison.txt",
)
SOURCE_NOTEBOOK = (
    "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
    "esmc_sae_feature_interpretation.ipynb"
)
INTERPRETATION_LIMIT = (
    "These are exact-dictionary Biohub ESMC SAE activations. They support semantic review, "
    "not activity, processivity, strand-displacement, or candidate acceptance claims."
)
METHOD_SUMMARY = (
    "Rank top SAE features from Biohub ESMC sparse activations by peak activation and by the Biohub "
    "tutorial's activation-thresholded prevalence, inspect where those features activate over WT residues, "
    "and render one selected feature at a time across WT plus candidate sequences in the marimo notebook. "
    "Feature names remain blank unless a source-backed interpretation exists for the exact SAE model, "
    "layer, sparsity, and codebook."
)
SOURCE_TABLES = [
    "biohub_esmc_sae_profile.parquet",
    "biohub_esmc_protein_features.parquet",
    "biohub_esmc_residue_features.parquet",
    "biohub_esmc_feature_catalog.parquet",
    "biohub_esmc_request_manifest.yaml",
]


def missing_row(panel_root: Path, missing: list[Path], *, reason: str | None = None) -> dict[str, Any]:
    message = reason or "Missing Biohub ESMC SAE feature-review input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_sae_interpretation",
        section=SECTION,
        artifact_kind="manifest",
        status="skipped_missing_input",
        path=panel_root / "missing_biohub_esmc_sae_interpretation.txt",
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="Biohub ESMC SAE feature-review visuals were skipped because sparse SAE inputs were missing.",
        description="The SAE feature-review section requires profile, protein-feature, and residue-feature tables.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title="Biohub ESMC SAE features support semantic review without acting as gates",
        method_summary=METHOD_SUMMARY,
        evidence_summary={"source_notebook": SOURCE_NOTEBOOK},
        role="review_only",
        skip_reason=message,
    )


def remove_retired_outputs(panel_root: Path) -> None:
    for name in RETIRED_OUTPUT_NAMES:
        path = panel_root / name
        if path.exists():
            path.unlink()


def top_wt_feature_rows(path: Path, *, residue_features_path: Path, top_n: int) -> list[dict[str, Any]]:
    rows = pq.read_table(
        path,
        filters=[("candidate_id", "==", "wild_type")],
        columns=["feature_index", "nonzero_residue_count", "activation_sum", "activation_mean", "activation_max"],
    ).to_pylist()
    thresholded = thresholded_feature_stats(residue_features_path)
    enriched = []
    for row in rows:
        enriched_row = dict(row)
        stats = thresholded_stats(enriched_row | {"candidate_id": "wild_type"}, thresholded)
        enriched_row["prevalent_residue_count"] = int(stats["prevalent_residue_count"])
        enriched_row["mean_prevalent_activation"] = float(stats["mean_prevalent_activation"])
        enriched.append(enriched_row)
    return sorted(
        enriched,
        key=lambda row: (
            float(row["activation_max"]),
            int(row["prevalent_residue_count"]),
            float(row["activation_sum"]),
        ),
        reverse=True,
    )[:top_n]


def position_count(residue_features_path: Path) -> int:
    table = pq.read_table(
        residue_features_path,
        filters=[("candidate_id", "==", "wild_type")],
        columns=["sequence_position_one_based"],
    )
    values = table.column("sequence_position_one_based").to_pylist()
    return max(int(value) for value in values)


def set_position_ticks(ax: Any, position_count_: int) -> None:
    step = 10 if position_count_ <= 360 else max(25, int(math.ceil(position_count_ / 36.0)))
    positions = list(range(1, position_count_ + 1, step))
    if positions[-1] != position_count_:
        positions.append(position_count_)
    ax.set_xticks(
        [position - 1 for position in positions], [str(position) for position in positions], fontsize=TICK_SIZE
    )
    ax.tick_params(axis="x", length=3)


def feature_axis_labels(feature_catalog_path: Path, selected_features: list[int]) -> list[str]:
    catalog = {
        int(row["feature_index"]): (str(row.get("label") or ""), str(row.get("description") or ""))
        for row in pq.read_table(feature_catalog_path, columns=["feature_index", "label", "description"]).to_pylist()
    }
    return [
        feature_axis_label(feature_index, *catalog.get(feature_index, ("", ""))) for feature_index in selected_features
    ]


def feature_axis_label(feature_index: int, label: str, description: str) -> str:
    text = _concise_feature_description(label=label, description=description)
    if not text:
        return f"F{feature_index}"
    return f"F{feature_index} - {text}"


def wt_sequence_from_mask(mask_residues: list[dict[str, Any]]) -> str:
    rows = sorted(mask_residues, key=lambda row: int(row["canonical_position"]))
    return "".join(str(row.get("wt_aa") or "X")[:1] for row in rows)


def relative_to(path_root: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), start=path_root.resolve())


def evidence_summary(
    feature_rows: list[dict[str, Any]],
    *,
    request_manifest_path: Path,
    sae_provenance_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = sae_request_manifest_summary(request_manifest_path) | {
        "source_notebook": SOURCE_NOTEBOOK,
        "feature_selection_rule": (
            "top WT features by activation_max, tie-broken by >0.01 activation prevalence and activation_sum"
        ),
        "prevalence_activation_threshold": FEATURE_PREVALENCE_THRESHOLD,
        "selected_feature_count": len(feature_rows),
        "selected_feature_indices": [int(row["feature_index"]) for row in feature_rows],
    }
    if sae_provenance_audit is not None:
        summary["sae_provenance_audit"] = dict(sae_provenance_audit)
    return summary


def _concise_feature_description(*, label: str, description: str) -> str:
    source = (label or description).strip()
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
    return value[: max_chars - 3].rstrip() + "..."
