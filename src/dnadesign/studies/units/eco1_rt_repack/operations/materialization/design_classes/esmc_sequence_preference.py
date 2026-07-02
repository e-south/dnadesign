"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/esmc_sequence_preference.py

Expanded ESMC additive candidate-preference materialization for Eco1 RT design classes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.downstream_inputs import (
    STAGED_CANDIDATE_TABLE_FILE_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    MaterializedDesignClassEsmcSequencePreference,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

from ..review_deliverables.biohub_esmc_sequence_preference import (
    TITLE_6B,
    VARIANT_LLR_FILE_NAME,
    write_biohub_esmc_sequence_preference_deliverables,
)
from ..review_deliverables.biohub_esmc_sequence_preference_model_stability import (
    write_biohub_esmc_model_stability_deliverables,
)

MANIFEST_FILE_NAME = "design_class_esmc_sequence_preference_manifest.yaml"
DELIVERABLE_ROOT = Path("review_deliverables/biohub_esmc_sequence_scoring")
DEFAULT_WT_SCORING_ROOT = Path("biohub_esmc/mutation_scoring")
SIX_B_WT_SCORING_ROOT = DEFAULT_WT_SCORING_ROOT / "esmc_6b_2024_12"
FOLDCHECK_RANKING_PATH = Path("foldcheck_review/foldcheck_candidate_ranking.parquet")


def materialize_design_class_esmc_sequence_preference(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
) -> MaterializedDesignClassEsmcSequencePreference:
    """Write expanded candidate ESMC additive LLR deliverables from WT grids."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_DESIGN_CLASSES_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    candidate_table_path = class_root / STAGED_CANDIDATE_TABLE_FILE_NAME
    foldcheck_ranking_path = class_root / FOLDCHECK_RANKING_PATH
    for required in (candidate_table_path, foldcheck_ranking_path):
        if not required.exists():
            raise FileNotFoundError(required)

    deliverables: list[dict[str, Any]] = []
    scoring_root = _scoring_root(class_root=class_root, source_root=source_root, relative_path=DEFAULT_WT_SCORING_ROOT)
    deliverables.extend(
        write_biohub_esmc_sequence_preference_deliverables(
            panel_root=class_root / DELIVERABLE_ROOT,
            candidate_table_path=candidate_table_path,
            wt_substitution_llr_path=scoring_root / "wt_substitution_llr.parquet",
            wt_mutation_scoring_manifest_path=scoring_root / "wt_mutation_scoring_manifest.yaml",
            foldcheck_ranking_path=foldcheck_ranking_path,
            source_tables=[
                STAGED_CANDIDATE_TABLE_FILE_NAME,
                str(FOLDCHECK_RANKING_PATH),
                str(DEFAULT_WT_SCORING_ROOT / "wt_substitution_llr.parquet"),
                str(DEFAULT_WT_SCORING_ROOT / "wt_mutation_scoring_manifest.yaml"),
            ],
        )
    )
    six_b_root = _scoring_root(class_root=class_root, source_root=source_root, relative_path=SIX_B_WT_SCORING_ROOT)
    six_b_panel_root = class_root / DELIVERABLE_ROOT / "esmc_6b_2024_12"
    deliverables.extend(
        write_biohub_esmc_sequence_preference_deliverables(
            panel_root=six_b_panel_root,
            candidate_table_path=candidate_table_path,
            wt_substitution_llr_path=six_b_root / "wt_substitution_llr.parquet",
            wt_mutation_scoring_manifest_path=six_b_root / "wt_mutation_scoring_manifest.yaml",
            foldcheck_ranking_path=foldcheck_ranking_path,
            deliverable_id_prefix="biohub_esmc_6b",
            title=TITLE_6B,
            source_tables=[
                STAGED_CANDIDATE_TABLE_FILE_NAME,
                str(FOLDCHECK_RANKING_PATH),
                str(SIX_B_WT_SCORING_ROOT / "wt_substitution_llr.parquet"),
                str(SIX_B_WT_SCORING_ROOT / "wt_mutation_scoring_manifest.yaml"),
            ],
        )
    )
    deliverables.extend(
        write_biohub_esmc_model_stability_deliverables(
            panel_root=class_root / DELIVERABLE_ROOT,
            left_table_path=class_root / DELIVERABLE_ROOT / VARIANT_LLR_FILE_NAME,
            right_table_path=six_b_panel_root / VARIANT_LLR_FILE_NAME,
        )
    )
    manifest_path = class_root / MANIFEST_FILE_NAME
    _write_manifest(
        manifest_path,
        output_root=class_root,
        source_output_root=source_root,
        deliverables=deliverables,
    )
    return MaterializedDesignClassEsmcSequencePreference(
        manifest_path=manifest_path,
        deliverable_count=len(deliverables),
    )


def _scoring_root(*, class_root: Path, source_root: Path, relative_path: Path) -> Path:
    staged = class_root / relative_path
    if staged.exists():
        return staged
    source = source_root / relative_path
    if source.exists():
        return source
    raise FileNotFoundError(staged)


def _write_manifest(
    path: Path,
    *,
    output_root: Path,
    source_output_root: Path,
    deliverables: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_deliverables = [_with_relative_path(row, output_root) for row in deliverables]
    payload = {
        "schema_id": "eco1_rt.design_class_esmc_sequence_preference_manifest",
        "schema_version": 1,
        "status": _status(relative_deliverables),
        "output_root": str(output_root),
        "source_output_root": str(source_output_root),
        "materialization_mode": "derived_from_existing_wt_single_substitution_grids",
        "additional_biohub_request_count": 0,
        "deliverable_count": len(relative_deliverables),
        "deliverables": relative_deliverables,
        "artifact_hashes": {
            str(row["deliverable_id"]): sha256_uri(output_root / str(row["path"]))
            for row in relative_deliverables
            if (output_root / str(row["path"])).exists()
        },
        "interpretation_limit": (
            "The additive LLR values are sums of WT-context masked-marginal single-substitution scores. "
            "They are not whole-protein likelihoods and are not activity measurements."
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _with_relative_path(row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    normalized = dict(row)
    normalized["path"] = _relative_or_absolute(Path(str(row["path"])), output_root)
    return normalized


def _status(deliverables: list[dict[str, Any]]) -> str:
    if any(str(row.get("status") or "").startswith(("errored", "skipped_missing_input")) for row in deliverables):
        return "materialized_degraded"
    return "materialized_complete"


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, start=root)
    except ValueError:
        return str(path)
