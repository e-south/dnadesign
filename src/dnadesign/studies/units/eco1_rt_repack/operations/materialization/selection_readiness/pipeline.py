"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/pipeline.py

Materialize Eco1 selection-readiness artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    CANDIDATE_SELECTION_PANEL_FILE_NAME,
    CANDIDATE_TRIAGE_TABLE_FILE_NAME,
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    FEASIBILITY_REPORT_FILE_NAME,
    MANIFEST_FILE_NAME,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.feasibility import (
    build_feasibility_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.io import (
    read_rows,
    write_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.models import (
    MaterializedSelectionReadiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    build_selection_panel_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    build_triage_rows,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_selection_readiness(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    selection_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedSelectionReadiness:
    """Materialize feasibility, triage, and class-balanced panel artifacts."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_OUTPUT_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    selected_root = _resolve(root, selection_root) if selection_root else class_root / DEFAULT_SELECTION_DIR_NAME
    paths = _input_paths(class_root=class_root, source_root=source_root)
    required_paths = [
        paths["candidate_pool"],
        paths["foldcheck_report"],
        paths["foldcheck_review"],
        paths["mask_set"],
    ]
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    candidate_rows = read_rows(paths["candidate_pool"])
    foldcheck_report_rows = read_rows(paths["foldcheck_report"])
    fold_review_rows = read_rows(paths["foldcheck_review"])
    llr_300m_rows = read_rows(paths["llr_300m"], required=False)
    llr_6b_rows = read_rows(paths["llr_6b"], required=False)
    sae_window_rows = read_rows(paths["sae_window"], required=False)
    feasibility_path = selected_root / FEASIBILITY_REPORT_FILE_NAME
    triage_path = selected_root / CANDIDATE_TRIAGE_TABLE_FILE_NAME
    panel_path = selected_root / CANDIDATE_SELECTION_PANEL_FILE_NAME
    feasibility_rows = build_feasibility_rows(
        candidate_rows=candidate_rows,
        foldcheck_report_rows=foldcheck_report_rows,
        input_candidate_pool_hash=sha256_uri(paths["candidate_pool"]),
        input_mask_policy_hash=sha256_uri(paths["mask_set"]),
        input_foldcheck_report_hash=sha256_uri(paths["foldcheck_report"]),
        created_at=created_at,
    )
    write_rows(feasibility_path, feasibility_rows, schema_id="eco1_rt.feasibility_report")
    input_hashes = {
        "candidate_pool": sha256_uri(paths["candidate_pool"]),
        "foldcheck_review": sha256_uri(paths["foldcheck_review"]),
        "feasibility_report": sha256_uri(feasibility_path),
        "sae_window_summary": sha256_uri(paths["sae_window"]) if paths["sae_window"].exists() else None,
    }
    triage_rows = build_triage_rows(
        candidate_rows=candidate_rows,
        fold_review_rows=fold_review_rows,
        feasibility_rows=feasibility_rows,
        llr_300m_rows=llr_300m_rows,
        llr_6b_rows=llr_6b_rows,
        sae_window_rows=sae_window_rows,
        input_hashes=input_hashes,
    )
    write_rows(triage_path, triage_rows, schema_id="eco1_rt.candidate_triage_table")
    panel_hashes = dict(input_hashes)
    panel_hashes["candidate_triage_table"] = sha256_uri(triage_path)
    panel_rows = build_selection_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=panel_hashes,
    )
    write_rows(panel_path, panel_rows, schema_id="eco1_rt.candidate_selection_panel")
    manifest_path = selected_root / MANIFEST_FILE_NAME
    _write_manifest(
        manifest_path,
        paths=paths,
        feasibility_path=feasibility_path,
        triage_path=triage_path,
        panel_path=panel_path,
        feasibility_rows=feasibility_rows,
        triage_rows=triage_rows,
        panel_rows=panel_rows,
        created_at=created_at,
    )
    return MaterializedSelectionReadiness(
        feasibility_report_path=feasibility_path,
        candidate_triage_table_path=triage_path,
        candidate_selection_panel_path=panel_path,
        manifest_path=manifest_path,
    )


def _input_paths(*, class_root: Path, source_root: Path) -> dict[str, Path]:
    scoring_root = class_root / "review_deliverables/biohub_esmc_sequence_scoring"
    return {
        "candidate_pool": class_root / "candidate_pool.parquet",
        "foldcheck_report": class_root / "foldcheck_report.parquet",
        "foldcheck_review": class_root / "foldcheck_review/foldcheck_candidate_ranking.parquet",
        "mask_set": source_root / "mask_set.yaml",
        "llr_300m": scoring_root / "biohub_esmc_variant_llr_scores.parquet",
        "llr_6b": scoring_root / "esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        "sae_window": class_root / "biohub_esmc/sae_feature_window_summary.parquet",
    }


def _write_manifest(
    path: Path,
    *,
    paths: dict[str, Path],
    feasibility_path: Path,
    triage_path: Path,
    panel_path: Path,
    feasibility_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    created_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "selection_policy_id": SELECTION_POLICY_ID,
        "governing_rule": "Select one feasible fold-preserved nonredundant representative from each design class.",
        "sae_window_policy": (
            "SAE windows are retained as review evidence but not used for selection because the current pool "
            "does not meaningfully stratify in SAE-window space."
        ),
        "source_tables": {key: str(value) for key, value in paths.items() if value.exists()},
        "artifacts": {
            "feasibility_report": str(feasibility_path),
            "candidate_triage_table": str(triage_path),
            "candidate_selection_panel": str(panel_path),
        },
        "artifact_hashes": {
            key: sha256_uri(value)
            for key, value in {
                **{key: value for key, value in paths.items() if value.exists()},
                "feasibility_report": feasibility_path,
                "candidate_triage_table": triage_path,
                "candidate_selection_panel": panel_path,
            }.items()
        },
        "row_counts": {
            "feasibility_report": len(feasibility_rows),
            "candidate_triage_table": len(triage_rows),
            "candidate_selection_panel": len(panel_rows),
        },
        "hard_gate_allowed_fold_classes": ["strong_fold_preserved", "good_fold_preserved"],
        "default_excluded_fold_classes": ["low_confidence", "review_band"],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()
