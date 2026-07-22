"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/pipeline.py

Materialize Eco1 SAE window-summary artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.constants import (
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    INTERPRETATION_LIMIT,
    MANIFEST_FILE_NAME,
    METHOD_ID,
    SUMMARY_FILE_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.io import (
    read_candidate_design_classes,
    read_feature_catalog,
    read_mask_rows,
    read_profiles,
    write_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.models import (
    MaterializedSaeWindowSummary,
    WindowSpec,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.vectors import (
    build_window_summary_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.windows import (
    default_window_specs,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_sae_window_summary(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    report_root: Path | None = None,
    residue_features_path: Path | None = None,
    profile_path: Path | None = None,
    feature_catalog_path: Path | None = None,
    candidate_pool_path: Path | None = None,
    mask_set_path: Path | None = None,
    window_specs: tuple[WindowSpec, ...] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedSaeWindowSummary:
    """Materialize SAE window summaries from existing residue-feature rows."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_OUTPUT_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    summary_root = class_root / (report_root or DEFAULT_REPORT_ROOT)
    residue_features = (
        _resolve(root, residue_features_path)
        if residue_features_path
        else class_root / "biohub_esmc_residue_features.parquet"
    )
    profile = _resolve(root, profile_path) if profile_path else class_root / "biohub_esmc_sae_profile.parquet"
    feature_catalog = (
        _resolve(root, feature_catalog_path)
        if feature_catalog_path
        else class_root / "biohub_esmc_feature_catalog.parquet"
    )
    candidate_pool = (
        _resolve(root, candidate_pool_path) if candidate_pool_path else class_root / "candidate_pool.parquet"
    )
    mask_set = _resolve(root, mask_set_path) if mask_set_path else source_root / "mask_set.yaml"
    required_paths = [residue_features, profile]
    if window_specs is None:
        required_paths.append(mask_set)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    specs = window_specs or default_window_specs(read_mask_rows(mask_set))
    profiles = read_profiles(profile)
    design_classes = read_candidate_design_classes(candidate_pool if candidate_pool.exists() else None)
    feature_catalog_rows = read_feature_catalog(feature_catalog if feature_catalog.exists() else None)
    rows = build_window_summary_rows(
        residue_features_path=residue_features,
        profiles=profiles,
        window_specs=specs,
        design_classes=design_classes,
        feature_catalog=feature_catalog_rows,
    )
    summary_path = summary_root / SUMMARY_FILE_NAME
    manifest_path = summary_root / MANIFEST_FILE_NAME
    write_summary(
        summary_path,
        rows,
        metadata={
            "schema_id": "eco1_rt.sae_feature_window_summary",
            "schema_version": "1",
            "status": "materialized",
            "method_id": METHOD_ID,
        },
    )
    _write_manifest(
        manifest_path,
        summary_path=summary_path,
        residue_features_path=residue_features,
        profile_path=profile,
        feature_catalog_path=feature_catalog,
        candidate_pool_path=candidate_pool,
        mask_set_path=mask_set,
        window_specs=specs,
        rows=rows,
        created_at=created_at,
    )
    return MaterializedSaeWindowSummary(summary_path=summary_path, manifest_path=manifest_path)


def _write_manifest(
    path: Path,
    *,
    summary_path: Path,
    residue_features_path: Path,
    profile_path: Path,
    feature_catalog_path: Path,
    candidate_pool_path: Path,
    mask_set_path: Path,
    window_specs: tuple[WindowSpec, ...],
    rows: list[dict[str, object]],
    created_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "eco1_rt.sae_feature_window_summary_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "method_id": METHOD_ID,
        "source_tables": {
            "residue_features": str(residue_features_path),
            "profile": str(profile_path),
            "feature_catalog": str(feature_catalog_path),
            "candidate_pool": str(candidate_pool_path),
            "mask_set": str(mask_set_path),
        },
        "artifact_hashes": _artifact_hashes(
            summary_path=summary_path,
            residue_features_path=residue_features_path,
            profile_path=profile_path,
            feature_catalog_path=feature_catalog_path,
            candidate_pool_path=candidate_pool_path,
            mask_set_path=mask_set_path,
        ),
        "window_count": len(window_specs),
        "row_count": len(rows),
        "windows": [
            {
                "window_id": spec.window_id,
                "window_label": spec.window_label,
                "residue_count": len(spec.residue_positions_1based),
                "residue_positions_1based": list(spec.residue_positions_1based),
                "purpose": spec.purpose,
            }
            for spec in window_specs
        ],
        "interpretation_limit": INTERPRETATION_LIMIT,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _artifact_hashes(
    *,
    summary_path: Path,
    residue_features_path: Path,
    profile_path: Path,
    feature_catalog_path: Path,
    candidate_pool_path: Path,
    mask_set_path: Path,
) -> dict[str, str]:
    paths = {
        "summary": summary_path,
        "residue_features": residue_features_path,
        "profile": profile_path,
    }
    if feature_catalog_path.exists():
        paths["feature_catalog"] = feature_catalog_path
    if candidate_pool_path.exists():
        paths["candidate_pool"] = candidate_pool_path
    if mask_set_path.exists():
        paths["mask_set"] = mask_set_path
    return {key: sha256_uri(value) for key, value in paths.items()}


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()
