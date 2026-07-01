"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/pipeline.py

Materialize Eco1 RT design-class ProteinMPNN request surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    CONSERVATION_SOURCES_PATH,
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    DESIGN_CLASS_MANIFEST_FILE_NAME,
    PROFILE_PATH,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.masking import (
    compose_design_class_mask_rows,
    summarize_design_class_mask_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    DesignClassArtifact,
    DesignClassSpec,
    MaterializedDesignClassRequests,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
    BASELINE_SPEC,
    select_specs,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority import (
    materialize_manual_mask_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request import (
    materialize_proteinmpnn_request,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import (
    materialize_thread_plan,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_design_class_requests(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    class_ids: list[str] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedDesignClassRequests:
    """Materialize class-specific mask sets, thread plans, and ProteinMPNN sidecars."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_DESIGN_CLASSES_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    class_root.mkdir(parents=True, exist_ok=True)
    shared = _load_shared_inputs(root=root, source_root=source_root, created_at=created_at)
    selected_specs = select_specs(class_ids)
    class_artifacts: list[DesignClassArtifact] = []
    manifest_rows = [_baseline_manifest_row(source_root=source_root)]
    for spec in selected_specs:
        artifact = _materialize_one_class(
            root=root,
            class_root=class_root,
            source_root=source_root,
            spec=spec,
            shared=shared,
            created_at=created_at,
        )
        class_artifacts.append(artifact)
        manifest_rows.append(_generated_manifest_row(spec=spec, artifact=artifact))
    manifest = {
        "schema_id": "eco1_rt.design_class_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "source_output_root": str(source_root),
        "design_classes_root": str(class_root),
        "deduplication_key": "sequence_hash",
        "sampling_note": (
            "The 5 A class is the existing baseline. Generated classes reuse the same ProteinMPNN "
            "sampling shape by default; candidate-pool materialization removes duplicate sequences "
            "across classes before fold checking."
        ),
        "available_design_class_ids": [spec.design_class_id for spec in ALL_SPECS],
        "design_classes": manifest_rows,
    }
    manifest_path = class_root / DESIGN_CLASS_MANIFEST_FILE_NAME
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return MaterializedDesignClassRequests(
        manifest_path=manifest_path,
        class_artifacts=tuple(class_artifacts),
    )


def _materialize_one_class(
    *,
    root: Path,
    class_root: Path,
    source_root: Path,
    spec: DesignClassSpec,
    shared: dict[str, Any],
    created_at: str,
) -> DesignClassArtifact:
    one_root = class_root / spec.design_class_id
    one_root.mkdir(parents=True, exist_ok=True)
    _copy_shared_runtime_inputs(source_root=source_root, class_root=one_root)
    rows = compose_design_class_mask_rows(
        spec=spec,
        residue_rows=shared["residue_rows"],
        contact_geometry_rows=shared["contact_geometry_rows"],
        conservation_rows=shared["conservation_rows"],
        manual_authority=shared["manual_authority"],
    )
    mask_set = _build_mask_set(
        spec=spec,
        rows=rows,
        upstream_hashes={
            "profile": sha256_uri(root / PROFILE_PATH),
            "conservation_sources": sha256_uri(root / CONSERVATION_SOURCES_PATH),
            "residue_map": sha256_uri(source_root / "residue_map.parquet"),
            "contact_geometry_profile": sha256_uri(source_root / "contact_geometry_profile.parquet"),
            "conservation_profile": sha256_uri(source_root / "conservation_profile.parquet"),
            "manual_mask_authority": sha256_uri(source_root / "manual_mask_authority.yaml"),
        },
        created_at=created_at,
    )
    mask_set_path = one_root / "mask_set.yaml"
    mask_set_path.write_text(yaml.safe_dump(mask_set, sort_keys=False), encoding="utf-8")
    thread_plan = materialize_thread_plan(
        repo_root=root,
        output_root=one_root,
        created_at=created_at,
        expected_mask_policy_id=spec.design_class_id,
        sampling_policy_overrides={"batch_id": spec.batch_id},
        artifact_id=f"eco1_rt_design_classes_v1.{spec.path_id}.thread_plan",
    )
    request = materialize_proteinmpnn_request(repo_root=root, output_root=one_root)
    return DesignClassArtifact(
        design_class_id=spec.design_class_id,
        class_root=one_root,
        mask_set_path=mask_set_path,
        thread_plan_path=thread_plan.thread_plan_path,
        request_manifest_path=request.request_manifest_path,
    )


def _build_mask_set(
    *,
    spec: DesignClassSpec,
    rows: list[dict[str, Any]],
    upstream_hashes: dict[str, str],
    created_at: str,
) -> dict[str, Any]:
    non_fixed_mapped_count = sum(1 for row in rows if row["non_fixed"])
    return {
        "schema_id": "thread.mask_set",
        "schema_version": 1,
        "artifact_id": f"eco1_rt_design_classes_v1.{spec.path_id}.mask_set",
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "profile_id": "eco1_rt_v1",
        "mask_policy_id": spec.design_class_id,
        "design_class_id": spec.design_class_id,
        "sampling_status": (
            "blocked_no_non_fixed_mapped_positions" if non_fixed_mapped_count == 0 else "pending_sampling_plan"
        ),
        "sampling_allowed": non_fixed_mapped_count > 0,
        "manual_mask_authority_status": "materialized_eco1_rt_manual_motif_wang_direct_contact_v1",
        "cysteine_policy": "no_new_cysteine_candidate_ingest",
        "source_method_id": "tao_style_fixed_backbone_rt_repack_sensitivity_v1",
        "premise": spec.premise,
        "rationale": spec.rationale,
        "upstream_artifact_hashes": upstream_hashes,
        "summary": summarize_design_class_mask_rows(rows, spec=spec),
        "residues": rows,
    }


def _load_shared_inputs(*, root: Path, source_root: Path, created_at: str) -> dict[str, Any]:
    for required in (
        source_root / "residue_map.parquet",
        source_root / "contact_geometry_profile.parquet",
        source_root / "conservation_profile.parquet",
        source_root / "backbone_bundle.yaml",
        root / PROFILE_PATH,
        root / CONSERVATION_SOURCES_PATH,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    manual_path = source_root / "manual_mask_authority.yaml"
    if not manual_path.exists():
        materialize_manual_mask_authority(repo_root=root, output_root=source_root, created_at=created_at)
    return {
        "residue_rows": pq.read_table(source_root / "residue_map.parquet").to_pylist(),
        "contact_geometry_rows": pq.read_table(source_root / "contact_geometry_profile.parquet").to_pylist(),
        "conservation_rows": pq.read_table(source_root / "conservation_profile.parquet").to_pylist(),
        "manual_authority": _load_yaml(manual_path),
    }


def _copy_shared_runtime_inputs(*, source_root: Path, class_root: Path) -> None:
    for file_name in ("backbone_bundle.yaml", "residue_map.parquet"):
        shutil.copyfile(source_root / file_name, class_root / file_name)


def _baseline_manifest_row(*, source_root: Path) -> dict[str, Any]:
    mask_set_path = source_root / "mask_set.yaml"
    candidate_table_path = source_root / "candidate_table.parquet"
    summary = _load_yaml(mask_set_path).get("summary", {}) if mask_set_path.exists() else {}
    return {
        "design_class_id": BASELINE_SPEC.design_class_id,
        "path_id": BASELINE_SPEC.path_id,
        "role": BASELINE_SPEC.role,
        "premise": BASELINE_SPEC.premise,
        "rationale": BASELINE_SPEC.rationale,
        "conservation_profile_id": BASELINE_SPEC.conservation_profile_id,
        "conservation_threshold": BASELINE_SPEC.conservation_threshold,
        "contact_threshold_angstrom": BASELINE_SPEC.contact_threshold_angstrom,
        "batch_id": BASELINE_SPEC.batch_id,
        "class_root": str(source_root),
        "mask_set_path": str(mask_set_path),
        "candidate_table_path": str(candidate_table_path),
        "protected_position_count": summary.get("protected_position_count"),
        "non_fixed_mapped_position_count": summary.get("non_fixed_mapped_position_count"),
        "status": "existing_baseline_referenced",
    }


def _generated_manifest_row(*, spec: DesignClassSpec, artifact: DesignClassArtifact) -> dict[str, Any]:
    mask_set = _load_yaml(artifact.mask_set_path)
    summary = mask_set["summary"]
    return {
        "design_class_id": spec.design_class_id,
        "path_id": spec.path_id,
        "role": spec.role,
        "premise": spec.premise,
        "rationale": spec.rationale,
        "conservation_profile_id": spec.conservation_profile_id,
        "conservation_threshold": spec.conservation_threshold,
        "contact_threshold_angstrom": spec.contact_threshold_angstrom,
        "batch_id": spec.batch_id,
        "class_root": str(artifact.class_root),
        "mask_set_path": str(artifact.mask_set_path),
        "thread_plan_path": str(artifact.thread_plan_path),
        "request_manifest_path": str(artifact.request_manifest_path),
        "mask_set_hash": sha256_uri(artifact.mask_set_path),
        "thread_plan_hash": sha256_uri(artifact.thread_plan_path),
        "request_manifest_hash": sha256_uri(artifact.request_manifest_path),
        "protected_position_count": summary["protected_position_count"],
        "non_fixed_mapped_position_count": summary["non_fixed_mapped_position_count"],
        "expected_sample_count": _load_yaml(artifact.thread_plan_path)["expected_sample_count"],
        "status": "request_materialized_pending_proteinmpnn_execution",
    }


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
