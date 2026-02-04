"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/library_artifacts.py

Stage-B library artifact loading and persistence helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import resolve_outputs_scoped_path
from ..artifacts.library import (
    LibraryArtifact,
    LibraryRecord,
    load_library_artifact,
    load_library_records,
    write_library_artifact,
)
from ..run_paths import display_path
from .attempts import _load_existing_library_index_by_plan
from .sequence_validation import _validate_library_constraints


@dataclass(frozen=True)
class LibrarySourceState:
    source: str
    artifact: LibraryArtifact | None
    records: dict[tuple[str, str], list[LibraryRecord]] | None
    cursor: dict[tuple[str, str], int] | None


def prepare_library_source(
    *,
    sampling_cfg,
    cfg_path: Path,
    run_root: Path,
    plan_items: list,
    plan_pools: dict[str, object],
    tables_root: Path,
) -> LibrarySourceState:
    library_records: dict[tuple[str, str], list[LibraryRecord]] | None = None
    library_cursor: dict[tuple[str, str], int] | None = None
    library_artifact: LibraryArtifact | None = None
    library_source = str(getattr(sampling_cfg, "library_source", "build")).lower()
    if library_source == "artifact":
        artifact_path = resolve_outputs_scoped_path(
            cfg_path,
            run_root,
            sampling_cfg.library_artifact_path,
            label="sampling.library_artifact_path",
        )
        if not artifact_path.exists():
            artifact_label = display_path(artifact_path, run_root, absolute=False)
            raise RuntimeError(f"Library artifact directory not found: {artifact_label}")
        library_artifact = load_library_artifact(artifact_path)
        library_records = load_library_records(library_artifact)
        library_cursor = {}
        existing_library_by_plan = _load_existing_library_index_by_plan(tables_root)
        for plan_item in plan_items:
            spec = plan_pools[plan_item.name]
            constraints = plan_item.regulator_constraints
            groups = list(constraints.groups or [])
            plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
            key = (spec.pool_name, plan_item.name)
            records = library_records.get(key)
            if not records:
                raise RuntimeError(
                    f"Library artifact missing libraries for {spec.pool_name}/{plan_item.name}. "
                    "Build libraries with `dense stage-b build-libraries` using this config."
                )
            max_used = existing_library_by_plan.get(key, 0)
            used_count = sum(1 for rec in records if int(rec.library_index) <= int(max_used))
            if max_used and used_count == 0:
                raise RuntimeError(
                    f"Library artifact indices do not cover previously used library_index={max_used} "
                    f"for {spec.pool_name}/{plan_item.name}."
                )
            library_cursor[key] = used_count
            for rec in records:
                if int(rec.library_index) <= 0:
                    raise RuntimeError(
                        f"Library artifact has non-positive library_index={rec.library_index} "
                        f"for {spec.pool_name}/{plan_item.name}."
                    )
                if rec.library_sampling_strategy is None or rec.pool_strategy is None:
                    raise RuntimeError(
                        f"Library artifact missing Stage-B sampling metadata for {spec.pool_name}/{plan_item.name} "
                        f"(library_index={rec.library_index})."
                    )
                _validate_library_constraints(
                    rec,
                    groups=groups,
                    min_count_by_regulator=plan_min_count_by_regulator,
                    input_name=spec.pool_name,
                    plan_name=plan_item.name,
                )
    elif library_source != "build":
        raise RuntimeError(f"Unsupported Stage-B sampling.library_source: {library_source}")
    return LibrarySourceState(
        source=library_source,
        artifact=library_artifact,
        records=library_records,
        cursor=library_cursor,
    )


def write_library_artifacts(
    *,
    library_source: str,
    library_artifact: LibraryArtifact | None,
    library_build_rows: list[dict],
    library_member_rows: list[dict],
    outputs_root: Path,
    cfg_path: Path,
    run_id: str,
    run_root: Path,
    config_hash: str,
    pool_manifest_hash: str | None,
) -> None:
    libraries_dir = outputs_root / "libraries"
    if library_source == "artifact":
        if library_artifact is None:
            raise RuntimeError("Stage-B sampling.library_source=artifact but no library artifact was loaded.")
        try:
            build_rows = pd.read_parquet(library_artifact.builds_path).to_dict("records")
            member_rows = pd.read_parquet(library_artifact.members_path).to_dict("records")
            write_library_artifact(
                out_dir=libraries_dir,
                builds=build_rows,
                members=member_rows,
                cfg_path=cfg_path,
                run_id=str(run_id),
                run_root=run_root,
                overwrite=True,
                config_hash=config_hash,
                pool_manifest_hash=pool_manifest_hash,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write library artifacts: {exc}") from exc
        return
    if not library_build_rows:
        return
    existing_builds: list[dict] = []
    existing_members: list[dict] = []
    builds_path = libraries_dir / "library_builds.parquet"
    members_path = libraries_dir / "library_members.parquet"
    if builds_path.exists():
        existing_builds = pd.read_parquet(builds_path).to_dict("records")
    if members_path.exists():
        existing_members = pd.read_parquet(members_path).to_dict("records")

    existing_indices = {
        int(row.get("library_index") or 0) for row in existing_builds if row.get("library_index") is not None
    }
    new_builds = [row for row in library_build_rows if int(row.get("library_index") or 0) not in existing_indices]
    build_rows = existing_builds + new_builds

    existing_member_keys = {
        (
            int(row.get("library_index") or 0),
            int(row.get("position") or 0),
        )
        for row in existing_members
    }
    new_members = [
        row
        for row in library_member_rows
        if (int(row.get("library_index") or 0), int(row.get("position") or 0)) not in existing_member_keys
    ]
    member_rows = existing_members + new_members

    try:
        write_library_artifact(
            out_dir=libraries_dir,
            builds=build_rows,
            members=member_rows,
            cfg_path=cfg_path,
            run_id=str(run_id),
            run_root=run_root,
            overwrite=True,
            config_hash=config_hash,
            pool_manifest_hash=pool_manifest_hash,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to write library artifacts: {exc}") from exc
