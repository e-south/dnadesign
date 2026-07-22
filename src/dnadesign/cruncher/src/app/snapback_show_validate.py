"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_show_validate.py

Validate preserved-site Snapback bundle drift and readback invariants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.snapback_publish import build_views_manifest
from dnadesign.cruncher.app.snapback_show_load import (
    SnapbackExplicitShowArtifacts,
    SnapbackShowArtifacts,
    SnapbackSolveShowArtifacts,
    load_snapback_explicit_show_artifacts,
)


def _required_existing_manifest_path(payload: dict[str, object], *, field: str, label: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} drift detected.")
    resolved = Path(value).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} missing: {resolved}")
    return resolved


def _declared_artifact_paths(payload: dict[str, object], *, label: str) -> dict[str, str]:
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError(f"{label} artifact inventory drift detected.")
    declared: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError(f"{label} artifact inventory drift detected.")
        name = item.get("name")
        path = item.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            raise ValueError(f"{label} artifact inventory drift detected.")
        declared[name] = path
    return declared


def _validate_explicit_publication_alignment(
    artifacts: SnapbackExplicitShowArtifacts,
    *,
    candidate: dict[str, object],
) -> None:
    if artifacts.views_manifest_payload is None:
        raise FileNotFoundError(f"Required snapback visual artifact missing: {artifacts.views_manifest_path}")
    required_optionals = [
        (artifacts.pre_nick_visual_payload, artifacts.pre_nick_visual_contract_path),
        (artifacts.post_nick_exposed_visual_payload, artifacts.post_nick_exposed_visual_contract_path),
        (artifacts.post_nick_foldback_visual_payload, artifacts.post_nick_foldback_visual_contract_path),
        (artifacts.triptych_visual_contracts, artifacts.triptych_visual_contracts_path),
    ]
    for payload, path in required_optionals:
        if payload is None:
            raise FileNotFoundError(f"Required snapback visual artifact missing: {path}")

    solution_id = artifacts.views_manifest_payload.get("solution_id")
    if not isinstance(solution_id, str) or not solution_id:
        raise ValueError("Snapback views manifest solution_id drift detected.")
    expected_views_manifest = build_views_manifest(
        solution_id=solution_id,
        include_jobs=artifacts.triptych_job_path.exists(),
    )
    if artifacts.views_manifest_payload != expected_views_manifest:
        raise ValueError("Snapback views manifest content drift detected.")
    for view in expected_views_manifest["views"]:
        view_path = artifacts.run_dir / str(view["path"])
        if not view_path.exists():
            raise FileNotFoundError(f"Snapback views manifest declared view missing: {view_path}")
    for job in expected_views_manifest["recommended_jobs"]:
        job_path = (artifacts.views_manifest_path.parent / str(job["path"])).resolve()
        if not job_path.exists():
            raise FileNotFoundError(f"Snapback views manifest recommended job missing: {job_path}")

    expected_state_ids = [
        f"{solution_id}.pre_nick_duplex",
        f"{solution_id}.post_nick_exposed",
        f"{solution_id}.post_nick_foldback",
    ]
    visuals = [
        artifacts.pre_nick_visual_payload,
        artifacts.post_nick_exposed_visual_payload,
        artifacts.post_nick_foldback_visual_payload,
    ]
    for expected_state_id, visual_payload in zip(expected_state_ids, visuals, strict=True):
        if visual_payload.get("state_id") != expected_state_id:
            raise ValueError(f"Snapback visual state_id drift detected for {expected_state_id}.")

    designed_sequence = candidate.get("designed_sequence")
    post_nick_sequence = candidate.get("post_nick_sequence")
    if artifacts.pre_nick_visual_payload.get("primary_sequence") != designed_sequence:
        raise ValueError("Snapback pre-nick visual primary_sequence drift detected.")
    if artifacts.post_nick_exposed_visual_payload.get("primary_sequence") != designed_sequence:
        raise ValueError("Snapback exposed visual primary_sequence drift detected.")
    if artifacts.post_nick_foldback_visual_payload.get("primary_sequence") != post_nick_sequence:
        raise ValueError("Snapback foldback visual primary_sequence drift detected.")
    if artifacts.post_nick_foldback_visual_payload.get("meta", {}).get("cap_extension_nt") != candidate.get(
        "cap_extension_nt"
    ):
        raise ValueError("Snapback foldback visual cap_extension_nt drift detected.")
    if artifacts.post_nick_foldback_visual_payload.get("meta", {}).get("terminal_ligatable_duplex_bp") != candidate.get(
        "terminal_ligatable_duplex_bp"
    ):
        raise ValueError("Snapback foldback visual terminal_ligatable_duplex_bp drift detected.")

    triptych_lines = artifacts.triptych_visual_contracts
    if len(triptych_lines) != 3:
        raise ValueError("Snapback triptych visual contract count drift detected.")
    if [payload.get("state_id") for payload in triptych_lines] != expected_state_ids:
        raise ValueError("Snapback triptych visual state ordering drift detected.")


def validate_snapback_explicit_show_artifacts(artifacts: SnapbackExplicitShowArtifacts) -> None:
    manifest = artifacts.manifest
    status = artifacts.status
    expected_run_dir = str(artifacts.run_dir)
    workspace_root = _required_existing_manifest_path(
        manifest,
        field="workspace_root",
        label="Snapback explicit manifest workspace_root",
    )
    expected_workspace_root = str(workspace_root)
    if manifest.get("kind") != "explicit":
        raise ValueError("Snapback explicit manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_design":
        raise ValueError("Snapback explicit manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_snapback_v2":
        raise ValueError("Unsupported snapback explicit contract version.")
    if status.get("workflow") != "snapback_design":
        raise ValueError("Snapback explicit status workflow drift detected.")
    if status.get("contract") != "single_nick_snapback_v2":
        raise ValueError("Unsupported snapback explicit status contract version.")
    if manifest.get("stage") != "snapback" or status.get("stage") != "snapback":
        raise ValueError("Snapback explicit stage drift detected.")
    if manifest.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback manifest run_dir drift detected.")
    if status.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback status run_dir drift detected.")
    if manifest.get("spec_name") != status.get("spec_name"):
        raise ValueError("Snapback manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Snapback manifest/status status drift detected.")
    for path in artifacts.required_artifact_paths():
        if not path.exists():
            raise FileNotFoundError(f"Required snapback artifact missing: {path}")
    declared_artifacts = _declared_artifact_paths(manifest, label="Snapback manifest")
    for key in ("pre_nick_duplex_view", "post_nick_exposed_view", "post_nick_foldback_view", "views_manifest"):
        if key in declared_artifacts:
            candidate = artifacts.run_dir / declared_artifacts[key]
            if not candidate.exists():
                raise FileNotFoundError(f"Declared snapback visual artifact missing: {candidate}")
    for key in (
        "pre_nick_duplex_visual_contract",
        "post_nick_exposed_visual_contract",
        "post_nick_foldback_visual_contract",
        "snapback_triptych_visual_contracts",
        "snapback_triptych_job",
    ):
        if key in declared_artifacts:
            candidate = artifacts.run_dir / declared_artifacts[key]
            if not candidate.exists():
                raise FileNotFoundError(f"Declared snapback visual artifact missing: {candidate}")
    if artifacts.report_payload.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback report run_dir drift detected.")
    if artifacts.report_payload.get("workspace_root") != expected_workspace_root:
        raise ValueError("Snapback report workspace_root drift detected.")
    if artifacts.views_manifest_path.exists():
        candidate = artifacts.report_payload.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("Snapback visual artifacts require a satisfied candidate payload.")
        _validate_explicit_publication_alignment(artifacts, candidate=candidate)


def _validate_materialized_hit_alignment(
    *,
    hit: dict[str, object],
    explicit_artifacts: SnapbackExplicitShowArtifacts,
    hit_run_dir: Path,
) -> None:
    candidate = explicit_artifacts.report_payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"Materialized snapback hit bundle missing candidate payload: {hit_run_dir}")
    intended_nick = candidate.get("intended_nick")
    if not isinstance(intended_nick, dict):
        raise ValueError(f"Materialized snapback hit bundle missing intended_nick payload: {hit_run_dir}")
    if intended_nick.get("variant_id") != hit.get("variant_id"):
        raise ValueError(f"Materialized snapback hit variant drift detected: {hit_run_dir}")
    for candidate_key, hit_key in (
        ("cap_sequence", "cap_sequence"),
        ("foldback_arm", "foldback_arm"),
        ("nick_boundary", "nick_boundary"),
        ("paired_bp", "paired_bp"),
        ("cap_extension_nt", "cap_extension_nt"),
        ("site_mutation_count", "site_mutation_count"),
    ):
        if candidate.get(candidate_key) != hit.get(hit_key):
            raise ValueError(f"Materialized snapback hit {candidate_key} drift detected: {hit_run_dir}")


def _validate_materialized_hit_bundles(artifacts: SnapbackSolveShowArtifacts) -> None:
    report_payload = artifacts.report_payload
    metadata = report_payload.get("metadata", {})
    hits = report_payload.get("hits", [])
    workspace_root = Path(report_payload["workspace_root"]).resolve()
    expected_hit_count = metadata.get("materialized_hit_count")
    if not isinstance(expected_hit_count, int):
        raise ValueError("Snapback solve materialized_hit_count drift detected.")
    if not isinstance(hits, list):
        raise ValueError("Snapback solve hits drift detected.")
    materialized_hits = [hit for hit in hits if isinstance(hit, dict) and hit.get("materialized_run_dir") is not None]
    if len(materialized_hits) != expected_hit_count:
        raise ValueError("Snapback solve materialized_hit_count drift detected.")
    observed_ranks = {hit.get("rank") for hit in materialized_hits}
    if observed_ranks != set(range(1, expected_hit_count + 1)):
        raise ValueError("Snapback solve materialized hit rank coverage drift detected.")
    seen_materialized_dirs: set[str] = set()
    for hit in materialized_hits:
        rank = hit.get("rank")
        materialized_run_dir = hit["materialized_run_dir"]
        expected_name = f"hit_{int(rank):02d}"
        if Path(materialized_run_dir).name != expected_name:
            raise ValueError("Snapback solve materialized hit path/rank drift detected.")
        if materialized_run_dir in seen_materialized_dirs:
            raise ValueError("Duplicate materialized snapback hit bundle path detected.")
        seen_materialized_dirs.add(materialized_run_dir)
        hit_run_dir = (workspace_root / materialized_run_dir).resolve()
        try:
            hit_run_dir.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(f"Materialized snapback hit bundle escaped workspace_root: {hit_run_dir}") from exc
        if not hit_run_dir.exists():
            raise FileNotFoundError(f"Materialized snapback hit bundle missing: {hit_run_dir}")
        explicit_artifacts = load_snapback_explicit_show_artifacts(hit_run_dir)
        validate_snapback_explicit_show_artifacts(explicit_artifacts)
        _validate_materialized_hit_alignment(hit=hit, explicit_artifacts=explicit_artifacts, hit_run_dir=hit_run_dir)


def validate_snapback_solve_show_artifacts(artifacts: SnapbackSolveShowArtifacts) -> None:
    manifest = artifacts.manifest
    status = artifacts.status
    expected_run_dir = str(artifacts.run_dir)
    workspace_root = _required_existing_manifest_path(
        manifest,
        field="workspace_root",
        label="Snapback solve manifest workspace_root",
    )
    expected_workspace_root = str(workspace_root)
    if manifest.get("kind") != "solve":
        raise ValueError("Snapback solve manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_snapback_solve_v3":
        raise ValueError("Unsupported snapback solve contract version.")
    if status.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve status workflow drift detected.")
    if status.get("contract") != "single_nick_snapback_solve_v3":
        raise ValueError("Unsupported snapback solve status contract version.")
    if manifest.get("stage") != "snapback" or status.get("stage") != "snapback":
        raise ValueError("Snapback solve stage drift detected.")
    if manifest.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve manifest run_dir drift detected.")
    if status.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve status run_dir drift detected.")
    if manifest.get("spec_name") != status.get("spec_name"):
        raise ValueError("Snapback solve manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Snapback solve manifest/status status drift detected.")
    for path in artifacts.required_artifact_paths():
        if not path.exists():
            raise FileNotFoundError(f"Required snapback solve artifact missing: {path}")
    if artifacts.report_payload.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve report run_dir drift detected.")
    if artifacts.report_payload.get("workspace_root") != expected_workspace_root:
        raise ValueError("Snapback solve report workspace_root drift detected.")
    _validate_materialized_hit_bundles(artifacts)


def validate_snapback_show_artifacts(artifacts: SnapbackShowArtifacts) -> None:
    if isinstance(artifacts, SnapbackExplicitShowArtifacts):
        validate_snapback_explicit_show_artifacts(artifacts)
        return
    validate_snapback_solve_show_artifacts(artifacts)


__all__ = [
    "validate_snapback_explicit_show_artifacts",
    "validate_snapback_show_artifacts",
    "validate_snapback_solve_show_artifacts",
]
