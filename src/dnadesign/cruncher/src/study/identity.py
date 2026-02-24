"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/identity.py

Deterministic Study identity helpers shared across workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.study.layout import resolve_study_run_dir
from dnadesign.cruncher.study.load import load_study_spec
from dnadesign.cruncher.study.schema_models import StudySpec
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path
from dnadesign.cruncher.utils.paths import resolve_workspace_root


def study_target_descriptor(spec: StudySpec) -> str:
    return json.dumps(
        spec.target,
        default=lambda item: item.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def study_spec_payload(spec: StudySpec) -> dict[str, object]:
    return {"study": spec.model_dump(mode="json")}


def compute_study_spec_hash(spec: StudySpec) -> str:
    payload = json.dumps(study_spec_payload(spec), sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def compute_study_id(spec: StudySpec, *, base_config_sha256: str) -> str:
    canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    payload = f"{canonical}|{base_config_sha256}|{study_target_descriptor(spec)}"
    return sha256_bytes(payload.encode("utf-8"))[:12]


def resolve_deterministic_study_run_dir(spec_path: Path) -> Path:
    resolved_spec = spec_path.expanduser().resolve()
    spec = load_study_spec(resolved_spec)
    base_config_path = Path(spec.base_config).expanduser().resolve()
    workspace_root = resolve_workspace_root(base_config_path)
    base_config_sha = sha256_path(base_config_path)
    study_id = compute_study_id(spec, base_config_sha256=base_config_sha)
    return resolve_study_run_dir(workspace_root, spec.name, study_id)
