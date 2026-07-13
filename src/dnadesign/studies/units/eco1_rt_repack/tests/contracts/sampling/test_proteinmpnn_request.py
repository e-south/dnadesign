"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/sampling/test_proteinmpnn_request.py

ProteinMPNN request contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import (
    validate_proteinmpnn_request_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request import (
    materialize_proteinmpnn_request,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import materialize_thread_plan
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)
from dnadesign.thread.adapters.proteinmpnn import request_hash as proteinmpnn_request_hash

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_proteinmpnn_request_validator_rejects_canonical_position_leakage(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)
    manifest = _load_yaml(result.request_manifest_path)
    manifest["fixed_positions_jsonl"]["chain_a_backbone"]["A"] = manifest["source_thread_plan"]["fixed_positions"]
    result.request_manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validate_proteinmpnn_request_content(
        result.request_manifest_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )

    assert "eco1_rt.sampling.proteinmpnn_fixed_positions_mismatch" in {issue.check_id for issue in issues}


def test_proteinmpnn_request_validator_rejects_rehashed_wrong_fixed_sidecar(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)
    manifest = _load_yaml(result.request_manifest_path)
    wrong_fixed_payload = {"chain_a_backbone": {"A": [1, 2, 3]}}
    result.fixed_positions_path.write_text(json.dumps(wrong_fixed_payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest["sidecar_hashes"]["fixed_positions_jsonl"] = "sha256:" + _sha256(result.fixed_positions_path)
    manifest["request_hash"] = proteinmpnn_request_hash(
        {key: value for key, value in manifest.items() if key != "request_hash"}
    )
    result.request_manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validate_proteinmpnn_request_content(
        result.request_manifest_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )

    assert "eco1_rt.sampling.proteinmpnn_sidecar_payload_mismatch" in {issue.check_id for issue in issues}


def test_proteinmpnn_request_validator_accepts_host_specific_recorded_paths(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)
    manifest = _load_yaml(result.request_manifest_path)
    manifest["source_thread_plan"]["path"] = "/other/host/thread_plan.yaml"
    manifest["sidecar_paths"] = {
        name: f"/other/host/proteinmpnn_request/{Path(path).name}" for name, path in manifest["sidecar_paths"].items()
    }
    manifest["request_hash"] = proteinmpnn_request_hash(
        {key: value for key, value in manifest.items() if key != "request_hash"}
    )
    result.request_manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validate_proteinmpnn_request_content(
        result.request_manifest_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )

    assert issues == []


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
