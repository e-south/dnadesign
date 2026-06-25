"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/sampling/thread_plan/test_contract.py

Thread-plan contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import (
    validate_thread_plan_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import materialize_thread_plan
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_thread_plan_validator_rejects_implicit_fallback(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    plan = _load_yaml(result.thread_plan_path)
    plan["fallback_policy"] = "fallback_to_available_backend"
    plan["backend_request_manifest"]["fallback_policy"] = "fallback_to_available_backend"
    result.thread_plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    issues = validate_thread_plan_content(
        result.thread_plan_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )

    assert "eco1_rt.sampling.thread_plan_fallback_policy_mismatch" in {issue.check_id for issue in issues}


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
