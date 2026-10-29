"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_artifact_integrity.py

Artifact reuse verifies output bytes and the producing implementation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import numpy as np
import pytest

from dnadesign.latentdna.src.services.freshness_service import evaluate_artifact_freshness
from dnadesign.latentdna.src.services.recipe_service import run_recipe
from dnadesign.latentdna.src.services.view_service import materialize_view
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config
from dnadesign.latentdna.tests.integrations.test_freshness_contracts import _build_overlay_backed_workspace


def test_recipe_rebuilds_modified_output_bytes(tmp_path):
    workspace, _ = _build_overlay_backed_workspace(tmp_path)
    materialize_view(workspace, "z20_60")
    context = load_workspace_config(workspace)
    path = context.output_root / "views/z20_60/matrix.npy"
    original = np.load(path)
    np.save(path, original + 1000)
    state = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="z20_60")
    assert state["status"] == "attention"
    assert "digest" in state["reason"]
    result = run_recipe(workspace, "materialize_only", refresh_catalog=False)
    assert result.metrics["rebuilt_steps"] == 1
    np.testing.assert_array_equal(np.load(path), original)


@pytest.mark.parametrize("change", ["version", "identity", "missing_digest"])
def test_artifacts_without_matching_execution_and_output_evidence_require_rebuild(tmp_path, change):
    workspace, _ = _build_overlay_backed_workspace(tmp_path)
    materialize_view(workspace, "z20_60")
    context = load_workspace_config(workspace)
    path = context.output_root / "views/z20_60/manifest.json"
    payload = json.loads(path.read_text())
    if change == "version":
        payload["tool_version"] = "0.0.0"
    elif change == "identity":
        payload["execution_identity"] = {"implementation_digest": "obsolete"}
    else:
        for output in payload["outputs"]:
            output.pop("digest", None)
    path.write_text(json.dumps(payload))
    state = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="z20_60")
    assert state["status"] == "attention"
