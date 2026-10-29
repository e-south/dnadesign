"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_retired_producer_boundary.py

Retired generators must not remain executable orchestration choices.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest
from typer.testing import CliRunner

from dnadesign.ops.contracts import resolve_usr_destination
from dnadesign.ops.runbooks.workflow_metadata import resolve_workflow_metadata


def test_retired_producer_rejected_before_config_access(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        resolve_usr_destination(tool="densegen", config_path=tmp_path / "missing.yaml")


@pytest.mark.parametrize("workflow", ["densegen_batch_submit", "densegen_batch_with_notify"])
def test_retired_workflows_are_unavailable(workflow):
    with pytest.raises(ValueError, match="unsupported"):
        resolve_workflow_metadata(workflow)


def test_retired_notify_integrations_are_unavailable():
    from dnadesign.notify.profiles.policy import supported_workflow_policies
    from dnadesign.notify.tool_events import supported_tool_event_packs

    assert "densegen" not in supported_workflow_policies()
    assert "densegen" not in supported_tool_event_packs()


def test_catalog_exports_remain_tool_independent():
    from dnadesign.cruncher.cli.app import app

    result = CliRunner().invoke(app, ["catalog", "--help"])
    assert result.exit_code == 0, result.output
    assert "export-meme" in result.output
    assert "export-sites" in result.output
    assert "export-densegen" not in result.output
