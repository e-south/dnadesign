"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_workflow_metadata.py

Contract tests for explicit orchestration workflow metadata and notify-policy
rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import get_args

import dnadesign.ops.runbooks.schema as runbook_schema
from dnadesign.ops.runbooks import workflow_metadata


def test_workflow_metadata_covers_all_schema_workflow_ids() -> None:
    workflow_ids = get_args(runbook_schema.OrchestrationRunbookV1.model_fields["workflow_id"].annotation)
    assert workflow_ids
    assert workflow_metadata.list_workflow_ids() == tuple(sorted(workflow_ids))


def test_infer_notify_workflow_metadata_is_explicit() -> None:
    definition = workflow_metadata.resolve_workflow_metadata("infer_batch_with_notify_slack")

    assert definition.tool == "infer"
    assert definition.requires_notify is True
    assert definition.requires_gpus is True
    assert definition.allowed_notify_policies == ("generic", "infer")


def test_densegen_submit_workflow_metadata_is_explicit() -> None:
    definition = workflow_metadata.resolve_workflow_metadata("densegen_batch_submit")

    assert definition.tool == "densegen"
    assert definition.requires_notify is False
    assert definition.requires_gpus is False
    assert definition.allowed_notify_policies == ()


def test_resolve_workflow_id_for_tool_and_notify_contract() -> None:
    assert workflow_metadata.resolve_workflow_id(tool="densegen", with_notify=False) == "densegen_batch_submit"
    assert workflow_metadata.resolve_workflow_id(tool="densegen", with_notify=True) == "densegen_batch_with_notify_slack"
    assert workflow_metadata.resolve_workflow_id(tool="infer", with_notify=False) == "infer_batch_submit"
    assert workflow_metadata.resolve_workflow_id(tool="infer", with_notify=True) == "infer_batch_with_notify_slack"


def test_validate_workflow_contract_rejects_infer_notify_policy_densegen() -> None:
    try:
        workflow_metadata.validate_workflow_contract(
            workflow_id="infer_batch_with_notify_slack",
            densegen_present=False,
            infer_present=True,
            notify_present=True,
            notify_tool="infer",
            notify_policy="densegen",
            has_gpus=True,
            has_gpu_capability=True,
            has_gpu_memory=False,
        )
    except ValueError as exc:
        assert str(exc) == "infer workflow does not accept notify.policy=densegen"
    else:
        raise AssertionError("expected ValueError for invalid infer notify policy")


def test_workflow_classification_helpers_return_false_for_unknown_id() -> None:
    assert workflow_metadata.is_densegen_workflow_id("unknown_workflow") is False
    assert workflow_metadata.is_infer_workflow_id("unknown_workflow") is False
