"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_plan_tools.py

Contract tests for ops plan-tool adapter registration and workflow coverage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import get_args

import dnadesign.ops.runbooks.schema as runbook_schema
from dnadesign.ops.orchestrator import plan_tools


def test_list_registered_plan_tools_returns_sorted_read_only_tuple() -> None:
    assert plan_tools.list_registered_plan_tools() == ("densegen", "infer")


def test_registered_plan_tools_exactly_match_schema_workflow_tools() -> None:
    assert plan_tools.list_registered_plan_tools() == runbook_schema.list_workflow_tools()


def test_plan_tool_adapters_cover_all_schema_workflow_ids() -> None:
    workflow_ids = get_args(runbook_schema.OrchestrationRunbookV1.model_fields["workflow_id"].annotation)
    assert workflow_ids
    for workflow_id in workflow_ids:
        adapter = plan_tools.resolve_plan_tool_adapter_for_workflow_id(workflow_id)
        assert adapter.tool in {"densegen", "infer"}
