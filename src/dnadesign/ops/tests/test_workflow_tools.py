"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_workflow_tools.py

Contract tests for shared workflow-tool registration and runbook-tool resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dnadesign.ops.orchestrator import workflow_tools


@dataclass(frozen=True)
class DummyAdapter:
    tool: str


def test_list_registered_workflow_tools_returns_sorted_read_only_tuple() -> None:
    registry: dict[str, DummyAdapter] = {}
    workflow_tools.register_workflow_tool_adapter(
        registry,
        contract_name="dummy tool adapter",
        tool="infer",
        adapter=DummyAdapter(tool="infer"),
    )
    workflow_tools.register_workflow_tool_adapter(
        registry,
        contract_name="dummy tool adapter",
        tool="densegen",
        adapter=DummyAdapter(tool="densegen"),
    )
    assert workflow_tools.list_registered_workflow_tools(registry) == ("densegen", "infer")


def test_validate_workflow_tool_registry_rejects_schema_drift() -> None:
    registry = {"infer": DummyAdapter(tool="infer")}
    with pytest.raises(RuntimeError, match="dummy tool adapter registry does not match workflow tool set"):
        workflow_tools.validate_workflow_tool_registry(registry, contract_name="dummy tool adapter")


def test_resolve_workflow_tool_adapter_for_runbook_requires_single_matching_tool() -> None:
    registry = {
        "densegen": DummyAdapter(tool="densegen"),
        "infer": DummyAdapter(tool="infer"),
    }

    with pytest.raises(ValueError, match="runbook workload contract must define exactly one tool block"):
        workflow_tools.resolve_workflow_tool_adapter_for_runbook(
            registry,
            contract_name="dummy tool adapter",
            runbook=SimpleNamespace(
                workflow_id="infer_batch_submit",
                densegen=object(),
                infer=object(),
            ),
        )

    with pytest.raises(ValueError, match="runbook workload contract does not match workflow tool"):
        workflow_tools.resolve_workflow_tool_adapter_for_runbook(
            registry,
            contract_name="dummy tool adapter",
            runbook=SimpleNamespace(
                workflow_id="infer_batch_submit",
                densegen=object(),
                infer=None,
            ),
        )

    adapter = workflow_tools.resolve_workflow_tool_adapter_for_runbook(
        registry,
        contract_name="dummy tool adapter",
        runbook=SimpleNamespace(
            workflow_id="infer_batch_submit",
            densegen=None,
            infer=object(),
        ),
    )
    assert adapter.tool == "infer"
