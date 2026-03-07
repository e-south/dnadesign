"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/workflow_tools.py

Shared workflow-tool contracts for adapter registration and runbook-tool
resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, MutableMapping, Protocol, TypeVar

from ..runbooks.schema import OrchestrationRunbookV1, list_workflow_tools, resolve_workflow_tool


class WorkflowToolAdapter(Protocol):
    tool: str


AdapterT = TypeVar("AdapterT", bound=WorkflowToolAdapter)


def register_workflow_tool_adapter(
    registry: MutableMapping[str, AdapterT],
    *,
    contract_name: str,
    tool: str,
    adapter: AdapterT,
) -> None:
    tool_name = str(tool or "").strip().lower()
    if not tool_name:
        raise ValueError(f"{contract_name} tool must be non-empty")
    if adapter.tool != tool_name:
        raise ValueError(f"{contract_name} tool mismatch: expected {tool_name}, got {adapter.tool}")
    if tool_name in registry:
        raise ValueError(f"{contract_name} already registered for tool: {tool_name}")
    registry[tool_name] = adapter


def list_registered_workflow_tools(registry: Mapping[str, AdapterT]) -> tuple[str, ...]:
    return tuple(sorted(registry))


def freeze_workflow_tool_registry(
    registry: Mapping[str, AdapterT],
    *,
    contract_name: str,
) -> Mapping[str, AdapterT]:
    validate_workflow_tool_registry(registry, contract_name=contract_name)
    return MappingProxyType(dict(registry))


def build_workflow_tool_registry(
    *,
    contract_name: str,
    adapters: tuple[AdapterT, ...],
) -> Mapping[str, AdapterT]:
    registry: dict[str, AdapterT] = {}
    for adapter in adapters:
        register_workflow_tool_adapter(
            registry,
            contract_name=contract_name,
            tool=adapter.tool,
            adapter=adapter,
        )
    return freeze_workflow_tool_registry(registry, contract_name=contract_name)


def validate_workflow_tool_registry(
    registry: Mapping[str, AdapterT],
    *,
    contract_name: str,
) -> None:
    registered_tools = list_registered_workflow_tools(registry)
    expected_tools = list_workflow_tools()
    if registered_tools != expected_tools:
        raise RuntimeError(
            f"{contract_name} registry does not match workflow tool set "
            f"(registered={registered_tools}, expected={expected_tools})"
        )


def resolve_workflow_tool_adapter_for_workflow_id(
    registry: Mapping[str, AdapterT],
    *,
    contract_name: str,
    workflow_id: str,
) -> AdapterT:
    tool = resolve_workflow_tool(workflow_id)
    adapter = registry.get(tool)
    if adapter is None:
        raise ValueError(f"missing {contract_name} for workflow tool: {tool}")
    return adapter


def resolve_runbook_workload_tool(runbook: OrchestrationRunbookV1) -> str:
    active_tools = tuple(tool for tool in list_workflow_tools() if getattr(runbook, tool, None) is not None)
    if len(active_tools) != 1:
        raise ValueError("runbook workload contract must define exactly one tool block")
    return active_tools[0]


def resolve_workflow_tool_adapter_for_runbook(
    registry: Mapping[str, AdapterT],
    *,
    contract_name: str,
    runbook: OrchestrationRunbookV1,
) -> AdapterT:
    adapter = resolve_workflow_tool_adapter_for_workflow_id(
        registry,
        contract_name=contract_name,
        workflow_id=runbook.workflow_id,
    )
    selected_tool = resolve_runbook_workload_tool(runbook)
    if selected_tool != adapter.tool:
        raise ValueError(
            "runbook workload contract does not match workflow tool "
            f"(workflow_id={runbook.workflow_id}, workflow_tool={adapter.tool}, workload_block={selected_tool})"
        )
    return adapter
