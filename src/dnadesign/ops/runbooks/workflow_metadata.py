"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/runbooks/workflow_metadata.py

Explicit orchestration workflow metadata and validation contracts for ops
runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WorkflowTool = Literal["densegen", "infer"]
NotifyPolicy = Literal["densegen", "infer", "generic"]
OrchestrationWorkflowId = Literal[
    "densegen_batch_submit",
    "densegen_batch_with_notify_slack",
    "infer_batch_submit",
    "infer_batch_with_notify_slack",
]


@dataclass(frozen=True)
class WorkflowMetadata:
    workflow_id: OrchestrationWorkflowId
    tool: WorkflowTool
    requires_notify: bool
    requires_gpus: bool
    allowed_notify_policies: tuple[NotifyPolicy, ...]


_WORKFLOW_METADATA_BY_ID: dict[str, WorkflowMetadata] = {
    "densegen_batch_submit": WorkflowMetadata(
        workflow_id="densegen_batch_submit",
        tool="densegen",
        requires_notify=False,
        requires_gpus=False,
        allowed_notify_policies=(),
    ),
    "densegen_batch_with_notify_slack": WorkflowMetadata(
        workflow_id="densegen_batch_with_notify_slack",
        tool="densegen",
        requires_notify=True,
        requires_gpus=False,
        allowed_notify_policies=("densegen", "generic"),
    ),
    "infer_batch_submit": WorkflowMetadata(
        workflow_id="infer_batch_submit",
        tool="infer",
        requires_notify=False,
        requires_gpus=True,
        allowed_notify_policies=(),
    ),
    "infer_batch_with_notify_slack": WorkflowMetadata(
        workflow_id="infer_batch_with_notify_slack",
        tool="infer",
        requires_notify=True,
        requires_gpus=True,
        allowed_notify_policies=("generic", "infer"),
    ),
}


def list_workflow_ids() -> tuple[OrchestrationWorkflowId, ...]:
    return tuple(sorted(_WORKFLOW_METADATA_BY_ID))


def list_workflow_tools() -> tuple[WorkflowTool, ...]:
    return tuple(sorted({metadata.tool for metadata in _WORKFLOW_METADATA_BY_ID.values()}))


def resolve_workflow_metadata(workflow_id: str) -> WorkflowMetadata:
    workflow = str(workflow_id or "").strip()
    metadata = _WORKFLOW_METADATA_BY_ID.get(workflow)
    if metadata is None:
        supported = ", ".join(list_workflow_ids())
        raise ValueError(f"unsupported orchestration workflow id: {workflow} (supported: {supported})")
    return metadata


def resolve_workflow_tool(workflow_id: str) -> WorkflowTool:
    return resolve_workflow_metadata(workflow_id).tool


def is_densegen_workflow_id(workflow_id: str) -> bool:
    metadata = _WORKFLOW_METADATA_BY_ID.get(str(workflow_id or "").strip())
    return metadata is not None and metadata.tool == "densegen"


def is_infer_workflow_id(workflow_id: str) -> bool:
    metadata = _WORKFLOW_METADATA_BY_ID.get(str(workflow_id or "").strip())
    return metadata is not None and metadata.tool == "infer"


def validate_workflow_contract(
    *,
    workflow_id: str,
    densegen_present: bool,
    infer_present: bool,
    notify_present: bool,
    notify_tool: WorkflowTool | None,
    notify_policy: NotifyPolicy | None,
    has_gpus: bool,
    has_gpu_capability: bool,
    has_gpu_memory: bool,
) -> None:
    metadata = resolve_workflow_metadata(workflow_id)
    workload_presence = {
        "densegen": densegen_present,
        "infer": infer_present,
    }
    expected_tool = metadata.tool
    if not workload_presence[expected_tool]:
        raise ValueError(f"{expected_tool} workflow requires runbook.{expected_tool} block")
    for tool, present in workload_presence.items():
        if tool != expected_tool and present:
            raise ValueError(f"{expected_tool} workflow does not accept runbook.{tool} block")

    if metadata.requires_gpus:
        if not has_gpus:
            raise ValueError(f"{expected_tool} workflow requires resources.gpus")
        if not has_gpu_capability:
            raise ValueError(f"{expected_tool} workflow requires resources.gpu_capability")
    elif has_gpus or has_gpu_capability or has_gpu_memory:
        raise ValueError(
            f"{expected_tool} workflow does not accept resources.gpus, "
            "resources.gpu_capability, or resources.gpu_memory_gib"
        )

    if metadata.requires_notify:
        if not notify_present:
            raise ValueError(f"{expected_tool} notify workflow requires runbook.notify block")
        if notify_tool != expected_tool:
            raise ValueError(f"{expected_tool} workflow requires notify.tool={expected_tool}")
        if notify_policy not in metadata.allowed_notify_policies:
            raise ValueError(f"{expected_tool} workflow does not accept notify.policy={notify_policy}")
    elif notify_present:
        raise ValueError(f"{metadata.workflow_id} does not accept runbook.notify block")
