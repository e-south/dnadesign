"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/mode_tools.py

Mode-tool adapter contracts for run-mode artifact probing and run-arg selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import yaml

from dnadesign.ops.contracts import resolve_usr_destination

from ..runbooks.schema import OrchestrationRunbookV1
from .workflow_tools import (
    build_workflow_tool_registry,
    freeze_workflow_tool_registry,
    list_registered_workflow_tools,
    register_workflow_tool_adapter,
    resolve_workflow_tool_adapter_for_runbook,
    resolve_workflow_tool_adapter_for_workflow_id,
)

ResolvedMode = Literal["fresh", "resume"]


class InferModeProbeError(ValueError):
    """Raised when infer mode probing cannot safely infer a single resume target."""


@dataclass(frozen=True)
class ModeToolAdapter:
    tool: str
    has_resume_artifacts: Callable[[OrchestrationRunbookV1], bool]
    run_args_for_mode: Callable[[OrchestrationRunbookV1, ResolvedMode], str]


def _dedupe_existing_paths(candidates: tuple[Path, ...]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return tuple(deduped)


def _infer_workspace_overlay_candidates(workspace_root: Path) -> tuple[Path, ...]:
    workspace_usr_root = workspace_root / "outputs" / "usr_datasets"
    if not workspace_usr_root.exists():
        return ()
    candidates: list[Path] = []
    candidates.extend(sorted(workspace_usr_root.glob("**/_derived/infer.parquet")))
    candidates.extend(sorted(workspace_usr_root.glob("**/_derived/infer/*.parquet")))
    return tuple(candidates)


def _infer_dataset_overlay_candidates(dataset_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    candidates.append(dataset_root / "_derived" / "infer.parquet")
    infer_parts_root = dataset_root / "_derived" / "infer"
    if infer_parts_root.exists():
        candidates.extend(sorted(infer_parts_root.glob("*.parquet")))
    return tuple(candidates)


def _infer_overlay_artifacts(workspace_root: Path, *, infer_config: Path | None) -> tuple[Path, ...]:
    if infer_config is None:
        return _dedupe_existing_paths(_infer_workspace_overlay_candidates(workspace_root))

    contract = _resolve_infer_usr_output_for_mode_probe(infer_config)
    if contract is None:
        return ()

    dataset_root = contract.usr_root / contract.usr_dataset
    return _dedupe_existing_paths(_infer_dataset_overlay_candidates(dataset_root))


def _resolve_infer_usr_output_for_mode_probe(infer_config: Path):
    try:
        return resolve_usr_destination(tool="infer", config_path=infer_config)
    except ValueError as exc:
        message = str(exc)
        if "at least one job with ingest.source='usr' and io.write_back=true" in message:
            return None
        if (
            "multiple SeqLedger destinations" in message
            or "requires ingest.root for source='usr' write-back jobs" in message
        ):
            raise InferModeProbeError(message) from exc
        raise ValueError(f"infer probe requires one SeqLedger destination in config {infer_config}: {message}") from exc


def _has_infer_resume_artifacts(runbook: OrchestrationRunbookV1) -> bool:
    infer_config = runbook.infer.config if runbook.infer is not None else None
    return bool(_infer_overlay_artifacts(runbook.workspace_root, infer_config=infer_config))


def _infer_config_uses_sequence_view_inputs(config_path: Path) -> bool:
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"infer config is not readable: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"infer config is not valid yaml: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"infer config root must be a mapping: {config_path}")
    jobs = payload.get("jobs") or ()
    if not isinstance(jobs, list):
        return False
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        feature_bundle = job.get("feature_bundle")
        if not isinstance(feature_bundle, Mapping):
            continue
        sequence_view_inputs = feature_bundle.get("sequence_view_inputs")
        if isinstance(sequence_view_inputs, list) and sequence_view_inputs:
            return True
    return False


def _run_args_for_infer(runbook: OrchestrationRunbookV1, mode: ResolvedMode) -> str:
    if runbook.infer is not None and _infer_config_uses_sequence_view_inputs(runbook.infer.config):
        return ""
    if mode == "fresh":
        return "--overwrite"
    return ""


_MODE_TOOL_ADAPTERS = build_workflow_tool_registry(
    contract_name="mode tool adapter",
    adapters=(
        ModeToolAdapter(
            tool="infer",
            has_resume_artifacts=_has_infer_resume_artifacts,
            run_args_for_mode=_run_args_for_infer,
        ),
    ),
)


def register_mode_tool_adapter(tool: str, adapter: ModeToolAdapter) -> None:
    global _MODE_TOOL_ADAPTERS
    updated = dict(_MODE_TOOL_ADAPTERS)
    register_workflow_tool_adapter(
        updated,
        contract_name="mode tool adapter",
        tool=tool,
        adapter=adapter,
    )
    _MODE_TOOL_ADAPTERS = freeze_workflow_tool_registry(updated, contract_name="mode tool adapter")


def list_registered_mode_tools() -> tuple[str, ...]:
    return list_registered_workflow_tools(_MODE_TOOL_ADAPTERS)


def resolve_mode_tool_adapter_for_workflow_id(workflow_id: str) -> ModeToolAdapter:
    return resolve_workflow_tool_adapter_for_workflow_id(
        _MODE_TOOL_ADAPTERS,
        contract_name="mode tool adapter",
        workflow_id=workflow_id,
    )


def resolve_mode_tool_adapter(runbook: OrchestrationRunbookV1) -> ModeToolAdapter:
    return resolve_workflow_tool_adapter_for_runbook(
        _MODE_TOOL_ADAPTERS,
        contract_name="mode tool adapter",
        runbook=runbook,
    )
