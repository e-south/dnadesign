"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workspaces/families.py

Typed workflow-family registration for family-aware workspace discovery and docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

WorkspaceKind = Literal["config", "runbook_family", "hybrid"]


@dataclass(frozen=True)
class WorkflowFamilyDescriptor:
    id: str
    display_name: str
    workspace_kind: WorkspaceKind
    runbook_command_roots: tuple[str, ...]
    spec_globs: tuple[str, ...]
    default_output_root: str
    docs_section_id: str


_WORKFLOW_FAMILIES: tuple[WorkflowFamilyDescriptor, ...] = (
    WorkflowFamilyDescriptor(
        id="sample",
        display_name="Fixed-length optimization",
        workspace_kind="config",
        runbook_command_roots=("fetch", "lock", "parse", "sample", "analyze", "export"),
        spec_globs=("configs/config.yaml",),
        default_output_root="outputs",
        docs_section_id="fixed_length",
    ),
    WorkflowFamilyDescriptor(
        id="cassette",
        display_name="Dual-nick hairpin cassette workflow",
        workspace_kind="runbook_family",
        runbook_command_roots=("cassette",),
        spec_globs=("configs/cassettes/*.cassette.yaml", "configs/cassettes/*.cassette.solve.yaml"),
        default_output_root="outputs/cassettes",
        docs_section_id="cassette",
    ),
    WorkflowFamilyDescriptor(
        id="yiu",
        display_name="payload-centric YIU rendering workflow",
        workspace_kind="runbook_family",
        runbook_command_roots=("yiu",),
        spec_globs=("configs/yiu/*.yiu.yaml",),
        default_output_root="outputs",
        docs_section_id="yiu",
    ),
    WorkflowFamilyDescriptor(
        id="snapback",
        display_name="single-nick snapback workflow",
        workspace_kind="runbook_family",
        runbook_command_roots=("snapback",),
        spec_globs=(
            "configs/snapback/*.snapback.yaml",
            "configs/snapback/*.snapback.solve.yaml",
            "configs/snapback/*.released.snapback.yaml",
        ),
        default_output_root="outputs",
        docs_section_id="snapback",
    ),
    WorkflowFamilyDescriptor(
        id="scar_nick",
        display_name="retained-scar terminal-nick workflow",
        workspace_kind="runbook_family",
        runbook_command_roots=("scar-nick",),
        spec_globs=("configs/scar_nick/*.scar_nick.yaml",),
        default_output_root="outputs/scar_nick",
        docs_section_id="scar_nick",
    ),
    WorkflowFamilyDescriptor(
        id="study",
        display_name="Study orchestration",
        workspace_kind="hybrid",
        runbook_command_roots=("study",),
        spec_globs=("configs/studies/*.study.yaml",),
        default_output_root="outputs/studies",
        docs_section_id="study",
    ),
    WorkflowFamilyDescriptor(
        id="portfolio",
        display_name="Portfolio orchestration",
        workspace_kind="runbook_family",
        runbook_command_roots=("portfolio",),
        spec_globs=("configs/*.portfolio.yaml",),
        default_output_root="outputs/portfolios",
        docs_section_id="portfolio",
    ),
)

_FAMILY_BY_ID = {descriptor.id: descriptor for descriptor in _WORKFLOW_FAMILIES}
_FAMILY_BY_COMMAND = {
    command: descriptor.id for descriptor in _WORKFLOW_FAMILIES for command in descriptor.runbook_command_roots
}

RUNBOOK_UTILITY_COMMAND_ROOTS: tuple[str, ...] = (
    "cache",
    "catalog",
    "config",
    "discover",
    "doctor",
    "optimizers",
    "runs",
    "sources",
    "status",
    "targets",
    "workspaces",
)


def workflow_family_descriptors() -> tuple[WorkflowFamilyDescriptor, ...]:
    return _WORKFLOW_FAMILIES


def workflow_family_descriptor(family_id: str) -> WorkflowFamilyDescriptor:
    try:
        return _FAMILY_BY_ID[family_id]
    except KeyError as exc:
        raise KeyError(f"Unknown workflow family: {family_id}") from exc


def allowed_runbook_command_roots() -> tuple[str, ...]:
    commands = set(RUNBOOK_UTILITY_COMMAND_ROOTS)
    commands.update(_FAMILY_BY_COMMAND)
    return tuple(sorted(commands))


def workspace_kind_from_presence(*, has_config: bool, has_runbook: bool) -> WorkspaceKind:
    if has_config and has_runbook:
        return "hybrid"
    if has_runbook:
        return "runbook_family"
    return "config"


def infer_runbook_workflow_families(payload: dict[str, object]) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    runbook = payload.get("runbook")
    if not isinstance(runbook, dict):
        return ()
    steps = runbook.get("steps")
    if not isinstance(steps, list):
        return ()
    discovered: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, list) or not run:
            continue
        root = str(run[0]).strip()
        family_id = _FAMILY_BY_COMMAND.get(root)
        if family_id and family_id not in discovered:
            discovered.append(family_id)
    return tuple(discovered)


def infer_runbook_workflow_families_from_path(runbook_path: Path) -> tuple[str, ...]:
    if not runbook_path.exists():
        return ()
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8")) or {}
    return infer_runbook_workflow_families(payload)


def discover_spec_workflow_families(workspace_root: Path) -> tuple[str, ...]:
    resolved_root = workspace_root.resolve()
    discovered: list[str] = []
    for descriptor in _WORKFLOW_FAMILIES:
        for pattern in descriptor.spec_globs:
            if any(path.is_file() for path in resolved_root.glob(pattern)):
                discovered.append(descriptor.id)
                break
    return tuple(discovered)
