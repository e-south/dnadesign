"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/workspaces/test_family_registry.py

Contracts for typed workflow-family registration and family-aware workspace
discovery.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.workspaces.families import (
    infer_runbook_workflow_families,
    workflow_family_descriptor,
    workspace_kind_from_presence,
)


def test_workflow_family_descriptor_registers_yiu_as_runbook_family() -> None:
    descriptor = workflow_family_descriptor("yiu")

    assert descriptor.id == "yiu"
    assert descriptor.workspace_kind == "runbook_family"
    assert descriptor.runbook_command_roots == ("yiu",)
    assert "configs/yiu/*.yiu.solve.yaml" in descriptor.spec_globs
    assert descriptor.default_output_root == "outputs/yiu"


def test_infer_runbook_workflow_families_reports_yiu() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo_yiu",
            "steps": [
                {
                    "id": "yiu_validate",
                    "run": ["yiu", "validate", "--spec", "configs/yiu/example.yiu.yaml"],
                }
            ],
        }
    }

    families = infer_runbook_workflow_families(payload)

    assert families == ("yiu",)


def test_workspace_kind_from_presence_formalizes_registry_kinds() -> None:
    assert workspace_kind_from_presence(has_config=True, has_runbook=False) == "config"
    assert workspace_kind_from_presence(has_config=False, has_runbook=True) == "runbook_family"
    assert workspace_kind_from_presence(has_config=True, has_runbook=True) == "hybrid"
