"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_status_service_inputs.py

Focused tests for metadata-driven status input validation and dispatch.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.ops.status.registry_loader import load_status_kind_spec
from dnadesign.ops.status.service import build_status_inputs


def test_build_status_inputs_rejects_unexpected_input() -> None:
    spec = load_status_kind_spec("ops-audit-json")

    with pytest.raises(
        ValueError,
        match="status kind 'ops-audit-json' does not accept inputs: unexpected",
    ):
        build_status_inputs(
            spec=spec,
            raw_inputs={"audit_json": "repo:docs/runbooks/README.md", "unexpected": "value"},
            repo_root=Path("/tmp/repo"),
        )


def test_build_status_inputs_requires_declared_flag() -> None:
    spec = load_status_kind_spec("ops-audit-json")

    with pytest.raises(ValueError, match="status kind 'ops-audit-json' requires --audit-json"):
        build_status_inputs(spec=spec, raw_inputs={}, repo_root=Path("/tmp/repo"))


def test_build_status_inputs_coerces_declared_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec = load_status_kind_spec("ops-audit-json")

    resolved = build_status_inputs(
        spec=spec,
        raw_inputs={"audit_json": "repo:docs/runbooks/README.md"},
        repo_root=repo_root,
    )

    assert resolved["audit_json"] == (repo_root / "docs" / "runbooks" / "README.md").resolve()


def test_build_status_inputs_applies_declared_default_scope() -> None:
    spec = load_status_kind_spec("promoter-study-preflight")

    resolved = build_status_inputs(spec=spec, raw_inputs={}, repo_root=Path("/tmp/repo"))

    assert resolved == {"scope": "next", "command_timeout_seconds": 180}


def test_build_status_inputs_coerces_latentdna_workspace_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec = load_status_kind_spec("latentdna-workspace-snapshot")

    resolved = build_status_inputs(
        spec=spec,
        raw_inputs={"workspace": "repo:src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth"},
        repo_root=repo_root,
    )

    assert (
        resolved["workspace"]
        == (repo_root / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth").resolve()
    )
