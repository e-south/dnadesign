"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_mode_tools.py

Contract tests for ops mode-tool adapter registration and infer overlay probe paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.ops.runbooks.schema as runbook_schema
from dnadesign.ops.orchestrator import mode_tools


def _write_infer_config(path: Path, *, usr_root: Path | None, usr_dataset: str = "demo") -> None:
    if usr_root is None:
        jobs_block = "jobs: []"
    else:
        jobs_block = f"""
jobs:
  - id: job_a
    operation: extract
    ingest:
      source: usr
      root: "{usr_root}"
      dataset: "{usr_dataset}"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
""".strip()
    path.write_text(
        f"""
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
{jobs_block}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_infer_overlay_probe_avoids_workspace_fallback_when_usr_destination_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_usr_root = workspace_root / "outputs" / "usr_datasets"
    workspace_usr_root.mkdir(parents=True, exist_ok=True)
    for idx in range(20):
        marker_dir = workspace_usr_root / f"set_{idx:03d}" / "_derived" / "other"
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / "part-000.parquet").write_text("x\n", encoding="utf-8")

    external_usr_root = tmp_path / "external_usr_root"
    overlay_path = external_usr_root / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text("x\n", encoding="utf-8")

    infer_config = tmp_path / "infer.yaml"
    _write_infer_config(infer_config, usr_root=external_usr_root)

    def _unexpected_workspace_probe(_workspace_root: Path) -> tuple[Path, ...]:
        raise AssertionError("workspace fallback should not be used when infer USR destination resolves")

    monkeypatch.setattr(mode_tools, "_infer_workspace_overlay_candidates", _unexpected_workspace_probe)
    artifacts = mode_tools._infer_overlay_artifacts(workspace_root, infer_config=infer_config)
    assert artifacts == (overlay_path,)


def test_infer_overlay_probe_returns_no_artifacts_when_no_usr_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    overlay_path = workspace_root / "outputs" / "usr_datasets" / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text("x\n", encoding="utf-8")

    infer_config = tmp_path / "infer.yaml"
    _write_infer_config(infer_config, usr_root=None)

    def _unexpected_workspace_probe(_workspace_root: Path) -> tuple[Path, ...]:
        raise AssertionError(
            "workspace fallback should stay disabled for infer configs without a scoped USR destination"
        )

    monkeypatch.setattr(mode_tools, "_infer_workspace_overlay_candidates", _unexpected_workspace_probe)
    artifacts = mode_tools._infer_overlay_artifacts(workspace_root, infer_config=infer_config)
    assert artifacts == ()


def test_infer_overlay_probe_returns_no_artifacts_when_usr_destination_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    overlay_path = workspace_root / "outputs" / "usr_datasets" / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text("x\n", encoding="utf-8")

    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text(
        f"""
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: job_a
    operation: extract
    ingest:
      source: usr
      root: "{tmp_path / "usr_a"}"
      dataset: "dataset_a"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
  - id: job_b
    operation: extract
    ingest:
      source: usr
      root: "{tmp_path / "usr_b"}"
      dataset: "dataset_b"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _unexpected_workspace_probe(_workspace_root: Path) -> tuple[Path, ...]:
        raise AssertionError("workspace fallback should stay disabled for ambiguous infer configs")

    monkeypatch.setattr(mode_tools, "_infer_workspace_overlay_candidates", _unexpected_workspace_probe)
    artifacts = mode_tools._infer_overlay_artifacts(workspace_root, infer_config=infer_config)
    assert artifacts == ()


def test_infer_overlay_probe_returns_no_artifacts_when_usr_root_is_implicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    overlay_path = workspace_root / "outputs" / "usr_datasets" / "demo" / "_derived" / "infer.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text("x\n", encoding="utf-8")

    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: job_a
    operation: extract
    ingest:
      source: usr
      dataset: "demo"
      field: sequence
    outputs:
      - id: ll_mean
        fn: log_likelihood
        format: float
        params:
          reduction: mean
    io:
      write_back: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _unexpected_workspace_probe(_workspace_root: Path) -> tuple[Path, ...]:
        raise AssertionError("workspace fallback should stay disabled when infer ingest.root is implicit")

    monkeypatch.setattr(mode_tools, "_infer_workspace_overlay_candidates", _unexpected_workspace_probe)
    artifacts = mode_tools._infer_overlay_artifacts(workspace_root, infer_config=infer_config)
    assert artifacts == ()


def test_register_mode_tool_adapter_rejects_duplicate_tool() -> None:
    adapter = mode_tools.resolve_mode_tool_adapter_for_workflow_id("infer_batch_submit")
    with pytest.raises(ValueError, match="mode tool adapter already registered"):
        mode_tools.register_mode_tool_adapter("infer", adapter)


def test_register_mode_tool_adapter_requires_tool_name_match() -> None:
    adapter = mode_tools.resolve_mode_tool_adapter_for_workflow_id("infer_batch_submit")
    with pytest.raises(ValueError, match="mode tool adapter tool mismatch"):
        mode_tools.register_mode_tool_adapter("densegen", adapter)


def test_list_registered_mode_tools_returns_sorted_read_only_tuple() -> None:
    registered_tools = mode_tools.list_registered_mode_tools()
    assert registered_tools == ("densegen", "infer")
    assert isinstance(registered_tools, tuple)


def test_registered_mode_tools_exactly_match_schema_workflow_tools() -> None:
    assert mode_tools.list_registered_mode_tools() == runbook_schema.list_workflow_tools()
