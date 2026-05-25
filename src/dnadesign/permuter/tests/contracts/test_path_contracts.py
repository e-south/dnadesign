"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/contracts/test_path_contracts.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.permuter.src.core.paths import resolve
from dnadesign.permuter.src.workspaces.datasets import resolve_workspace_dataset_path


def _refs_csv(workspace: Path) -> Path:
    path = workspace / "refs.csv"
    path.write_text("ref_name,sequence\nx,ACGT\n", encoding="utf-8")
    return path


def _config_yaml(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspaces" / "toy"
    workspace.mkdir(parents=True)
    _refs_csv(workspace)
    path = workspace / "config.yaml"
    path.write_text("scope:\n  name: toy\n", encoding="utf-8")
    return path


def _workspace_config(tmp_path: Path, *, layout: str = "flat") -> Path:
    workspace = tmp_path / "workspaces" / "toy"
    workspace.mkdir(parents=True)
    _refs_csv(workspace)
    path = workspace / "config.yaml"
    path.write_text(
        f"""
scope:
  name: toy
  bio_type: dna
  input:
    refs: "${{WORKSPACE_DIR}}/refs.csv"
    name_col: ref_name
    seq_col: sequence
  permute:
    protocol: scan_dna
    params: {{}}
  output:
    dir: "${{WORKSPACE_DIR}}/outputs"
    layout: {layout}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_resolve_rejects_invalid_output_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid output.layout"):
        resolve(
            config_yaml=_config_yaml(tmp_path),
            refs="${WORKSPACE_DIR}/refs.csv",
            output_dir="${WORKSPACE_DIR}/outputs",
            ref_name="x",
            out_override=None,
            layout="mystery",
        )


def test_resolve_rejects_legacy_flat_ref_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid output.layout"):
        resolve(
            config_yaml=_config_yaml(tmp_path),
            refs="${WORKSPACE_DIR}/refs.csv",
            output_dir="${WORKSPACE_DIR}/outputs",
            ref_name="x",
            out_override=None,
            layout="flat_ref",
        )


def test_permuter_output_root_overrides_configured_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "cluster-results"
    monkeypatch.setenv("PERMUTER_OUTPUT_ROOT", str(env_root))

    paths = resolve(
        config_yaml=_config_yaml(tmp_path),
        refs="${WORKSPACE_DIR}/refs.csv",
        output_dir="${WORKSPACE_DIR}/outputs",
        ref_name="x",
        out_override=None,
        layout="flat",
    )

    assert paths.output_root == env_root / "toy"
    assert paths.dataset_dir == paths.output_root


def test_explicit_out_overrides_permuter_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERMUTER_OUTPUT_ROOT", str(tmp_path / "cluster-results"))
    explicit = tmp_path / "explicit"

    paths = resolve(
        config_yaml=_config_yaml(tmp_path),
        refs="${WORKSPACE_DIR}/refs.csv",
        output_dir="${WORKSPACE_DIR}/outputs",
        ref_name="x",
        out_override=explicit,
        layout="flat",
    )

    assert paths.output_root == explicit
    assert paths.dataset_dir == explicit


def test_resolve_rejects_legacy_job_dir_variable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="WORKSPACE_DIR"):
        resolve(
            config_yaml=_config_yaml(tmp_path),
            refs="${JOB_DIR}/refs.csv",
            output_dir="${WORKSPACE_DIR}/outputs",
            ref_name="x",
            out_override=None,
            layout="flat",
        )


def test_resolve_rejects_undocumented_package_root_variable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PACKAGE_ROOT"):
        resolve(
            config_yaml=_config_yaml(tmp_path),
            refs="${PACKAGE_ROOT}/refs.csv",
            output_dir="${WORKSPACE_DIR}/outputs",
            ref_name="x",
            out_override=None,
            layout="flat",
        )


def test_workspace_dataset_resolution_does_not_probe_legacy_flat_ref_layout(tmp_path: Path) -> None:
    config = _workspace_config(tmp_path, layout="flat")
    legacy_dir = config.parent / "outputs__x"
    legacy_dir.mkdir()
    (legacy_dir / "records.parquet").write_text("not a parquet file", encoding="utf-8")

    resolved = resolve_workspace_dataset_path(workspace_hint=config, ref="x", out=None)

    assert resolved.records == config.parent / "outputs" / "records.parquet"
    assert not (config.parent / "outputs").exists()
