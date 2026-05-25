"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/cli/test_workspace.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_workspace_validate_minimal_config(tmp_path: Path) -> None:
    workspace = tmp_path / "toy_workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
scope:
  name: toy_workspace
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
    name_col: ref_name
    seq_col: sequence
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "validate", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "toy_workspace" in result.output
    assert "scan_dna" in result.output


def test_workspace_validate_rejects_scope_name_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "toy_workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
scope:
  name: other_scope
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "validate", "--workspace", str(workspace)])

    assert result.exit_code != 0
    assert "scope id must match scope.name" in result.output


def test_workspace_validate_rejects_unknown_config_fields(tmp_path: Path) -> None:
    workspace = tmp_path / "toy_workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
scope:
  name: toy_workspace
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
    layoutt: flat
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "validate", "--workspace", str(workspace)])

    assert result.exit_code != 0
    assert "Extra inputs are not permitted" in result.output


def test_run_reuses_workspace_validation_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "toy_workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
scope:
  name: other_scope
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["run", "--workspace", str(workspace), "--ref", "toy"])

    assert result.exit_code != 0
    assert "scope id must match scope.name" in str(result.exception)
    assert not (workspace / "outputs").exists()


def test_workspace_list_skips_unrelated_config_yaml(tmp_path: Path) -> None:
    valid = tmp_path / "valid_workspace"
    valid.mkdir()
    (valid / "config.yaml").write_text(
        """
scope:
  name: valid_workspace
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (valid / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "config.yaml").write_text("model:\n  id: not_permuter\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "list", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "valid_workspace" in result.output
    assert "not_permuter" not in result.output


def test_workspace_list_json_emits_parseable_machine_output_for_long_paths(tmp_path: Path) -> None:
    long_root = tmp_path / ("very_long_workspace_root_" + "x" * 120)
    workspace = long_root / "valid_workspace"
    workspace.mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        """
scope:
  name: valid_workspace
  bio_type: dna
  input:
    refs: "${WORKSPACE_DIR}/refs.csv"
  permute:
    protocol: scan_dna
    params: {}
  output:
    dir: "${WORKSPACE_DIR}/outputs"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "refs.csv").write_text("ref_name,sequence\ntoy,ACGT\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "list", "--root", str(long_root), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["id"] == "valid_workspace"
