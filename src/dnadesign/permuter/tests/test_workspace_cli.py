"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_workspace_cli.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_workspace_validate_minimal_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workspace:
  id: toy_workspace
runs:
  - id: dna_scan
    protocol: scan_dna
    inputs:
      ref_name: toy
      sequence: ACGT
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["workspace", "validate", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "toy_workspace" in result.output
    assert "1 run" in result.output


def test_workspace_validate_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workspace:
  id: toy_workspace
runs:
  - id: dna_scan
    protocol: scan_dna
  - id: dna_scan
    protocol: scan_dna
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["workspace", "validate", "--workspace", str(workspace)])

    assert result.exit_code != 0
    assert "duplicate run id" in result.output


def test_workspace_list_skips_unrelated_config_yaml(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "config.yaml").write_text(
        """
workspace:
  id: valid_workspace
runs:
  - id: dna_scan
    protocol: scan_dna
""".strip()
        + "\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "config.yaml").write_text("model:\n  id: not_permuter\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["workspace", "list", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "valid_workspace" in result.output
    assert "not_permuter" not in result.output
