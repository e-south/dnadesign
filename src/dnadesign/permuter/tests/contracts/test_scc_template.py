"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/contracts/test_scc_template.py

Permuter SCC wrapper contract checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_permuter_scc_template_is_public_surface_wrapper() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    template = repo_root / "docs" / "bu-scc" / "jobs" / "permuter-evaluate.qsub"
    text = template.read_text(encoding="utf-8")

    assert "PERMUTER_WORKSPACE is required" in text
    assert "PERMUTER_REF is required" in text
    assert 'export USR_ACTOR_TOOL="${USR_ACTOR_TOOL:-permuter}"' in text
    assert "uv run permuter workspace validate --workspace" in text
    assert "uv run permuter run" in text
    assert "uv run permuter evaluate" in text
    assert "--json" in text
    assert "outputs/logs/ops/runtime" in text


def test_permuter_scc_template_is_shell_parseable() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    template = repo_root / "docs" / "bu-scc" / "jobs" / "permuter-evaluate.qsub"

    result = subprocess.run(
        ["bash", "-n", str(template)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
