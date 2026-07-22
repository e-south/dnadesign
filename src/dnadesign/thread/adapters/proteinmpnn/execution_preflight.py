"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/execution_preflight.py

Official ProteinMPNN checkout validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dnadesign.thread.adapters.proteinmpnn.models import ProteinMpnnRequestIssue

REQUIRED_SCRIPT_PATHS = (
    "protein_mpnn_run.py",
    "helper_scripts/parse_multiple_chains.py",
    "helper_scripts/assign_fixed_chains.py",
    "helper_scripts/make_fixed_positions_dict.py",
)
DEFAULT_MODEL_NAME = "v_48_020"


def resolve_proteinmpnn_root(root: Path | None = None) -> Path:
    """Resolve the explicit ProteinMPNN checkout path."""

    if root is not None:
        return root.expanduser().resolve()
    env_value = os.environ.get("PROTEINMPNN_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    raise ValueError("ProteinMPNN execution requires --proteinmpnn-root or PROTEINMPNN_ROOT")


def validate_proteinmpnn_root(root: Path, *, model_name: str = DEFAULT_MODEL_NAME) -> list[ProteinMpnnRequestIssue]:
    """Validate that a ProteinMPNN checkout has the official scripts and model weights."""

    issues: list[ProteinMpnnRequestIssue] = []
    root = root.expanduser().resolve()
    for rel_path in REQUIRED_SCRIPT_PATHS:
        path = root / rel_path
        if not path.is_file():
            issues.append(
                ProteinMpnnRequestIssue(
                    check_id="thread.proteinmpnn.tool_missing_script",
                    message=f"ProteinMPNN checkout is missing official script {rel_path!r}",
                    path=str(path),
                )
            )
    weights = root / "vanilla_model_weights" / f"{model_name}.pt"
    if not weights.is_file():
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.tool_missing_weights",
                message=f"ProteinMPNN checkout is missing vanilla model weights {model_name!r}",
                path=str(weights),
            )
        )
    return issues


def proteinmpnn_git_commit(root: Path) -> str:
    """Return the ProteinMPNN git commit when the checkout carries Git metadata."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
