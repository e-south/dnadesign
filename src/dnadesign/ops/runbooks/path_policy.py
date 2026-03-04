"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/runbooks/path_policy.py

Shared path-policy constants for Ops runbook and operational artifact contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR = Path("src/dnadesign/ops/runbooks/presets")
WORKSPACE_RUNBOOKS_RELATIVE_DIR = Path("outputs/logs/ops/runbooks")
WORKSPACE_AUDIT_RELATIVE_DIR = Path("outputs/logs/ops/audit")
WORKSPACE_SGE_STDOUT_RELATIVE_DIR = Path("outputs/logs/ops/sge")
WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR = Path("outputs/logs/ops/runtime")
REPO_TRANSIENT_OPERATIONAL_DIR_NAMES = (".codex_tmp", ".tmp_ops", "tmp_ops")
