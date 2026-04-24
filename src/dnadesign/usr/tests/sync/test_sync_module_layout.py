"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/test_sync_module_layout.py

Layout contract tests for the root sync facade decomposition.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src import sync as sync_module


def test_sync_remote_execution_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync.remote.execution")
    assert hasattr(module, "SyncRuntime")
    assert hasattr(module, "plan_diff")
    assert hasattr(module, "plan_diff_file")
    assert hasattr(module, "execute_pull")
    assert hasattr(module, "execute_pull_file")
    assert hasattr(module, "execute_push")
    assert hasattr(module, "execute_push_file")


def test_sync_module_delegates_execution_orchestration() -> None:
    source = inspect.getsource(sync_module)
    assert "sync_execution.plan_diff(" in source
    assert "sync_execution.plan_diff_file(" in source
    assert "sync_execution.execute_pull(" in source
    assert "sync_execution.execute_pull_file(" in source
    assert "sync_execution.execute_push(" in source
    assert "sync_execution.execute_push_file(" in source
