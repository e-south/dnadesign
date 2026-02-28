"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_package_layout.py

Package layout contract tests for notify non-CLI module decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _notify_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_notify_non_cli_packages_importable() -> None:
    assert importlib.import_module("dnadesign.notify.delivery")
    assert importlib.import_module("dnadesign.notify.delivery.http")
    assert importlib.import_module("dnadesign.notify.delivery.payload")
    assert importlib.import_module("dnadesign.notify.delivery.secrets")
    assert importlib.import_module("dnadesign.notify.delivery.secrets.contract")
    assert importlib.import_module("dnadesign.notify.delivery.secrets.keyring_backend")
    assert importlib.import_module("dnadesign.notify.delivery.secrets.file_backend")
    assert importlib.import_module("dnadesign.notify.delivery.secrets.shell_backend")
    assert importlib.import_module("dnadesign.notify.delivery.secrets.ops")
    assert importlib.import_module("dnadesign.notify.delivery.validation")
    assert importlib.import_module("dnadesign.notify.events")
    assert importlib.import_module("dnadesign.notify.events.source")
    assert importlib.import_module("dnadesign.notify.events.source_builtin")
    assert importlib.import_module("dnadesign.notify.events.transforms")
    assert importlib.import_module("dnadesign.notify.profiles")
    assert importlib.import_module("dnadesign.notify.profiles.flows")
    assert importlib.import_module("dnadesign.notify.profiles.flow_types")
    assert importlib.import_module("dnadesign.notify.profiles.flow_webhook")
    assert importlib.import_module("dnadesign.notify.profiles.flow_events")
    assert importlib.import_module("dnadesign.notify.profiles.flow_profile")
    assert importlib.import_module("dnadesign.notify.profiles.ops")
    assert importlib.import_module("dnadesign.notify.profiles.schema")
    assert importlib.import_module("dnadesign.notify.profiles.schema.contract")
    assert importlib.import_module("dnadesign.notify.profiles.schema.reader")
    assert importlib.import_module("dnadesign.notify.profiles.schema.resolver")
    assert importlib.import_module("dnadesign.notify.profiles.policy")
    assert importlib.import_module("dnadesign.notify.profiles.workspace")
    assert importlib.import_module("dnadesign.notify.tool_events")
    assert importlib.import_module("dnadesign.notify.tool_events.core")
    assert importlib.import_module("dnadesign.notify.tool_events.densegen")
    assert importlib.import_module("dnadesign.notify.tool_events.densegen_common")
    assert importlib.import_module("dnadesign.notify.tool_events.densegen_metrics")
    assert importlib.import_module("dnadesign.notify.tool_events.densegen_messages")
    assert importlib.import_module("dnadesign.notify.tool_events.densegen_eval")
    assert importlib.import_module("dnadesign.notify.tool_events.packs_builtin")
    assert importlib.import_module("dnadesign.notify.tool_events.types")


def test_notify_runtime_cursor_submodules_importable() -> None:
    assert importlib.import_module("dnadesign.notify.runtime.cursor")
    assert importlib.import_module("dnadesign.notify.runtime.cursor.offsets")
    assert importlib.import_module("dnadesign.notify.runtime.cursor.locking")
    assert importlib.import_module("dnadesign.notify.runtime.cursor.iteration")


def test_notify_monolith_files_removed() -> None:
    notify_root = _notify_root()
    assert not (notify_root / "delivery" / "secrets.py").exists()
    assert not (notify_root / "runtime" / "cursor.py").exists()
    assert not (notify_root / "profiles" / "schema.py").exists()


def test_notify_legacy_top_level_module_paths_are_not_importable() -> None:
    legacy_modules = [
        "dnadesign.notify.event_transforms",
        "dnadesign.notify.events_source",
        "dnadesign.notify.events_source_builtin",
        "dnadesign.notify.http",
        "dnadesign.notify.payload",
        "dnadesign.notify.profile_flows",
        "dnadesign.notify.profile_ops",
        "dnadesign.notify.profile_schema",
        "dnadesign.notify.secrets",
        "dnadesign.notify.tool_event_packs_builtin",
        "dnadesign.notify.tool_event_types",
        "dnadesign.notify.tool_events_densegen",
        "dnadesign.notify.validation",
        "dnadesign.notify.workflow_policy",
        "dnadesign.notify.workspace_source",
    ]
    for module_path in legacy_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_path)
