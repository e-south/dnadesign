"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/tests_layout_inventory.py

Single source of truth for the sanctioned USR test layout.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

ROOT_TEST_SUPPORT_FILES = {
    "__init__.py",
    "registry_helpers.py",
    "source_layout_inventory.py",
    "tests_layout_inventory.py",
}

ROOT_TEST_MODULES = {
    "test_module_layout.py",
    "test_public_api_imports.py",
    "test_root_resolution_contract.py",
    "test_tests_layout.py",
    "test_usr_docs_contract.py",
    "test_usr_harness_script.py",
    "test_usr_sync_audit_drill_script.py",
}

TOP_LEVEL_TEST_PACKAGES = {
    "cli",
    "datasets",
    "legacy",
    "remote_sync",
    "sync",
}
