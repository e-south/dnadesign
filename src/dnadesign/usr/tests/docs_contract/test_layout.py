"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/test_layout.py

Structural layout and maintainer-boundary contracts for USR docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess

from ..source_layout_inventory import TOP_LEVEL_SOURCE_PACKAGES
from .helpers import read_text, repo_root


def test_usr_layout_docs_cover_sanctioned_source_packages_and_new_support_surfaces() -> None:
    usr_agents = read_text("src/dnadesign/usr/AGENTS.md")
    code_map = read_text("src/dnadesign/usr/docs/reference/dataset-layout-and-code-map.md")
    api_doc = read_text("src/dnadesign/usr/docs/reference/python-api.md")
    introspection = read_text("src/dnadesign/usr/docs/architecture-introspection.md")

    for package_name in sorted(TOP_LEVEL_SOURCE_PACKAGES - {"storage"}):
        token = f"{package_name}/"
        assert token in usr_agents
        assert token in code_map

    for token in (
        "cli/support/resolution/dataset_targets.py",
        "cli/support/wiring/dependencies.py",
        "cli/support/wiring/registration.py",
        "cli/support/presentation/runtime.py",
        "src/dnadesign/usr/ops/sync_audit_drill.py",
        "src/dnadesign/testsupport/usr.py",
    ):
        assert token in code_map or token in introspection or token in usr_agents

    assert "Public import surface: `dnadesign.usr`" in api_doc
    assert "uv run usr-sync-audit-drill" in api_doc


def test_usr_agents_describe_ops_scripts_assets_and_testsupport_boundaries() -> None:
    usr_agents = read_text("src/dnadesign/usr/AGENTS.md")

    assert "src/dnadesign/usr/ops/" in usr_agents
    assert "sync_audit_drill.py" in usr_agents
    assert "src/dnadesign/usr/scripts/" in usr_agents
    assert "not a public cross-tool API surface" in usr_agents
    assert "src/dnadesign/usr/assets/demo_material/" in usr_agents
    assert "src/dnadesign/testsupport/" in usr_agents
    assert "Shared test fixtures consumed outside USR belong under `src/dnadesign/testsupport/`" in usr_agents


def test_usr_archive_and_assets_docs_use_canonical_paths() -> None:
    datasets_index = read_text("src/dnadesign/usr/datasets/README.md")
    archive_readme = read_text("src/dnadesign/usr/datasets/archived/README.md")
    quickstart = read_text("src/dnadesign/usr/docs/getting-started/cli-quickstart.md")

    assert "archived/" in datasets_index
    assert "_archive/" not in datasets_index
    assert "canonical location for archived datasets" in datasets_index
    assert "sanctioned location for archived usr datasets" in archive_readme.lower()
    assert "src/dnadesign/usr/assets/demo_material/demo_sequences.csv" in quickstart
    assert "uv run usr validate usr_demo_cli_examples --strict" in quickstart
    assert "USR_SHOW_DEV_COMMANDS=1 uv run usr dev make-mock --help" in quickstart
    assert "src/dnadesign/usr/demo_material/demo_sequences.csv" not in quickstart
    assert not (repo_root() / "src" / "dnadesign" / "usr" / "demo_material").exists()
    assert not (repo_root() / "src" / "dnadesign" / "usr" / "archived").exists()


def test_usr_surface_does_not_track_transient_cache_artifacts() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src/dnadesign/usr"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tracked = [
        path
        for path in result.stdout.splitlines()
        if path.endswith(".DS_Store") or "/__pycache__/" in path or path.endswith(".pyc")
    ]
    assert tracked == []
