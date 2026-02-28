"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_package_layout.py

Tests for package-root path resolution after baserender internal src/ consolidation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.baserender.src.config import resolve_job_path, resolve_preset_path, resolve_style
from dnadesign.baserender.src.config.jobs import sequence_rows_v3
from dnadesign.baserender.src.workspace import default_workspaces_root


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_default_style_preset_resolves_from_package_root() -> None:
    preset = resolve_preset_path("presentation_default")
    assert preset is not None
    assert preset.resolve() == (_baserender_root() / "styles" / "style_v1" / "presentation_default.yaml").resolve()


def test_presentation_default_uses_solid_connectors() -> None:
    style = resolve_style(preset="presentation_default", overrides={})
    assert style.connector_dash == ()


def test_default_workspaces_root_resolves_from_current_working_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_workspaces_root().resolve() == (tmp_path / "workspaces").resolve()


def test_resolve_job_path_finds_docs_example_by_name() -> None:
    assert (
        resolve_job_path("densegen_job").resolve()
        == (_baserender_root() / "docs" / "examples" / "densegen_job.yaml").resolve()
    )


def test_sequence_rows_job_namespace_exports_contract() -> None:
    assert sequence_rows_v3.SequenceRowsJobV3 is not None
