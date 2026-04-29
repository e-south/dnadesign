"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_package_layout.py

Tests for package-root path resolution after baserender internal src/ consolidation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import subprocess
import tomllib
from pathlib import Path

import pytest

from dnadesign.baserender.src.config import resolve_job_path, resolve_preset_path, resolve_style
from dnadesign.baserender.src.config.jobs import base_render_v3, sequence_rows_v3
from dnadesign.baserender.src.workspaces import default_workspaces_root


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


def test_resolve_job_path_missing_job_message_does_not_point_to_missing_jobs_dir() -> None:
    with pytest.raises(FileNotFoundError, match="docs/examples/ or as an explicit path"):
        resolve_job_path("definitely_missing_job_name")


def test_sequence_rows_job_namespace_exports_contract() -> None:
    assert sequence_rows_v3.SequenceRowsJobV3 is not None


def test_base_render_job_namespace_is_canonical_generic_contract() -> None:
    assert base_render_v3.BaseRenderJobV3 is not None
    assert base_render_v3.RenderJobV3 is base_render_v3.BaseRenderJobV3
    assert base_render_v3.SequenceRowsJobV3 is base_render_v3.BaseRenderJobV3


def test_cli_implementation_lives_under_src_cli_package() -> None:
    root = _baserender_root()
    assert (root / "src" / "cli" / "__init__.py").exists()
    assert (root / "src" / "cli" / "app.py").exists()
    assert (root / "src" / "cli" / "actions.py").exists()
    assert not (root / "cli.py").exists()
    assert not (root / "src" / "cli.py").exists()
    assert not (root / "src" / "cli_actions.py").exists()


def test_package_root_contains_only_public_initializer() -> None:
    loose_modules = sorted(p.name for p in _baserender_root().glob("*.py"))
    assert loose_modules == ["__init__.py"]


def test_output_implementation_lives_under_src_outputs_package() -> None:
    root = _baserender_root()
    assert (root / "src" / "outputs" / "__init__.py").exists()
    assert (root / "src" / "outputs" / "images.py").exists()
    assert (root / "src" / "outputs" / "names.py").exists()
    assert (root / "src" / "outputs" / "video.py").exists()
    assert not (root / "src" / "outputs.py").exists()


def test_src_root_contains_only_package_initializer() -> None:
    loose_modules = sorted(p.name for p in (_baserender_root() / "src").glob("*.py"))
    assert loose_modules == ["__init__.py"]


def test_src_nested_ia_packages_exist() -> None:
    root = _baserender_root()
    for rel in (
        "src/public",
        "src/execution",
        "src/runtime",
        "src/workspaces",
        "src/styles/curated",
    ):
        assert (root / rel / "__init__.py").exists(), rel


def test_console_script_targets_src_cli_app() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["scripts"]["baserender"] == "dnadesign.baserender:app"


def test_no_tracked_generated_baserender_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "src/dnadesign/baserender"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    offenders = []
    for path in tracked:
        p = Path(path)
        parts = p.parts
        if not p.exists():
            continue
        is_workspace_artifact = "workspaces" in parts and (
            "results" in parts or ("outputs" in parts and not path.endswith("/.gitkeep"))
        )
        if any(part in {".mplconfig", "__pycache__"} for part in parts) or path.endswith(".DS_Store"):
            offenders.append(path)
        elif is_workspace_artifact:
            offenders.append(path)
    assert offenders == []
