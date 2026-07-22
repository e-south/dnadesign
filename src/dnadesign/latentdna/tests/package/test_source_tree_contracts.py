"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/package/test_source_tree_contracts.py

Source-tree contracts for the latentdna package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _latentdna_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "latentdna"
    raise RuntimeError("repo root not found")


def test_latentdna_root_keeps_progressive_disclosure_directories() -> None:
    latentdna_root = _latentdna_root()
    assert (latentdna_root / "README.md").is_file()
    assert (latentdna_root / "docs").is_dir()
    assert (latentdna_root / "ops").is_dir()
    assert (latentdna_root / "docs" / "reference").is_dir()
    assert (latentdna_root / "docs" / "dev").is_dir()
    assert (latentdna_root / "src").is_dir()
    assert (latentdna_root / "tests").is_dir()
    assert (latentdna_root / "tests" / "sources").is_dir()
    assert (latentdna_root / "tests" / "views").is_dir()
    assert (latentdna_root / "tests" / "perf").is_dir()
    assert (latentdna_root / "workspaces").is_dir()


def test_latentdna_root_keeps_minimal_top_level_surface() -> None:
    latentdna_root = _latentdna_root()
    observed = {
        path.name
        for path in latentdna_root.iterdir()
        if path.name != "__pycache__" and not path.name.startswith(".") and path.name != "AGENTS.md"
    }
    assert observed == {
        "README.md",
        "docs",
        "ops",
        "src",
        "tests",
        "workspaces",
    }


def test_latentdna_internal_cli_is_nested_under_src() -> None:
    latentdna_src = _latentdna_root() / "src"
    cli_dir = latentdna_src / "cli"
    cluster_dir = latentdna_src / "clusters"
    metadata_dir = latentdna_src / "metadata"
    notebook_dir = latentdna_src / "notebooks"
    presentation_dir = latentdna_src / "presentation"
    references_dir = latentdna_src / "references"
    stats_dir = latentdna_src / "stats"
    workspaces_dir = latentdna_src / "workspaces"
    assert not (latentdna_src / "api.py").exists()
    assert sorted(path.name for path in latentdna_src.glob("*.py")) == ["__init__.py", "version.py"]
    assert cli_dir.is_dir()
    assert (cli_dir / "__init__.py").is_file()
    assert (cli_dir / "app.py").is_file()
    assert (cli_dir / "commands").is_dir()
    assert cluster_dir.is_dir()
    assert (cluster_dir / "__init__.py").is_file()
    assert (cluster_dir / "fit.py").is_file()
    assert metadata_dir.is_dir()
    assert (metadata_dir / "__init__.py").is_file()
    assert (metadata_dir / "axes.py").is_file()
    assert (metadata_dir / "join_keys.py").is_file()
    assert notebook_dir.is_dir()
    assert (notebook_dir / "__init__.py").is_file()
    assert (notebook_dir / "browser_runtime.py").is_file()
    assert (notebook_dir / "browser_runtime_docs.py").is_file()
    assert (notebook_dir / "browser_runtime_compare.py").is_file()
    assert (notebook_dir / "browser_runtime_projection.py").is_file()
    assert (notebook_dir / "browser_runtime_plot_review_axes.py").is_file()
    assert (notebook_dir / "browser_runtime_support.py").is_file()
    assert (notebook_dir / "browser_runtime_ui.py").is_file()
    assert (notebook_dir / "rendering.py").is_file()
    assert (notebook_dir / "scaffold.py").is_file()
    assert (notebook_dir / "scaffold_geometry_panels.py").is_file()
    assert (notebook_dir / "scaffold_panels.py").is_file()
    assert (notebook_dir / "scaffold_pages.py").is_file()
    assert (notebook_dir / "scaffold_plot_review.py").is_file()
    assert (notebook_dir / "scaffold_selectors.py").is_file()
    assert presentation_dir.is_dir()
    assert (presentation_dir / "__init__.py").is_file()
    assert (presentation_dir / "annotation_layout.py").is_file()
    assert (presentation_dir / "labels.py").is_file()
    assert (presentation_dir / "visual_style.py").is_file()
    assert references_dir.is_dir()
    assert (references_dir / "__init__.py").is_file()
    assert (references_dir / "sets.py").is_file()
    assert stats_dir.is_dir()
    assert (stats_dir / "__init__.py").is_file()
    assert (stats_dir / "rank.py").is_file()
    assert workspaces_dir.is_dir()
    assert (workspaces_dir / "__init__.py").is_file()
    assert (workspaces_dir / "loader.py").is_file()
    assert (workspaces_dir / "paths.py").is_file()
    assert (workspaces_dir / "scaffold.py").is_file()
    assert (workspaces_dir / "validation.py").is_file()


def test_latentdna_source_modules_stay_below_monolith_limit() -> None:
    latentdna_src = _latentdna_root() / "src"
    max_lines = 1500
    oversized = {}
    for path in latentdna_src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > max_lines:
            oversized[path.relative_to(latentdna_src).as_posix()] = line_count
    assert oversized == {}


def test_latentdna_source_modules_use_canonical_headers() -> None:
    latentdna_src = _latentdna_root() / "src"
    repo_root = _latentdna_root().parents[2]
    separator = "-" * 80
    invalid: dict[str, str] = {}
    for path in sorted(latentdna_src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(repo_root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        header = "\n".join(lines[:12])
        if len(lines) < 9:
            invalid[relative_path] = "missing canonical header"
            continue
        if lines[0] != '"""' or lines[1] != separator or lines[2] != "dnadesign":
            invalid[relative_path] = "header prefix mismatch"
            continue
        if lines[3] != relative_path:
            invalid[relative_path] = f"path mismatch: {lines[3]!r}"
            continue
        if not lines[5].strip() or len(lines[5]) > 140:
            invalid[relative_path] = "missing concise module purpose"
            continue
        if "Module Author(s): Eric J. South" not in header:
            invalid[relative_path] = "author mismatch"
            continue
        if "Codex" in header or "OpenAI" in header:
            invalid[relative_path] = "agent author claim is not allowed"
            continue
    assert invalid == {}


def test_latentdna_package_data_uses_workspace_shape_globs() -> None:
    repo_root = _latentdna_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"dnadesign.latentdna"' in pyproject
    assert "workspaces/templates/*/*.yaml" in pyproject
    assert "workspaces/templates/*/*.md" in pyproject


def test_latentdna_integration_tests_use_descriptive_module_names() -> None:
    integrations_dir = _latentdna_root() / "tests" / "integrations"
    legacy_modules = sorted(path.name for path in integrations_dir.glob("test_phase*.py"))
    assert legacy_modules == []
