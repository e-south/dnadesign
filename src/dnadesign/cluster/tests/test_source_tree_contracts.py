"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_source_tree_contracts.py

Information-architecture source-tree contracts for cluster package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _cluster_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "cluster"
    raise RuntimeError("repo root not found")


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def test_cluster_runtime_modules_live_under_src_directory() -> None:
    cluster_root = _cluster_root()
    assert (cluster_root / "src").is_dir()

    top_level_py = sorted(path.name for path in cluster_root.glob("*.py"))
    assert top_level_py == ["__init__.py", "api.py", "cli.py", "contracts.py"]


def test_cluster_root_keeps_progressive_disclosure_directories() -> None:
    cluster_root = _cluster_root()
    assert (cluster_root / "README.md").is_file()
    assert (cluster_root / "docs").is_dir()
    assert (cluster_root / "ops").is_dir()
    assert (cluster_root / "src").is_dir()
    assert (cluster_root / "tests").is_dir()
    assert (cluster_root / "workspaces").is_dir()


def test_cluster_root_keeps_minimal_top_level_surface() -> None:
    cluster_root = _cluster_root()
    observed = {
        path.name
        for path in cluster_root.iterdir()
        if path.name != "__pycache__" and not path.name.startswith(".") and path.name != "AGENTS.md"
    }
    assert observed == {
        "README.md",
        "__init__.py",
        "api.py",
        "assets",
        "cli.py",
        "contracts.py",
        "docs",
        "ops",
        "presets",
        "scripts",
        "src",
        "tests",
        "workspaces",
    }


def test_cluster_ops_surface_is_limited_to_status_registry_files() -> None:
    ops_root = _cluster_root() / "ops"
    observed = {
        path.name for path in ops_root.iterdir() if path.name != "__pycache__" and not path.name.startswith(".")
    }
    assert observed == {"__init__.py", "status.registry.yaml", "status_providers.py"}


def test_cluster_workspaces_scaffold_exists() -> None:
    cluster_root = _cluster_root()
    workspaces_root = cluster_root / "workspaces"
    assert (workspaces_root / "README.md").is_file()
    assert (workspaces_root / "promoter_clusters_v1" / "config.yaml").is_file()
    assert (workspaces_root / "perm_v1" / "config.yaml").is_file()


def test_cluster_package_data_includes_builtin_workspaces_and_presets() -> None:
    repo_root = _cluster_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"dnadesign.cluster"' in pyproject
    assert "workspaces/*/config.yaml" in pyproject
    assert "presets/*/*.yaml" in pyproject


def test_cluster_root_does_not_track_jobs_or_runtime_results() -> None:
    cluster_root = _cluster_root()
    assert not (cluster_root / "jobs").exists()
    assert not (cluster_root / "results").exists()


def test_cluster_tree_does_not_track_platform_cruft() -> None:
    cluster_root = _cluster_root()
    offenders = sorted(str(path.relative_to(cluster_root)) for path in cluster_root.rglob(".DS_Store"))
    assert offenders == []


def test_cluster_src_headers_do_not_use_template_project_placeholder() -> None:
    cluster_src_root = _cluster_root() / "src"
    offenders: list[str] = []
    for path in sorted(cluster_src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "<dnadesign project>" in text:
            offenders.append(str(path.relative_to(_cluster_root())))
    assert offenders == []


def test_cluster_cli_source_avoids_stale_jobs_and_old_artifact_help_strings() -> None:
    cli_app = (_cluster_root() / "src" / "cli" / "app.py").read_text(encoding="utf-8")
    assert "Defaults to sweeps/<method>/." not in cli_app
    assert "defaults to <results-root>/<FIT>/analysis/" not in cli_app


def test_cluster_runtime_keeps_execution_and_cli_boundaries_split() -> None:
    src_root = _cluster_root() / "src"
    for relative in (
        "execution.py",
        "execution_support.py",
        "execution_fit.py",
        "execution_fit_support.py",
        "execution_analysis_support.py",
        "execution_sweep.py",
        "execution_umap.py",
        "execution_analysis.py",
        "execution_table.py",
        "cli/app.py",
        "cli/commands.py",
        "cli/commands_fit.py",
        "cli/commands_umap.py",
        "cli/umap_resolution.py",
        "cli/commands_analysis.py",
        "cli/commands_table.py",
        "cli/resolution.py",
        "cli/subapps.py",
        "methods/kmeans.py",
        "methods/params.py",
        "presets/runtime.py",
        "io/parquet_attach.py",
        "runs/index.py",
        "runs/index_store.py",
        "umap/contracts.py",
        "umap/frame.py",
        "umap/hues.py",
        "umap/overlays.py",
        "umap/plot.py",
        "umap/requests.py",
        "workspaces/errors.py",
        "workspaces/paths.py",
        "workspaces/schema.py",
    ):
        assert (src_root / relative).is_file()

    assert _line_count(src_root / "execution.py") <= 120
    assert _line_count(src_root / "execution_support.py") <= 325
    assert _line_count(src_root / "execution_fit.py") <= 330
    assert _line_count(src_root / "execution_fit_support.py") <= 220
    assert _line_count(src_root / "execution_analysis_support.py") <= 280
    assert _line_count(src_root / "execution_sweep.py") <= 220
    assert _line_count(src_root / "execution_umap.py") <= 280
    assert _line_count(src_root / "execution_analysis.py") <= 160
    assert _line_count(src_root / "execution_table.py") <= 250
    assert _line_count(src_root / "cli" / "app.py") <= 120
    assert _line_count(src_root / "cli" / "commands.py") <= 80
    assert _line_count(src_root / "cli" / "commands_fit.py") <= 260
    assert _line_count(src_root / "cli" / "commands_umap.py") <= 170
    assert _line_count(src_root / "cli" / "umap_resolution.py") <= 260
    assert _line_count(src_root / "cli" / "commands_analysis.py") <= 240
    assert _line_count(src_root / "cli" / "commands_table.py") <= 160
    assert _line_count(src_root / "cli" / "resolution.py") <= 180
    assert _line_count(src_root / "cli" / "subapps.py") <= 220
    assert _line_count(src_root / "methods" / "kmeans.py") <= 180
    assert _line_count(src_root / "methods" / "params.py") <= 80
    assert _line_count(src_root / "io" / "parquet_attach.py") <= 220
    assert _line_count(src_root / "runs" / "index.py") <= 160
    assert _line_count(src_root / "runs" / "index_store.py") <= 140
    assert _line_count(src_root / "umap" / "contracts.py") <= 80
    assert _line_count(src_root / "umap" / "frame.py") <= 180
    assert _line_count(src_root / "umap" / "hues.py") <= 260
    assert _line_count(src_root / "umap" / "overlays.py") <= 120
    assert _line_count(src_root / "umap" / "plot.py") <= 380
    assert _line_count(src_root / "umap" / "requests.py") <= 220
    assert _line_count(src_root / "workspaces" / "errors.py") <= 40
    assert _line_count(src_root / "workspaces" / "loader.py") <= 80
    assert _line_count(src_root / "workspaces" / "paths.py") <= 180
    assert _line_count(src_root / "workspaces" / "schema.py") <= 300
