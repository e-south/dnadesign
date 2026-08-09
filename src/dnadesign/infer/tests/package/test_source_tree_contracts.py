"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_source_tree_contracts.py

Information-architecture source-tree contracts for infer package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _infer_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "infer"
    raise RuntimeError("repo root not found")


def _is_local_empty_surface_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.name == "__pycache__" or child.name.startswith("."):
            continue
        if child.is_dir() and _is_local_empty_surface_dir(child):
            continue
        return False
    return True


def test_infer_runtime_modules_live_under_src_directory() -> None:
    infer_root = _infer_root()
    assert (infer_root / "src").is_dir()

    top_level_py = sorted(path.name for path in infer_root.glob("*.py"))
    assert top_level_py == ["__init__.py", "__main__.py", "cli.py", "contracts.py"]


def test_infer_root_keeps_progressive_disclosure_directories() -> None:
    infer_root = _infer_root()
    assert (infer_root / "README.md").is_file()
    assert (infer_root / "docs").is_dir()
    assert (infer_root / "src").is_dir()
    assert (infer_root / "tests").is_dir()
    assert (infer_root / "workspaces").is_dir()


def test_infer_root_keeps_minimal_top_level_surface() -> None:
    infer_root = _infer_root()
    observed = {
        path.name
        for path in infer_root.iterdir()
        if path.name != "__pycache__"
        and not path.name.startswith(".")
        and path.name != "AGENTS.md"
        and not _is_local_empty_surface_dir(path)
    }
    assert observed == {
        "README.md",
        "__init__.py",
        "__main__.py",
        "assets",
        "cli.py",
        "contracts.py",
        "docs",
        "src",
        "tests",
        "workspaces",
    }


def test_infer_workspaces_scaffold_exists() -> None:
    infer_root = _infer_root()
    workspaces_root = infer_root / "workspaces"
    assert (workspaces_root / "README.md").is_file()


def test_infer_packaged_workspace_examples_exist() -> None:
    workspaces_root = _infer_root() / "workspaces"
    assert (workspaces_root / "evo2_feature_bundle_smoke" / "README.md").is_file()
    assert (workspaces_root / "evo2_feature_bundle_smoke" / "config.yaml").is_file()
    assert (workspaces_root / "evo2_feature_bundle_smoke" / "inputs" / "anchor_only_records.jsonl").is_file()
    assert (workspaces_root / "evo2_feature_bundle_smoke" / "inputs" / "template_1kb_records.jsonl").is_file()
    assert (workspaces_root / "demo_usr_pressure" / "config.yaml").is_file()


def test_infer_internal_cli_is_packaged_and_not_flat() -> None:
    infer_src = _infer_root() / "src"
    cli_dir = infer_src / "cli"
    assert (infer_src / "bootstrap.py").is_file()
    assert cli_dir.is_dir()
    assert (cli_dir / "__init__.py").is_file()
    assert (cli_dir / "app.py").is_file()
    assert (cli_dir / "console.py").is_file()
    assert (cli_dir / "builders.py").is_file()
    assert (cli_dir / "ingest.py").is_file()
    assert (cli_dir / "requests.py").is_file()
    assert not (infer_src / "cli.py").exists()
    assert not (infer_src / "cli_builders.py").exists()
    assert not (infer_src / "cli_ingest.py").exists()
    assert not (infer_src / "cli_requests.py").exists()


def test_infer_cli_commands_are_split_by_group() -> None:
    commands_dir = _infer_root() / "src" / "cli" / "commands"
    assert commands_dir.is_dir()
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "run.py").is_file()
    assert (commands_dir / "extract.py").is_file()
    assert (commands_dir / "generate.py").is_file()
    assert (commands_dir / "prune.py").is_file()
    assert (commands_dir / "migrate.py").is_file()
    assert (commands_dir / "presets.py").is_file()
    assert (commands_dir / "adapters.py").is_file()
    assert (commands_dir / "validate.py").is_file()
    assert (commands_dir / "workspace.py").is_file()


def test_infer_runtime_modules_are_grouped_under_runtime_package() -> None:
    runtime_dir = _infer_root() / "src" / "runtime"
    assert runtime_dir.is_dir()
    assert (runtime_dir / "__init__.py").is_file()
    assert (runtime_dir / "adapter_dispatch.py").is_file()
    assert (runtime_dir / "adapter_runtime.py").is_file()
    assert (runtime_dir / "batch_policy.py").is_file()
    assert (runtime_dir / "ingest_loading.py").is_file()
    assert (runtime_dir / "extract_chunk_writeback.py").is_file()
    assert (runtime_dir / "extract_execution.py").is_file()
    assert (runtime_dir / "generate_execution.py").is_file()
    assert (runtime_dir / "progress.py").is_file()
    assert (runtime_dir / "resume_planner.py").is_file()
    assert (runtime_dir / "writeback_dispatch.py").is_file()


def test_infer_feature_contracts_are_grouped_under_features_package() -> None:
    features_dir = _infer_root() / "src" / "features"
    assert features_dir.is_dir()
    assert (features_dir / "__init__.py").is_file()
    assert (features_dir / "contracts.py").is_file()
    assert (features_dir / "selectors.py").is_file()
    assert (features_dir / "context.py").is_file()
    assert (features_dir / "execution.py").is_file()
    assert (features_dir / "export.py").is_file()
    assert (features_dir / "overlay_pruning.py").is_file()
    assert not (features_dir / "legacy_alias_migration.py").exists()
    assert not (features_dir / "legacy_payload_retirement.py").exists()


def test_infer_root_does_not_track_runtime_log_artifacts() -> None:
    infer_root = _infer_root()
    tracked_logs = sorted(path.name for path in infer_root.glob("*.log"))
    assert tracked_logs == []


def test_infer_package_data_includes_workspace_templates_and_presets() -> None:
    repo_root = _infer_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"dnadesign.infer"' in pyproject
    assert "docs/operations/examples/*.yaml" in pyproject
    assert "src/presets/*/*.yaml" in pyproject
    assert "workspaces/*.md" in pyproject
    assert "workspaces/*/*.md" in pyproject
    assert "workspaces/*/*.yaml" in pyproject
    assert "workspaces/*/inputs/*" in pyproject


def test_infer_evo2_extra_stays_linux_x86_64_only() -> None:
    repo_root = _infer_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\"evo2>=0.5.3; sys_platform == 'linux' and platform_machine == 'x86_64'\"" in pyproject


def test_infer_tests_are_grouped_by_area() -> None:
    tests_root = _infer_root() / "tests"
    assert (tests_root / "cli").is_dir()
    assert (tests_root / "runtime").is_dir()
    assert (tests_root / "contracts").is_dir()
    assert (tests_root / "docs").is_dir()
    assert (tests_root / "package").is_dir()


def test_infer_src_headers_do_not_use_template_project_placeholder() -> None:
    infer_src_root = _infer_root() / "src"
    offenders: list[str] = []
    for path in sorted(infer_src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "<dnadesign project>" in text:
            offenders.append(str(path.relative_to(_infer_root())))
    assert offenders == []
