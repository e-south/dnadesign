"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/architecture/test_lifecycle.py

Lifecycle contracts for source, command and distribution boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest

from dnadesign.devtools.architecture.boundaries import find_top_level_layout_violations
from dnadesign.devtools.architecture.lifecycle import find_installation_issues, load_lifecycle, wheel_issues
from dnadesign.devtools.docs.public_surface_contracts import _find_readme_tool_catalog_issues


def scaffold(root: Path, status: str = "active") -> None:
    (root / ".github").mkdir()
    (root / ".github/tool-lifecycle.toml").write_text(
        "schema_version = 1\n[components]\n"
        f'alpha = {{status="{status}", catalog=true}}\n'
        'old = {status="retired", catalog=false}\n'
    )
    (root / "src/dnadesign/alpha").mkdir(parents=True)
    (root / "src/dnadesign/alpha/__init__.py").write_text("")
    (root / "src/dnadesign/alpha/README.md").write_text("# Alpha\n")
    (root / "src/dnadesign/__init__.py").write_text("")
    (root / "src/dnadesign/_entrypoints.py").write_text("")
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text('[project.scripts]\nalpha="dnadesign.alpha:main"\n')
    (root / "README.md").write_text(
        "## Available tools\n\n### Design, modeling, and analysis\n\n"
        "| Tool | Description |\n| --- | --- |\n| [**alpha**](src/dnadesign/alpha/) | Do work. |\n"
    )


def test_live_commands_and_wheel_members_match_lifecycle(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert find_installation_issues(tmp_path) == []
    assert wheel_issues(load_lifecycle(tmp_path), ["dnadesign/alpha/__init__.py"]) == []


@pytest.mark.parametrize("target", ["dnadesign.old:main", "dnadesign.unknown:main"])
def test_command_owner_must_be_declared_and_not_retired(tmp_path: Path, target: str) -> None:
    scaffold(tmp_path)
    (tmp_path / "pyproject.toml").write_text(f'[project.scripts]\nprobe="{target}"\n')
    assert find_installation_issues(tmp_path)


def test_lazy_launcher_cannot_hide_a_retired_command(tmp_path: Path) -> None:
    scaffold(tmp_path)
    launcher = tmp_path / "src/dnadesign/_entrypoints.py"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text('def old():\n    return _run("dnadesign.old:main", "full")\n')
    (tmp_path / "pyproject.toml").write_text('[project.scripts]\nprobe="dnadesign._entrypoints:old"\n')
    assert any("retired" in item for item in find_installation_issues(tmp_path))


@pytest.mark.parametrize("member", ["dnadesign/old/config.yaml", "dnadesign/unknown/__init__.py"])
def test_wheel_rejects_retired_and_unclassified_payload(tmp_path: Path, member: str) -> None:
    scaffold(tmp_path)
    assert wheel_issues(load_lifecycle(tmp_path), ["dnadesign/alpha/__init__.py", member])


def test_wheel_requires_retained_components(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert wheel_issues(load_lifecycle(tmp_path), [])


def test_repository_components_remain_in_source_but_not_installed(tmp_path: Path) -> None:
    scaffold(tmp_path)
    metadata = tmp_path / ".github/tool-lifecycle.toml"
    metadata.write_text(metadata.read_text() + 'checks={status="active", catalog=false, installed=false}\n')
    (tmp_path / "src/dnadesign/checks").mkdir()
    assert find_top_level_layout_violations(repo_root=tmp_path) == []
    components = load_lifecycle(tmp_path)
    assert wheel_issues(components, ["dnadesign/alpha/__init__.py"]) == []
    assert wheel_issues(components, ["dnadesign/alpha/__init__.py", "dnadesign/checks/__init__.py"])
    (tmp_path / "pyproject.toml").write_text('[project.scripts]\nchecks="dnadesign.checks:main"\n')
    assert find_installation_issues(tmp_path)


@pytest.mark.parametrize("installed", ['"false"', "1"])
def test_installation_flag_requires_a_boolean(tmp_path: Path, installed: str) -> None:
    scaffold(tmp_path)
    metadata = tmp_path / ".github/tool-lifecycle.toml"
    metadata.write_text(metadata.read_text() + f'checks={{status="active", catalog=false, installed={installed}}}\n')
    with pytest.raises(ValueError, match="boolean"):
        load_lifecycle(tmp_path)


def test_retired_component_cannot_be_installed(tmp_path: Path) -> None:
    scaffold(tmp_path)
    metadata = tmp_path / ".github/tool-lifecycle.toml"
    metadata.write_text(
        metadata.read_text().replace(
            'status="retired", catalog=false', 'status="retired", catalog=false, installed=true'
        )
    )
    with pytest.raises(ValueError, match="retired"):
        load_lifecycle(tmp_path)


def test_supported_catalog_does_not_require_migration_prose(tmp_path: Path) -> None:
    scaffold(tmp_path, "maintenance")
    assert _find_readme_tool_catalog_issues(tmp_path) == []


def test_directories_do_not_implicitly_enter_catalog(tmp_path: Path) -> None:
    scaffold(tmp_path)
    (tmp_path / "src/dnadesign/sketch").mkdir()
    assert _find_readme_tool_catalog_issues(tmp_path) == []
    violations = find_top_level_layout_violations(repo_root=tmp_path)
    assert [item.path.name for item in violations] == ["sketch"]


def test_retired_source_is_rejected_even_without_a_command(tmp_path: Path) -> None:
    scaffold(tmp_path)
    (tmp_path / "src/dnadesign/old").mkdir()
    assert [item.path.name for item in find_top_level_layout_violations(repo_root=tmp_path)] == ["old"]


def test_catalog_rejects_retired_rows_when_no_public_tools_remain(tmp_path: Path) -> None:
    scaffold(tmp_path)
    metadata = tmp_path / ".github/tool-lifecycle.toml"
    metadata.write_text(
        metadata.read_text().replace(
            'status="active", catalog=true',
            'status="retired", catalog=false',
        )
    )
    assert _find_readme_tool_catalog_issues(tmp_path)


@pytest.mark.parametrize("value", ['status="typo", catalog=true', 'status="retired", catalog=true'])
def test_invalid_lifecycle_declarations_fail_closed(tmp_path: Path, value: str) -> None:
    scaffold(tmp_path)
    (tmp_path / ".github/tool-lifecycle.toml").write_text(f"schema_version=1\n[components]\nalpha={{{value}}}\n")
    with pytest.raises(ValueError):
        load_lifecycle(tmp_path)


def test_missing_lifecycle_metadata_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tool-lifecycle"):
        load_lifecycle(tmp_path)
