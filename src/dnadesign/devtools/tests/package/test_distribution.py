"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_distribution.py

Distribution discovery and content-boundary contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import tomllib
import zipfile
from pathlib import Path

from setuptools import Distribution, find_namespace_packages
from setuptools.config.pyprojecttoml import apply_configuration

_DISTRIBUTION_EXCLUDES = [
    "dnadesign.archived",
    "dnadesign.archived.*",
    "dnadesign.prototypes",
    "dnadesign.prototypes.*",
    "dnadesign.studies",
    "dnadesign.studies.*",
    "dnadesign.usr.archived",
    "dnadesign.usr.archived.*",
    "dnadesign.usr.datasets.usr_regulondb_native_promoters",
    "dnadesign.usr.datasets.usr_regulondb_native_promoters.*",
    "dnadesign.opal.campaigns.*.notebooks",
    "dnadesign.opal.campaigns.*.notebooks.*",
    "dnadesign.densegen.workspaces.*.outputs.notebooks",
    "dnadesign.densegen.workspaces.*.outputs.notebooks.*",
    "dnadesign.latentdna.workspaces.*.outputs.notebooks",
    "dnadesign.latentdna.workspaces.*.outputs.notebooks.*",
]
_DISTRIBUTION_DATA_EXCLUDES = [
    "reader_spop_*.parquet",
    "remotes.yaml",
]
_DENSEGEN_DATA_EXCLUDES = [
    "workspaces/*/config.probe.*.yaml",
]
_EXCLUDED_PACKAGE_SENTINELS = {
    "dnadesign.archived.legacy",
    "dnadesign.prototypes.sketch",
    "dnadesign.usr.archived.legacy",
    "dnadesign.usr.datasets.usr_regulondb_native_promoters.local",
    "dnadesign.opal.campaigns.demo.notebooks",
    "dnadesign.densegen.workspaces.demo.outputs.notebooks",
    "dnadesign.latentdna.workspaces.demo.outputs.notebooks",
}
_RETAINED_PACKAGE_SENTINELS = {
    "dnadesign.opal.campaigns.demo.notebooks_api",
    "dnadesign.densegen.workspaces.demo.outputs.notebooks_api",
    "dnadesign.latentdna.workspaces.demo.outputs.notebooks_api",
}
_REQUIRED_WHEEL_MEMBERS = {
    "dnadesign/baserender/styles/style_v1/presentation_default.yaml",
    "dnadesign/junction/docs/assets/annealed-fragments.svg",
    "dnadesign/junction/docs/assets/assembly-overview.svg",
    "dnadesign/junction/docs/assets/junction-detail.svg",
    "dnadesign/junction/examples/gene-scale/request.yaml",
    "dnadesign/junction/examples/three-fragment-review/request.yaml",
    "dnadesign/junction/examples/three-fragment-review/jobs/annealed-fragments.yaml",
    "dnadesign/junction/examples/three-fragment-review/jobs/assembly-overview.yaml",
    "dnadesign/junction/examples/three-fragment-review/jobs/junction-detail.yaml",
    "dnadesign/opal/campaigns/demo_gp_topn/configs/campaign.yaml",
    "dnadesign/opal/campaigns/_fixtures/scalar-regression/records.parquet",
    "dnadesign/usr/datasets/registry.yaml",
}
_FORBIDDEN_WHEEL_PREFIXES = (
    "dnadesign/archived/",
    "dnadesign/prototypes/",
    "dnadesign/usr/archived/",
    "dnadesign/usr/datasets/usr_regulondb_native_promoters/",
    "dnadesign/studies/",
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(parent for parent in current.parents if (parent / "pyproject.toml").exists())


def test_distribution_excludes_internal_source_shelves(tmp_path: Path) -> None:
    repo_root = _repo_root()
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = project["tool"]["setuptools"]["packages"]["find"]
    excluded = discovery["exclude"]

    assert project["tool"]["setuptools"]["include-package-data"] is True
    assert excluded == _DISTRIBUTION_EXCLUDES
    assert project["tool"]["setuptools"]["exclude-package-data"]["*"] == _DISTRIBUTION_DATA_EXCLUDES
    assert project["tool"]["setuptools"]["exclude-package-data"]["dnadesign.densegen"] == _DENSEGEN_DATA_EXCLUDES

    source_root = tmp_path / "src"
    (source_root / "dnadesign" / "active").mkdir(parents=True)
    for package in _EXCLUDED_PACKAGE_SENTINELS | _RETAINED_PACKAGE_SENTINELS:
        (source_root / package.replace(".", "/")).mkdir(parents=True)

    discovered = set(
        find_namespace_packages(
            where=str(source_root),
            exclude=excluded,
        )
    )

    assert {"dnadesign", "dnadesign.active"} <= discovered
    assert discovered.isdisjoint(_EXCLUDED_PACKAGE_SENTINELS)
    assert _RETAINED_PACKAGE_SENTINELS <= discovered


def test_distribution_contains_no_ignored_python_sources() -> None:
    repo_root = _repo_root()
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = project["tool"]["setuptools"]["packages"]["find"]
    source_root = repo_root / discovery["where"][0]
    packages = find_namespace_packages(
        where=str(source_root),
        exclude=discovery["exclude"],
    )
    packaged_sources = {
        path.resolve() for package in packages for path in (source_root / package.replace(".", "/")).glob("*.py")
    }

    accepted = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src/dnadesign",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    accepted_sources = {
        (repo_root / relative.decode()).resolve() for relative in accepted if relative and relative.endswith(b".py")
    }

    assert packaged_sources <= accepted_sources


def test_distribution_excludes_private_and_generated_package_data() -> None:
    repo_root = _repo_root()
    distribution = Distribution()
    distribution.script_name = "pyproject.toml"
    apply_configuration(distribution, repo_root / "pyproject.toml")
    command = distribution.get_command_obj("build_py")
    command.ensure_finalized()

    excluded_cases = (
        (
            "dnadesign.densegen",
            repo_root / "src/dnadesign/densegen",
            "workspaces/demo/config.probe.stall10.yaml",
        ),
        (
            "dnadesign.usr",
            repo_root / "src/dnadesign/usr",
            "remotes.yaml",
        ),
        (
            "dnadesign.latentdna.workspaces.demo.study_inputs",
            repo_root / "src/dnadesign/latentdna/workspaces/demo/study_inputs",
            "reader_spop_observations.parquet",
        ),
    )
    for package, source_dir, relative_path in excluded_cases:
        candidate = str(source_dir / relative_path)
        assert command.exclude_data_files(package, str(source_dir), [candidate]) == []

    retained_cases = (
        (
            "dnadesign.usr",
            repo_root / "src/dnadesign/usr",
            "ops/status.registry.yaml",
        ),
        (
            "dnadesign.opal.campaigns.demo",
            repo_root / "src/dnadesign/opal/campaigns/demo",
            "records.parquet",
        ),
    )
    for package, source_dir, relative_path in retained_cases:
        candidate = str(source_dir / relative_path)
        assert command.exclude_data_files(package, str(source_dir), [candidate]) == [candidate]


def test_built_wheel_retains_runtime_resources_without_internal_shelves(tmp_path: Path) -> None:
    repo_root = _repo_root()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())

    assert _REQUIRED_WHEEL_MEMBERS <= members
    assert not any(member.startswith(_FORBIDDEN_WHEEL_PREFIXES) for member in members)
    assert not any("config.probe." in member for member in members)
    assert "dnadesign/usr/remotes.yaml" not in members
    assert not any("/reader_spop_" in member for member in members)
