"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_distribution.py

Distribution discovery and content-boundary contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
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
    "dnadesign/junction/docs/assets/assembly-process.svg",
    "dnadesign/junction/docs/assets/junction-detail.svg",
    "dnadesign/junction/examples/gene-scale/request.yaml",
    "dnadesign/junction/examples/gene-scale/jobs/annealed-fragments.yaml",
    "dnadesign/junction/examples/gene-scale/jobs/assembly-process.yaml",
    "dnadesign/junction/examples/gene-scale/jobs/junction-detail.yaml",
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
_FORBIDDEN_WHEEL_MEMBERS = {
    "dnadesign/junction/docs/assets/three-fragment-review.svg",
    "dnadesign/junction/examples/three-fragment-review/review.job.yaml",
}


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(parent for parent in current.parents if (parent / "pyproject.toml").exists())


def _copy_build_source(repo_root: Path, destination: Path) -> None:
    """Copy only Git-accepted files so ignored build residue cannot enter a wheel."""

    accepted = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in accepted:
        if not raw_path:
            continue
        relative = Path(raw_path.decode())
        source = repo_root / relative
        if not source.exists() and not source.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)


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
    source_root = tmp_path / "source"
    wheel_dir = tmp_path / "dist"
    _copy_build_source(repo_root, source_root)
    subprocess.run(
        ["uv", "build", "--out-dir", str(wheel_dir)],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    sdists = list(wheel_dir.glob("*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0]) as source_archive:
        source_members = [Path(member.name).parts[1:] for member in source_archive.getmembers()]
    assert not any(".." in parts for parts in source_members)
    assert not any(
        {"workspaces", "campaigns"}.intersection(parts) and {"outputs", "runs"}.intersection(parts)
        for parts in source_members
    )
    assert not any(
        "/".join(parts).removeprefix("src/").startswith(_FORBIDDEN_WHEEL_PREFIXES) for parts in source_members
    )
    assert not any("/reader_spop_" in "/".join(parts) for parts in source_members)

    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
        metadata = BytesParser().parsebytes(wheel.read(next(m for m in members if m.endswith(".dist-info/METADATA"))))

    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    assert {r.name for r in requirements if r.marker is None or r.marker.evaluate({"extra": ""})} == {"numpy"}
    dense_arrays = next(r for r in requirements if r.name == "dense-arrays")
    assert dense_arrays.url is not None, "The full wheel must declare its unavailable-on-PyPI dependency source"
    assert dense_arrays.url.startswith("git+https://github.com/e-south/dense-arrays@")
    assert len(dense_arrays.url.rsplit("@", 1)[1]) == 40

    assert _REQUIRED_WHEEL_MEMBERS <= members
    assert members.isdisjoint(_FORBIDDEN_WHEEL_MEMBERS)
    assert not any(member.startswith(_FORBIDDEN_WHEEL_PREFIXES) for member in members)
    assert not any("config.probe." in member for member in members)
    assert "dnadesign/usr/remotes.yaml" not in members
    assert not any("/reader_spop_" in member for member in members)

    environment = tmp_path / "scoring-env"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True, capture_output=True)
    python = environment / "bin/python"
    lock = tomllib.loads((repo_root / "uv.lock").read_text())
    numpy_version = next(p["version"] for p in lock["package"] if p["name"] == "numpy")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheels[0]), f"numpy=={numpy_version}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    smoke = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
from importlib.util import find_spec
import numpy as np
from importlib.metadata import version
from dnadesign import __version__
from dnadesign.opal import score_multistate_response_behavior
assert __version__ == version('dnadesign')
assert find_spec('pandas') is None
assert find_spec('sklearn') is None
assert find_spec('marimo') is None
result = score_multistate_response_behavior(
    np.zeros((1, 8)), state_ids=('00', '10', '01', '11'),
    target_mask=(0, 0, 1, 1), softmin_scale=0.3,
)
assert result.behavior_score.tolist() == [0.0]
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert smoke.returncode == 0, smoke.stderr
    cli = subprocess.run([str(environment / "bin/opal"), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert cli.returncode != 0
    assert "dnadesign[full]" in cli.stderr
    assert "Traceback" not in cli.stderr
    _check_full_install(repo_root, tmp_path, environment, wheels[0])


def _check_full_install(repo_root: Path, tmp_path: Path, environment: Path, wheel: Path) -> None:
    requirements = tmp_path / "full-requirements.txt"
    subprocess.run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes", "-o", str(requirements)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    python = environment / "bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), f"{wheel}[full]", "-r", str(requirements)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True, capture_output=True, text=True)
    smoke = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import sys
from pathlib import Path
from hashlib import sha256
import dnadesign
from dnadesign.construct import place_annotated_part
from dnadesign.contracts.sequence import AnnotatedSequencePartV1
assert Path(dnadesign.__file__).is_relative_to(sys.prefix)
sequence = 'AACCGG'
digest = 'sha256:' + sha256(sequence.encode()).hexdigest()
part = AnnotatedSequencePartV1.model_validate({
    'part_id': 'example', 'strandedness': 'double', 'topology': 'linear',
    'sequence': sequence, 'sequence_digest': digest,
    'source_refs': [{'kind': 'artifact', 'authority': 'example', 'identifier': 'example', 'digest': digest}],
    'features': [{
        'feature_id': 'example-feature', 'role': 'sequence_region', 'owner': 'example',
        'start': 0, 'end': 2, 'orientation': 'forward', 'sequence': 'AA', 'source_digest': digest,
    }],
})
result = place_annotated_part(
    template_id='example', template_sequence='TTTTTT', part=part,
    placement_kind='replace', start=2, end=4, orientation='reverse_complement',
)
assert result.sequence == 'TTCCGGTTTT'
assert (result.part_start, result.part_end) == (2, 8)
assert (result.features[0].realized_start, result.features[0].realized_end) == (6, 8)
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert smoke.returncode == 0, smoke.stderr
    for command in ("opal", "construct", "latentdna", "dense"):
        help_result = subprocess.run(
            [str(environment / "bin" / command), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert help_result.returncode == 0, help_result.stderr


def test_scoring_install_is_small_and_full_tools_remain_explicit() -> None:
    project = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
    base = {Requirement(value).name for value in project["project"]["dependencies"]}
    assert base == {"numpy"}
    full = {Requirement(value).name for value in project["project"]["optional-dependencies"]["full"]}
    assert {"pandas", "pyarrow", "pydantic", "typer", "scikit-learn", "dense-arrays", "torch", "marimo"} <= full
    assert project["dependency-groups"]["tools"] == ["dnadesign[full]"]
    assert project["tool"]["uv"]["default-groups"] == ["tools"]


def test_tool_software_versions_follow_the_installed_distribution() -> None:
    expected = version("dnadesign")
    for module in (
        "dnadesign",
        "dnadesign.opal.src",
        "dnadesign.usr.src.version",
        "dnadesign.latentdna.src.version",
        "dnadesign.permuter",
    ):
        assert import_module(module).__version__ == expected
