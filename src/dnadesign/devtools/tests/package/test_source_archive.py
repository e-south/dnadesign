"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_source_archive.py

Source distribution reproducibility and machine-metadata boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import runpy
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("epoch", [None, "1700000000"])
def test_source_archive_has_reproducible_metadata_and_preserves_contents(tmp_path: Path, epoch: str | None) -> None:
    repo = Path(__file__).resolve().parents[5]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(repo / "setup.py", source / "setup.py")
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=84.1.0", "wheel>=0.48.0"]\n'
        'build-backend="setuptools.build_meta"\n'
        '[project]\nname="archive-fixture"\nversion="0.1.0"\ndynamic=["readme"]\n'
        '[tool.setuptools]\npy-modules=["example"]\n'
    )
    (source / "README.md").write_text("Archive fixture.\n")
    module = source / "example.py"
    module.write_text('VALUE = "retained content"\n')
    module.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "SOURCE_DATE_EPOCH"}
    if epoch is not None:
        env["SOURCE_DATE_EPOCH"] = epoch
    archives = []
    for index in (1, 2):
        os.utime(module, (1600000000 + index, 1600000000 + index))
        output = tmp_path / str(index)
        subprocess.run(
            ["uv", "build", "--python", sys.executable, "--sdist", "--out-dir", str(output)],
            cwd=source,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        archives.append(next(output.glob("*.tar.gz")))

    with tarfile.open(archives[0]) as archive:
        members = archive.getmembers()
        assert all(m.uid == m.gid == 0 and m.uname == m.gname == "" for m in members)
        assert all(m.mtime == int(epoch or "0") for m in members)
        member = archive.getmember("archive_fixture-0.1.0/example.py")
        assert archive.extractfile(member).read() == b'VALUE = "retained content"\n'
        assert member.mode & 0o111
    assert struct.unpack("<I", archives[0].read_bytes()[4:8])[0] == int(epoch or "0")
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_wheel_rebuild_drops_stale_python_and_data_files(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[5]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(repo / "setup.py", source / "setup.py")
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=84.1.0", "wheel>=0.48.0"]\n'
        'build-backend="setuptools.build_meta"\n'
        '[project]\nname="incremental-fixture"\nversion="0.1.0"\ndynamic=["readme"]\n'
        '[tool.setuptools]\npy-modules=["example"]\n'
    )
    (source / "README.md").write_text("Incremental fixture.\n")
    (source / "example.py").write_text('VALUE = "current"\n')
    staging = source / "build/lib"
    staging.mkdir(parents=True)
    (staging / "removed.py").write_text('VALUE = "retired"\n')
    (staging / "private-record.txt").write_text("stale fixture data\n")
    output = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--python", sys.executable, "--wheel", "--out-dir", str(output)],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with zipfile.ZipFile(next(output.glob("*.whl"))) as wheel:
        assert "removed.py" not in wheel.namelist()
        assert "private-record.txt" not in wheel.namelist()
        assert wheel.read("example.py") == b'VALUE = "current"\n'


@pytest.mark.parametrize("destination", ["source", "build_base", "symlink"])
def test_build_staging_cleanup_rejects_unsafe_destinations(tmp_path: Path, monkeypatch, destination: str) -> None:
    import setuptools
    from setuptools.dist import Distribution

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    repo = Path(__file__).resolve().parents[5]
    build_class = runpy.run_path(str(repo / "setup.py"))["PortableBuild"]
    command = build_class(Distribution())
    command.ensure_finalized()
    build_base = tmp_path / "build"
    build_base.mkdir()
    command.get_finalized_command("build").build_base = str(build_base)
    protected = tmp_path / "protected"
    protected.mkdir()
    marker = protected / "keep.txt"
    marker.write_text("preserve")
    if destination == "source":
        command.build_lib = str(repo / "src")
    elif destination == "build_base":
        command.build_lib = str(build_base)
    else:
        link = build_base / "lib"
        link.symlink_to(protected, target_is_directory=True)
        command.build_lib = str(link)
    with pytest.raises(ValueError, match="staging subdirectory"):
        command.run()
    assert marker.read_text() == "preserve"
