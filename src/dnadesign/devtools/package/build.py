"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/package/build.py

Build distributions from a clean committed source with a retained build lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from .provenance import file_record, sha256, verify_bundle


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def export_source(repo_root: Path, destination: Path) -> dict:
    """Export HEAD, rejecting changes that a commit-based receipt cannot describe."""
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Build requires clean tracked and untracked source; commit the intended changes first")
    revision = _git(repo_root, "rev-parse", "HEAD")
    record = {
        "revision": revision,
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "source_date_epoch": int(_git(repo_root, "show", "-s", "--format=%ct", revision)),
    }
    content = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", revision], check=True, capture_output=True
    ).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        archive.extractall(destination, filter="data")
    return record


def _run(command: list[str], cwd: Path, log: Path, env: dict) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError(f"Build step failed; see {log}")
    return result.stdout.strip()


def _sdist_contents(path: Path) -> dict:
    with tarfile.open(path) as archive:
        return {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}


def build_bundle(repo_root: Path, output: Path, *, build_lock: Path | None = None) -> dict:
    """Build twice from HEAD and retain enough evidence to repeat or check the build."""
    repo_root, output = repo_root.resolve(), output.resolve()
    if output.is_relative_to(repo_root):
        raise ValueError("Build output must be outside the source checkout")
    if output.exists():
        raise ValueError("Build output must be a new directory")
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(UV_NO_CONFIG="1", PYTHONNOUSERSITE="1")
    with tempfile.TemporaryDirectory(prefix="dnadesign-build-") as workspace:
        work = Path(workspace)
        source = work / "source"
        identity = export_source(repo_root, source)
        project = tomllib.loads((source / "pyproject.toml").read_text())
        name, version = project["project"]["name"], project["project"]["version"]
        citation = (source / "CITATION.cff").read_text()
        output.mkdir(parents=True)
        env["SOURCE_DATE_EPOCH"] = str(identity["source_date_epoch"])
        (output / "build-requirements.in").write_text("\n".join(project["build-system"]["requires"]) + "\n")
        lock = output / "build-requirements.lock"
        if build_lock is not None:
            shutil.copyfile(build_lock, lock)
        else:
            _run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "build-requirements.in",
                    "--python",
                    sys.executable,
                    "--generate-hashes",
                    "--no-header",
                    "--no-annotate",
                    "-o",
                    lock.name,
                ],
                output,
                output / "resolve.log",
                env,
            )
        _run(["uv", "venv", "--python", sys.executable, str(work / "env")], work, output / "environment.log", env)
        python = work / "env" / "bin" / "python"
        _run(
            ["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r", str(lock)],
            work,
            output / "install.log",
            env,
        )
        toolchain = json.loads(
            _run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import json; from importlib.metadata import distributions; "
                    "print(json.dumps({d.metadata['Name']: d.version for d in distributions()}))",
                ],
                work,
                output / "toolchain.log",
                env,
            )
        )
        toolchain.update(
            python=platform.python_version(), uv=subprocess.check_output(["uv", "--version"], text=True).strip()
        )
        for attempt in ("first", "repeat"):
            _run(
                ["uv", "build", "--no-build-isolation", "--python", str(python), "--out-dir", str(work / attempt)],
                source,
                output / f"{attempt}.log",
                env,
            )
        first, repeat = work / "first", work / "repeat"
        wheels, sdists = list(first.glob("*.whl")), list(first.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError("Build must produce exactly one wheel and one source distribution")
        wheel, sdist = wheels[0], sdists[0]
        with ZipFile(wheel) as archive:
            metadata = [item for item in archive.namelist() if item.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Wheel must contain one metadata record")
            parsed = BytesParser().parsebytes(archive.read(metadata[0]))
            if parsed["Name"] != name or parsed["Version"] != version:
                raise ValueError("Wheel metadata differs from the source project")
        comparison = {
            "wheel_bytes_identical": sha256(wheel) == sha256(repeat / wheel.name),
            "sdist_contents_identical": _sdist_contents(sdist) == _sdist_contents(repeat / sdist.name),
            "sdist_bytes_identical": sha256(sdist) == sha256(repeat / sdist.name),
        }
        if not comparison["wheel_bytes_identical"] or not comparison["sdist_contents_identical"]:
            raise ValueError("Repeated build differs in wheel bytes or source-distribution contents")
        for artifact in (wheel, sdist):
            shutil.copyfile(artifact, output / artifact.name)
        # The checkout citation omits a version: pyproject.toml owns that value.
        (output / "CITATION.cff").write_text(citation.rstrip() + f'\nversion: "{version}"\n')
        payloads = [
            output / item
            for item in (wheel.name, sdist.name, "build-requirements.in", "build-requirements.lock", "CITATION.cff")
        ]
        record = {
            "schema": "dnadesign-build/v1",
            "name": name,
            "version": version,
            "source": identity,
            "pyproject_sha256": sha256(source / "pyproject.toml"),
            "runtime_lock_sha256": sha256(source / "uv.lock"),
            "toolchain": toolchain,
            "repeat": comparison,
            "files": [file_record(path, output) for path in payloads],
            "boundary": "Local build integrity, not an authenticated public release or downstream numerical acceptance",
        }
        (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return verify_bundle(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify a clean committed distribution")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-lock", type=Path, help="Reuse a prior hash-locked build environment")
    args = parser.parse_args(argv)
    try:
        record = build_bundle(args.repo_root, args.output, build_lock=args.build_lock)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Built {record['name']} {record['version']}; provenance: {args.output / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
