"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_build.py

Clean-source build provenance and portable artifact verification contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import shutil
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools==84.0.0", "wheel==0.48.0"]\n'
        'build-backend="setuptools.build_meta"\n'
        '[project]\nname="build-fixture"\nversion="0.2.0"\n'
        '[tool.setuptools]\npy-modules=["example"]\n'
    )
    (root / "example.py").write_text('VALUE = "accepted source"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / "CITATION.cff").write_text('cff-version: 1.2.0\ntitle: "Build fixture"\n')
    (root / ".gitignore").write_text("ignored.py\n")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "fixture")
    return root


def test_snapshot_exports_only_a_clean_commit(source: Path, tmp_path: Path) -> None:
    from dnadesign.devtools.package.build import export_source

    (source / "ignored.py").write_text("not build input\n")
    target = tmp_path / "export"
    record = export_source(source, target)
    assert record["revision"] == _git(source, "rev-parse", "HEAD")
    assert record["tree"] == _git(source, "rev-parse", "HEAD^{tree}")
    assert (target / "example.py").read_text() == 'VALUE = "accepted source"\n'
    assert not (target / "ignored.py").exists()
    assert not (target / ".git").exists()


@pytest.mark.parametrize("change", ["tracked", "untracked", "staged"])
def test_snapshot_rejects_uncommitted_source(source: Path, tmp_path: Path, change: str) -> None:
    from dnadesign.devtools.package.build import export_source

    (source / ("new.py" if change == "untracked" else "example.py")).write_text("changed\n")
    if change == "staged":
        _git(source, "add", ".")
    with pytest.raises(ValueError, match="clean"):
        export_source(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_build_records_reusable_lock_source_and_distribution_bytes(source: Path, tmp_path: Path) -> None:
    from dnadesign.devtools.package.build import build_bundle

    output = tmp_path / "bundle"
    receipt = build_bundle(source, output)
    assert receipt["schema"] == "dnadesign-build/v1"
    assert receipt["name"] == "build-fixture"
    assert receipt["version"] == "0.2.0"
    assert receipt["source"]["revision"] == _git(source, "rev-parse", "HEAD")
    assert receipt["source"]["tree"] == _git(source, "rev-parse", "HEAD^{tree}")
    assert receipt["repeat"]["wheel_bytes_identical"] is True
    assert receipt["repeat"]["sdist_contents_identical"] is True
    assert receipt["toolchain"]["setuptools"] == "84.0.0"
    assert receipt["toolchain"]["wheel"] == "0.48.0"
    assert "--hash=sha256:" in (output / "build-requirements.lock").read_text()
    assert 'version: "0.2.0"' in (output / "CITATION.cff").read_text()
    for record in receipt["files"]:
        path = output / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
    wheel = next(output.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("example.py") == b'VALUE = "accepted source"\n'
    assert json.loads((output / "provenance.json").read_text()) == receipt
    assert not (output / "source").exists()
    assert not (output / "build-env").exists()
    assert _git(source, "status", "--porcelain") == ""
    # Receipts remain usable after moving a downloaded bundle.
    moved = tmp_path / "download"
    shutil.copytree(output, moved)
    from dnadesign.devtools.package.provenance import verify_bundle

    assert verify_bundle(moved)["source"] == receipt["source"]
    repeated = build_bundle(source, tmp_path / "repeat", build_lock=output / "build-requirements.lock")
    assert repeated["toolchain"] == receipt["toolchain"]
    assert repeated["files"][0] == receipt["files"][0]


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "download"
    root.mkdir()
    paths = ["example.whl", "example.tar.gz", "build-requirements.lock", "build-requirements.in", "CITATION.cff"]
    files = []
    for name in paths:
        (root / name).write_bytes(b"retained bytes")
        files.append({"path": name, "bytes": 14, "sha256": sha256(b"retained bytes").hexdigest()})
    (root / "provenance.json").write_text(json.dumps({"schema": "dnadesign-build/v1", "files": files}))
    return root


@pytest.mark.parametrize("malformation", ["empty", "duplicate", "escape", "absolute", "symlink", "changed"])
def test_download_verification_rejects_unsafe_or_incomplete_bundles(bundle: Path, malformation: str) -> None:
    from dnadesign.devtools.package.provenance import verify_bundle

    record = json.loads((bundle / "provenance.json").read_text())
    if malformation == "empty":
        record["files"] = []
    elif malformation == "duplicate":
        record["files"].append(record["files"][0])
    elif malformation in {"escape", "absolute", "symlink"}:
        outside = bundle.parent / "outside.whl"
        outside.write_bytes(b"retained bytes")
        if malformation == "symlink":
            (bundle / "example.whl").unlink()
            (bundle / "example.whl").symlink_to(outside)
        else:
            record["files"][0]["path"] = "../outside.whl" if malformation == "escape" else str(outside)
    else:
        (bundle / "example.whl").write_bytes(b"modified bytes")
    (bundle / "provenance.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_download_can_require_an_independently_recorded_receipt_digest(bundle: Path) -> None:
    from dnadesign.devtools.package.provenance import verify_bundle

    accepted = sha256((bundle / "provenance.json").read_bytes()).hexdigest()
    verify_bundle(bundle, expected_sha256=accepted)
    with pytest.raises(ValueError, match="provenance"):
        verify_bundle(bundle, expected_sha256="0" * 64)


def test_build_rejects_output_inside_source_or_existing_delivery(source: Path, tmp_path: Path) -> None:
    from dnadesign.devtools.package.build import build_bundle

    with pytest.raises(ValueError, match="outside"):
        build_bundle(source, source / "dist")
    with pytest.raises(ValueError, match="new directory"):
        build_bundle(source, tmp_path)
