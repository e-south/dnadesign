"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_preflight.py

Preflight tests for a pinned official LigandMPNN checkout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import LigandMpnnUpstreamPin, preflight_ligandmpnn


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_preflight_accepts_exact_checkout_commit_and_checkpoint_hashes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "LigandMPNN"
    (root / "model_params").mkdir(parents=True)
    (root / "run.py").write_text("# official entrypoint\n", encoding="utf-8")
    (root / "score.py").write_text("# official scoring entrypoint\n", encoding="utf-8")
    (root / "data_utils.py").write_text("def parse_PDB(): ...\n", encoding="utf-8")
    model = b"ligand checkpoint"
    packing = b"packing checkpoint"
    (root / "model_params/ligandmpnn_v_32_010_25.pt").write_bytes(model)
    (root / "model_params/ligandmpnn_sc_v_32_002_16.pt").write_bytes(packing)

    def _git_output(command, **kwargs):
        if "rev-parse" in command:
            return b"1" * 40 + b"\n"
        if "show" in command:
            return (root / command[-1].split(":", 1)[1]).read_bytes()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(subprocess, "check_output", _git_output)
    pin = LigandMpnnUpstreamPin(
        commit="1" * 40,
        checkpoint_sha256=_sha256(model),
        packing_checkpoint_sha256=_sha256(packing),
    )

    report = preflight_ligandmpnn(root, pin, require_packing_checkpoint=True)

    assert report.ok
    assert report.issues == ()
    assert report.provenance.upstream_commit == "1" * 40


def test_preflight_reports_each_pin_violation_without_running_upstream(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "LigandMPNN"
    (root / "model_params").mkdir(parents=True)
    (root / "model_params/ligandmpnn_v_32_010_25.pt").write_bytes(b"wrong")

    def _wrong_commit(*args, **kwargs) -> bytes:
        return b"2" * 40 + b"\n"

    monkeypatch.setattr(subprocess, "check_output", _wrong_commit)
    pin = LigandMpnnUpstreamPin(commit="1" * 40, checkpoint_sha256="a" * 64)

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert {issue.check_id for issue in report.issues} == {
        "thread.ligandmpnn.missing_entrypoint",
        "thread.ligandmpnn.missing_score_entrypoint",
        "thread.ligandmpnn.missing_parser_module",
        "thread.ligandmpnn.upstream_commit_mismatch",
        "thread.ligandmpnn.checkpoint_hash_mismatch",
    }


@pytest.mark.parametrize("entrypoint_name", ["run.py", "score.py"])
def test_preflight_rejects_dirty_pinned_entrypoints(tmp_path: Path, entrypoint_name: str) -> None:
    root, pin = _pinned_checkout(tmp_path)
    (root / entrypoint_name).write_text("# modified entrypoint\n", encoding="utf-8")

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.dirty_entrypoint", str(root / entrypoint_name))
    ]


@pytest.mark.parametrize("entrypoint_name", ["run.py", "score.py"])
def test_preflight_rejects_assume_unchanged_entrypoints(tmp_path: Path, entrypoint_name: str) -> None:
    root, pin = _pinned_checkout(tmp_path)
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--assume-unchanged", entrypoint_name],
        check=True,
    )
    (root / entrypoint_name).write_text("# hidden modified entrypoint\n", encoding="utf-8")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--", entrypoint_name],
        text=True,
    )
    assert status == ""

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.dirty_entrypoint", str(root / entrypoint_name))
    ]


def test_preflight_reads_pinned_blobs_without_replacement_refs(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    (root / "run.py").write_text("# replacement entrypoint\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "run.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "replacement tree",
        ],
        check=True,
    )
    replacement_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    subprocess.run(["git", "-C", str(root), "replace", pin.commit, replacement_commit], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", pin.commit], check=True)
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        == pin.commit
    )
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "show", f"{pin.commit}:run.py"],
        )
        == (root / "run.py").read_bytes()
    )

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.dirty_entrypoint", str(root / "run.py"))
    ]


def _pinned_checkout(tmp_path: Path) -> tuple[Path, LigandMpnnUpstreamPin]:
    root = tmp_path / "LigandMPNN"
    (root / "model_params").mkdir(parents=True)
    (root / "run.py").write_text("# official entrypoint\n", encoding="utf-8")
    (root / "score.py").write_text("# official scoring entrypoint\n", encoding="utf-8")
    (root / "data_utils.py").write_text("def parse_PDB(): ...\n", encoding="utf-8")
    model = b"ligand checkpoint"
    (root / "model_params/ligandmpnn_v_32_010_25.pt").write_bytes(model)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    pin = LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_sha256(model))
    return root, pin
