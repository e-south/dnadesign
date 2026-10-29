"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_preflight.py

Preflight tests for a pinned official LigandMPNN checkout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
import hashlib
import subprocess
import sys
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
        if "ls-files" in command:
            return b"data_utils.py\n"
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


def test_preflight_rejects_dirty_parser_after_context_evidence(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    (root / "data_utils.py").write_text("def parse_PDB(): return 'modified'\n", encoding="utf-8")

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.dirty_parser_module", str(root / "data_utils.py"))
    ]


def test_preflight_rejects_parser_removed_from_git_index(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    subprocess.run(
        ["git", "-C", str(root), "rm", "--cached", "--quiet", "data_utils.py"],
        check=True,
    )
    assert (root / "data_utils.py").is_file()

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.untracked_parser_module", str(root / "data_utils.py"))
    ]


def test_preflight_rejects_staged_parser_when_worktree_bytes_match_pin(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    parser = root / "data_utils.py"
    pinned_payload = parser.read_bytes()
    parser.write_text("def parse_PDB(): return 'staged-modification'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "data_utils.py"], check=True)
    parser.write_bytes(pinned_payload)

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.dirty_parser_index", str(parser))
    ]


def test_preflight_rejects_symlinked_parser_even_when_target_matches_pinned_bytes(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    parser = root / "data_utils.py"
    target = tmp_path / "matching-data-utils.py"
    parser.replace(target)
    parser.symlink_to(target)

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.parser_module_not_regular", str(parser))
    ]


@pytest.mark.parametrize(
    "lstat_error",
    [
        PermissionError(errno.EACCES, "parser leaf is inaccessible"),
        PermissionError(errno.EACCES, "parser ancestor is inaccessible"),
        OSError(errno.EIO, "parser status I/O failure"),
    ],
    ids=["inaccessible-leaf", "inaccessible-ancestor", "other-oserror"],
)
def test_preflight_normalizes_unreadable_parser_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lstat_error: OSError,
) -> None:
    root, pin = _pinned_checkout(tmp_path)
    parser = root / "data_utils.py"
    original_lstat = Path.lstat

    def _raise_for_parser(path: Path):
        if path == parser:
            raise lstat_error
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", _raise_for_parser)

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.message, issue.path) for issue in report.issues] == [
        (
            "thread.ligandmpnn.unreadable_parser_module",
            "pinned LigandMPNN data_utils.py status could not be read",
            str(parser),
        )
    ]


def test_preflight_rejects_symlinked_checkpoint_before_execution(tmp_path: Path) -> None:
    root, pin = _pinned_checkout(tmp_path)
    checkpoint = root / pin.checkpoint_path
    target = tmp_path / "checkpoint.pt"
    checkpoint.replace(target)
    checkpoint.symlink_to(target)

    report = preflight_ligandmpnn(root, pin)

    assert not report.ok
    assert [(issue.check_id, issue.path) for issue in report.issues] == [
        ("thread.ligandmpnn.checkpoint_not_regular", str(checkpoint))
    ]


def test_preflight_rejects_checkpoint_replaced_by_fifo_without_blocking(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    expected = _sha256(checkpoint.read_bytes())

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from contextlib import contextmanager\n"
                "from pathlib import Path\n"
                "import dnadesign.thread.adapters.ligandmpnn.preflight as module\n"
                "target = Path(sys.argv[1])\n"
                "original = module.open_regular_file\n"
                "@contextmanager\n"
                "def replace_before_open(path):\n"
                "    if path == target:\n"
                "        path.unlink()\n"
                "        os.mkfifo(path)\n"
                "    with original(path) as handle:\n"
                "        yield handle\n"
                "module.open_regular_file = replace_before_open\n"
                "issues = []\n"
                "module._check_digest(target, sys.argv[2], 'checkpoint', issues)\n"
                "assert [issue.check_id for issue in issues] == "
                "['thread.ligandmpnn.checkpoint_not_regular']\n"
            ),
            str(checkpoint),
            expected,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert probe.stderr == ""


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
