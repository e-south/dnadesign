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

from dnadesign.thread.adapters.ligandmpnn import LigandMpnnUpstreamPin, preflight_ligandmpnn


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_preflight_accepts_exact_checkout_commit_and_checkpoint_hashes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "LigandMPNN"
    (root / "model_params").mkdir(parents=True)
    (root / "run.py").write_text("# official entrypoint\n", encoding="utf-8")
    (root / "score.py").write_text("# official scoring entrypoint\n", encoding="utf-8")
    model = b"ligand checkpoint"
    packing = b"packing checkpoint"
    (root / "model_params/ligandmpnn_v_32_010_25.pt").write_bytes(model)
    (root / "model_params/ligandmpnn_sc_v_32_002_16.pt").write_bytes(packing)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"1" * 40 + b"\n")
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
        "thread.ligandmpnn.upstream_commit_mismatch",
        "thread.ligandmpnn.checkpoint_hash_mismatch",
    }
