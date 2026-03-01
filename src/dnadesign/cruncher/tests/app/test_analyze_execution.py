"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_analyze_execution.py

Covers analyze run-execution context resolution and fail-fast lockfile handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dnadesign.cruncher.app.analyze.execution import resolve_analysis_run_execution_context
from dnadesign.cruncher.utils.hashing import sha256_path


def _fake_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        sequences_df=pd.DataFrame({"sequence": ["AAAA"]}),
        elites_df=pd.DataFrame({"id": ["E1"]}),
        hits_df=pd.DataFrame({"elite_id": ["E1"]}),
        baseline_df=pd.DataFrame({"sequence": ["CCCC"]}),
        baseline_hits_df=pd.DataFrame({"sequence": ["CCCC"]}),
        trace_idata=None,
        elites_meta={"status": "ok"},
    )


def test_resolve_analysis_run_execution_context_rejects_missing_lockfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    missing_lock = tmp_path / "missing.lock"
    manifest = {"lockfile_path": str(missing_lock), "lockfile_sha256": "deadbeef"}

    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_run_dir", lambda *_a, **_k: run_dir)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution.load_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_optimizer_stats", lambda *_a, **_k: {})

    with pytest.raises(FileNotFoundError, match="Lockfile referenced by run manifest missing"):
        resolve_analysis_run_execution_context(
            cfg=SimpleNamespace(sample=None),
            config_path=tmp_path / "config.yaml",
            run_name="r1",
        )


def test_resolve_analysis_run_execution_context_rejects_lockfile_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("content")
    manifest = {"lockfile_path": str(lockfile), "lockfile_sha256": "deadbeef"}

    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_run_dir", lambda *_a, **_k: run_dir)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution.load_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_optimizer_stats", lambda *_a, **_k: {})

    with pytest.raises(ValueError, match="Lockfile checksum mismatch"):
        resolve_analysis_run_execution_context(
            cfg=SimpleNamespace(sample=None),
            config_path=tmp_path / "config.yaml",
            run_name="r1",
        )


def test_resolve_analysis_run_execution_context_returns_typed_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("content")
    manifest = {"lockfile_path": str(lockfile), "lockfile_sha256": sha256_path(lockfile)}

    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_run_dir", lambda *_a, **_k: run_dir)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution.load_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_optimizer_stats", lambda *_a, **_k: "n/a")
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze.execution.load_pwms_from_config",
        lambda *_a, **_k: ({"lexA": object()}, {"sample": {}}),
    )
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._resolve_tf_names", lambda *_a, **_k: ["lexA"])
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze.execution._resolve_sample_meta",
        lambda *_a, **_k: SimpleNamespace(mode="opt", top_k=1),
    )
    monkeypatch.setattr("dnadesign.cruncher.app.analyze.execution._analysis_id", lambda: "aid")
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze.execution._load_run_artifacts_for_analysis",
        lambda *_a, **_k: _fake_artifacts(),
    )

    context = resolve_analysis_run_execution_context(
        cfg=SimpleNamespace(sample=None),
        config_path=tmp_path / "config.yaml",
        run_name="r1",
    )

    assert context.run_name == "r1"
    assert context.run_dir == run_dir
    assert context.optimizer_stats is None
    assert context.tf_names == ["lexA"]
    assert context.analysis_used_file.parent.parent == context.tmp_root
    assert context.tmp_root.exists()
