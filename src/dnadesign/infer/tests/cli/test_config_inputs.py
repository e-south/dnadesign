"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_config_inputs.py

Config-driven ingest input resolution contracts for infer run workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.infer.src.cli.config_inputs import resolve_config_job_inputs
from dnadesign.infer.src.config import JobConfig
from dnadesign.infer.src.errors import ConfigError, ValidationError


def _job(*, source: str, job_id: str = "j1", path: str | None = None) -> JobConfig:
    payload: dict[str, object] = {
        "id": job_id,
        "operation": "extract",
        "ingest": {"source": source, "field": "sequence"},
        "outputs": [{"id": "o1", "fn": "evo2.logits", "format": "list"}],
    }
    if path is not None:
        payload["ingest"] = {"source": source, "field": "sequence", "path": path}
    return JobConfig(**payload)


def test_resolve_config_job_inputs_usr_requires_explicit_root_or_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DNADESIGN_USR_ROOT", raising=False)
    job = JobConfig(
        id="j1",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "field": "sequence"},
        outputs=[{"id": "o1", "fn": "evo2.logits", "format": "list"}],
    )

    with pytest.raises(ValidationError, match="USR ingest requires ingest.root or DNADESIGN_USR_ROOT"):
        resolve_config_job_inputs(
            job=job,
            config_dir=tmp_path,
            i_know_this_is_pickle=False,
            guard_pickle=lambda _x: None,
        )


def test_resolve_config_job_inputs_usr_uses_env_root_when_explicit_root_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_root = tmp_path / "usr_root"
    monkeypatch.setenv("DNADESIGN_USR_ROOT", str(env_root))
    job = JobConfig(
        id="j1",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "field": "sequence"},
        outputs=[{"id": "o1", "fn": "evo2.logits", "format": "list"}],
    )

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path,
        i_know_this_is_pickle=False,
        guard_pickle=lambda _x: None,
    )

    assert out is None
    assert job.ingest.root == env_root.resolve().as_posix()


def test_resolve_config_job_inputs_usr_normalizes_relative_package_root(tmp_path: Path) -> None:
    usr_pkg_root = tmp_path / "shared_usr"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# test package root\n", encoding="utf-8")
    job = JobConfig(
        id="j1",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": "../shared_usr", "field": "sequence"},
        outputs=[{"id": "o1", "fn": "evo2.logits", "format": "list"}],
    )

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path / "configs",
        i_know_this_is_pickle=False,
        guard_pickle=lambda _x: None,
    )

    assert out is None
    assert job.ingest.root == (usr_pkg_root / "datasets").resolve().as_posix()


def test_resolve_config_job_inputs_sequences_reads_relative_file(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "seqs.txt").write_text("ACGT\nTGCA\n", encoding="utf-8")
    job = _job(source="sequences", path="inputs/seqs.txt")

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path,
        i_know_this_is_pickle=False,
        guard_pickle=lambda _x: None,
    )

    assert out == ["ACGT", "TGCA"]


def test_resolve_config_job_inputs_records_reads_relative_jsonl(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "records.jsonl").write_text(
        '{"sequence":"ACGT","id":"a"}\n{"sequence":"TGCA","id":"b"}\n',
        encoding="utf-8",
    )
    job = _job(source="records", path="inputs/records.jsonl")

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path,
        i_know_this_is_pickle=False,
        guard_pickle=lambda _x: None,
    )

    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["sequence"] == "ACGT"
    assert out[1]["sequence"] == "TGCA"


def test_resolve_config_job_inputs_records_requires_path(tmp_path: Path) -> None:
    job = _job(source="records")

    with pytest.raises(
        ConfigError,
        match="ingest.source='records' requires ingest.path for infer run config workflows",
    ):
        resolve_config_job_inputs(
            job=job,
            config_dir=tmp_path,
            i_know_this_is_pickle=False,
            guard_pickle=lambda _x: None,
        )


def test_resolve_config_job_inputs_pt_file_uses_relative_path_and_guard(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "batch.pt").write_text("x", encoding="utf-8")
    job = _job(source="pt_file", path="inputs/batch.pt")
    calls: list[bool] = []

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path,
        i_know_this_is_pickle=True,
        guard_pickle=lambda flag: calls.append(flag),
    )

    assert out == (tmp_path / "inputs" / "batch.pt").resolve().as_posix()
    assert calls == [True]


def test_resolve_config_job_inputs_pt_file_falls_back_to_job_id_path(tmp_path: Path) -> None:
    job = _job(source="pt_file", job_id="demo_pt", path=None)
    calls: list[bool] = []

    out = resolve_config_job_inputs(
        job=job,
        config_dir=tmp_path,
        i_know_this_is_pickle=False,
        guard_pickle=lambda flag: calls.append(flag),
    )

    assert out == (tmp_path / "demo_pt.pt").resolve().as_posix()
    assert calls == [False]
