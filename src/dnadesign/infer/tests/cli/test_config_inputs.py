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
from dnadesign.infer.src.errors import ConfigError


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


def test_resolve_config_job_inputs_usr_returns_none(tmp_path: Path) -> None:
    job = JobConfig(
        id="j1",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "field": "sequence"},
        outputs=[{"id": "o1", "fn": "evo2.logits", "format": "list"}],
    )

    out = resolve_config_job_inputs(job=job, config_dir=tmp_path, i_know_this_is_pickle=False, guard_pickle=lambda _x: None)

    assert out is None


def test_resolve_config_job_inputs_sequences_reads_relative_file(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "seqs.txt").write_text("ACGT\nTGCA\n", encoding="utf-8")
    job = _job(source="sequences", path="inputs/seqs.txt")

    out = resolve_config_job_inputs(job=job, config_dir=tmp_path, i_know_this_is_pickle=False, guard_pickle=lambda _x: None)

    assert out == ["ACGT", "TGCA"]


def test_resolve_config_job_inputs_records_reads_relative_jsonl(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "records.jsonl").write_text('{"sequence":"ACGT","id":"a"}\n{"sequence":"TGCA","id":"b"}\n', encoding="utf-8")
    job = _job(source="records", path="inputs/records.jsonl")

    out = resolve_config_job_inputs(job=job, config_dir=tmp_path, i_know_this_is_pickle=False, guard_pickle=lambda _x: None)

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
