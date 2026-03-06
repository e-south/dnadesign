"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_namespace_contracts.py

Namespace contract and agnostic-model pressure tests for infer runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from dnadesign.infer.src.config import JobConfig, ModelConfig
from dnadesign.infer.src.engine import run_extract_job, run_generate_job
from dnadesign.infer.src.errors import ConfigError
from dnadesign.infer.src.registry import register_fn


class _AttachCaptureDataset:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def attach(
        self,
        path: Path,
        namespace: str,
        *,
        key: str,
        key_col: str | None = None,
        columns=None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        backend: str = "pyarrow",
        note: str = "",
    ) -> int:
        payload = pq.read_table(path)
        self.calls.append(
            {
                "namespace": namespace,
                "key": key,
                "key_col": key_col,
                "columns": list(columns) if columns is not None else [],
                "allow_overwrite": allow_overwrite,
                "allow_missing": allow_missing,
                "parse_json": parse_json,
                "backend": backend,
                "note": note,
                "payload_schema_names": list(payload.schema.names),
            }
        )
        return 1


def test_extract_fails_fast_when_output_namespace_mismatches_model(monkeypatch) -> None:
    register_fn("audit_ns_extract.logits", "logits")
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._get_adapter",
        lambda _model: SimpleNamespace(logits=lambda chunk, **_kwargs: [[1.0] for _ in chunk]),
    )

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="mismatch_extract",
        operation="extract",
        ingest={"source": "sequences"},
        outputs=[{"id": "logits", "fn": "audit_ns_extract.logits", "format": "list", "params": {}}],
    )

    with pytest.raises(ConfigError, match="output namespace"):
        run_extract_job(inputs=["ACGT"], model=model, job=job, progress_factory=None)


def test_generate_fails_fast_when_explicit_fn_namespace_mismatches_model(monkeypatch) -> None:
    register_fn("audit_ns_generate.generate", "generate")
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._get_adapter",
        lambda _model: SimpleNamespace(generate=lambda prompts, **_kwargs: {"gen_seqs": prompts}),
    )

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="mismatch_generate",
        operation="generate",
        ingest={"source": "sequences"},
        fn="audit_ns_generate.generate",
        params={"max_new_tokens": 4, "temperature": 1.0},
    )

    with pytest.raises(ConfigError, match="generate namespace"):
        run_generate_job(inputs=["ACGT"], model=model, job=job, progress_factory=None)


def test_agnostic_model_usr_pressure_path_writes_namespaced_columns(monkeypatch) -> None:
    namespace = "agnosticns"
    model_id = f"{namespace}_1b"
    register_fn(f"{namespace}.logits", "logits")
    register_fn(f"{namespace}.log_likelihood", "log_likelihood")

    seqs = ["ACGT", "TGCA"]
    ids = ["id-1", "id-2"]
    ds = _AttachCaptureDataset()

    monkeypatch.setattr(
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._get_adapter",
        lambda _model: SimpleNamespace(
            logits=lambda chunk, **_kwargs: [[float(i)] for i, _ in enumerate(chunk)],
            log_likelihood=lambda chunk, **_kwargs: [float(i) for i, _ in enumerate(chunk)],
        ),
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._plan_resume_for_usr",
        lambda **_kwargs: (list(range(len(ids))), {"logits": [None] * len(ids), "llr": [None] * len(ids)}),
    )

    model = ModelConfig(id=model_id, device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="pressure_job",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo"},
        outputs=[
            {"id": "logits", "fn": f"{namespace}.logits", "format": "list", "params": {}},
            {
                "id": "llr",
                "fn": f"{namespace}.log_likelihood",
                "format": "float",
                "params": {"method": "native", "reduction": "mean"},
            },
        ],
        io={"write_back": True, "overwrite": False},
    )

    out = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert set(out.keys()) == {"logits", "llr"}
    assert len(ds.calls) == 2

    attached_columns = {col for call in ds.calls for col in call["columns"]}  # type: ignore[index]
    assert f"infer__{model_id}__pressure_job__logits" in attached_columns
    assert f"infer__{model_id}__pressure_job__llr" in attached_columns
