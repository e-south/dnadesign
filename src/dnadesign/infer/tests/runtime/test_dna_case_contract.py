from __future__ import annotations

from types import SimpleNamespace

from dnadesign.infer.src.config import JobConfig, ModelConfig
from dnadesign.infer.src.engine import run_extract_job
from dnadesign.infer.src.ingest.validators import canonicalize_dna


def test_canonicalize_dna_uppercases_lowercase_model_inputs() -> None:
    assert canonicalize_dna(["ACGT", "acgt"], allow_iupac=False) == ["ACGT", "ACGT"]


def test_run_extract_job_uppercases_dna_before_adapter(monkeypatch) -> None:
    observed: list[list[str]] = []

    def _logits(chunk, **_kwargs):
        observed.append(list(chunk))
        return [[float(len(seq))] for seq in chunk]

    monkeypatch.setattr(
        "dnadesign.infer.src.engine._get_adapter",
        lambda _model: SimpleNamespace(logits=_logits),
    )
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="case_contract",
        operation="extract",
        ingest={"source": "sequences"},
        outputs=[{"id": "logits", "fn": "evo2.logits", "format": "list", "params": {}}],
    )

    out = run_extract_job(inputs=["acgt", "ACgt"], model=model, job=job, progress_factory=None)

    assert observed == [["ACGT", "ACGT"]]
    assert out == {"logits": [[4.0], [4.0]]}
