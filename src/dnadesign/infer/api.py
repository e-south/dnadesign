"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/api.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .config import JobConfig, ModelConfig, OutputSpec
from .engine import run_extract_job, run_generate_job
from .errors import ConfigError

ProgressFactory = Optional[Callable[[str, int], Any]]  # returns handle with .update(n), .close()


def run_extract(
    seqs: List[str],
    *,
    model_id: str,
    outputs: List[dict],
    device: str | None = None,
    precision: str | None = None,
    alphabet: str | None = None,
    batch_size: int | None = None,
    progress_factory: ProgressFactory = None,
) -> Dict[str, List[object]]:
    if not isinstance(seqs, list) or not all(isinstance(x, str) for x in seqs):
        raise ConfigError("seqs must be list[str]")
    model = ModelConfig(
        id=model_id,
        device=device or "cpu",
        precision=(precision or "fp32"),
        alphabet=(alphabet or "dna"),
        batch_size=batch_size,
    )
    job = JobConfig(
        id="adhoc_extract",
        operation="extract",
        ingest={"source": "sequences"},
        outputs=[OutputSpec(**o) for o in outputs],
    )
    return run_extract_job(seqs, model=model, job=job, progress_factory=progress_factory)


def run_generate(
    prompts: List[str],
    *,
    model_id: str,
    params: dict,
    device: str | None = None,
    precision: str | None = None,
    alphabet: str | None = None,
    batch_size: int | None = None,
    progress_factory: ProgressFactory = None,
) -> Dict[str, List[object]]:
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ConfigError("prompts must be list[str]")
    model = ModelConfig(
        id=model_id,
        device=device or "cpu",
        precision=(precision or "fp32"),
        alphabet=(alphabet or "dna"),
        batch_size=batch_size,
    )
    job = JobConfig(
        id="adhoc_generate",
        operation="generate",
        ingest={"source": "sequences"},
        params=params,
    )
    return run_generate_job(prompts, model=model, job=job, progress_factory=progress_factory)


def run_job(
    inputs,
    *,
    model: ModelConfig | dict,
    job: JobConfig | dict,
    progress_factory: ProgressFactory = None,
):
    model_cfg = model if isinstance(model, ModelConfig) else ModelConfig(**model)
    job_cfg = job if isinstance(job, JobConfig) else JobConfig(**job)
    if job_cfg.operation == "extract":
        return run_extract_job(inputs, model=model_cfg, job=job_cfg, progress_factory=progress_factory)
    elif job_cfg.operation == "generate":
        return run_generate_job(inputs, model=model_cfg, job=job_cfg, progress_factory=progress_factory)
    else:
        raise ConfigError(f"Unknown operation: {job_cfg.operation}")
