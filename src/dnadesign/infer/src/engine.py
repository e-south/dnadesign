"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/engine.py

Inference execution orchestration for extract and generate jobs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

from ._logging import get_logger
from .runtime.adapter_runtime import (
    auto_derate_enabled as _auto_derate_enabled,
    clear_adapter_cache,
    get_adapter as _get_adapter,
    is_oom as _is_oom,
)
from .runtime.adapter_dispatch import resolve_extract_callable, resolve_generate_callable
from .runtime.batch_policy import resolve_extract_batch_policy, resolve_micro_batch_size
from .config import JobConfig, ModelConfig
from .contracts import resolve_generate_namespaced_fn, validate_extract_output_namespace
from .errors import (
    ValidationError,
)
from .runtime.extract_chunk_writeback import build_extract_chunk_write_back
from .runtime.extract_execution import execute_extract_output
from .runtime.generate_execution import execute_generate_batches, validate_generate_payload
from .runtime.ingest_loading import load_extract_ingest, load_generate_ingest
from .ingest.validators import validate_dna, validate_protein
from .runtime.progress import ProgressFactory, create_progress_handle
from .runtime.resume_planner import plan_resume_for_usr as _plan_resume_for_usr
from .runtime.writeback_dispatch import run_extract_write_back
from .writers.usr import write_back_usr

_LOG = get_logger(__name__)


def _validate_alphabet(alphabet: str, seqs: List[str]) -> None:
    if alphabet == "dna":
        validate_dna(seqs, allow_iupac=False)
    elif alphabet == "protein":
        validate_protein(seqs, allow_extended_aas=False)
    else:
        raise ValidationError(f"Unknown alphabet: {alphabet}")


def run_extract_job(
    inputs,
    *,
    model: ModelConfig,
    job: JobConfig,
    progress_factory: ProgressFactory = None,
) -> Dict[str, List[object]]:
    # ingest
    source = job.ingest.source
    payload = load_extract_ingest(inputs, ingest=job.ingest)
    seqs = payload.seqs
    ids = payload.ids
    records = payload.records
    pt_path = payload.pt_path
    ds = payload.dataset

    validate_extract_output_namespace(model_id=model.id, outputs=job.outputs or [])
    _validate_alphabet(model.alphabet, seqs)
    adapter = _get_adapter(model)

    # micro-batch
    micro_bs, default_bs = resolve_extract_batch_policy(model_batch_size=model.batch_size)
    auto_derate = _auto_derate_enabled()

    # resume plan
    if source == "usr" and ids is not None:
        todo_idx, existing = _plan_resume_for_usr(
            ds=ds,
            ids=ids,
            model_id=model.id,
            job_id=job.id,
            outputs=job.outputs or [],
            overwrite=bool(job.io.overwrite),
        )
    else:
        todo_idx = list(range(len(seqs)))
        existing = {o.id: [None] * len(seqs) for o in (job.outputs or [])}

    # execute
    columnar: Dict[str, List[object]] = {}

    for out in job.outputs or []:
        method_name, fn = resolve_extract_callable(adapter=adapter, namespaced_fn=out.fn)

        all_vals: List[object] = list(existing[out.id])
        need_idx = [j for j in todo_idx if all_vals[j] is None]
        if len(need_idx) == 0:
            _LOG.info(f"{job.id}/{out.id}: nothing to do (already complete).")
            columnar[out.id] = all_vals
            continue

        # progress handle
        pbar = create_progress_handle(
            progress_factory=progress_factory,
            label=f"{job.id}/{out.id}",
            total=len(need_idx),
            unit="seq",
        )

        on_chunk = build_extract_chunk_write_back(
            source=source,
            write_back=bool(job.io.write_back),
            ds=ds,
            ids=ids,
            model_id=model.id,
            job_id=job.id,
            out_id=out.id,
            overwrite=bool(job.io.overwrite),
            writer=write_back_usr,
        )

        try:
            all_vals = execute_extract_output(
                seqs=seqs,
                need_idx=need_idx,
                existing=all_vals,
                method_name=method_name,
                fn=fn,
                params=out.params,
                output_format=out.format,
                micro_batch_size=micro_bs,
                default_batch_size=default_bs,
                auto_derate=auto_derate,
                is_oom=_is_oom,
                on_progress=pbar.update,
                on_chunk=on_chunk,
            )
        finally:
            pbar.close()

        if len(all_vals) != len(seqs):
            raise RuntimeError("Adapter returned wrong number of outputs")
        columnar[out.id] = all_vals

    run_extract_write_back(
        write_back=bool(job.io.write_back),
        source=source,
        records=records,
        pt_path=pt_path,
        ds=ds,
        ids=ids,
        model_id=model.id,
        job_id=job.id,
        columnar=columnar,
        overwrite=bool(job.io.overwrite),
    )

    return columnar


def run_generate_job(
    inputs,
    *,
    model: ModelConfig,
    job: JobConfig,
    progress_factory: ProgressFactory = None,
) -> Dict[str, List[object]]:
    prompts = load_generate_ingest(inputs, ingest=job.ingest)

    gen_name = resolve_generate_namespaced_fn(model_id=model.id, fn=getattr(job, "fn", None))
    _validate_alphabet(model.alphabet, prompts)
    adapter = _get_adapter(model)

    # Choose generate fn from explicit contract-validated namespaced function.
    fn = resolve_generate_callable(adapter=adapter, namespaced_fn=gen_name)

    micro_bs = resolve_micro_batch_size(model_batch_size=model.batch_size)
    params = job.params or {}
    if not micro_bs or micro_bs <= 0:
        return validate_generate_payload(fn(prompts, **params))

    pbar = create_progress_handle(
        progress_factory=progress_factory,
        label=f"{job.id}/generate",
        total=len(prompts),
        unit="prompt",
    )

    try:
        return execute_generate_batches(
            prompts=prompts,
            fn=fn,
            params=params,
            micro_batch_size=micro_bs,
            auto_derate=_auto_derate_enabled(),
            is_oom=_is_oom,
            on_progress=pbar.update,
        )
    finally:
        pbar.close()
