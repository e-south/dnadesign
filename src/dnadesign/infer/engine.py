"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/engine.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._logging import get_logger
from .adapter_dispatch import resolve_extract_callable, resolve_generate_callable
from .config import JobConfig, ModelConfig
from .contracts import resolve_generate_namespaced_fn, validate_extract_output_namespace
from .errors import (
    ConfigError,
    InferError,
    ModelLoadError,
    ValidationError,
)
from .extract_execution import execute_extract_output
from .generate_execution import execute_generate_batches, validate_generate_payload
from .ingest.sources import (
    load_pt_file_input,
    load_records_input,
    load_sequences_input,
    load_usr_input,
)
from .ingest.validators import validate_dna, validate_protein
from .progress import ProgressFactory, create_progress_handle
from .registry import get_adapter_cls
from .resume_planner import plan_resume_for_usr as _plan_resume_for_usr
from .writeback_dispatch import run_extract_write_back
from .writers.usr import write_back_usr

_LOG = get_logger(__name__)

# Cache adapter instances per (model_id, device, precision)
_ADAPTER_CACHE: Dict[Tuple[str, str, str], object] = {}


def clear_adapter_cache() -> None:
    _ADAPTER_CACHE.clear()


def _validate_alphabet(alphabet: str, seqs: List[str]) -> None:
    if alphabet == "dna":
        validate_dna(seqs, allow_iupac=False)
    elif alphabet == "protein":
        validate_protein(seqs, allow_extended_aas=False)
    else:
        raise ValidationError(f"Unknown alphabet: {alphabet}")


def _get_adapter(model: ModelConfig):
    key = (model.id, model.device, model.precision)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    adapter_cls = get_adapter_cls(model.id)
    try:
        adapter = adapter_cls(model.id, model.device, model.precision)
    except InferError:
        raise
    except Exception as e:
        raise ModelLoadError(str(e))
    _ADAPTER_CACHE[key] = adapter
    return adapter


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _auto_derate_enabled() -> bool:
    return os.environ.get("INFER_AUTO_DERATE_OOM", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _load_extract_ingest(inputs, *, ingest) -> Tuple[List[str], Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[str], object]:
    source = ingest.source
    if source == "sequences":
        seqs = load_sequences_input(inputs)
        return seqs, None, None, None, None
    if source == "records":
        seqs, records = load_records_input(inputs, ingest.field or "sequence")
        return seqs, None, records, None, None
    if source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        seqs, records = load_pt_file_input(inputs, ingest.field or "sequence")
        return seqs, None, records, inputs, None
    if source == "usr":
        seqs, ids, ds = load_usr_input(
            dataset_name=ingest.dataset,  # type: ignore[arg-type]
            field=ingest.field or "sequence",
            root=ingest.root,
            ids=ingest.ids,
        )
        return seqs, ids, None, None, ds
    raise ConfigError(f"Unknown ingest source: {source}")


def _load_generate_ingest(inputs, *, ingest) -> List[str]:
    prompts, _ids, _records, _pt_path, _ds = _load_extract_ingest(inputs, ingest=ingest)
    return prompts


def run_extract_job(
    inputs,
    *,
    model: ModelConfig,
    job: JobConfig,
    progress_factory: ProgressFactory = None,
) -> Dict[str, List[object]]:
    # ingest
    source = job.ingest.source
    seqs, ids, records, pt_path, ds = _load_extract_ingest(inputs, ingest=job.ingest)

    validate_extract_output_namespace(model_id=model.id, outputs=job.outputs or [])
    _validate_alphabet(model.alphabet, seqs)
    adapter = _get_adapter(model)

    # micro-batch
    micro_bs = model.batch_size or int(os.environ.get("DNADESIGN_INFER_BATCH", "0"))
    micro_bs = int(micro_bs) if micro_bs else 0
    default_bs = int(os.environ.get("DNADESIGN_INFER_DEFAULT_BS", "64"))
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

        def _write_back_chunk(idx_chunk: List[int], vals: List[object]) -> None:
            if source != "usr" or ids is None or not job.io.write_back:
                return
            chunk_ids = [ids[j] for j in idx_chunk]
            write_back_usr(
                ds,
                ids=chunk_ids,
                model_id=model.id,
                job_id=job.id,
                columnar={out.id: vals},
                overwrite=bool(job.io.overwrite),
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
                on_chunk=_write_back_chunk,
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
    prompts = _load_generate_ingest(inputs, ingest=job.ingest)

    gen_name = resolve_generate_namespaced_fn(model_id=model.id, fn=getattr(job, "fn", None))
    _validate_alphabet(model.alphabet, prompts)
    adapter = _get_adapter(model)

    # Choose generate fn from explicit contract-validated namespaced function.
    fn = resolve_generate_callable(adapter=adapter, namespaced_fn=gen_name)

    micro_bs = model.batch_size or int(os.environ.get("DNADESIGN_INFER_BATCH", "0"))
    micro_bs = int(micro_bs) if micro_bs else 0
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
