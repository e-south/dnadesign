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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._logging import get_logger
from .adapter_dispatch import resolve_extract_callable, resolve_generate_callable
from .config import JobConfig, ModelConfig
from .contracts import infer_usr_column_name, resolve_generate_namespaced_fn, validate_extract_output_namespace
from .errors import (
    ConfigError,
    InferError,
    ModelLoadError,
    ValidationError,
    WriteBackError,
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
from .registry import get_adapter_cls
from .writers.pt_file import write_back_pt_file
from .writers.records import write_back_records
from .writers.usr import write_back_usr

_LOG = get_logger(__name__)

# Cache adapter instances per (model_id, device, precision)
_ADAPTER_CACHE: Dict[Tuple[str, str, str], object] = {}


def clear_adapter_cache() -> None:
    _ADAPTER_CACHE.clear()


# --------------------------------------------------------------------- Progress
def _get_tqdm():
    show = os.environ.get("DNADESIGN_PROGRESS", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    if not show:

        class _NoTQDM:
            def __init__(self, total=None, **kwargs):
                self.total = total

            def update(self, n):
                pass

            def close(self):
                pass

        return _NoTQDM, False
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm, True
    except Exception:
        _LOG.info("tqdm not available; continuing without a progress bar.")

        class _NoTQDM:
            def __init__(self, total=None, **kwargs):
                self.total = total

            def update(self, n):
                pass

            def close(self):
                pass

        return _NoTQDM, False


ProgressFactory = Optional[Callable[[str, int], Any]]  # returns handle with update(n), close()


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


def _plan_resume_for_usr(
    *,
    ds,  # dnadesign.usr.Dataset
    ids: List[str],
    model_id: str,
    job_id: str,
    outputs: List,  # list[OutputSpec]
    overwrite: bool,
) -> Tuple[List[int], Dict[str, List[object]]]:
    N = len(ids)
    existing: Dict[str, List[object]] = {o.id: [None] * N for o in outputs}
    if overwrite or ds is None or N == 0:
        return list(range(N)), existing

    infer_cols = {
        o.id: infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=o.id)
        for o in outputs
    }

    try:
        import pyarrow.parquet as pq

        rec_path = ds.records_path  # type: ignore[attr-defined]
        pf = pq.ParquetFile(rec_path)
        present = set(pf.schema_arrow.names)  # type: ignore[attr-defined]
        want_cols = ["id"] + [c for c in infer_cols.values() if c in present]
        if len(want_cols) > 1:
            tbl = pq.read_table(rec_path, columns=want_cols)
            t_ids = tbl.column("id").to_pylist()
            pos = {rid: i for i, rid in enumerate(t_ids)}

            for out in outputs:
                colname = infer_cols[out.id]
                if colname in present:
                    vals = tbl.column(colname).to_pylist()
                    for j, rid in enumerate(ids):
                        i_tbl = pos.get(rid)
                        if i_tbl is not None:
                            existing[out.id][j] = vals[i_tbl]

        if hasattr(ds, "list_overlays"):
            overlays = ds.list_overlays()  # type: ignore[attr-defined]
            infer_overlay = next((ov for ov in overlays if getattr(ov, "namespace", None) == "infer"), None)
            if infer_overlay is not None:
                overlay_path = Path(str(infer_overlay.path))
                overlay_pf = pq.ParquetFile(str(overlay_path))
                overlay_present = set(overlay_pf.schema_arrow.names)
                overlay_cols = ["id"] + [c for c in infer_cols.values() if c in overlay_present]
                if len(overlay_cols) > 1:
                    overlay_tbl = pq.read_table(str(overlay_path), columns=overlay_cols)
                    overlay_ids = overlay_tbl.column("id").to_pylist()
                    overlay_pos = {rid: i for i, rid in enumerate(overlay_ids)}
                    for out in outputs:
                        colname = infer_cols[out.id]
                        if colname not in overlay_present:
                            continue
                        vals = overlay_tbl.column(colname).to_pylist()
                        for j, rid in enumerate(ids):
                            i_tbl = overlay_pos.get(rid)
                            if i_tbl is not None and vals[i_tbl] is not None:
                                existing[out.id][j] = vals[i_tbl]
    except Exception as e:
        raise WriteBackError(f"USR resume scan failed for records table {ds.records_path}: {e}") from e

    todo_idx: List[int] = []
    for j in range(N):
        if any(existing[o.id][j] is None for o in outputs):
            todo_idx.append(j)
    return todo_idx, existing


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
        if progress_factory:
            pbar = progress_factory(f"{job.id}/{out.id}", len(need_idx))
        else:
            tqdm, _ = _get_tqdm()
            pbar = tqdm(total=len(need_idx), unit="seq", desc=f"{job.id}/{out.id}")

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

    # final write-back
    if job.io.write_back:
        if source == "records":
            write_back_records(records, model_id=model.id, job_id=job.id, columnar=columnar, overwrite=job.io.overwrite)  # type: ignore[arg-type]
        elif source == "pt_file":
            write_back_pt_file(
                pt_path,
                records,
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )  # noqa
        elif source == "usr":
            if ids is None or ds is None:
                raise WriteBackError("USR write-back requires ids and dataset handle")
        else:
            raise WriteBackError("write_back not supported for this ingest source")

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

    if progress_factory:
        pbar = progress_factory(f"{job.id}/generate", len(prompts))
    else:
        tqdm, _ = _get_tqdm()
        pbar = tqdm(total=len(prompts), unit="prompt", desc=f"{job.id}/generate")

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
