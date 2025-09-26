"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/engine.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import os

from .config import JobConfig, ModelConfig
from .errors import (
    CapabilityError,
    ConfigError,
    RuntimeOOMError,
    ValidationError,
    WriteBackError,
)
from .ingest.sources import (
    load_pt_file_input,
    load_records_input,
    load_sequences_input,
    load_usr_input,
)
from .ingest.validators import validate_dna, validate_protein
from ._logging import get_logger
from .registry import get_adapter_cls, resolve_fn
from .writers.pt_file import write_back_pt_file
from .writers.records import write_back_records
from .writers.usr import write_back_usr

_LOG = get_logger(__name__)

# Cache adapter instances per (model_id, device, precision)
_ADAPTER_CACHE: Dict[Tuple[str, str, str], object] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Optional progress bar
# ──────────────────────────────────────────────────────────────────────────────
def _get_tqdm():
    show = os.environ.get("DNADESIGN_PROGRESS", "1").lower() not in {"0", "false", "off", "no"}
    if not show:
        # no-op shim
        class _NoTQDM:
            def __init__(self, total=None, **kwargs): self.total = total
            def update(self, n): pass
            def close(self): pass
            def set_postfix(self, **kw): pass
        return _NoTQDM, False
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm, True
    except Exception:
        _LOG.info("tqdm not available; continuing without a progress bar.")
        class _NoTQDM:
            def __init__(self, total=None, **kwargs): self.total = total
            def update(self, n): pass
            def close(self): pass
            def set_postfix(self, **kw): pass
        return _NoTQDM, False


def _get_adapter(model: ModelConfig):
    key = (model.id, model.device, model.precision)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    adapter_cls = get_adapter_cls(model.id)
    try:
        adapter = adapter_cls(model.id, model.device, model.precision)
    except Exception as e:
        raise RuntimeError(str(e))
    _ADAPTER_CACHE[key] = adapter
    return adapter


def _validate_alphabet(alphabet: str, seqs: List[str]) -> None:
    if alphabet == "dna":
        validate_dna(seqs, allow_iupac=False)
    elif alphabet == "protein":
        validate_protein(seqs, allow_extended_aas=False)
    else:  # pragma: no cover
        raise ValidationError(f"Unknown alphabet: {alphabet}")


def _plan_resume_for_usr(
    *,
    ds,               # dnadesign.usr.Dataset
    ids: List[str],
    model_id: str,
    job_id: str,
    outputs: List,    # list[OutputSpec]
    overwrite: bool,
) -> Tuple[List[int], Dict[str, List[object]]]:
    """
    Build a plan for which rows to (re)compute and return any existing values
    for already-computed rows so we can return a complete column set.

    Returns:
      todo_idx : list[int] indices into ids that still need work
      existing : dict[out_id] -> list[object|None] length=len(ids)
    """
    N = len(ids)
    existing: Dict[str, List[object]] = {o.id: [None] * N for o in outputs}
    if overwrite or ds is None or N == 0:
        return list(range(N)), existing

    # Column names in the dataset (writer prefixes "infer__")
    infer_cols = {
        o.id: f"infer__{model_id}__{job_id}__{o.id}"
        for o in outputs
    }

    # Try to read only the columns that exist.
    try:
        import pyarrow.parquet as pq  # local import to keep dependency soft
        rec_path = ds.records_path  # type: ignore[attr-defined]
        pf = pq.ParquetFile(rec_path)
        present = set(pf.schema_arrow.names)  # type: ignore[attr-defined]
        want_cols = ["id"] + [c for c in infer_cols.values() if c in present]
        if len(want_cols) == 1:
            # None of our infer columns exist → nothing to resume
            return list(range(N)), existing

        tbl = pq.read_table(rec_path, columns=want_cols)
        t_ids = tbl.column("id").to_pylist()
        # Build id -> row index in table
        pos = {rid: i for i, rid in enumerate(t_ids)}

        # For each output with a present column, map values by id
        for out in outputs:
            colname = infer_cols[out.id]
            if colname in present:
                vals = tbl.column(colname).to_pylist()
                # Scatter to our subset/order
                for j, rid in enumerate(ids):
                    i_tbl = pos.get(rid)
                    if i_tbl is not None:
                        existing[out.id][j] = vals[i_tbl]
    except Exception as e:
        _LOG.debug("Resume scan skipped (could not read USR table): %s", e)
        return list(range(N)), existing

    # Decide which rows need work: rows missing ANY requested output
    todo_idx: List[int] = []
    for j in range(N):
        has_all = True
        for out in outputs:
            if existing[out.id][j] is None:
                has_all = False
                break
        if not has_all:
            todo_idx.append(j)
    return todo_idx, existing


def run_extract_job(
    inputs, *, model: ModelConfig, job: JobConfig
) -> Dict[str, List[object]]:
    # ── ingest
    source = job.ingest.source
    if source == "sequences":
        seqs = load_sequences_input(inputs)
        ids = None
        records = None
        pt_path = None
        ds = None
    elif source == "records":
        seqs, records = load_records_input(inputs, job.ingest.field or "sequence")
        ids = None
        pt_path = None
        ds = None
    elif source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        seqs, records = load_pt_file_input(inputs, job.ingest.field or "sequence")
        ids = None
        pt_path = inputs
        ds = None
    elif source == "usr":
        seqs, ids, ds = load_usr_input(
            dataset_name=job.ingest.dataset,  # type: ignore[arg-type]
            field=job.ingest.field or "sequence",
            root=job.ingest.root,
            ids=job.ingest.ids,
        )
        records = None
        pt_path = None
    else:
        raise ConfigError(f"Unknown ingest source: {source}")

    # ── validate alphabet
    _validate_alphabet(model.alphabet, seqs)

    # ── adapter
    adapter = _get_adapter(model)

    # ── ensure one namespace per job (e.g., all 'evo2.*')
    namespace = job.outputs[0].fn.split(".")[0] if job.outputs else None
    for out in job.outputs or []:
        ns = out.fn.split(".")[0]
        if ns != namespace:
            raise ConfigError(
                "All outputs in a job must share the same adapter namespace"
            )

    # ── micro-batch sizing
    micro_bs = model.batch_size or int(os.environ.get("DNADESIGN_INFER_BATCH", "0"))
    micro_bs = int(micro_bs) if micro_bs else 0

    # ── resume/skip plan (USR only; else recompute all)
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

    # Progress bar per output
    tqdm, _pbar_enabled = _get_tqdm()

    # ── execute with resume + micro-batching + incremental write-back
    try:
        columnar: Dict[str, List[object]] = {}

        for out in job.outputs or []:
            method_name = resolve_fn(out.fn)
            fn = getattr(adapter, method_name, None)
            if fn is None:
                raise CapabilityError(f"Adapter does not implement '{out.fn}'")

            # Initialize the full output vector with any existing values
            all_vals: List[object] = list(existing[out.id])

            # Which indices still need THIS output? (if resume, only rows missing any output were in todo_idx;
            # now further filter to those missing THIS specific out)
            need_idx = [j for j in todo_idx if all_vals[j] is None]

            if len(need_idx) == 0:
                _LOG.info(f"{job.id}/{out.id}: nothing to do (already complete).")
                columnar[out.id] = all_vals
                continue

            pbar = tqdm(total=len(need_idx), unit="seq", desc=f"{job.id}/{out.id}")

            # Process in chunks
            start = 0
            while start < len(need_idx):
                take = len(need_idx) - start if not micro_bs or micro_bs <= 0 else min(micro_bs, len(need_idx) - start)
                idx_chunk = need_idx[start : start + take]
                chunk = [seqs[i] for i in idx_chunk]

                # Run adapter
                if method_name == "log_likelihood":
                    vals = fn(chunk, **out.params)
                elif method_name in {"logits", "embedding"}:
                    vals = fn(chunk, **out.params, fmt=out.format)
                else:
                    raise CapabilityError(f"Unsupported extract function '{out.fn}' in v1")

                if len(vals) != len(idx_chunk):
                    raise RuntimeError("Adapter returned wrong number of outputs for chunk")

                # Scatter results into full vector
                for k, j in enumerate(idx_chunk):
                    all_vals[j] = vals[k]

                # Incremental write-back to allow safe resume on crash
                if source == "usr" and ids is not None and job.io.write_back:
                    chunk_ids = [ids[j] for j in idx_chunk]
                    write_back_usr(
                        ds,
                        ids=chunk_ids,
                        model_id=model.id,
                        job_id=job.id,
                        columnar={out.id: vals},
                        overwrite=bool(job.io.overwrite),
                    )

                pbar.update(len(idx_chunk))
                start += take

            pbar.close()

            # Sanity
            if len(all_vals) != len(seqs):
                raise RuntimeError("Adapter returned wrong number of outputs")

            columnar[out.id] = all_vals

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            raise RuntimeOOMError(msg)
        raise

    # ── final write-back (for non-USR sources, or to ensure all outputs land together)
    if job.io.write_back:
        if source == "records":
            write_back_records(
                records,
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )
        elif source == "pt_file":
            write_back_pt_file(
                pt_path,  # type: ignore[arg-type]
                records,  # type: ignore[arg-type]
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )
        elif source == "usr":
            # We already flushed by-chunk, but do a final pass to ensure completeness.
            if ids is None or ds is None:
                raise WriteBackError("USR write-back requires ids and dataset handle")
            write_back_usr(
                ds,
                ids=ids,
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )
        else:
            raise WriteBackError("write_back not supported for this ingest source")

    return columnar


def run_generate_job(
    inputs, *, model: ModelConfig, job: JobConfig
) -> Dict[str, List[object]]:
    source = job.ingest.source
    if source == "sequences":
        prompts = load_sequences_input(inputs)
    elif source == "records":
        prompts, _ = load_records_input(inputs, job.ingest.field or "sequence")
    elif source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        prompts, _ = load_pt_file_input(inputs, job.ingest.field or "sequence")
    elif source == "usr":
        prompts, _, _ = load_usr_input(
            dataset_name=job.ingest.dataset,  # type: ignore[arg-type]
            field=job.ingest.field or "sequence",
            root=job.ingest.root,
            ids=job.ingest.ids,
        )
    else:
        raise ConfigError(f"Unknown ingest source: {source}")

    _validate_alphabet(model.alphabet, prompts)

    adapter = _get_adapter(model)

    # Force 'evo2.generate' resolve; adapters may expand later
    gen_fn_name = resolve_fn("evo2.generate") if hasattr(adapter, "generate") else None
    if not gen_fn_name or not getattr(adapter, gen_fn_name, None):
        raise CapabilityError("Adapter does not support generation in v1")

    fn = getattr(adapter, gen_fn_name)
    micro_bs = model.batch_size or int(os.environ.get("DNADESIGN_INFER_BATCH", "0"))
    micro_bs = int(micro_bs) if micro_bs else 0
    if not micro_bs or micro_bs <= 0:
        return fn(prompts, **(job.params or {}))
    gen_all: Dict[str, List[object]] = {"gen_seqs": []}
    tqdm, _ = _get_tqdm()
    pbar = tqdm(total=len(prompts), unit="prompt", desc=f"{job.id}/generate")
    for s in range(0, len(prompts), micro_bs):
        chunk = prompts[s : s + micro_bs]
        out = fn(chunk, **(job.params or {}))
        gen_all["gen_seqs"].extend(out.get("gen_seqs", []))
        pbar.update(len(chunk))
    pbar.close()
    return gen_all
