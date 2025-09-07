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
from .logging import get_logger
from .registry import get_adapter_cls, resolve_fn
from .writers.pt_file import write_back_pt_file
from .writers.records import write_back_records
from .writers.usr import write_back_usr

_LOG = get_logger(__name__)

# Cache adapter instances per (model_id, device, precision)
_ADAPTER_CACHE: Dict[Tuple[str, str, str], object] = {}


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

    # ── execute
    try:
        columnar: Dict[str, List[object]] = {}
        for out in job.outputs or []:
            method_name = resolve_fn(out.fn)
            fn = getattr(adapter, method_name, None)
            if fn is None:
                raise CapabilityError(f"Adapter does not implement '{out.fn}'")

            if method_name == "log_likelihood":
                vals = fn(seqs, **out.params)
            elif method_name in {"logits", "embedding"}:
                vals = fn(seqs, **out.params, fmt=out.format)
            else:
                raise CapabilityError(f"Unsupported extract function '{out.fn}' in v1")

            if len(vals) != len(seqs):
                raise RuntimeError("Adapter returned wrong number of outputs")

            columnar[out.id] = vals
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            raise RuntimeOOMError(msg)
        raise

    # ── optional write-back
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
    out = fn(prompts, **(job.params or {}))
    return out
