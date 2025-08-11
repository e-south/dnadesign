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
from .ingest.sources import load_pt_file_input, load_records_input, load_sequences_input
from .ingest.validators import validate_dna, validate_protein
from .logging import get_logger
from .registry import get_adapter_cls, resolve_fn
from .writers.pt_file import write_back_pt_file
from .writers.records import write_back_records

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
    # ingest
    if job.ingest.source == "sequences":
        seqs = load_sequences_input(inputs)
        records = None
        pt_path = None
    elif job.ingest.source == "records":
        seqs, records = load_records_input(inputs, job.ingest.field)
        pt_path = None
    elif job.ingest.source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        seqs, records = load_pt_file_input(inputs, job.ingest.field)
        pt_path = inputs
    else:
        raise ConfigError(f"Unknown ingest source: {job.ingest.source}")

    # Validate alphabet (strict defaults)
    _validate_alphabet(model.alphabet, seqs)

    # Adapter
    adapter = _get_adapter(model)

    # Validate all fns are supported by adapter namespace
    namespace = job.outputs[0].fn.split(".")[0] if job.outputs else None
    for out in job.outputs or []:
        ns = out.fn.split(".")[0]
        if ns != namespace:
            raise ConfigError(
                "All outputs in a job must share the same adapter namespace"
            )

    # Execute
    try:
        columnar: Dict[str, List[object]] = {}
        for out in job.outputs or []:
            method_name = resolve_fn(out.fn)
            fn = getattr(adapter, method_name, None)
            if fn is None:
                raise CapabilityError(f"Adapter does not implement '{out.fn}'")
            # Dispatch by function name
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

    # Optional write-back
    if job.io.write_back:
        if job.ingest.source == "records":
            write_back_records(
                records,
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )
        elif job.ingest.source == "pt_file":
            write_back_pt_file(
                pt_path,
                records,
                model_id=model.id,
                job_id=job.id,
                columnar=columnar,
                overwrite=job.io.overwrite,
            )
        else:
            raise WriteBackError(
                "write_back is only supported for records/pt_file sources"
            )

    return columnar


def run_generate_job(
    inputs, *, model: ModelConfig, job: JobConfig
) -> Dict[str, List[object]]:
    # ingest → prompts
    if job.ingest.source == "sequences":
        prompts = load_sequences_input(inputs)
    elif job.ingest.source == "records":
        prompts, _ = load_records_input(inputs, job.ingest.field)
    elif job.ingest.source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        prompts, _ = load_pt_file_input(inputs, job.ingest.field)
    else:
        raise ConfigError(f"Unknown ingest source: {job.ingest.source}")

    # Validate alphabet (based on model)
    _validate_alphabet(model.alphabet, prompts)

    adapter = _get_adapter(model)

    # Dispatch generate
    from .registry import resolve_fn

    # Namespaced must match adapter; we accept only evo2.generate in v1
    gen_fn_name = resolve_fn("evo2.generate") if hasattr(adapter, "generate") else None
    if not gen_fn_name or not getattr(adapter, gen_fn_name, None):
        raise CapabilityError("Adapter does not support generation in v1")

    fn = getattr(adapter, gen_fn_name)
    out = fn(prompts, **(job.params or {}))
    # returns is advisory; simply pass through adapter result keys
    return out
