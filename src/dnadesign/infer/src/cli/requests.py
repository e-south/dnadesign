"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/requests.py

CLI request assembly helpers for infer extract and generate commands.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import JobConfig, ModelConfig, OutputSpec
from ..errors import ConfigError
from ..presets import load_preset
from .builders import build_model_config


@dataclass(frozen=True)
class ExtractRequest:
    model: ModelConfig
    job: JobConfig
    output_rows: List[Dict[str, Any]]


@dataclass(frozen=True)
class GenerateRequest:
    model: ModelConfig
    job: JobConfig
    output_row: Dict[str, Any]


def build_extract_request(
    *,
    model_id: Optional[str],
    device: Optional[str],
    precision: Optional[str],
    alphabet: Optional[str],
    batch_size: Optional[int],
    preset: Optional[str],
    fn: Optional[str],
    format: Optional[str],
    out_id: str,
    pool_method: Optional[str],
    pool_dim: Optional[int],
    layer: Optional[str],
    write_back: bool,
    overwrite: bool,
) -> ExtractRequest:
    outputs: List[OutputSpec] = []

    if preset:
        p = load_preset(preset)
        if p["kind"] != "extract":
            raise ConfigError(f"Preset '{preset}' is not an extract preset.")
        outputs = [OutputSpec(**o) for o in p.get("outputs", [])]
        model = build_model_config(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            batch_size=batch_size,
            preset_model=p.get("model", {}),
        )
    else:
        if not (fn and format):
            raise ConfigError("Provide --fn and --format, or use --preset.")
        params: Dict[str, Any] = {}
        if fn.split(".")[-1] in {"logits", "embedding"}:
            if pool_method or pool_dim is not None:
                params["pool"] = {}
                if pool_method:
                    params["pool"]["method"] = pool_method
                if pool_dim is not None:
                    params["pool"]["dim"] = pool_dim
            if fn.split(".")[-1] == "embedding" and not layer:
                raise ConfigError("embedding extract requires --layer with a valid Evo2 layer name")
            if layer:
                params["layer"] = layer
        elif fn.split(".")[-1] == "log_likelihood":
            params.setdefault("method", "native")
            params.setdefault("reduction", "sum")
        outputs = [OutputSpec(id=out_id, fn=fn, params=params, format=format)]
        model = build_model_config(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            batch_size=batch_size,
        )

    job = JobConfig(
        id="adhoc_extract",
        operation="extract",
        ingest={"source": "sequences"},
        outputs=outputs,
        io={"write_back": write_back, "overwrite": overwrite},
    )
    return ExtractRequest(
        model=model,
        job=job,
        output_rows=[o.model_dump() for o in outputs],
    )


def build_generate_request(
    *,
    model_id: Optional[str],
    device: Optional[str],
    precision: Optional[str],
    alphabet: Optional[str],
    batch_size: Optional[int],
    preset: Optional[str],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    seed: Optional[int],
) -> GenerateRequest:
    params: Dict[str, Any] = {}

    if preset:
        p = load_preset(preset)
        if p["kind"] != "generate":
            raise ConfigError(f"Preset '{preset}' is not a generate preset.")
        params.update(p.get("params") or {})
        model = build_model_config(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            batch_size=batch_size,
            preset_model=p.get("model", {}),
        )
    else:
        if max_new_tokens is None:
            max_new_tokens = 64
        if temperature is None:
            temperature = 1.0
        params.update(
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        if top_k is not None:
            params["top_k"] = top_k
        if top_p is not None:
            params["top_p"] = top_p
        if seed is not None:
            params["seed"] = seed
        model = build_model_config(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            batch_size=batch_size,
        )

    job = JobConfig(
        id="adhoc_generate",
        operation="generate",
        ingest={"source": "sequences"},
        params=params,
    )
    return GenerateRequest(
        model=model,
        job=job,
        output_row={"id": "gen", "fn": "generate", "format": "—", "params": params},
    )
