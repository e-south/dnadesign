"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/products/specs.py

Stable product-specification identifiers for Construct runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from ..contracts.config import JobConfig
from ..contracts.errors import ValidationError


class SpecTemplate(Protocol):
    id: str
    kind: str
    source: str
    dataset: str | None
    field: str | None
    record_id: str | None
    circular: bool


def build_classic_spec_id(
    cfg: JobConfig,
    *,
    template: SpecTemplate,
    template_sha256: str,
    input_root: Path,
    output_root: Path,
) -> str:
    if cfg.job.template is None or cfg.job.realize is None:
        raise ValidationError("job.template and job.realize are required when building a realized-template spec id.")
    template_cfg = cfg.job.template
    realize = cfg.job.realize
    window = realize.window
    payload = {
        "job_id": cfg.job.id,
        "input": {
            "source": {
                "kind": cfg.job.input.source.kind,
                "dataset": cfg.job.input.source.dataset,
                "root": str(input_root),
            },
            "field": cfg.job.input.field,
            "ids": list(cfg.job.input.ids or []),
        },
        "template": {
            "id": template_cfg.id,
            "circular": template.circular,
            "source": {
                "kind": template.kind,
                "label": template.source,
                "dataset": template.dataset,
                "field": template.field,
                "record_id": template.record_id,
                "sha256": template_sha256,
            },
        },
        "parts": [
            {
                "name": part.name,
                "role": part.role,
                "sequence": {
                    "source": part.sequence.source,
                    "field": part.sequence.field,
                    "literal": part.sequence.literal,
                },
                "placement": {
                    "kind": part.placement.kind,
                    "orientation": part.placement.orientation,
                    "locator": part.placement.locator.model_dump(exclude_none=True),
                    "guards": (
                        part.placement.guards.model_dump(exclude_none=True)
                        if part.placement.guards is not None
                        else None
                    ),
                },
            }
            for part in cfg.job.parts
        ],
        "realize": {
            "mode": realize.mode,
            "focal_part": realize.focal_part,
            "required_slots": list(realize.required_slots),
            "window": (
                {
                    "semantics": window.semantics,
                    "reference": window.reference,
                    "direction": window.direction,
                    "size_bp": window.size_bp,
                    "upstream_bp": window.upstream_bp,
                    "downstream_bp": window.downstream_bp,
                    "offset_bp": window.offset_bp,
                }
                if window is not None
                else None
            ),
        },
        "output": {
            "target": {
                "kind": cfg.job.output.target.kind,
                "dataset": cfg.job.output.target.dataset,
                "root": str(output_root),
            },
            "record_source": cfg.job.output.record_source,
            "on_conflict": cfg.job.output.on_conflict,
            "allow_same_as_input": cfg.job.output.allow_same_as_input,
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_normalize_spec_id(
    *,
    cfg: JobConfig,
    input_root: object,
    output_root: object,
) -> str:
    if cfg.job.normalize_anchor is None:
        raise ValidationError("job.normalize_anchor is required when building a normalize-anchor spec id.")
    payload = {
        "job": {
            "id": cfg.job.id,
            "mode": cfg.job.mode,
            "input": {
                "dataset": cfg.job.input.source.dataset,
                "root": str(input_root),
                "field": cfg.job.input.field,
                "ids": list(cfg.job.input.ids or []),
            },
            "normalize_anchor": cfg.job.normalize_anchor.model_dump(mode="json"),
            "output": {
                "dataset": cfg.job.output.target.dataset,
                "root": str(output_root),
                "on_conflict": cfg.job.output.on_conflict,
            },
        }
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
