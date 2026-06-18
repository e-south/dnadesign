"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/materialization/views.py

Construct config and view declaration builders for RT-lnRNA materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from .common import _list, _mapping, _span_0
from .contracts import (
    _INPUT_DATASET,
    _MATERIALIZATION_SOURCE,
    _OUTPUT_DATASET,
    _REQUIRED_SLOT_IDS,
    MaterializationContractError,
)
from .manifest import _target_context_bounds


def _context_output_variants() -> list[dict[str, object]]:
    return [
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "seq_mean",
            "view_name": "dual_cassette_2000bp_seq_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "seq_mean",
            "view_name": "dual_cassette_2000bp_reverse_complement_seq_mean",
        },
    ]


def _slot_anchor_output_variants() -> list[dict[str, object]]:
    return [
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "lnrna",
            "anchor_window_size_bp": 384,
            "view_name": "lnrna_fixed_384bp_window_in_construct_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "lnrna",
            "anchor_window_size_bp": 384,
            "view_name": "lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "rt_cds",
            "anchor_window_size_bp": 1600,
            "view_name": "rt_cds_fixed_1600bp_window_in_construct_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "rt_cds",
            "anchor_window_size_bp": 1600,
            "view_name": "rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean",
        },
    ]


def _construct_config(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    usr_root: Path,
    input_ids_by_subject_id: Mapping[str, str],
    job_id: str,
    output_on_conflict: str,
    output_variants: list[dict[str, object]],
    construct_subject_ids: tuple[str, ...] | None = None,
    window_offset_bp: int | None = None,
) -> dict[str, object]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    target_start, target_end = _target_context_bounds(manifest)
    resolved_window_offset_bp = (
        _centered_window_offset_bp(slots=slots, target_start=target_start, target_end=target_end)
        if window_offset_bp is None
        else window_offset_bp
    )
    resolved_construct_subject_ids = construct_subject_ids or tuple(
        str(candidate["construct_subject_id"]) for candidate in _list(manifest["candidates"], label="candidates")
    )
    return {
        "job": {
            "id": job_id,
            "input": {
                "source": {
                    "kind": "usr",
                    "dataset": _INPUT_DATASET,
                    "root": str(usr_root),
                },
                "field": None,
                "ids": [input_ids_by_subject_id[subject_id] for subject_id in resolved_construct_subject_ids],
            },
            "template": {
                "id": str(
                    _mapping(manifest["construct_template"], label="construct_template")["construct_template_id"]
                ),
                "source": {
                    "kind": "literal",
                    "sequence": template_sequence,
                    "label": "genbank:pes-retron-26.gb#record",
                },
                "circular": True,
            },
            "parts": [_part_config(slot=slot, template_sequence=template_sequence) for slot in slots],
            "realize": {
                "mode": "window",
                "focal_part": "lnrna",
                "required_slots": list(_REQUIRED_SLOT_IDS),
                "window": {
                    "semantics": "fixed_total",
                    "reference": "center",
                    "direction": "symmetric",
                    "size_bp": target_end - target_start,
                    "offset_bp": resolved_window_offset_bp,
                },
            },
            "output_variants": output_variants,
            "output": {
                "record_source": _MATERIALIZATION_SOURCE,
                "on_conflict": output_on_conflict,
                "target": {
                    "kind": "usr",
                    "dataset": _OUTPUT_DATASET,
                    "root": str(usr_root),
                },
            },
        }
    }


def _centered_window_offset_bp(
    *,
    slots: tuple[dict[str, object], ...],
    target_start: int,
    target_end: int,
) -> int:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    lnrna_start, lnrna_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    base_center = lnrna_start + ((lnrna_end - lnrna_start) // 2)
    window_length = target_end - target_start
    return target_start - (base_center - (window_length // 2))


def _candidate_window_bounds(
    *,
    slots: tuple[dict[str, object], ...],
    realized_spans: dict[str, tuple[int, int]],
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    base_start, base_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    realized_start, realized_end = realized_spans["lnrna"]
    base_center = base_start + ((base_end - base_start) // 2)
    realized_center = realized_start + ((realized_end - realized_start) // 2)
    window_start = target_start + (realized_center - base_center)
    return window_start, window_start + (target_end - target_start)


def _part_config(*, slot: dict[str, object], template_sequence: str) -> dict[str, object]:
    start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
    return {
        "name": str(slot["slot_id"]),
        "role": str(slot["role"]),
        "sequence": {
            "source": "input_field",
            "field": str(slot["sequence_field"]),
        },
        "placement": {
            "kind": "replace",
            "orientation": str(slot["orientation"]),
            "locator": {
                "kind": "coordinates",
                "start": start,
                "end": end,
            },
            "guards": {
                "replaced_sequence": template_sequence[start:end],
                "replaced_span_bp": end - start,
            },
        },
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
