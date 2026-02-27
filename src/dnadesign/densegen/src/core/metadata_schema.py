"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/metadata_schema.py

Typed metadata registry for DenseGen output fields.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetaField:
    name: str
    types: tuple[type, ...]
    description: str
    required: bool = True
    allow_none: bool = False


META_FIELDS: list[MetaField] = [
    MetaField("schema_version", (str,), "DenseGen schema version (e.g., 2.7)."),
    MetaField("created_at", (str,), "UTC ISO8601 timestamp for record creation."),
    MetaField("run_id", (str,), "Run identifier (densegen.run.id)."),
    MetaField("length", (int,), "Actual output sequence length."),
    MetaField("plan", (str,), "Plan item name."),
    MetaField("input_name", (str,), "Input source name."),
    MetaField("input_mode", (str,), "Input mode (binding_sites | sequence_library | pwm_sampled | plan_pool)."),
    MetaField("input_pwm_ids", (list,), "Stage-A PWM motif IDs used for sampling (pwm_* inputs)."),
    MetaField("used_tfbs", (list,), "TF:TFBS strings used in the final sequence."),
    MetaField(
        "used_tfbs_detail",
        (list,),
        "Per-placement parts detail for TFBS and fixed elements.",
    ),
    MetaField("used_tf_counts", (list,), "Per-TF placement counts ({tf, count})."),
    MetaField("library_unique_tf_count", (int,), "Unique TF count in the Stage-B sampled library."),
    MetaField("library_unique_tfbs_count", (int,), "Unique TFBS count in the Stage-B sampled library."),
    MetaField("covers_all_tfs_in_solution", (bool,), "Whether min_count_per_tf coverage was satisfied."),
    MetaField(
        "required_regulators",
        (list,),
        "Regulators required for this library (selected from regulator_constraints groups).",
    ),
    MetaField("min_count_by_regulator", (list,), "Per-regulator minimum counts ({tf, min_count})."),
    MetaField("compression_ratio", (numbers.Real,), "Solution compression ratio.", allow_none=True),
    MetaField("sampling_library_hash", (str,), "Stage-B stable hash for the sampled library."),
    MetaField("sampling_library_index", (int,), "Stage-B 1-based index of the sampled library."),
    MetaField("pad_used", (bool,), "Whether pad bases were applied."),
    MetaField("pad_bases", (int,), "Number of bases padded.", allow_none=True),
    MetaField("pad_end", (str,), "Pad end (5prime/3prime).", allow_none=True),
    MetaField("pad_literal", (str,), "Literal sequence used for pad bases.", allow_none=True),
    MetaField(
        "sequence_validation",
        (dict,),
        "Final-sequence validation summary ({validation_passed, violations}).",
    ),
    MetaField("gc_total", (numbers.Real,), "GC fraction of the final sequence."),
    MetaField("gc_core", (numbers.Real,), "GC fraction of the pre-pad core sequence."),
]

_FIELD_BY_NAME = {field.name: field for field in META_FIELDS}
_ALLOWED_INPUT_MODES = {"binding_sites", "sequence_library", "pwm_sampled", "plan_pool"}


def validate_metadata(meta: Mapping[str, Any]) -> None:
    missing = [field.name for field in META_FIELDS if field.required and field.name not in meta]
    if missing:
        raise ValueError(f"Missing metadata keys: {', '.join(missing)}")

    extra = set(meta.keys()) - set(_FIELD_BY_NAME.keys())
    if extra:
        raise ValueError(f"Unknown metadata keys: {', '.join(sorted(extra))}")

    for name, field in _FIELD_BY_NAME.items():
        if name not in meta:
            continue
        val = meta[name]
        if val is None:
            if field.allow_none:
                continue
            raise ValueError(f"Metadata field '{name}' is required and cannot be null")
        if not isinstance(val, field.types):
            expected = ", ".join(t.__name__ for t in field.types)
            raise TypeError(f"Metadata field '{name}' expected {expected}, got {type(val).__name__}")

    _validate_list_fields(meta)
    _validate_struct_fields(meta)


def _validate_list_fields(meta: Mapping[str, Any]) -> None:
    list_of_str = {
        "used_tfbs",
        "input_pwm_ids",
        "required_regulators",
    }
    for name in list_of_str:
        if name not in meta:
            continue
        values = meta[name]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"Metadata field '{name}' must be a list of strings")
        for item in values:
            if not isinstance(item, str):
                raise TypeError(f"Metadata field '{name}' must contain only strings")

    if "used_tfbs_detail" in meta:
        vals = meta["used_tfbs_detail"]
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise TypeError("Metadata field 'used_tfbs_detail' must be a list of dicts")
        input_mode = str(meta.get("input_mode") or "").strip().lower()
        for item in vals:
            if not isinstance(item, dict):
                raise TypeError("Metadata field 'used_tfbs_detail' must contain dict entries")
            part_kind = str(item.get("part_kind") or "tfbs").strip().lower()
            if part_kind == "tfbs":
                for key in (
                    "part_index",
                    "regulator",
                    "sequence",
                    "core_sequence",
                    "orientation",
                    "offset",
                    "offset_raw",
                    "pad_left",
                    "length",
                    "end",
                    "source",
                    "motif_id",
                    "tfbs_id",
                ):
                    if key not in item or item.get(key) is None:
                        raise ValueError(f"used_tfbs_detail tfbs entries must include '{key}'")
                for key in ("regulator", "sequence", "core_sequence", "source", "motif_id", "tfbs_id"):
                    if not str(item.get(key) or "").strip():
                        raise ValueError(f"used_tfbs_detail tfbs entries must use non-empty '{key}'")
                orientation = str(item.get("orientation") or "").strip().lower()
                if orientation not in {"fwd", "rev"}:
                    raise ValueError("used_tfbs_detail tfbs entries must use orientation 'fwd' or 'rev'")
                if input_mode == "pwm_sampled":
                    for key in (
                        "score_best_hit_raw",
                        "score_theoretical_max",
                        "score_relative_to_theoretical_max",
                        "rank_among_mined_positive",
                        "rank_among_selected",
                        "selection_policy",
                        "matched_start",
                        "matched_stop",
                        "matched_strand",
                    ):
                        if key not in item or item.get(key) is None:
                            raise ValueError(f"used_tfbs_detail tfbs entries for pwm_sampled must include '{key}'")
                    if not str(item.get("matched_strand") or "").strip():
                        raise ValueError(
                            "used_tfbs_detail tfbs entries for pwm_sampled must use non-empty matched_strand"
                        )
                continue
            if part_kind == "fixed_element":
                for key in ("role", "sequence", "offset_raw", "pad_left", "offset", "length", "end"):
                    if key not in item or item.get(key) is None:
                        raise ValueError(f"used_tfbs_detail fixed_element entries must include '{key}'")
                continue
            raise ValueError(f"used_tfbs_detail entries must use supported part_kind values; got '{part_kind}'")

    if "used_tf_counts" in meta:
        vals = meta["used_tf_counts"]
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise TypeError("Metadata field 'used_tf_counts' must be a list of dicts")
        for item in vals:
            if not isinstance(item, dict):
                raise TypeError("Metadata field 'used_tf_counts' must contain dict entries")
            if "tf" not in item or "count" not in item:
                raise ValueError("used_tf_counts entries must include 'tf' and 'count'")

    if "min_count_by_regulator" in meta:
        vals = meta["min_count_by_regulator"]
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise TypeError("Metadata field 'min_count_by_regulator' must be a list of dicts")
        for item in vals:
            if not isinstance(item, dict):
                raise TypeError("Metadata field 'min_count_by_regulator' must contain dict entries")
            if "tf" not in item or "min_count" not in item:
                raise ValueError("min_count_by_regulator entries must include 'tf' and 'min_count'")
            if not isinstance(item["tf"], str):
                raise TypeError("min_count_by_regulator.tf must be a string")
            if not isinstance(item["min_count"], int):
                raise TypeError("min_count_by_regulator.min_count must be an int")


def _validate_struct_fields(meta: Mapping[str, Any]) -> None:
    input_mode = str(meta.get("input_mode") or "").strip()
    if input_mode not in _ALLOWED_INPUT_MODES:
        allowed = ", ".join(sorted(_ALLOWED_INPUT_MODES))
        raise ValueError(f"Metadata field 'input_mode' must be one of: {allowed}")

    sampling_library_hash = str(meta.get("sampling_library_hash") or "").strip()
    if not sampling_library_hash:
        raise ValueError("Metadata field 'sampling_library_hash' is required and cannot be empty")

    if "sequence_validation" in meta:
        block = meta["sequence_validation"]
        if not isinstance(block, dict):
            raise TypeError("sequence_validation must be a dict")
        if not isinstance(block.get("validation_passed"), bool):
            raise TypeError("sequence_validation.validation_passed must be a bool")
        violations = block.get("violations")
        if not isinstance(violations, list):
            raise TypeError("sequence_validation.violations must be a list")
        for violation in violations:
            if not isinstance(violation, dict):
                raise TypeError("sequence_validation.violations entries must be dicts")
