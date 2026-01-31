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
    MetaField("run_root", (str,), "Resolved run root path (densegen.run.root)."),
    MetaField("run_config_path", (str,), "Run config path (relative to run root when possible)."),
    MetaField("run_config_sha256", (str,), "SHA256 hash of the run config file."),
    MetaField("length", (int,), "Actual output sequence length."),
    MetaField("random_seed", (int,), "Global RNG seed used for the run."),
    MetaField("policy_pad", (str,), "Pad policy applied (off|strict|adaptive)."),
    MetaField("policy_sampling", (str,), "Stage-B sampling policy label (pool strategy)."),
    MetaField("policy_solver", (str,), "Solver policy label (strategy name)."),
    MetaField("solver_backend", (str,), "Solver backend name (null when approximate).", allow_none=True),
    MetaField("solver_strategy", (str,), "Solver strategy used."),
    MetaField(
        "solver_time_limit_seconds",
        (numbers.Real,),
        "Solver time limit in seconds.",
        allow_none=True,
    ),
    MetaField(
        "solver_threads",
        (int,),
        "Solver thread count.",
        allow_none=True,
    ),
    MetaField("solver_strands", (str,), "Solver strands mode (single|double)."),
    MetaField("dense_arrays_version", (str,), "dense-arrays package version.", allow_none=True),
    MetaField("dense_arrays_version_source", (str,), "dense-arrays version source (installed|lock|pyproject|unknown)."),
    MetaField("solver_status", (str,), "Solver status label.", allow_none=True),
    MetaField("solver_objective", (numbers.Real,), "Solver objective value.", allow_none=True),
    MetaField("solver_solve_time_s", (numbers.Real,), "Solver solve time in seconds.", allow_none=True),
    MetaField("plan", (str,), "Plan item name."),
    MetaField("tf_list", (list,), "All TFs present in the Stage-B sampled library."),
    MetaField("tfbs_parts", (list,), "TF:TFBS strings used to build the Stage-B library."),
    MetaField("used_tfbs", (list,), "TF:TFBS strings used in the final sequence."),
    MetaField(
        "used_tfbs_detail",
        (list,),
        "Per-placement detail: tf/tfbs/motif_id/tfbs_id/orientation/offset (offset uses final coordinates).",
    ),
    MetaField("used_tf_counts", (list,), "Per-TF placement counts ({tf, count})."),
    MetaField("used_tf_list", (list,), "TFs used in the final sequence."),
    MetaField("covers_all_tfs_in_solution", (bool,), "Whether min_count_per_tf coverage was satisfied."),
    MetaField("min_count_per_tf", (int,), "Coverage threshold for per-TF placements."),
    MetaField("required_regulators", (list,), "Regulators required per plan item."),
    MetaField("min_required_regulators", (int,), "Minimum required regulators (k-of-n).", allow_none=True),
    MetaField("min_count_by_regulator", (list,), "Per-regulator minimum counts ({tf, min_count})."),
    MetaField("covers_required_regulators", (bool,), "Whether required regulators were present in the solution."),
    MetaField("input_type", (str,), "Input source type."),
    MetaField("input_name", (str,), "Input source name."),
    MetaField("input_path", (str,), "Resolved path for file-based inputs.", allow_none=True),
    MetaField("input_dataset", (str,), "USR dataset name for USR inputs.", allow_none=True),
    MetaField("input_root", (str,), "Resolved root for USR inputs.", allow_none=True),
    MetaField("input_mode", (str,), "Input mode (binding_sites | sequence_library | pwm_sampled)."),
    MetaField("input_pwm_ids", (list,), "Stage-A PWM motif IDs used for sampling (pwm_* inputs)."),
    MetaField("input_row_count", (int,), "Total rows/sequences in the input pool."),
    MetaField("input_tf_count", (int,), "Unique TF count in the input pool (binding_sites only)."),
    MetaField("input_tfbs_count", (int,), "Unique TFBS/sequence count in the input pool."),
    MetaField(
        "input_tf_tfbs_pair_count",
        (int,),
        "Unique (TF, TFBS) pair count in the input pool (binding_sites only).",
        allow_none=True,
    ),
    MetaField(
        "sampling_fraction",
        (numbers.Real,),
        "Stage-B unique TFBS count in the sampled library divided by input_tfbs_count (1.0 when pool_strategy=full).",
        allow_none=True,
    ),
    MetaField(
        "sampling_fraction_pairs",
        (numbers.Real,),
        "Stage-B unique TF:TFBS pair count in the sampled library divided by input_tf_tfbs_pair_count.",
        allow_none=True,
    ),
    MetaField("input_pwm_strategy", (str,), "Stage-A PWM sampling strategy.", allow_none=True),
    MetaField("input_pwm_mining_batch_size", (int,), "Stage-A PWM mining batch size (FIMO).", allow_none=True),
    MetaField(
        "input_pwm_mining_log_every_batches",
        (int,),
        "Stage-A PWM mining log frequency (batches).",
        allow_none=True,
    ),
    MetaField("input_pwm_budget_mode", (str,), "Stage-A PWM mining budget mode.", allow_none=True),
    MetaField("input_pwm_budget_candidates", (int,), "Stage-A PWM mining fixed candidates.", allow_none=True),
    MetaField(
        "input_pwm_budget_target_tier_fraction",
        (numbers.Real,),
        "Stage-A PWM mining target tier fraction.",
        allow_none=True,
    ),
    MetaField("input_pwm_budget_max_candidates", (int,), "Stage-A PWM mining max candidates.", allow_none=True),
    MetaField(
        "input_pwm_budget_max_seconds",
        (numbers.Real,),
        "Stage-A PWM mining max seconds.",
        allow_none=True,
    ),
    MetaField("input_pwm_budget_min_candidates", (int,), "Stage-A PWM mining min candidates.", allow_none=True),
    MetaField(
        "input_pwm_budget_growth_factor",
        (numbers.Real,),
        "Stage-A PWM mining growth factor.",
        allow_none=True,
    ),
    MetaField("input_pwm_bgfile", (str,), "Stage-A PWM background model path (FIMO).", allow_none=True),
    MetaField("input_pwm_keep_all_candidates_debug", (bool,), "Stage-A PWM FIMO debug TSV enabled.", allow_none=True),
    MetaField("input_pwm_include_matched_sequence", (bool,), "Stage-A PWM matched-sequence capture.", allow_none=True),
    MetaField("input_pwm_n_sites", (int,), "Stage-A PWM sampling n_sites.", allow_none=True),
    MetaField("input_pwm_uniqueness_key", (str,), "Stage-A PWM uniqueness key.", allow_none=True),
    MetaField("input_pwm_selection_policy", (str,), "Stage-A PWM selection policy.", allow_none=True),
    MetaField("input_pwm_selection_alpha", (numbers.Real,), "Stage-A PWM selection alpha.", allow_none=True),
    MetaField(
        "input_pwm_selection_pool_min_score_norm",
        (numbers.Real,),
        "Stage-A PWM selection pool min score norm.",
        allow_none=True,
    ),
    MetaField(
        "input_pwm_selection_pool_max_candidates",
        (int,),
        "Stage-A PWM selection pool max candidates.",
        allow_none=True,
    ),
    MetaField(
        "input_pwm_selection_pool_relevance_norm",
        (str,),
        "Stage-A PWM selection pool relevance normalization.",
        allow_none=True,
    ),
    MetaField("fixed_elements", (dict,), "Fixed-element constraints (promoters + side biases)."),
    MetaField("visual", (str,), "ASCII visual layout of placements."),
    MetaField("compression_ratio", (numbers.Real,), "Solution compression ratio.", allow_none=True),
    MetaField("library_size", (int,), "Number of motifs in the Stage-B sampled library."),
    MetaField("library_unique_tf_count", (int,), "Unique TF count in the Stage-B sampled library."),
    MetaField("library_unique_tfbs_count", (int,), "Unique TFBS count in the Stage-B sampled library."),
    MetaField("sequence_length", (int,), "Target sequence length."),
    MetaField("promoter_constraint", (str,), "Primary promoter constraint name (if set).", allow_none=True),
    MetaField("sampling_target_length", (int,), "Stage-B target library length for sampling."),
    MetaField("sampling_achieved_length", (int,), "Stage-B achieved library length for sampling."),
    MetaField("sampling_relaxed_cap", (bool,), "Stage-B sampling caps were relaxed."),
    MetaField("sampling_final_cap", (int,), "Stage-B final per-TF cap after relaxation.", allow_none=True),
    MetaField("sampling_pool_strategy", (str,), "Stage-B sampling pool strategy (full|subsample|iterative_subsample)."),
    MetaField("sampling_library_size", (int,), "Stage-B configured library size for subsampling."),
    MetaField("sampling_library_strategy", (str,), "Stage-B library sampling strategy.", allow_none=True),
    MetaField("sampling_iterative_max_libraries", (int,), "Stage-B max libraries for iterative subsampling."),
    MetaField("sampling_iterative_min_new_solutions", (int,), "Stage-B min new solutions per library."),
    MetaField("sampling_library_index", (int,), "Stage-B 1-based index of the sampled library."),
    MetaField("sampling_library_hash", (str,), "Stage-B SHA256 hash of the sampled library contents."),
    MetaField("pad_used", (bool,), "Whether pad bases were applied."),
    MetaField("pad_bases", (int,), "Number of bases padded.", allow_none=True),
    MetaField("pad_end", (str,), "Pad end (5prime/3prime).", allow_none=True),
    MetaField("pad_gc_mode", (str,), "Pad GC mode (off|range|target).", allow_none=True),
    MetaField("pad_gc_min", (numbers.Real,), "Final GC minimum used.", allow_none=True),
    MetaField("pad_gc_max", (numbers.Real,), "Final GC maximum used.", allow_none=True),
    MetaField("pad_gc_target_min", (numbers.Real,), "Requested GC minimum.", allow_none=True),
    MetaField("pad_gc_target_max", (numbers.Real,), "Requested GC maximum.", allow_none=True),
    MetaField("pad_gc_actual", (numbers.Real,), "Observed GC fraction.", allow_none=True),
    MetaField("pad_relaxed", (bool,), "Whether GC bounds were relaxed.", allow_none=True),
    MetaField("pad_relaxed_reason", (str,), "Reason GC bounds were relaxed.", allow_none=True),
    MetaField("pad_attempts", (int,), "Number of attempts to fill pad.", allow_none=True),
    MetaField("gc_total", (numbers.Real,), "GC fraction of the final sequence."),
    MetaField("gc_core", (numbers.Real,), "GC fraction of the pre-pad core sequence."),
]

_FIELD_BY_NAME = {field.name: field for field in META_FIELDS}


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
        "tf_list",
        "tfbs_parts",
        "used_tfbs",
        "used_tf_list",
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
        for item in vals:
            if not isinstance(item, dict):
                raise TypeError("Metadata field 'used_tfbs_detail' must contain dict entries")
            for key in ("tf", "tfbs", "orientation", "offset"):
                if key not in item:
                    raise ValueError(f"used_tfbs_detail entries must include '{key}'")

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
    if "fixed_elements" not in meta:
        return
    fixed = meta["fixed_elements"]
    if not isinstance(fixed, dict):
        raise TypeError("fixed_elements must be a dict")
    if "promoter_constraints" not in fixed or "side_biases" not in fixed:
        raise ValueError("fixed_elements must include 'promoter_constraints' and 'side_biases'")
    pcs = fixed.get("promoter_constraints")
    if not isinstance(pcs, list):
        raise TypeError("fixed_elements.promoter_constraints must be a list")
    for pc in pcs:
        if not isinstance(pc, dict):
            raise TypeError("fixed_elements.promoter_constraints entries must be dicts")
    sb = fixed.get("side_biases")
    if not isinstance(sb, dict):
        raise TypeError("fixed_elements.side_biases must be a dict")
    for side in ("left", "right"):
        vals = sb.get(side)
        if not isinstance(vals, list):
            raise TypeError(f"fixed_elements.side_biases.{side} must be a list")
