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
    MetaField("schema_version", (str,), "DenseGen schema version (e.g., 2.1)."),
    MetaField("created_at", (str,), "UTC ISO8601 timestamp for record creation."),
    MetaField("run_id", (str,), "Run identifier (densegen.run.id)."),
    MetaField("run_root", (str,), "Resolved run root path (densegen.run.root)."),
    MetaField("run_config_path", (str,), "Run config path (relative to run root when possible)."),
    MetaField("run_config_sha256", (str,), "SHA256 hash of the run config file."),
    MetaField("length", (int,), "Actual output sequence length."),
    MetaField("random_seed", (int,), "Global RNG seed used for the run."),
    MetaField("policy_gc_fill", (str,), "Gap-fill policy applied (off|strict|adaptive)."),
    MetaField("policy_sampling", (str,), "Sampling policy label (pool strategy)."),
    MetaField("policy_solver", (str,), "Solver policy label (strategy name)."),
    MetaField("solver_backend", (str,), "Solver backend name (null when approximate).", allow_none=True),
    MetaField("solver_strategy", (str,), "Solver strategy used."),
    MetaField("solver_options", (list,), "Solver options list."),
    MetaField("solver_strands", (str,), "Solver strands mode (single|double)."),
    MetaField("dense_arrays_version", (str,), "dense-arrays package version.", allow_none=True),
    MetaField("dense_arrays_version_source", (str,), "dense-arrays version source (installed|lock|pyproject|unknown)."),
    MetaField("solver_status", (str,), "Solver status label.", allow_none=True),
    MetaField("solver_objective", (numbers.Real,), "Solver objective value.", allow_none=True),
    MetaField("solver_solve_time_s", (numbers.Real,), "Solver solve time in seconds.", allow_none=True),
    MetaField("plan", (str,), "Plan item name."),
    MetaField("tf_list", (list,), "All TFs present in the sampled library."),
    MetaField("tfbs_parts", (list,), "TF:TFBS strings used to build the library."),
    MetaField("used_tfbs", (list,), "TF:TFBS strings used in the final sequence."),
    MetaField(
        "used_tfbs_detail",
        (list,),
        "Per-placement detail: tf/tfbs/orientation/offset (offset uses final sequence coordinates).",
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
    MetaField("input_pwm_ids", (list,), "PWM motif IDs used for sampling (pwm_* inputs)."),
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
        "Unique TFBS count in the sampled library divided by input_tfbs_count (1.0 when pool_strategy=full).",
        allow_none=True,
    ),
    MetaField(
        "sampling_fraction_pairs",
        (numbers.Real,),
        "Unique TF:TFBS pair count in the sampled library divided by input_tf_tfbs_pair_count.",
        allow_none=True,
    ),
    MetaField("input_pwm_strategy", (str,), "PWM sampling strategy.", allow_none=True),
    MetaField("input_pwm_scoring_backend", (str,), "PWM scoring backend (densegen|fimo).", allow_none=True),
    MetaField("input_pwm_score_threshold", (numbers.Real,), "PWM score threshold.", allow_none=True),
    MetaField("input_pwm_score_percentile", (numbers.Real,), "PWM score percentile.", allow_none=True),
    MetaField("input_pwm_pvalue_threshold", (numbers.Real,), "PWM p-value threshold (FIMO).", allow_none=True),
    MetaField("input_pwm_pvalue_bins", (list,), "PWM p-value bins (FIMO).", allow_none=True),
    MetaField(
        "input_pwm_pvalue_bin_ids",
        (list,),
        "Deprecated: selected p-value bin indices (use input_pwm_mining_retain_bin_ids).",
        allow_none=True,
    ),
    MetaField("input_pwm_mining_batch_size", (int,), "PWM mining batch size (FIMO).", allow_none=True),
    MetaField("input_pwm_mining_max_batches", (int,), "PWM mining max batches (FIMO).", allow_none=True),
    MetaField("input_pwm_mining_max_seconds", (numbers.Real,), "PWM mining max seconds (FIMO).", allow_none=True),
    MetaField(
        "input_pwm_mining_retain_bin_ids",
        (list,),
        "PWM mining retained p-value bin indices (FIMO).",
        allow_none=True,
    ),
    MetaField(
        "input_pwm_mining_log_every_batches",
        (int,),
        "PWM mining log frequency (batches).",
        allow_none=True,
    ),
    MetaField("input_pwm_selection_policy", (str,), "PWM selection policy (FIMO).", allow_none=True),
    MetaField("input_pwm_bgfile", (str,), "PWM background model path (FIMO).", allow_none=True),
    MetaField("input_pwm_keep_all_candidates_debug", (bool,), "PWM FIMO debug TSV enabled.", allow_none=True),
    MetaField("input_pwm_include_matched_sequence", (bool,), "PWM matched-sequence capture.", allow_none=True),
    MetaField("input_pwm_n_sites", (int,), "PWM sampling n_sites.", allow_none=True),
    MetaField("input_pwm_oversample_factor", (int,), "PWM sampling oversample factor.", allow_none=True),
    MetaField("fixed_elements", (dict,), "Fixed-element constraints (promoters + side biases)."),
    MetaField("visual", (str,), "ASCII visual layout of placements."),
    MetaField("compression_ratio", (numbers.Real,), "Solution compression ratio.", allow_none=True),
    MetaField("library_size", (int,), "Number of motifs in the sampled library."),
    MetaField("library_unique_tf_count", (int,), "Unique TF count in the sampled library."),
    MetaField("library_unique_tfbs_count", (int,), "Unique TFBS count in the sampled library."),
    MetaField("sequence_length", (int,), "Target sequence length."),
    MetaField("promoter_constraint", (str,), "Primary promoter constraint name (if set).", allow_none=True),
    MetaField("sampling_target_length", (int,), "Target library length for sampling."),
    MetaField("sampling_achieved_length", (int,), "Achieved library length for sampling."),
    MetaField("sampling_relaxed_cap", (bool,), "Whether sampling caps were relaxed."),
    MetaField("sampling_final_cap", (int,), "Final per-TF cap after relaxation.", allow_none=True),
    MetaField("sampling_pool_strategy", (str,), "Sampling pool strategy (full|subsample|iterative_subsample)."),
    MetaField("sampling_library_size", (int,), "Configured library size for subsampling."),
    MetaField("sampling_library_strategy", (str,), "Library sampling strategy.", allow_none=True),
    MetaField("sampling_iterative_max_libraries", (int,), "Max libraries for iterative subsampling."),
    MetaField("sampling_iterative_min_new_solutions", (int,), "Min new solutions per library."),
    MetaField("sampling_library_index", (int,), "1-based index of the sampled library."),
    MetaField("sampling_library_hash", (str,), "SHA256 hash of the sampled library contents."),
    MetaField("gap_fill_used", (bool,), "Whether gap fill was applied."),
    MetaField("gap_fill_bases", (int,), "Number of bases filled.", allow_none=True),
    MetaField("gap_fill_end", (str,), "Gap fill end (5prime/3prime).", allow_none=True),
    MetaField("gap_fill_gc_min", (numbers.Real,), "Final GC minimum used.", allow_none=True),
    MetaField("gap_fill_gc_max", (numbers.Real,), "Final GC maximum used.", allow_none=True),
    MetaField("gap_fill_gc_target_min", (numbers.Real,), "Requested GC minimum.", allow_none=True),
    MetaField("gap_fill_gc_target_max", (numbers.Real,), "Requested GC maximum.", allow_none=True),
    MetaField("gap_fill_gc_actual", (numbers.Real,), "Observed GC fraction.", allow_none=True),
    MetaField("gap_fill_relaxed", (bool,), "Whether GC bounds were relaxed.", allow_none=True),
    MetaField("gap_fill_attempts", (int,), "Number of attempts to fill gap.", allow_none=True),
    MetaField("gc_total", (numbers.Real,), "GC fraction of the final sequence."),
    MetaField("gc_core", (numbers.Real,), "GC fraction of the pre-gap-fill core sequence."),
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
        "solver_options",
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

    if "input_pwm_pvalue_bins" in meta:
        vals = meta["input_pwm_pvalue_bins"]
        if vals is not None:
            if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
                raise TypeError("Metadata field 'input_pwm_pvalue_bins' must be a list of numbers")
            for item in vals:
                if not isinstance(item, numbers.Real):
                    raise TypeError("Metadata field 'input_pwm_pvalue_bins' must contain only numbers")

    if "input_pwm_pvalue_bin_ids" in meta:
        vals = meta["input_pwm_pvalue_bin_ids"]
        if vals is not None:
            if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
                raise TypeError("Metadata field 'input_pwm_pvalue_bin_ids' must be a list of integers")
            for item in vals:
                if not isinstance(item, int):
                    raise TypeError("Metadata field 'input_pwm_pvalue_bin_ids' must contain only integers")

    if "input_pwm_mining_retain_bin_ids" in meta:
        vals = meta["input_pwm_mining_retain_bin_ids"]
        if vals is not None:
            if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
                raise TypeError("Metadata field 'input_pwm_mining_retain_bin_ids' must be a list of integers")
            for item in vals:
                if not isinstance(item, int):
                    raise TypeError("Metadata field 'input_pwm_mining_retain_bin_ids' must contain only integers")

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
