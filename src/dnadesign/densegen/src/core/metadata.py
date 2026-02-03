"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/metadata.py

Centralized metadata derivation for DenseGen outputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from .metadata_schema import validate_metadata


def _promoter_constraint_name(fixed_elements) -> str | None:
    if fixed_elements is None:
        return None
    if hasattr(fixed_elements, "promoter_constraints"):
        pcs = getattr(fixed_elements, "promoter_constraints") or []
    else:
        pcs = (fixed_elements or {}).get("promoter_constraints") or []
    if not pcs:
        return None
    pc0 = pcs[0]
    if hasattr(pc0, "name"):
        return getattr(pc0, "name")
    if isinstance(pc0, dict):
        return pc0.get("name")
    return None


def build_metadata(
    *,
    sol,
    plan_name: str,
    tfbs_parts: List[str],
    regulator_labels: List[str],
    library_for_opt: List[str],
    fixed_elements,
    chosen_solver: str | None,
    solver_strategy: str,
    solver_time_limit_seconds: float | None,
    solver_threads: int | None,
    solver_strands: str,
    seq_len: int,
    actual_length: int,
    pad_meta: dict,
    sampling_meta: dict,
    schema_version: str,
    created_at: str,
    run_id: str,
    run_root: str,
    run_config_path: str,
    run_config_sha256: str,
    random_seed: int,
    policy_pad: str,
    policy_sampling: str,
    policy_solver: str,
    input_meta: dict,
    fixed_elements_dump: dict,
    used_tfbs: List[str],
    used_tfbs_detail: List[dict],
    used_tf_counts: dict,
    used_tf_list: List[str],
    min_count_per_tf: int,
    covers_all_tfs_in_solution: bool,
    required_regulators: List[str],
    min_required_regulators: int | None,
    min_count_by_regulator: dict[str, int] | None,
    covers_required_regulators: bool,
    gc_total: float,
    gc_core: float,
    input_row_count: int,
    input_tf_count: int,
    input_tfbs_count: int,
    input_tf_tfbs_pair_count: int | None,
    sampling_fraction: float | None,
    sampling_fraction_pairs: float | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
) -> dict:
    tf_list = sorted(set(regulator_labels or []))
    library_unique_tfbs = len(set(library_for_opt)) if library_for_opt else 0
    used_tf_counts_list = [
        {"tf": str(tf), "count": int(n)} for tf, n in sorted(used_tf_counts.items(), key=lambda kv: kv[0])
    ]
    min_count_by_regulator_list = [
        {"tf": str(tf), "min_count": int(n)}
        for tf, n in sorted((min_count_by_regulator or {}).items(), key=lambda kv: kv[0])
    ]
    meta = {
        "schema_version": schema_version,
        "created_at": created_at,
        "run_id": run_id,
        "run_root": run_root,
        "run_config_path": run_config_path,
        "run_config_sha256": run_config_sha256,
        "length": int(actual_length),
        "random_seed": int(random_seed),
        "policy_pad": policy_pad,
        "policy_sampling": policy_sampling,
        "policy_solver": policy_solver,
        "solver_backend": chosen_solver,
        "solver_strategy": solver_strategy,
        "solver_time_limit_seconds": solver_time_limit_seconds,
        "solver_threads": solver_threads,
        "solver_strands": solver_strands,
        "dense_arrays_version": dense_arrays_version,
        "dense_arrays_version_source": dense_arrays_version_source,
        "solver_status": solver_status,
        "solver_objective": solver_objective,
        "solver_solve_time_s": solver_solve_time_s,
        "plan": plan_name,
        "tf_list": tf_list,
        "tfbs_parts": tfbs_parts or [],
        "used_tfbs": used_tfbs,
        "used_tfbs_detail": used_tfbs_detail,
        "used_tf_counts": used_tf_counts_list,
        "used_tf_list": used_tf_list,
        "covers_all_tfs_in_solution": bool(covers_all_tfs_in_solution),
        "min_count_per_tf": int(min_count_per_tf),
        "input_type": input_meta.get("input_type"),
        "input_name": input_meta.get("input_name"),
        "input_path": input_meta.get("input_path"),
        "input_dataset": input_meta.get("input_dataset"),
        "input_root": input_meta.get("input_root"),
        "input_mode": input_meta.get("input_mode"),
        "input_pwm_ids": input_meta.get("input_pwm_ids") or [],
        "input_row_count": int(input_row_count),
        "input_tf_count": int(input_tf_count),
        "input_tfbs_count": int(input_tfbs_count),
        "input_tf_tfbs_pair_count": input_tf_tfbs_pair_count,
        "sampling_fraction": sampling_fraction,
        "sampling_fraction_pairs": sampling_fraction_pairs,
        "input_pwm_strategy": input_meta.get("input_pwm_strategy"),
        "input_pwm_mining_batch_size": input_meta.get("input_pwm_mining_batch_size"),
        "input_pwm_mining_log_every_batches": input_meta.get("input_pwm_mining_log_every_batches"),
        "input_pwm_budget_mode": input_meta.get("input_pwm_budget_mode"),
        "input_pwm_budget_candidates": input_meta.get("input_pwm_budget_candidates"),
        "input_pwm_budget_target_tier_fraction": input_meta.get("input_pwm_budget_target_tier_fraction"),
        "input_pwm_budget_max_candidates": input_meta.get("input_pwm_budget_max_candidates"),
        "input_pwm_budget_max_seconds": input_meta.get("input_pwm_budget_max_seconds"),
        "input_pwm_budget_min_candidates": input_meta.get("input_pwm_budget_min_candidates"),
        "input_pwm_budget_growth_factor": input_meta.get("input_pwm_budget_growth_factor"),
        "input_pwm_bgfile": input_meta.get("input_pwm_bgfile"),
        "input_pwm_keep_all_candidates_debug": input_meta.get("input_pwm_keep_all_candidates_debug"),
        "input_pwm_include_matched_sequence": input_meta.get("input_pwm_include_matched_sequence"),
        "input_pwm_n_sites": input_meta.get("input_pwm_n_sites"),
        "input_pwm_tier_fractions": input_meta.get("input_pwm_tier_fractions"),
        "input_pwm_uniqueness_key": input_meta.get("input_pwm_uniqueness_key"),
        "input_pwm_selection_policy": input_meta.get("input_pwm_selection_policy"),
        "input_pwm_selection_alpha": input_meta.get("input_pwm_selection_alpha"),
        "input_pwm_selection_pool_min_score_norm": input_meta.get("input_pwm_selection_pool_min_score_norm"),
        "input_pwm_selection_pool_max_candidates": input_meta.get("input_pwm_selection_pool_max_candidates"),
        "input_pwm_selection_pool_relevance_norm": input_meta.get("input_pwm_selection_pool_relevance_norm"),
        "fixed_elements": fixed_elements_dump,
        "visual": str(sol),
        "compression_ratio": getattr(sol, "compression_ratio", None),
        "library_size": len(library_for_opt),
        "library_unique_tf_count": len(tf_list),
        "library_unique_tfbs_count": library_unique_tfbs,
        "sequence_length": seq_len,
        "promoter_constraint": _promoter_constraint_name(fixed_elements),
        "sampling_achieved_length": sampling_meta.get("achieved_length"),
        "sampling_relaxed_cap": sampling_meta.get("relaxed_cap"),
        "sampling_final_cap": sampling_meta.get("final_cap"),
        "sampling_pool_strategy": sampling_meta.get("pool_strategy"),
        "sampling_library_size": sampling_meta.get("library_size"),
        "sampling_library_strategy": sampling_meta.get("library_sampling_strategy"),
        "sampling_iterative_max_libraries": sampling_meta.get("iterative_max_libraries"),
        "sampling_iterative_min_new_solutions": sampling_meta.get("iterative_min_new_solutions"),
        "sampling_library_index": int(sampling_library_index),
        "sampling_library_hash": sampling_library_hash,
        "required_regulators": required_regulators,
        "min_required_regulators": min_required_regulators,
        "min_count_by_regulator": min_count_by_regulator_list,
        "covers_required_regulators": bool(covers_required_regulators),
        "pad_used": pad_meta.get("used", False),
        "pad_bases": pad_meta.get("bases"),
        "pad_end": pad_meta.get("end"),
        "pad_gc_mode": pad_meta.get("gc_mode"),
        "pad_gc_min": pad_meta.get("gc_min"),
        "pad_gc_max": pad_meta.get("gc_max"),
        "pad_gc_target_min": pad_meta.get("gc_target_min"),
        "pad_gc_target_max": pad_meta.get("gc_target_max"),
        "pad_gc_actual": pad_meta.get("gc_actual"),
        "pad_relaxed": pad_meta.get("relaxed"),
        "pad_relaxed_reason": pad_meta.get("relaxed_reason"),
        "pad_attempts": pad_meta.get("attempts"),
        "gc_total": gc_total,
        "gc_core": gc_core,
    }
    validate_metadata(meta)
    return meta
