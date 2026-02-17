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

BINDING_SITES_ONLY_FIELDS = {
    "input_tf_tfbs_pair_count",
    "sampling_fraction_pairs",
}


def _apply_retention_policy(meta: dict) -> None:
    input_mode = str(meta.get("input_mode") or "")
    if input_mode != "pwm_sampled":
        meta["input_pwm_ids"] = []
    if input_mode != "binding_sites":
        for field in BINDING_SITES_ONLY_FIELDS:
            meta[field] = None


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
    promoter_detail: dict | None = None,
    sequence_validation: dict | None = None,
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
    if promoter_detail is None:
        promoter_detail = {"placements": []}
    if sequence_validation is None:
        sequence_validation = {"validation_passed": True, "violations": []}
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
        "run_config_path": run_config_path,
        "length": int(actual_length),
        "random_seed": int(random_seed),
        "policy_sampling": policy_sampling,
        "solver_backend": chosen_solver,
        "solver_strands": solver_strands,
        "dense_arrays_version": dense_arrays_version,
        "plan": plan_name,
        "tf_list": tf_list,
        "tfbs_parts": tfbs_parts or [],
        "used_tfbs": used_tfbs,
        "used_tfbs_detail": used_tfbs_detail,
        "used_tf_counts": used_tf_counts_list,
        "covers_all_tfs_in_solution": bool(covers_all_tfs_in_solution),
        "input_name": input_meta.get("input_name"),
        "input_mode": input_meta.get("input_mode"),
        "input_pwm_ids": input_meta.get("input_pwm_ids") or [],
        "input_tf_tfbs_pair_count": input_tf_tfbs_pair_count,
        "sampling_fraction": sampling_fraction,
        "sampling_fraction_pairs": sampling_fraction_pairs,
        "fixed_elements": fixed_elements_dump,
        "visual": str(sol),
        "compression_ratio": getattr(sol, "compression_ratio", None),
        "library_size": len(library_for_opt),
        "library_unique_tf_count": len(tf_list),
        "library_unique_tfbs_count": library_unique_tfbs,
        "promoter_constraint": _promoter_constraint_name(fixed_elements),
        "sampling_pool_strategy": sampling_meta.get("pool_strategy"),
        "sampling_library_size": sampling_meta.get("library_size"),
        "sampling_library_strategy": sampling_meta.get("library_sampling_strategy"),
        "sampling_iterative_max_libraries": sampling_meta.get("iterative_max_libraries"),
        "sampling_library_hash": str(sampling_library_hash),
        "sampling_library_index": int(sampling_library_index),
        "required_regulators": required_regulators,
        "min_count_by_regulator": min_count_by_regulator_list,
        "pad_used": pad_meta.get("used", False),
        "pad_bases": pad_meta.get("bases"),
        "pad_end": pad_meta.get("end"),
        "gc_total": gc_total,
        "gc_core": gc_core,
        "promoter_detail": promoter_detail,
        "sequence_validation": sequence_validation,
    }
    _apply_retention_policy(meta)
    validate_metadata(meta)
    return meta
