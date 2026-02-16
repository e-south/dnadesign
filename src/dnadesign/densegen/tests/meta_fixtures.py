"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/meta_fixtures.py

Reusable metadata fixtures for DenseGen tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def output_meta(*, library_hash: str, library_index: int) -> dict:
    return {
        "schema_version": "2.9",
        "run_id": "demo",
        "run_config_path": "config.yaml",
        "created_at": "2026-01-14T00:00:00+00:00",
        "length": 10,
        "random_seed": 0,
        "policy_sampling": "subsample",
        "solver_backend": "CBC",
        "solver_strands": "double",
        "dense_arrays_version": None,
        "plan": "demo_plan",
        "tf_list": ["lexA", "cpxR"],
        "tfbs_parts": ["lexA:AAA", "cpxR:CCC"],
        "used_tfbs": ["lexA:AAA", "cpxR:CCC"],
        "used_tfbs_detail": [
            {"tf": "lexA", "tfbs": "AAA", "orientation": "fwd", "offset": 0},
            {"tf": "cpxR", "tfbs": "CCC", "orientation": "fwd", "offset": 4},
        ],
        "used_tf_counts": [{"tf": "lexA", "count": 1}, {"tf": "cpxR", "count": 1}],
        "covers_all_tfs_in_solution": True,
        "input_name": "plan_pool__demo_plan",
        "input_mode": "binding_sites",
        "input_pwm_ids": [],
        "input_tf_tfbs_pair_count": 1,
        "sampling_fraction": 0.5,
        "sampling_fraction_pairs": 0.5,
        "fixed_elements": {"promoter_constraints": [], "side_biases": {"left": [], "right": []}},
        "visual": "",
        "compression_ratio": None,
        "library_size": 2,
        "library_unique_tf_count": 2,
        "library_unique_tfbs_count": 2,
        "promoter_constraint": None,
        "promoter_detail": {"placements": []},
        "sequence_validation": {"validation_passed": True, "violations": []},
        "sampling_pool_strategy": "subsample",
        "sampling_library_size": 2,
        "sampling_library_strategy": "tf_balanced",
        "sampling_iterative_max_libraries": 1,
        "sampling_library_index": library_index,
        "sampling_library_hash": library_hash,
        "required_regulators": [],
        "min_count_by_regulator": [],
        "pad_used": False,
        "pad_bases": None,
        "pad_end": None,
        "gc_total": 0.5,
        "gc_core": 0.5,
    }
