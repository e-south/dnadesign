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
        "created_at": "2026-01-14T00:00:00+00:00",
        "length": 10,
        "plan": "demo_plan",
        "input_name": "plan_pool__demo_plan",
        "input_mode": "plan_pool",
        "input_pwm_ids": [],
        "used_tfbs": ["lexA:AAA", "cpxR:CCC"],
        "used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "part_index": 0,
                "regulator": "lexA",
                "sequence": "AAA",
                "core_sequence": "AAA",
                "orientation": "fwd",
                "offset": 0,
                "offset_raw": 0,
                "pad_left": 0,
                "length": 3,
                "end": 3,
                "source": "demo",
                "motif_id": "motif_lexA",
                "tfbs_id": "tfbs_lexA_AAA",
            },
            {
                "part_kind": "tfbs",
                "part_index": 1,
                "regulator": "cpxR",
                "sequence": "CCC",
                "core_sequence": "CCC",
                "orientation": "fwd",
                "offset": 4,
                "offset_raw": 4,
                "pad_left": 0,
                "length": 3,
                "end": 7,
                "source": "demo",
                "motif_id": "motif_cpxR",
                "tfbs_id": "tfbs_cpxR_CCC",
            },
        ],
        "used_tf_counts": [{"tf": "lexA", "count": 1}, {"tf": "cpxR", "count": 1}],
        "library_unique_tf_count": 2,
        "library_unique_tfbs_count": 2,
        "covers_all_tfs_in_solution": True,
        "compression_ratio": None,
        "required_regulators": [],
        "min_count_by_regulator": [],
        "sampling_library_hash": library_hash,
        "sampling_library_index": library_index,
        "sequence_validation": {"validation_passed": True, "violations": []},
        "pad_used": False,
        "pad_bases": None,
        "pad_end": None,
        "pad_literal": None,
        "gc_total": 0.5,
        "gc_core": 0.5,
    }
