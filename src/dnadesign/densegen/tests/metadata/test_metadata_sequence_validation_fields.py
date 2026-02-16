"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/metadata/test_metadata_sequence_validation_fields.py

Metadata contract tests for promoter-detail and sequence-validation fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.metadata import build_metadata


class _DummySol:
    compression_ratio = None

    def __str__(self) -> str:
        return "visual"


def test_build_metadata_includes_promoter_and_validation_fields() -> None:
    meta = build_metadata(
        sol=_DummySol(),
        plan_name="demo_plan",
        tfbs_parts=["TF1:AAAA"],
        regulator_labels=["TF1"],
        library_for_opt=["AAAA"],
        fixed_elements=None,
        chosen_solver="CBC",
        solver_strategy="iterate",
        solver_time_limit_seconds=None,
        solver_threads=None,
        solver_strands="double",
        seq_len=10,
        actual_length=10,
        pad_meta={"used": False},
        sampling_meta={
            "achieved_length": 4,
            "relaxed_cap": False,
            "final_cap": None,
            "pool_strategy": "subsample",
            "library_size": 1,
            "library_sampling_strategy": "tf_balanced",
            "iterative_max_libraries": 1,
            "iterative_min_new_solutions": 0,
        },
        schema_version="2.9",
        created_at="2026-02-10T00:00:00+00:00",
        run_id="run_1",
        run_root=".",
        run_config_path="config.yaml",
        run_config_sha256="dummy",
        random_seed=1,
        policy_pad="off",
        policy_sampling="subsample",
        policy_solver="iterate",
        input_meta={
            "input_type": "binding_sites",
            "input_name": "demo_input",
            "input_source_names": ["demo_input"],
            "input_mode": "binding_sites",
            "input_pwm_ids": [],
        },
        fixed_elements_dump={"promoter_constraints": [], "side_biases": {"left": [], "right": []}},
        used_tfbs=["TF1:AAAA"],
        used_tfbs_detail=[{"tf": "TF1", "tfbs": "AAAA", "orientation": "fwd", "offset": 0}],
        used_tf_counts={"TF1": 1},
        used_tf_list=["TF1"],
        min_count_per_tf=0,
        covers_all_tfs_in_solution=True,
        required_regulators=[],
        min_required_regulators=None,
        min_count_by_regulator=None,
        covers_required_regulators=True,
        gc_total=0.5,
        gc_core=0.5,
        input_row_count=1,
        input_tf_count=1,
        input_tfbs_count=1,
        input_tf_tfbs_pair_count=1,
        sampling_fraction=1.0,
        sampling_fraction_pairs=1.0,
        sampling_library_index=1,
        sampling_library_hash="hash",
        dense_arrays_version=None,
        dense_arrays_version_source="unknown",
        solver_status=None,
        solver_objective=None,
        solver_solve_time_s=None,
    )
    assert "promoter_detail" in meta
    assert "sequence_validation" in meta
    assert meta["sequence_validation"]["validation_passed"] is True
    assert meta["sequence_validation"]["violations"] == []
