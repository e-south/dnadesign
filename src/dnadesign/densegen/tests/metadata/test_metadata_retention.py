"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/metadata/test_metadata_retention.py

Retention and sparsity guard tests for DenseGen output metadata fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.metadata import build_metadata


class _DummySol:
    compression_ratio = None

    def __str__(self) -> str:
        return "visual"


def _build_for_input_mode(input_meta: dict) -> dict:
    return build_metadata(
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
        input_meta=input_meta,
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


def _input_meta_with_pwm_values(input_mode: str) -> dict:
    return {
        "input_type": "binding_sites",
        "input_name": "demo_input",
        "input_source_names": ["demo_input"],
        "input_mode": input_mode,
        "input_pwm_ids": ["MOTIF_A"],
    }


def test_build_metadata_clears_pwm_only_fields_for_non_pwm_mode() -> None:
    meta = _build_for_input_mode(_input_meta_with_pwm_values("binding_sites"))
    assert meta["input_pwm_ids"] == []


def test_build_metadata_keeps_pwm_fields_for_pwm_mode() -> None:
    input_meta = _input_meta_with_pwm_values("pwm_sampled")
    meta = _build_for_input_mode(input_meta)
    assert meta["input_pwm_ids"] == ["MOTIF_A"]
