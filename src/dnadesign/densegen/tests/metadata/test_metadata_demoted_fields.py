"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/metadata/test_metadata_demoted_fields.py

Curated DenseGen metadata schema contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.densegen.src.core.metadata import build_metadata
from dnadesign.densegen.src.core.metadata_schema import META_FIELDS

EXPECTED_META_FIELDS = {
    "schema_version",
    "created_at",
    "run_id",
    "run_config_path",
    "length",
    "random_seed",
    "policy_sampling",
    "solver_backend",
    "solver_strands",
    "dense_arrays_version",
    "plan",
    "tf_list",
    "tfbs_parts",
    "used_tfbs",
    "used_tfbs_detail",
    "used_tf_counts",
    "covers_all_tfs_in_solution",
    "required_regulators",
    "min_count_by_regulator",
    "input_name",
    "input_mode",
    "input_pwm_ids",
    "input_tf_tfbs_pair_count",
    "sampling_fraction",
    "sampling_fraction_pairs",
    "fixed_elements",
    "visual",
    "compression_ratio",
    "library_size",
    "library_unique_tf_count",
    "library_unique_tfbs_count",
    "promoter_constraint",
    "sampling_pool_strategy",
    "sampling_library_size",
    "sampling_library_strategy",
    "sampling_iterative_max_libraries",
    "sampling_library_index",
    "pad_used",
    "pad_bases",
    "pad_end",
    "gc_total",
    "gc_core",
}

EXPECTED_DENSEGEN_REGISTRY_COLUMNS = [f"densegen__{name}" for name in sorted(EXPECTED_META_FIELDS)] + [
    "densegen__npz_ref",
    "densegen__npz_sha256",
    "densegen__npz_bytes",
    "densegen__npz_fields",
]


class _DummySol:
    compression_ratio = None

    def __str__(self) -> str:
        return "visual"


def _build_demo_meta() -> dict:
    input_meta = {
        "input_type": "plan_pool",
        "input_name": "plan_pool__demo",
        "input_source_names": ["demo_input"],
        "input_mode": "binding_sites",
        "input_pwm_ids": [],
    }
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


def test_metadata_schema_matches_curated_contract() -> None:
    schema_fields = {field.name for field in META_FIELDS}
    assert schema_fields == EXPECTED_META_FIELDS


def test_build_metadata_emits_curated_contract() -> None:
    meta = _build_demo_meta()
    assert set(meta.keys()) == EXPECTED_META_FIELDS


def test_densegen_registry_columns_match_curated_contract() -> None:
    reg_path = Path("src/dnadesign/usr/datasets/registry.yaml")
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    densegen_cols = [
        str(item.get("name"))
        for item in reg["namespaces"]["densegen"]["columns"]
        if str(item.get("name", "")).startswith("densegen__")
    ]
    assert sorted(densegen_cols) == sorted(EXPECTED_DENSEGEN_REGISTRY_COLUMNS)
