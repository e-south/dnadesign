"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_orchestrator_solution_requirements.py

Unit coverage for Stage-B solution requirement evaluation in the orchestrator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.pipeline.stage_b_runtime_checks import _evaluate_solution_requirements


@pytest.mark.parametrize(
    ("used_tf_counts", "min_count_per_tf", "required_regulators", "min_count_by_regulator", "expected"),
    [
        (
            {"TF1": 1},
            1,
            [],
            {},
            (
                False,
                True,
                "min_count_per_tf",
                {"min_count_per_tf": 1, "missing_tfs": ["TF2"]},
            ),
        ),
        (
            {"TF1": 1},
            0,
            ["TF1", "TF2"],
            {},
            (
                True,
                False,
                "required_regulators",
                {"required_regulators": ["TF1", "TF2"], "missing_tfs": ["TF2"]},
            ),
        ),
        (
            {"TF1": 1},
            0,
            [],
            {"TF1": 2},
            (
                True,
                True,
                "min_count_by_regulator",
                {"min_count_by_regulator": [{"tf": "TF1", "min_count": 2, "found": 1}]},
            ),
        ),
        (
            {"TF1": 1, "TF2": 1},
            1,
            ["TF1", "TF2"],
            {"TF1": 1},
            (True, True, None, {}),
        ),
    ],
)
def test_evaluate_solution_requirements(
    used_tf_counts: dict[str, int],
    min_count_per_tf: int,
    required_regulators: list[str],
    min_count_by_regulator: dict[str, int],
    expected: tuple[bool, bool, str | None, dict],
) -> None:
    result = _evaluate_solution_requirements(
        min_count_per_tf=min_count_per_tf,
        tf_list_from_library=["TF1", "TF2"],
        required_regulators=required_regulators,
        plan_min_count_by_regulator=min_count_by_regulator,
        used_tf_counts=used_tf_counts,
    )
    assert result == expected
