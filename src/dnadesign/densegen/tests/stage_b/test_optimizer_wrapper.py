from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.optimizer import DenseArrayOptimizer
from dnadesign.densegen.src.adapters.optimizer.dense_arrays import DenseArraysAdapter, _apply_solver_controls
from dnadesign.densegen.src.core.pipeline.stage_b_runtime_callbacks import (
    _apply_solver_min_total_sites_constraint,
)


def test_promoter_constraint_name_is_ignored() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={"promoter_constraints": [{"name": "sigma70", "upstream": "TTGACA", "downstream": "TATAAT"}]},
    )
    opt.get_optimizer_instance()


def test_promoter_constraint_variant_ids_are_accepted() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={
            "promoter_constraints": [
                {
                    "name": "sigma70",
                    "upstream": "TTGACA",
                    "downstream": "TATAAT",
                    "upstream_variant_id": "consensus",
                    "downstream_variant_id": "consensus",
                }
            ]
        },
    )
    opt.get_optimizer_instance()


def test_promoter_constraint_variant_ids_must_be_non_empty_strings() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={
            "promoter_constraints": [
                {
                    "name": "sigma70",
                    "upstream": "TTGACA",
                    "downstream": "TATAAT",
                    "upstream_variant_id": "",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="upstream_variant_id must be a non-empty string"):
        opt.get_optimizer_instance()


def test_solver_time_limit_applies() -> None:
    class _DummyModel:
        def __init__(self) -> None:
            self.time_limit_ms = None
            self.threads = None

        def SetTimeLimit(self, ms: int) -> None:
            self.time_limit_ms = ms

        def SetNumThreads(self, threads: int) -> None:
            self.threads = threads

    class _DummyOptimizer:
        def __init__(self) -> None:
            self.model = None

        def build_model(self, solver="CBC", solver_options=None) -> None:
            self.model = _DummyModel()

    opt = _DummyOptimizer()
    _apply_solver_controls(opt, solver_attempt_timeout_seconds=2, threads=3)
    opt.build_model()
    assert opt.model.time_limit_ms == 2000
    assert opt.model.threads == 3


def test_solver_enforces_min_total_sites_when_target_is_unreachable() -> None:
    adapter = DenseArraysAdapter()
    run_baseline = adapter.build(
        library=["AAA", "CCC"],
        sequence_length=3,
        solver="CBC",
        strategy="optimal",
        fixed_elements=None,
        strands="single",
        regulator_by_index=["TF1", "TF2"],
        required_regulators=None,
        min_count_by_regulator=None,
        min_required_regulators=None,
        solver_attempt_timeout_seconds=2.0,
        solver_threads=1,
        extra_label=None,
    )
    baseline_solution = next(iter(run_baseline.generator))
    assert baseline_solution.sequence

    run_with_constraint = adapter.build(
        library=["AAA", "CCC"],
        sequence_length=3,
        solver="CBC",
        strategy="optimal",
        fixed_elements=None,
        strands="single",
        regulator_by_index=["TF1", "TF2"],
        required_regulators=None,
        min_count_by_regulator=None,
        min_required_regulators=None,
        solver_attempt_timeout_seconds=2.0,
        solver_threads=1,
        extra_label=None,
    )
    _apply_solver_min_total_sites_constraint(
        optimizer=run_with_constraint.optimizer,
        min_total_sites=2,
        countable_indices=[0, 1],
        source_label="demo",
        plan_name="default",
    )
    with pytest.raises(ValueError, match="No feasible solution was found"):
        _ = next(iter(run_with_constraint.generator))
