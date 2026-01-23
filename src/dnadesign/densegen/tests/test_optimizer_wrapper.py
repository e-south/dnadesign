from __future__ import annotations

import pytest

import dnadesign.densegen.src.adapters.optimizer.dense_arrays as dense_arrays_adapter
from dnadesign.densegen.src.adapters.optimizer import DenseArrayOptimizer


def test_promoter_constraint_name_is_ignored() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={"promoter_constraints": [{"name": "sigma70", "upstream": "TTGACA", "downstream": "TATAAT"}]},
    )
    opt.get_optimizer_instance()


def test_solver_time_limit_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyModel:
        def __init__(self) -> None:
            self.time_limit_ms = None
            self.threads = None

        def SetTimeLimit(self, ms: int) -> None:
            self.time_limit_ms = ms

        def SetNumThreads(self, threads: int) -> None:
            self.threads = threads

    class _DummyOptimizer:
        def __init__(self, library, sequence_length, strands="double") -> None:
            self.library = list(library)
            self.sequence_length = sequence_length
            self.strands = strands
            self.model = None

        def add_promoter_constraints(self, **_kwargs) -> None:
            return None

        def add_side_biases(self, **_kwargs) -> None:
            return None

        def add_regulator_constraints(self, *_args, **_kwargs) -> None:
            return None

        def build_model(self, solver="CBC", solver_options=None) -> None:
            self.model = _DummyModel()

        def solutions(self, solver="CBC", solver_options=None):
            if False:
                yield None

    monkeypatch.setattr(dense_arrays_adapter.da, "Optimizer", _DummyOptimizer)

    adapter = dense_arrays_adapter.DenseArraysAdapter()
    run = adapter.build(
        library=["AT"],
        sequence_length=10,
        solver="CBC",
        strategy="iterate",
        fixed_elements=None,
        solver_time_limit_seconds=2,
        solver_threads=3,
    )
    run.optimizer.build_model()
    assert run.optimizer.model.time_limit_ms == 2000
    assert run.optimizer.model.threads == 3
