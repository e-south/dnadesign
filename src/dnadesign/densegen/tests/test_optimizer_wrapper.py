from __future__ import annotations

from dnadesign.densegen.src.adapters.optimizer import DenseArrayOptimizer
from dnadesign.densegen.src.adapters.optimizer.dense_arrays import _apply_solver_controls


def test_promoter_constraint_name_is_ignored() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={"promoter_constraints": [{"name": "sigma70", "upstream": "TTGACA", "downstream": "TATAAT"}]},
    )
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
    _apply_solver_controls(opt, time_limit_seconds=2, threads=3)
    opt.build_model()
    assert opt.model.time_limit_ms == 2000
    assert opt.model.threads == 3
