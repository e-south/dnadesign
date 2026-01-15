from __future__ import annotations

from dnadesign.densegen.src.adapters.optimizer import DenseArrayOptimizer


def test_promoter_constraint_name_is_ignored() -> None:
    opt = DenseArrayOptimizer(
        library=["TTGACA", "TATAAT", "AT"],
        sequence_length=60,
        fixed_elements={"promoter_constraints": [{"name": "sigma70", "upstream": "TTGACA", "downstream": "TATAAT"}]},
    )
    opt.get_optimizer_instance()
