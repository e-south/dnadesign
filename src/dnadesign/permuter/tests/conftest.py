"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/conftest.py

Deterministic evaluator fixture for inference-free Permuter tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.permuter.src.core import registry


class FixtureEvaluator:
    def __init__(self, **params):
        pass

    def score(self, sequences, **kwargs):
        return [float(len(sequence)) for sequence in sequences]


@pytest.fixture(autouse=True)
def fixture_evaluator(monkeypatch):
    evaluators = registry._builtins("evaluators")
    evaluators["test_fixture"] = FixtureEvaluator
    monkeypatch.setattr(registry, "_EVAL", evaluators)
