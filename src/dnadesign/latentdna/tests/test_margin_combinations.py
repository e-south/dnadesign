"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_margin_combinations.py

Explicit margin-combination contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pyarrow as pa
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.build import _combine_margins


def test_margin_combinations_are_explicit_and_validate_columns():
    table = pa.table({"a": [0.1, -0.2], "b": [-0.4, 0.5]})
    assert _combine_margins(table, {}).equals(table)
    result = _combine_margins(
        table,
        {"joint": {"operation": "min", "columns": ["a", "b"]}, "best": {"operation": "max", "columns": ["a", "b"]}},
    )
    assert result["joint"].to_pylist() == pytest.approx([-0.4, -0.2])
    assert result["best"].to_pylist() == pytest.approx([0.1, 0.5])
    with pytest.raises(ContractViolationError, match="missing"):
        _combine_margins(table, {"joint": {"operation": "min", "columns": ["missing"]}})
    with pytest.raises(ContractViolationError, match="operation"):
        _combine_margins(table, {"joint": {"operation": "sum", "columns": ["a"]}})
