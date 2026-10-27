"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_scalar_common.py

Regression tests for scalar common LatentDNA.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.common import (
    _pairwise_cosine_distance_summary,
    _workspace_input_path,
)


def test_pairwise_cosine_distance_summary_caps_rows_deterministically() -> None:
    rng = np.random.default_rng(17)
    matrix = np.asarray(rng.normal(size=(12, 4)), dtype=np.float32)

    first = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=11)
    second = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=11)

    assert first.method == "seeded_row_sample_all_pairs"
    assert first.source_rows == 12
    assert first.evaluated_rows == 5
    assert first.pair_count == 10
    assert first.median == second.median
    assert first.iqr == second.iqr


def test_pairwise_cosine_distance_summary_uses_exact_pairs_under_cap() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    summary = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=17)

    assert summary.method == "exact_all_pairs"
    assert summary.source_rows == 3
    assert summary.evaluated_rows == 3
    assert summary.pair_count == 3
    assert not math.isnan(summary.median)


def test_workspace_input_path_rejects_paths_outside_allowed_roots(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    allowed = workspace_dir / "inputs" / "table.parquet"
    allowed.parent.mkdir()
    allowed.write_text("placeholder", encoding="utf-8")
    outside = tmp_path / "outside.parquet"
    outside.write_text("placeholder", encoding="utf-8")
    context = SimpleNamespace(workspace_dir=workspace_dir)

    assert _workspace_input_path(context, "inputs/table.parquet") == allowed.resolve()
    with pytest.raises(ContractViolationError, match="escapes allowed scalar.build roots"):
        _workspace_input_path(context, "../outside.parquet")
