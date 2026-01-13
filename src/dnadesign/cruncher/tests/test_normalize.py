"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_normalize.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.cruncher.ingest.normalize import normalize_prob_matrix


def test_normalize_prob_matrix_tolerates_rounding() -> None:
    matrix = [[0.25, 0.25, 0.25, 0.249999]]

    norm = normalize_prob_matrix(matrix)

    assert abs(sum(norm[0]) - 1.0) < 1e-9


def test_normalize_prob_matrix_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        normalize_prob_matrix([[0.1, 0.1, 0.1, 0.1]])
