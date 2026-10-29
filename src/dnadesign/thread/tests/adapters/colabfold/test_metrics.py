"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/colabfold/test_metrics.py

Tests for ColabFold metric helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np

from dnadesign.thread.adapters.colabfold.metrics import ca_rmsd


def test_ca_rmsd_superposes_rotated_translated_row_vectors() -> None:
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mobile = reference @ rotation + np.asarray([10.0, -4.0, 2.5], dtype=float)

    rmsd = ca_rmsd(mobile, reference)

    assert rmsd is not None
    assert math.isclose(rmsd, 0.0, abs_tol=1e-9)
