"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_reference_contrast_configuration.py

Reference contrasts use explicit cohort and reference selectors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars import preassay_reference


def test_reference_contrast_uses_declared_groups_and_reference_values() -> None:
    rows = [{"group": "a", "name": "a"}, {"group": "b", "name": "b"}, {"group": "reference", "name": "standard"}]
    matrix = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    values = preassay_reference._configured_reference_contrasts(
        rows,
        matrix,
        [
            {
                "cohort_column": "group",
                "target": "a",
                "background": "b",
                "reference_column": "name",
                "reference_values": ["standard"],
                "metric_id": "a_reference_margin",
            }
        ],
    )
    assert values == {"a_reference_margin": pytest.approx(1.0)}


def test_reference_contrast_rejects_missing_reference() -> None:
    with pytest.raises(ContractViolationError, match="reference selector matched no rows"):
        preassay_reference._configured_reference_contrasts(
            [{"group": "a"}, {"group": "b"}],
            np.eye(2, dtype=np.float32),
            [
                {
                    "cohort_column": "group",
                    "target": "a",
                    "background": "b",
                    "reference_column": "name",
                    "reference_values": ["standard"],
                    "metric_id": "margin",
                }
            ],
        )
