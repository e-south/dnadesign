from __future__ import annotations

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.classification_metrics import (
    average_precision,
    dual_joint_margin,
    roc_auc,
)


def test_binary_metrics_give_half_credit_to_tied_positive_negative_scores() -> None:
    labels = np.asarray([0, 1], dtype=np.int8)
    scores = np.asarray([0.5, 0.5], dtype=np.float64)

    assert roc_auc(labels, scores) == 0.5
    assert average_precision(labels, scores) == 0.5


def test_binary_metrics_reject_non_finite_scores() -> None:
    labels = np.asarray([0, 1], dtype=np.int8)
    scores = np.asarray([0.1, np.nan], dtype=np.float64)

    with pytest.raises(ContractViolationError, match="finite score values"):
        roc_auc(labels, scores)


def test_dual_joint_margin_rejects_unaligned_vectors() -> None:
    with pytest.raises(ContractViolationError, match="aligned score vectors"):
        dual_joint_margin(np.asarray([0.1, 0.2]), np.asarray([0.1]))
