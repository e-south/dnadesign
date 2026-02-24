"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_optimizer_kind_resolution.py

Validate strict optimizer-kind resolution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.core.optimizers.kinds import resolve_optimizer_kind


def test_resolve_optimizer_kind_accepts_gibbs_anneal() -> None:
    assert resolve_optimizer_kind("gibbs_anneal", context="sample.optimizer.kind") == "gibbs_anneal"


def test_resolve_optimizer_kind_rejects_missing_kind() -> None:
    with pytest.raises(ValueError, match="sample.optimizer.kind"):
        resolve_optimizer_kind(None, context="sample.optimizer.kind")
