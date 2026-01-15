from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.postprocess import random_fill


def test_gc_fill_strict_infeasible() -> None:
    with pytest.raises(ValueError):
        random_fill(length=1, gc_min=0.4, gc_max=0.6, mode="strict")


def test_gc_fill_adaptive_relaxes() -> None:
    seq, info = random_fill(length=1, gc_min=0.4, gc_max=0.6, mode="adaptive")
    assert len(seq) == 1
    assert info["relaxed"] is True
    assert info["final_gc_min"] == 0.0
    assert info["final_gc_max"] == 1.0
