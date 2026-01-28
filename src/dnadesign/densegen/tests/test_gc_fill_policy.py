from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.postprocess import generate_pad


def test_pad_strict_infeasible_range() -> None:
    with pytest.raises(ValueError):
        generate_pad(
            length=1,
            mode="strict",
            gc_mode="range",
            gc_min=0.4,
            gc_max=0.6,
            gc_target=0.5,
            gc_tolerance=0.1,
            gc_min_pad_length=4,
        )


def test_pad_adaptive_relaxes_short_pad() -> None:
    seq, info = generate_pad(
        length=1,
        mode="adaptive",
        gc_mode="range",
        gc_min=0.4,
        gc_max=0.6,
        gc_target=0.5,
        gc_tolerance=0.1,
        gc_min_pad_length=4,
    )
    assert len(seq) == 1
    assert info["relaxed"] is True
    assert info["relaxed_reason"] == "short_pad"
    assert info["final_gc_min"] == 0.0
    assert info["final_gc_max"] == 1.0
