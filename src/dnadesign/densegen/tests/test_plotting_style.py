from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dnadesign.densegen.src.viz.plotting import _apply_style


def test_seaborn_style_missing_raises() -> None:
    fig, ax = plt.subplots()
    saved = dict(plt.style.library)
    try:
        for name in ("seaborn-v0_8-ticks", "seaborn-ticks"):
            plt.style.library.pop(name, None)
        with pytest.raises(ValueError):
            _apply_style(ax, {"seaborn_style": True})
    finally:
        plt.style.library.clear()
        plt.style.library.update(saved)
        plt.close(fig)
