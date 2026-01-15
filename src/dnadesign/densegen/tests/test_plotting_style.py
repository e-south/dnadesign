from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dnadesign.densegen.src.viz.plotting import _apply_style


def test_seaborn_style_missing_raises(monkeypatch) -> None:
    fig, ax = plt.subplots()

    def _raise(_style: str) -> None:
        raise OSError("style not available")

    monkeypatch.setattr(plt.style, "use", _raise)

    with pytest.raises(ValueError):
        _apply_style(ax, {"seaborn_style": True})

    plt.close(fig)
