from __future__ import annotations

from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS


def test_plot_registry_has_descriptions() -> None:
    for name, meta in PLOT_SPECS.items():
        assert "description" in meta, f"Missing description for plot '{name}'"
        assert str(meta["description"]).strip(), f"Empty description for plot '{name}'"
