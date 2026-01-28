"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_demo_config_selection_policy.py

Ensures the demo config declares Stage-A selection policy explicitly.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.config import load_config


def _demo_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workspaces" / "demo_meme_two_tf" / "config.yaml"


def test_demo_config_declares_selection_policy() -> None:
    cfg_path = _demo_config_path()
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    for inp in cfg.inputs:
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        selection = getattr(sampling, "selection", None)
        uniqueness = getattr(sampling, "uniqueness", None)
        assert selection is not None, f"{inp.name} should declare selection settings"
        assert selection.policy == "mmr", f"{inp.name} should use mmr selection in the demo"
        assert uniqueness is not None
        assert uniqueness.key == "core"
