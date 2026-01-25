"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_demo_config_selection_policy.py

Ensures the demo config hides selection policy for FIMO sampling.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.config import load_config


def _demo_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workspaces" / "demo_meme_two_tf" / "config.yaml"


def test_demo_config_hides_selection_policy_for_fimo() -> None:
    cfg_path = _demo_config_path()
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    for inp in cfg.inputs:
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        backend = str(getattr(sampling, "scoring_backend", "") or "").lower()
        if backend != "fimo":
            continue
        assert not hasattr(sampling, "selection_policy"), f"{inp.name} should not expose selection_policy"
