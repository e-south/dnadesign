from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.cli import _candidate_logging_enabled
from dnadesign.densegen.src.config import load_config


def test_candidate_logging_enabled_toggle() -> None:
    cfg_path = Path("src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml")
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen

    assert _candidate_logging_enabled(cfg) is False

    cfg.inputs[0].sampling.keep_all_candidates_debug = True
    assert _candidate_logging_enabled(cfg) is True
