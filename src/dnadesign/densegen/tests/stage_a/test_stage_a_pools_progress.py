"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_pools_progress.py

Stage-A pool preparation progress toggles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import dnadesign.densegen.src.core.pipeline.stage_a_pools as stage_a_pools
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.utils import logging_utils


def _demo_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces" / "demo_sampling_baseline" / "config.yaml"


def test_prepare_stage_a_pools_sets_progress_style(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _demo_config_path()
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen

    prev_enabled = logging_utils.is_progress_enabled()
    prev_style = logging_utils.get_progress_style()
    logging_utils.set_progress_enabled(False)
    logging_utils.set_progress_style("stream")

    outputs_root = tmp_path / "outputs"
    pool_dir = outputs_root / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_manifest = pool_dir / "pool_manifest.json"
    pool_manifest.write_text("{}")
    events_path = outputs_root / "meta" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    def _fake_build_pool_artifact(**_kwargs):
        return None, {}

    def _fake_load_pool_data(_pool_dir: Path):
        return None, {}

    monkeypatch.setattr(stage_a_pools, "build_pool_artifact", _fake_build_pool_artifact)
    monkeypatch.setattr(stage_a_pools, "load_pool_data", _fake_load_pool_data)
    monkeypatch.setattr(stage_a_pools, "_active_input_names", lambda _plan_items: {"demo_pwm"})

    try:
        stage_a_pools.prepare_stage_a_pools(
            cfg=cfg,
            cfg_path=loaded.path,
            run_root=tmp_path,
            outputs_root=outputs_root,
            rng=None,
            build_stage_a=True,
            candidate_logging=False,
            candidates_dir=outputs_root / "pools" / "candidates",
            plan_items=[],
            events_path=events_path,
            run_id=str(cfg.run.id),
            deps=None,
        )
        expected_style, _ = logging_utils.resolve_progress_style(
            str(cfg.logging.progress_style),
            stdout=sys.stdout,
        )
        assert logging_utils.get_progress_style() == expected_style
        assert logging_utils.is_progress_enabled() is (expected_style in {"stream", "screen"})
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)
