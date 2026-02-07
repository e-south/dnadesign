"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_demo_configs_pt_surface.py

Validate that all demo configs expose the hardened PT adaptation surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.config.load import load_config


def _workspace_config_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    return [
        root / "demo_basics_two_tf" / "config.yaml",
        root / "demo_campaigns_multi_tf" / "config.yaml",
        root / "densegen_prep_three_tf" / "config.yaml",
    ]


def test_demo_configs_enable_adaptive_pt_surface() -> None:
    for config_path in _workspace_config_paths():
        cfg = load_config(config_path)
        sample_cfg = cfg.sample
        assert sample_cfg is not None

        overrides = sample_cfg.moves.overrides
        assert overrides.adaptive_weights.enabled, f"{config_path} must enable moves.overrides.adaptive_weights"
        assert overrides.proposal_adapt.enabled, f"{config_path} must enable moves.overrides.proposal_adapt"

        adapt_cfg = sample_cfg.pt.adapt
        assert adapt_cfg.enabled, f"{config_path} must enable sample.pt.adapt"
        assert adapt_cfg.strict, f"{config_path} must enable sample.pt.adapt.strict"
        assert adapt_cfg.saturation_windows >= 3, (
            f"{config_path} must set sample.pt.adapt.saturation_windows to at least 3"
        )
