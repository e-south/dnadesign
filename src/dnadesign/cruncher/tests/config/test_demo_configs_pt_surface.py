"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_demo_configs_pt_surface.py

Validate that all demo configs expose the gibbs annealing optimizer surface.

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


def test_demo_configs_enable_optimizer_surface() -> None:
    for config_path in _workspace_config_paths():
        cfg = load_config(config_path)
        sample_cfg = cfg.sample
        assert sample_cfg is not None

        overrides = sample_cfg.moves.overrides
        assert overrides.adaptive_weights.enabled, f"{config_path} must enable moves.overrides.adaptive_weights"
        assert overrides.proposal_adapt.enabled, f"{config_path} must enable moves.overrides.proposal_adapt"

        optimizer_cfg = sample_cfg.optimizer
        assert optimizer_cfg.kind == "gibbs_anneal", f"{config_path} must set sample.optimizer.kind=gibbs_anneal"
        assert optimizer_cfg.chains >= 1, f"{config_path} must set sample.optimizer.chains >= 1"
        assert optimizer_cfg.cooling.kind == "linear", f"{config_path} must use linear cooling for demos"
        assert optimizer_cfg.cooling.beta_start is not None
        assert optimizer_cfg.cooling.beta_end is not None
        assert optimizer_cfg.cooling.beta_end >= optimizer_cfg.cooling.beta_start


def test_densegen_demo_uses_tighter_site_windows() -> None:
    config_path = Path(__file__).resolve().parents[2] / "workspaces" / "densegen_prep_three_tf" / "config.yaml"
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.catalog.site_window_lengths["lexA"] <= 16
    assert cfg.catalog.site_window_lengths["cpxR"] <= 16
    max_window = max(cfg.catalog.site_window_lengths.values())
    assert cfg.sample.sequence_length >= max_window
