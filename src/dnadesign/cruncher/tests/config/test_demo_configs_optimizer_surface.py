"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_demo_configs_optimizer_surface.py

Validate that all demo configs expose the gibbs annealing optimizer surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

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
        assert not overrides.adaptive_weights.enabled, (
            f"{config_path} must disable moves.overrides.adaptive_weights for stable demo tails."
        )
        assert overrides.proposal_adapt.enabled, f"{config_path} must enable moves.overrides.proposal_adapt"
        assert overrides.move_schedule.enabled, f"{config_path} must enable moves.overrides.move_schedule"

        optimizer_cfg = sample_cfg.optimizer
        assert optimizer_cfg.kind == "gibbs_anneal", f"{config_path} must set sample.optimizer.kind=gibbs_anneal"
        assert optimizer_cfg.chains >= 1, f"{config_path} must set sample.optimizer.chains >= 1"
        if optimizer_cfg.cooling.kind == "linear":
            assert optimizer_cfg.cooling.beta_start is not None
            assert optimizer_cfg.cooling.beta_end is not None
            assert optimizer_cfg.cooling.beta_end >= optimizer_cfg.cooling.beta_start
        else:
            assert optimizer_cfg.cooling.kind == "piecewise"
            assert optimizer_cfg.cooling.stages


def test_densegen_demo_uses_tighter_site_windows() -> None:
    config_path = Path(__file__).resolve().parents[2] / "workspaces" / "densegen_prep_three_tf" / "config.yaml"
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.catalog.site_window_lengths["lexA"] <= 16
    assert cfg.catalog.site_window_lengths["cpxR"] <= 16
    max_window = max(cfg.catalog.site_window_lengths.values())
    assert cfg.sample.sequence_length >= max_window


def test_demo_configs_use_tuned_gibbs_annealing_defaults() -> None:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    expected = {
        root / "demo_basics_two_tf" / "config.yaml": {
            "chains": 6,
            "cooling_kind": "piecewise",
            "final_beta": 14.0,
            "draws": 150000,
            "tune": 25000,
        },
        root / "demo_campaigns_multi_tf" / "config.yaml": {
            "chains": 10,
            "cooling_kind": "piecewise",
            "final_beta": 18.0,
            "draws": 12000,
            "tune": 4000,
        },
        root / "densegen_prep_three_tf" / "config.yaml": {
            "chains": 8,
            "cooling_kind": "piecewise",
            "final_beta": 18.0,
            "draws": 8000,
            "tune": 2500,
        },
    }
    for config_path, values in expected.items():
        cfg = load_config(config_path)
        assert cfg.sample is not None
        budget = cfg.sample.budget
        optimizer_cfg = cfg.sample.optimizer
        assert optimizer_cfg.chains == values["chains"], f"{config_path} must use tuned demo chain count."
        assert optimizer_cfg.cooling.kind == values["cooling_kind"]
        assert budget.draws == values["draws"]
        assert budget.tune == values["tune"]
        assert optimizer_cfg.cooling.stages[-1].sweeps == budget.draws
        assert optimizer_cfg.cooling.stages[-1].beta == values["final_beta"]


def test_demo_configs_use_modern_schema_keys() -> None:
    for config_path in _workspace_config_paths():
        payload = yaml.safe_load(config_path.read_text())
        cruncher = payload["cruncher"]

        discover = cruncher.get("discover") or {}
        assert "enabled" in discover, f"{config_path} should set discover.enabled explicitly."

        sample = cruncher["sample"]
        optimizer = sample["optimizer"]
        assert "early_stop" in optimizer, f"{config_path} should expose sample.optimizer.early_stop."

        analysis = cruncher["analysis"]
        assert "trajectory_chain_overlay" in analysis, f"{config_path} should use analysis.trajectory_chain_overlay."
        assert "trajectory_sweep_mode" in analysis, f"{config_path} should set analysis.trajectory_sweep_mode."
        assert "trajectory_slot_overlay" not in analysis, f"{config_path} must not use trajectory_slot_overlay."

        catalog = cruncher["catalog"]
        assert "pwm_window_lengths" not in catalog, (
            f"{config_path} must not include removed catalog.pwm_window_lengths."
        )
        assert "pwm_window_strategy" not in catalog, (
            f"{config_path} must not include removed catalog.pwm_window_strategy."
        )
