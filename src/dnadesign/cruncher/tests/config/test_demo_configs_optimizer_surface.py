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


def _demo_workspace_config_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    return [
        root / "demo_pairwise" / "configs" / "config.yaml",
        root / "demo_multitf" / "configs" / "config.yaml",
    ]


def test_demo_configs_enable_optimizer_surface() -> None:
    for config_path in _demo_workspace_config_paths():
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


def test_matrix_mode_demo_configs_do_not_define_site_window_lengths() -> None:
    for config_path in _demo_workspace_config_paths():
        cfg = load_config(config_path)
        assert cfg.catalog.pwm_source == "matrix"
        assert cfg.catalog.site_window_lengths == {}


def test_multitf_demo_pins_merged_discovery_source() -> None:
    config_path = Path(__file__).resolve().parents[2] / "workspaces" / "demo_multitf" / "configs" / "config.yaml"
    cfg = load_config(config_path)

    assert cfg.catalog.pwm_source == "matrix"
    assert cfg.catalog.combine_sites is True
    assert cfg.catalog.source_preference == ["demo_merged_meme_oops_multitf"]
    assert cfg.discover.source_id == "demo_merged_meme_oops_multitf"
    assert cfg.discover.tool == "meme"
    assert cfg.discover.meme_mod == "oops"


def test_core_demo_configs_pin_merged_meme_oops_sources() -> None:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    expected = {
        root / "demo_pairwise" / "configs" / "config.yaml": "demo_merged_meme_oops",
        root / "demo_multitf" / "configs" / "config.yaml": "demo_merged_meme_oops_multitf",
    }

    for config_path, source_id in expected.items():
        cfg = load_config(config_path)
        assert cfg.catalog.pwm_source == "matrix", f"{config_path} must sample against discovered motif matrices."
        assert cfg.catalog.combine_sites is True, f"{config_path} must merge TFBS sets before discovery."
        assert cfg.catalog.source_preference == [source_id], f"{config_path} must pin discovered source preference."
        assert cfg.discover.source_id == source_id, f"{config_path} discover.source_id must match catalog pin."
        assert cfg.discover.tool == "meme", f"{config_path} must use MEME for OOPS discovery."
        assert cfg.discover.meme_mod == "oops", f"{config_path} must run MEME in OOPS mode."


def test_demo_configs_use_tuned_gibbs_annealing_defaults() -> None:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    expected = {
        root / "demo_pairwise" / "configs" / "config.yaml": {
            "chains": 8,
            "cooling_kind": "piecewise",
            "final_beta": 24.0,
            "draws": 150000,
            "tune": 25000,
        },
        root / "demo_multitf" / "configs" / "config.yaml": {
            "chains": 8,
            "cooling_kind": "piecewise",
            "final_beta": 24.0,
            "draws": 150000,
            "tune": 25000,
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


def test_project_workspace_defaults_match_tuned_surface() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "workspaces"
        / "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs"
        / "configs"
        / "config.yaml"
    )
    payload = yaml.safe_load(config_path.read_text())
    cruncher = payload["cruncher"]
    sample = cruncher["sample"]

    assert sample["sequence_length"] == 18
    assert sample["budget"]["draws"] == 300000
    assert sample["budget"]["tune"] == 50000
    assert sample["optimizer"]["chains"] == 8
    assert sample["optimizer"]["cooling"]["kind"] == "piecewise"
    assert sample["optimizer"]["cooling"]["stages"][-1]["sweeps"] == sample["budget"]["draws"]
    assert sample["optimizer"]["cooling"]["stages"][-1]["beta"] == 24.0


def test_project_workspace_uses_matrix_meme_oops_discovery_contract() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "workspaces"
        / "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs"
        / "configs"
        / "config.yaml"
    )
    cfg = load_config(config_path)

    assert cfg.catalog.pwm_source == "matrix"
    assert cfg.catalog.combine_sites is True
    assert cfg.catalog.source_preference == ["project_merged_meme_oops_all_tfs"]
    assert cfg.discover.tool == "meme"
    assert cfg.discover.meme_mod == "oops"
    assert any(source.source_id == "demo_local_meme" for source in cfg.ingest.local_sources)


def test_demo_configs_use_modern_schema_keys() -> None:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    for config_path in (
        *(_demo_workspace_config_paths()),
        root / "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs" / "configs" / "config.yaml",
    ):
        payload = yaml.safe_load(config_path.read_text())
        cruncher = payload["cruncher"]
        assert "campaigns" not in cruncher, f"{config_path} must not include removed campaigns key."
        assert "campaign" not in cruncher, f"{config_path} must not include removed campaign key."

        discover = cruncher.get("discover") or {}
        assert "enabled" in discover, f"{config_path} should set discover.enabled explicitly."

        sample = cruncher["sample"]
        optimizer = sample["optimizer"]
        assert "early_stop" in optimizer, f"{config_path} should expose sample.optimizer.early_stop."
        elites_select = sample["elites"]["select"]
        assert "diversity" in elites_select, f"{config_path} should expose sample.elites.select.diversity."

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


def test_lexa_cpxr_local_meme_inputs_match_across_workspaces() -> None:
    root = Path(__file__).resolve().parents[2] / "workspaces"
    pairwise = root / "demo_pairwise" / "inputs" / "local_motifs"
    multitf = root / "demo_multitf" / "inputs" / "local_motifs"
    project = root / "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs" / "inputs" / "local_motifs"

    for tf in ("lexA", "cpxR"):
        pairwise_bytes = (pairwise / f"{tf}.txt").read_bytes()
        multitf_bytes = (multitf / f"{tf}.txt").read_bytes()
        project_bytes = (project / f"{tf}.txt").read_bytes()
        assert pairwise_bytes == multitf_bytes, f"{tf} local motif should be shared by demo_pairwise and demo_multitf."
        assert pairwise_bytes == project_bytes, f"{tf} local motif should be shared by demo_pairwise and project."
