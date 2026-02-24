"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_compute_config.py

Validate compute + sequence length schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.config.moves import resolve_move_config


def _base_config() -> dict:
    return {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": ".cruncher", "pwm_source": "matrix"},
            "sample": {
                "seed": 7,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 3},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                },
                "moves": {"profile": "balanced"},
                "output": {
                    "save_trace": False,
                    "save_sequences": True,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
            },
        }
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_compute_config_loads(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config())
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.sequence_length == 12
    assert cfg.sample.budget.tune == 1
    assert cfg.sample.budget.draws == 3
    assert cfg.sample.objective.softmin.schedule == "fixed"
    assert cfg.sample.objective.softmin.beta_end == pytest.approx(6.0)
    assert cfg.sample.optimizer.kind == "gibbs_anneal"
    assert cfg.sample.optimizer.chains == 1
    assert cfg.sample.optimizer.cooling.kind == "linear"
    assert cfg.sample.optimizer.cooling.beta_start == pytest.approx(0.20)
    assert cfg.sample.optimizer.cooling.beta_end == pytest.approx(4.0)


def test_sample_optimizer_kind_is_required_when_optimizer_block_is_set(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["optimizer"] = {"chains": 2}
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.optimizer.kind"):
        load_config(config_path)


def test_balanced_move_defaults_are_stability_oriented(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config())
    cfg = load_config(config_path)
    assert cfg.sample is not None
    move_cfg = resolve_move_config(cfg.sample.moves)

    assert move_cfg.block_len_range == (2, 6)
    assert move_cfg.multi_k_range == (2, 3)
    assert move_cfg.insertion_consensus_prob == pytest.approx(0.35)
    assert move_cfg.move_probs["S"] == pytest.approx(0.85)
    assert move_cfg.move_probs["B"] == pytest.approx(0.07)
    assert move_cfg.move_probs["M"] == pytest.approx(0.04)
    assert move_cfg.move_probs["I"] == pytest.approx(0.04)


@pytest.mark.parametrize("value", [0, -1])
def test_draws_requires_positive(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["budget"]["draws"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.budget.draws"):
        load_config(config_path)


@pytest.mark.parametrize("value", [-1])
def test_tune_requires_non_negative(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["budget"]["tune"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.budget.tune"):
        load_config(config_path)


def test_analysis_particle_trajectory_fields_load(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {
        "run_selector": "latest",
        "trajectory_stride": 5,
        "trajectory_scatter_scale": "llr",
        "trajectory_scatter_retain_elites": True,
        "trajectory_sweep_y_column": "objective_scalar",
        "trajectory_sweep_mode": "all",
        "trajectory_particle_alpha_min": 0.2,
        "trajectory_particle_alpha_max": 0.9,
        "trajectory_chain_overlay": False,
    }
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.analysis is not None
    assert cfg.analysis.trajectory_stride == 5
    assert cfg.analysis.trajectory_scatter_scale == "llr"
    assert cfg.analysis.trajectory_scatter_retain_elites is True
    assert cfg.analysis.trajectory_sweep_y_column == "objective_scalar"
    assert cfg.analysis.trajectory_sweep_mode == "all"


def test_analysis_legacy_trajectory_slot_overlay_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {"trajectory_slot_overlay": False}
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_sample_output_save_sequences_false_is_allowed(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["output"]["save_sequences"] = False
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.output.save_sequences is False


def test_sample_output_random_baseline_n_loads(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["output"]["random_baseline_n"] = 64
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.output.random_baseline_n == 64


def test_sample_output_random_baseline_is_enabled_by_default(tmp_path: Path) -> None:
    payload = _base_config()
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.output.save_random_baseline is True
    assert cfg.sample.output.random_baseline_n == 10_000


@pytest.mark.parametrize("value", [0, -1])
def test_sample_output_random_baseline_n_requires_positive(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["output"]["random_baseline_n"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.output.random_baseline_n"):
        load_config(config_path)


def test_analysis_trajectory_defaults_prefer_best_so_far_and_dense_lineage(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {"enabled": True}
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.analysis is not None
    assert cfg.analysis.trajectory_scatter_scale == "llr"
    assert cfg.analysis.trajectory_scatter_retain_elites is True
    assert cfg.analysis.trajectory_stride == 5
    assert cfg.analysis.trajectory_sweep_y_column == "objective_scalar"
    assert cfg.analysis.trajectory_sweep_mode == "best_so_far"
    assert cfg.analysis.trajectory_summary_overlay is False
    assert cfg.analysis.trajectory_particle_alpha_max == pytest.approx(0.45)
    assert cfg.analysis.mmr_sweep.enabled is False
    assert cfg.analysis.fimo_compare.enabled is False


@pytest.mark.parametrize(
    "stale_key, stale_value",
    [
        ("trajectory_scatter_mode", "best_progression"),
        ("trajectory_scatter_elite_context_radius", 12),
        ("trajectory_scatter_objective_column", "objective_scalar"),
        ("trajectory_scatter_elite_collision_strategy", "none_or_count"),
        ("trajectory_scatter_elite_link_mode", "exact_only"),
    ],
)
def test_analysis_stale_scatter_keys_are_rejected(tmp_path: Path, stale_key: str, stale_value: object) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {stale_key: stale_value}
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_analysis_trajectory_summary_overlay_toggle_loads(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {"enabled": True, "trajectory_summary_overlay": False}
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.analysis is not None
    assert cfg.analysis.trajectory_summary_overlay is False


def test_analysis_fimo_compare_toggle_loads(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {"run_selector": "latest", "fimo_compare": {"enabled": True}}
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.analysis is not None
    assert cfg.analysis.fimo_compare.enabled is True


def test_elites_removed_filter_key_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["elites"]["filter"] = {"min_per_tf_norm": 0.6}
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_elites_removed_select_alpha_key_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["elites"]["select"]["alpha"] = 0.7
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_move_overrides_load_gibbs_inertia_and_freeze_controls(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["moves"]["overrides"] = {
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "gibbs_inertia": {
            "enabled": True,
            "kind": "fixed",
            "p_stay_end": 0.95,
        },
        "adaptive_weights": {
            "enabled": True,
            "freeze_after_sweep": 100,
        },
        "proposal_adapt": {
            "enabled": True,
            "freeze_after_beta": 2.0,
        },
    }
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)

    assert cfg.sample is not None
    assert cfg.sample.moves.overrides is not None
    assert cfg.sample.moves.overrides.gibbs_inertia is not None
    assert cfg.sample.moves.overrides.gibbs_inertia.enabled is True
    assert cfg.sample.moves.overrides.gibbs_inertia.kind == "fixed"
    assert cfg.sample.moves.overrides.gibbs_inertia.p_stay_end == pytest.approx(0.95)
    assert cfg.sample.moves.overrides.adaptive_weights is not None
    assert cfg.sample.moves.overrides.adaptive_weights.freeze_after_sweep == 100
    assert cfg.sample.moves.overrides.proposal_adapt is not None
    assert cfg.sample.moves.overrides.proposal_adapt.freeze_after_beta == pytest.approx(2.0)
