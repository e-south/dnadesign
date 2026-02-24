"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_sample_workflow_mpl.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

import dnadesign.cruncher.app.sample_workflow as sample_workflow
from dnadesign.cruncher.config.load import load_config


def _config_payload(*, catalog_root: Path, save_trace: bool) -> dict:
    return {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {
                "root": str(catalog_root.resolve()),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "sample": {
                "seed": 1,
                "sequence_length": 10,
                "budget": {"tune": 1, "draws": 1},
                "objective": {"bidirectional": True, "score_scale": "llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                },
                "output": {
                    "save_trace": save_trace,
                    "save_sequences": True,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
            },
        }
    }


def test_run_sample_calls_runtime_cache_setup_only_when_trace_enabled(tmp_path, monkeypatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=False)))
    cfg = load_config(config_path)

    mpl_calls: list[Path] = []
    arviz_calls: list[Path] = []

    def _fake_ensure_mpl_cache(path: Path) -> Path:
        mpl_calls.append(path)
        return path

    def _fake_ensure_arviz_data_dir(path: Path) -> Path:
        arviz_calls.append(path)
        return path

    monkeypatch.setattr(sample_workflow, "ensure_mpl_cache", _fake_ensure_mpl_cache)
    monkeypatch.setattr(sample_workflow, "ensure_arviz_data_dir", _fake_ensure_arviz_data_dir)
    monkeypatch.setattr(sample_workflow, "_lockmap_for", lambda cfg, config_path: {})
    monkeypatch.setattr(sample_workflow, "target_statuses", lambda **kwargs: [])
    monkeypatch.setattr(sample_workflow, "has_blocking_target_errors", lambda statuses: False)
    monkeypatch.setattr(
        sample_workflow,
        "_run_sample_for_set",
        lambda *args, **kwargs: tmp_path / "results" / "sample" / "dummy",
    )

    sample_workflow.run_sample(cfg, config_path)
    assert mpl_calls == []
    assert arviz_calls == []

    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=True)))
    cfg = load_config(config_path)
    sample_workflow.run_sample(cfg, config_path)
    assert len(mpl_calls) == 1
    assert len(arviz_calls) == 1


def test_run_sample_forwards_runtime_progress_controls(tmp_path, monkeypatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=False)))
    cfg = load_config(config_path)

    captured: list[dict[str, object]] = []

    def _fake_run_set(*args, **kwargs):
        captured.append({"progress_bar": kwargs.get("progress_bar"), "progress_every": kwargs.get("progress_every")})
        return tmp_path / "results" / "sample" / "dummy"

    monkeypatch.setattr(sample_workflow, "_lockmap_for", lambda cfg, config_path: {})
    monkeypatch.setattr(sample_workflow, "target_statuses", lambda **kwargs: [])
    monkeypatch.setattr(sample_workflow, "has_blocking_target_errors", lambda statuses: False)
    monkeypatch.setattr(sample_workflow, "_run_sample_for_set", _fake_run_set)

    sample_workflow.run_sample(cfg, config_path, progress_bar=False, progress_every=37)
    assert captured == [{"progress_bar": False, "progress_every": 37}]


def test_run_sample_forwards_run_index_registration_flag(tmp_path, monkeypatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=False)))
    cfg = load_config(config_path)

    captured: list[bool] = []

    def _fake_run_set(*args, **kwargs):
        captured.append(bool(kwargs.get("register_run_in_index", True)))
        return tmp_path / "results" / "sample" / "dummy"

    monkeypatch.setattr(sample_workflow, "_lockmap_for", lambda cfg, config_path: {})
    monkeypatch.setattr(sample_workflow, "target_statuses", lambda **kwargs: [])
    monkeypatch.setattr(sample_workflow, "has_blocking_target_errors", lambda statuses: False)
    monkeypatch.setattr(sample_workflow, "_run_sample_for_set", _fake_run_set)

    sample_workflow.run_sample(cfg, config_path, register_run_in_index=False)
    assert captured == [False]
