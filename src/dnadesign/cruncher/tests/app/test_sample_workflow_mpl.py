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
                    "filter": {"min_per_tf_norm": None, "require_all_tfs": True, "pwm_sum_min": 0.0},
                    "select": {"alpha": 0.85, "pool_size": "auto"},
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


def test_run_sample_calls_mpl_cache_setup_only_when_trace_enabled(tmp_path, monkeypatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=False)))
    cfg = load_config(config_path)

    calls: list[Path] = []

    def _fake_ensure_mpl_cache(path: Path) -> Path:
        calls.append(path)
        return path

    monkeypatch.setattr(sample_workflow, "ensure_mpl_cache", _fake_ensure_mpl_cache)
    monkeypatch.setattr(sample_workflow, "_lockmap_for", lambda cfg, config_path: {})
    monkeypatch.setattr(sample_workflow, "target_statuses", lambda **kwargs: [])
    monkeypatch.setattr(sample_workflow, "has_blocking_target_errors", lambda statuses: False)
    monkeypatch.setattr(
        sample_workflow,
        "_run_sample_for_set",
        lambda *args, **kwargs: tmp_path / "results" / "sample" / "dummy",
    )

    sample_workflow.run_sample(cfg, config_path)
    assert calls == []

    config_path.write_text(yaml.safe_dump(_config_payload(catalog_root=catalog_root, save_trace=True)))
    cfg = load_config(config_path)
    sample_workflow.run_sample(cfg, config_path)
    assert len(calls) == 1
