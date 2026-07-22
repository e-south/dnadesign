"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_config_manifest_pinned_labels.py

Configuration contracts for manifest-pinned observed-label sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.core.utils import ConfigError


def _payload() -> dict:
    return {
        "schema_version": "opal.campaign.v3",
        "ownership": {
            "owner_scope": "study_campaign",
            "study_id": "stress_promoter",
            "dataset_id": "promoter_candidates",
            "portable": False,
        },
        "campaign": {"name": "Promoter campaign", "slug": "promoter", "workdir": "."},
        "data": {
            "location": {"kind": "usr", "path": "./usr/datasets", "dataset": "promoter_candidates"},
            "x_column_name": "X",
            "y_column_name": "response_window_vector",
            "y_expected_length": 8,
        },
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {
            "name": "vector_from_table_v1",
            "params": {"value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]},
        },
        "model": {"name": "random_forest", "params": {"n_estimators": 5, "random_state": 0}},
        "selection_views": [
            {
                "id": "primary",
                "objective": {"name": "scalar_identity_v1", "params": {}},
                "selection": {
                    "name": "top_n",
                    "params": {
                        "top_k": 2,
                        "score_ref": "scalar",
                        "objective_mode": "maximize",
                        "tie_handling": "competition_rank",
                    },
                },
            }
        ],
        "labels": {
            "source": {
                "kind": "usr_sidecar",
                "dataset": "promoter_candidates",
                "path": "_opal/observed_labels.parquet",
                "manifest_path": "_opal/observed_labels.manifest.json",
            },
            "y_space": "response_window_vector_v1",
        },
        "writeback": {"prediction_records": "ledger_only"},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_preserves_manifest_pinned_label_source(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "campaign.yaml", _payload()))

    assert cfg.labels.source.manifest_path == "_opal/observed_labels.manifest.json"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../observed_labels.parquet"),
        ("manifest_path", "/tmp/observed_labels.manifest.json"),
        ("manifest_path", "../observed_labels.manifest.json"),
    ],
)
def test_load_config_rejects_label_source_paths_outside_dataset_root(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    payload = _payload()
    payload["labels"]["source"][field] = value

    with pytest.raises(ConfigError, match=rf"labels\.source\.{field} must be relative to the USR dataset root"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_rejects_manifest_pin_without_study_owner(tmp_path: Path) -> None:
    payload = _payload()
    payload["ownership"] = {"owner_scope": "opal_demo", "portable": True}

    with pytest.raises(ConfigError, match="manifest_path requires study_campaign ownership"):
        load_config(_write(tmp_path / "campaign.yaml", payload))
