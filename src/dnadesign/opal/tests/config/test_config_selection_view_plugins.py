"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_config_selection_view_plugins.py

Campaign v3 selection-view plugin and adjacent config contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.core.utils import ConfigError


def _base_config(
    *,
    objective_name: str = "scalar_identity_v1",
    objective_params: dict | None = None,
    score_ref: str = "scalar",
) -> dict:
    return {
        "schema_version": "opal.campaign.v3",
        "ownership": {"owner_scope": "opal_demo", "portable": True},
        "campaign": {"name": "Demo", "slug": "demo", "workdir": "."},
        "data": {
            "location": {"kind": "local", "path": "./records.parquet"},
            "x_column_name": "X",
            "y_column_name": "Y",
        },
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {"name": "scalar_from_table_v1", "params": {}},
        "model": {"name": "random_forest", "params": {"n_estimators": 5, "random_state": 0}},
        "selection_views": [
            {
                "id": "primary",
                "objective": {"name": objective_name, "params": objective_params or {}},
                "selection": {
                    "name": "top_n",
                    "params": {
                        "top_k": 2,
                        "score_ref": score_ref,
                        "objective_mode": "maximize",
                        "tie_handling": "competition_rank",
                    },
                },
            }
        ],
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_preserves_view_identity_and_campaign_metadata(tmp_path: Path) -> None:
    payload = _base_config()
    payload["campaign"]["metadata"] = {"scenario_kind": "positive", "split_id": "random"}

    cfg = load_config(_write(tmp_path / "campaign.yaml", payload))

    assert cfg.selection_views[0].id == "primary"
    assert cfg.selection_views[0].objective.name == "scalar_identity_v1"
    assert cfg.selection_views[0].selection.params["score_ref"] == "scalar"
    assert cfg.campaign.metadata == {"scenario_kind": "positive", "split_id": "random"}


def test_load_config_accepts_spop_view(tmp_path: Path) -> None:
    cfg = load_config(
        _write(
            tmp_path / "campaign.yaml",
            _base_config(objective_name="spop_v1", score_ref="spop"),
        )
    )

    assert cfg.selection_views[0].objective.name == "spop_v1"
    assert cfg.selection_views[0].selection.params["score_ref"] == "spop"


def test_load_config_resolves_candidate_scope_from_campaign_root(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    payload = _base_config()
    payload["data"]["candidate_scope"] = {
        "kind": "id_list",
        "path": "scopes/heldout_ids.csv",
        "id_column": "id",
    }

    cfg = load_config(_write(root / "configs" / "campaign.yaml", payload))

    assert cfg.data.candidate_scope is not None
    assert cfg.data.candidate_scope.path == str((root / "scopes" / "heldout_ids.csv").resolve())


def test_load_config_accepts_restriction_site_eligibility(tmp_path: Path) -> None:
    payload = _base_config()
    payload["candidate_eligibility"] = {
        "rules": [
            {
                "name": "restriction_site_exclusion",
                "params": {
                    "sequence_column": "sequence",
                    "scan_space": "final_assembled_insert",
                    "assembly_strategy_ref": "sfxi_promoter_insert:v1",
                    "left_flank": "accgggatcctgcag",
                    "right_flank": "tgagggaattcgcga",
                    "expected_core_length": 60,
                    "min_remaining_candidates": 1,
                    "forbidden_sites": [{"enzyme": "BamHI", "motif": "GGATCC", "allowed_regions": ["left_flank"]}],
                },
            }
        ]
    }

    cfg = load_config(_write(tmp_path / "campaign.yaml", payload))

    assert cfg.candidate_eligibility.rules[0].params["forbidden_sites"][0]["enzyme"] == "BamHI"


def _sfxi_payload(*, uncertainty_method: str) -> dict:
    payload = _base_config(
        objective_name="sfxi_v1",
        objective_params={
            "setpoint_vector": [0, 0, 0, 1],
            "uncertainty_method": uncertainty_method,
            "scaling": {"min_n": 1},
        },
        score_ref="sfxi",
    )
    payload["transforms_y"] = {"name": "sfxi_vec8_from_table_v1", "params": {}}
    return payload


def test_load_config_accepts_sfxi_delta_uncertainty(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "campaign.yaml", _sfxi_payload(uncertainty_method="delta")))
    assert cfg.selection_views[0].objective.params["uncertainty_method"] == "delta"


@pytest.mark.parametrize("invalid_method", ["analytical", "bogus", "auto"])
def test_load_config_rejects_invalid_sfxi_uncertainty(tmp_path: Path, invalid_method: str) -> None:
    with pytest.raises(ConfigError, match="uncertainty_method"):
        load_config(_write(tmp_path / "campaign.yaml", _sfxi_payload(uncertainty_method=invalid_method)))


def _rmf_payload(target_mask: list[int]) -> dict:
    payload = _base_config(
        objective_name="response_magnitude_feasibility_v1",
        objective_params={
            "state_ids": ["00", "10", "01", "11"],
            "target_mask": target_mask,
            "calibration": {
                "response_separation_min": 0.2,
                "on_magnitude_min": 0.0,
                "off_magnitude_max": -0.2,
                "response_separation_scale": 0.1,
                "on_magnitude_scale": 0.2,
                "off_magnitude_scale": 0.2,
            },
        },
        score_ref="feasibility_margin",
    )
    payload["data"].update({"y_column_name": "response_window_vector", "y_expected_length": 8})
    payload["transforms_y"] = {
        "name": "vector_from_table_v1",
        "params": {"value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]},
    }
    return payload


def test_load_config_accepts_response_magnitude_feasibility_view(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "campaign.yaml", _rmf_payload([0, 1, 0, 1])))
    objective = cfg.selection_views[0].objective
    assert objective.name == "response_magnitude_feasibility_v1"
    assert objective.params["target_mask"] == [0, 1, 0, 1]


@pytest.mark.parametrize("target_mask", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_load_config_rejects_degenerate_response_target_mask(tmp_path: Path, target_mask: list[int]) -> None:
    with pytest.raises(ConfigError, match="target_mask"):
        load_config(_write(tmp_path / "campaign.yaml", _rmf_payload(target_mask)))


def _multistate_response_behavior_payload() -> dict:
    payload = _base_config(
        objective_name="multistate_response_behavior_v1",
        objective_params={
            "state_ids": ["00", "10", "01", "11"],
            "target_mask": [0, 1, 0, 1],
            "softmin_scale": 0.25,
        },
        score_ref="behavior_score",
    )
    payload["data"].update({"y_column_name": "response_window_vector", "y_expected_length": 8})
    payload["transforms_y"] = {
        "name": "vector_from_table_v1",
        "params": {"value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]},
    }
    return payload


def test_load_config_accepts_multistate_response_behavior_view(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "campaign.yaml", _multistate_response_behavior_payload()))

    objective = cfg.selection_views[0].objective
    assert objective.name == "multistate_response_behavior_v1"
    assert objective.params == {
        "state_ids": ["00", "10", "01", "11"],
        "target_mask": [0, 1, 0, 1],
        "softmin_scale": 0.25,
    }


def test_load_config_rejects_multistate_response_behavior_temperature(tmp_path: Path) -> None:
    payload = _multistate_response_behavior_payload()
    payload["selection_views"][0]["objective"]["params"]["temperature"] = 2.0

    with pytest.raises(ConfigError, match="temperature"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_rejects_removed_two_scale_behavior_shape(tmp_path: Path) -> None:
    payload = _multistate_response_behavior_payload()
    params = payload["selection_views"][0]["objective"]["params"]
    params.pop("softmin_scale")
    params["normalization"] = {"response_scale": 0.25, "signal_scale": 0.5}

    with pytest.raises(ConfigError, match="normalization"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


@pytest.mark.parametrize("softmin_scale", [True, 0.0, -1.0, float("inf")])
def test_load_config_rejects_invalid_behavior_softmin_scale(tmp_path: Path, softmin_scale: object) -> None:
    payload = _multistate_response_behavior_payload()
    payload["selection_views"][0]["objective"]["params"]["softmin_scale"] = softmin_scale

    with pytest.raises(ConfigError, match="softmin_scale"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_rejects_boolean_behavior_target_mask_aliases(tmp_path: Path) -> None:
    payload = _multistate_response_behavior_payload()
    payload["selection_views"][0]["objective"]["params"]["target_mask"] = [False, True, False, True]

    with pytest.raises(ConfigError, match="boolean aliases"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


@pytest.mark.parametrize("state_ids", [["00", " 10", "01", "11"], ["00", "10", "01", "   "]])
def test_load_config_rejects_behavior_state_identity_whitespace(tmp_path: Path, state_ids: list[str]) -> None:
    payload = _multistate_response_behavior_payload()
    payload["selection_views"][0]["objective"]["params"]["state_ids"] = state_ids

    with pytest.raises(ConfigError, match="leading or trailing whitespace"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def _usr_sidecar_payload() -> dict:
    payload = _base_config()
    payload["data"].update(
        {
            "location": {"kind": "usr", "path": "./usr/datasets", "dataset": "demo_candidates"},
            "y_column_name": "opal__demo__y",
        }
    )
    payload["labels"] = {
        "source": {
            "kind": "usr_sidecar",
            "dataset": "demo_candidates",
            "path": "_opal/observed_labels.parquet",
        },
        "y_space": "scalar_test",
    }
    payload["writeback"] = {"prediction_records": "ledger_only"}
    return payload


def test_load_config_accepts_usr_sidecar(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "campaign.yaml", _usr_sidecar_payload()))
    assert cfg.labels.source.kind == "usr_sidecar"
    assert cfg.writeback.prediction_records == "ledger_only"


def test_load_config_rejects_usr_sidecar_without_explicit_writeback(tmp_path: Path) -> None:
    payload = _usr_sidecar_payload()
    payload.pop("writeback")
    with pytest.raises(ConfigError, match="writeback.prediction_records"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_rejects_usr_sidecar_for_different_dataset(tmp_path: Path) -> None:
    payload = _usr_sidecar_payload()
    payload["labels"]["source"]["dataset"] = "other_candidates"
    with pytest.raises(ConfigError, match="same dataset"):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_accepts_artifact_retention(tmp_path: Path) -> None:
    payload = _base_config()
    payload["artifact_retention"] = {
        "mode": "production_review",
        "prediction_ledger": "latest_full_plus_selected_history",
        "plot_tidy_data": "compact",
        "model_artifacts": "latest",
        "tabular_format": "parquet_zstd",
        "max_estimated_bytes": 50_000_000_000,
        "fail_if_estimate_exceeds": True,
        "final_round": 11,
    }
    cfg = load_config(_write(tmp_path / "campaign.yaml", payload))
    assert cfg.artifact_retention.mode == "production_review"
    assert cfg.artifact_retention.final_round == 11


def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text("schema_version: opal.campaign.v3\nschema_version: opal.campaign.v3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate key in YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["selection_views"][0]["selection"]["params"].pop("score_ref"), "score_ref"),
        (
            lambda payload: payload["selection_views"][0]["selection"]["params"].update({"unknown_key": True}),
            "unknown_key",
        ),
    ],
)
def test_load_config_rejects_invalid_selection_params(tmp_path: Path, mutation, message: str) -> None:
    payload = _base_config()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path / "campaign.yaml", payload))


def test_load_config_accepts_gaussian_process_kernel(tmp_path: Path) -> None:
    payload = _base_config()
    payload["model"] = {
        "name": "gaussian_process",
        "params": {
            "alpha": 1.0e-6,
            "normalize_y": True,
            "n_restarts_optimizer": 2,
            "kernel": {
                "name": "matern",
                "length_scale": 0.5,
                "nu": 1.5,
                "with_white_noise": True,
            },
        },
    }
    cfg = load_config(_write(tmp_path / "campaign.yaml", payload))
    assert cfg.model.params["kernel"]["name"] == "matern"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["model"].update({"name": "not_a_model"}), "Unknown model plugin"),
        (
            lambda payload: payload["model"]["params"].update({"kernel": {"name": "unknown_kernel"}}),
            "kernel",
        ),
    ],
)
def test_load_config_rejects_invalid_model_config(tmp_path: Path, mutate, message: str) -> None:
    payload = deepcopy(_base_config())
    if "kernel" in message:
        payload["model"] = {"name": "gaussian_process", "params": {"alpha": 1.0e-6}}
    mutate(payload)
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path / "campaign.yaml", payload))
