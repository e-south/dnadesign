"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_config_selection_views_v3.py

Campaign v3 selection-view configuration contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.core.utils import ConfigError


def _base_config(tmp_path: Path) -> dict:
    records = tmp_path / "records.parquet"
    records.touch()
    return {
        "schema_version": "opal.campaign.v3",
        "ownership": {"owner_scope": "opal_demo", "portable": True},
        "campaign": {"name": "Multi-view", "slug": "multi_view", "workdir": str(tmp_path)},
        "data": {
            "location": {"kind": "local", "path": str(records)},
            "x_column_name": "X",
            "y_column_name": "Y",
            "y_expected_length": 1,
        },
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {"name": "scalar_from_table_v1", "params": {}},
        "model": {"name": "random_forest", "params": {"n_estimators": 5, "random_state": 7}},
        "selection_views": [
            {
                "id": "target_a",
                "objective": {"name": "scalar_identity_v1", "params": {}},
                "selection": {
                    "name": "top_n",
                    "params": {
                        "top_k": 1,
                        "score_ref": "scalar",
                        "objective_mode": "maximize",
                        "tie_handling": "competition_rank",
                    },
                },
            },
            {
                "id": "target_b",
                "objective": {"name": "scalar_identity_v1", "params": {}},
                "selection": {
                    "name": "top_n",
                    "params": {
                        "top_k": 1,
                        "score_ref": "scalar",
                        "objective_mode": "maximize",
                        "tie_handling": "competition_rank",
                    },
                },
            },
        ],
        "selection_batch": {"deduplicate_by": "sequence", "expected_unique_count": 2},
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _enable_unique_allocation(payload: dict) -> None:
    for view in payload["selection_views"]:
        view["selection"]["params"]["tie_handling"] = "ordinal"
        view["selection"]["params"]["require_exact_top_k"] = True
    payload["selection_batch"]["allocation"] = {
        "strategy": "round_robin_next_best_unallocated",
        "view_priority": ["target_a", "target_b"],
    }


def test_v3_allows_repeated_objective_plugins_with_distinct_view_ids(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, _base_config(tmp_path)))

    assert cfg.schema_version == "opal.campaign.v3"
    assert [view.id for view in cfg.selection_views] == ["target_a", "target_b"]
    assert [view.objective.name for view in cfg.selection_views] == [
        "scalar_identity_v1",
        "scalar_identity_v1",
    ]
    assert cfg.selection_batch.deduplicate_by == "sequence"
    assert cfg.selection_batch.expected_unique_count == 2


def test_v3_accepts_explicit_round_robin_unique_allocation(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)

    cfg = load_config(_write_config(tmp_path, payload))

    assert cfg.selection_batch.allocation is not None
    assert cfg.selection_batch.allocation.strategy == "round_robin_next_best_unallocated"
    assert cfg.selection_batch.allocation.view_priority == ["target_a", "target_b"]


@pytest.mark.parametrize(
    "view_priority",
    [
        ["target_a"],
        ["target_a", "unknown"],
    ],
)
def test_v3_rejects_incomplete_or_unknown_allocation_priority(
    tmp_path: Path,
    view_priority: list[str],
) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)
    payload["selection_batch"]["allocation"]["view_priority"] = view_priority

    with pytest.raises(ConfigError, match="view_priority must be an exact permutation"):
        load_config(_write_config(tmp_path, payload))


def test_v3_rejects_duplicate_allocation_priority(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)
    payload["selection_batch"]["allocation"]["view_priority"] = ["target_a", "target_a"]

    with pytest.raises(ConfigError, match="view_priority must not contain duplicates"):
        load_config(_write_config(tmp_path, payload))


def test_v3_requires_ordinal_ties_for_unique_allocation(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)
    payload["selection_views"][1]["selection"]["params"]["tie_handling"] = "competition_rank"

    with pytest.raises(ConfigError, match="target_b.*tie_handling='ordinal'"):
        load_config(_write_config(tmp_path, payload))


def test_v3_requires_exact_view_quotas_for_unique_allocation(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)
    payload["selection_views"][1]["selection"]["params"]["require_exact_top_k"] = False

    with pytest.raises(ConfigError, match="target_b.*require_exact_top_k=true"):
        load_config(_write_config(tmp_path, payload))


def test_v3_requires_batch_count_to_equal_allocation_quotas(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    _enable_unique_allocation(payload)
    payload["selection_batch"]["expected_unique_count"] = 3

    with pytest.raises(ConfigError, match=r"must equal the sum.*\(2\)"):
        load_config(_write_config(tmp_path, payload))


def test_v3_requires_explicit_campaign_ownership(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    payload.pop("ownership")

    with pytest.raises(ConfigError, match="ownership"):
        load_config(_write_config(tmp_path, payload))


def test_v3_rejects_duplicate_selection_view_ids(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    payload["selection_views"][1]["id"] = "target_a"

    with pytest.raises(ConfigError, match="selection view ids must be unique"):
        load_config(_write_config(tmp_path, payload))


def test_v3_rejects_namespaced_score_ref_inside_view(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    payload["selection_views"][0]["selection"]["params"]["score_ref"] = "scalar_identity_v1/scalar"

    with pytest.raises(ConfigError, match="selection_views.*score_ref.*channel name"):
        load_config(_write_config(tmp_path, payload))


def test_v2_shape_is_rejected_without_compatibility_path(tmp_path: Path) -> None:
    payload = _base_config(tmp_path)
    payload.pop("schema_version")
    views = payload.pop("selection_views")
    payload.pop("selection_batch")
    payload["objectives"] = [views[0]["objective"]]
    payload["selection"] = views[0]["selection"]

    with pytest.raises(ConfigError, match="schema_version"):
        load_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("ownership", "error"),
    [
        (
            {
                "owner_scope": "opal_demo",
                "study_id": "example_study",
                "portable": True,
            },
            "opal_demo.*study_id",
        ),
        (
            {"owner_scope": "opal_demo", "portable": False},
            "opal_demo.*portable",
        ),
        (
            {"owner_scope": "study_campaign", "dataset_id": "dataset", "portable": False},
            "study_campaign.*study_id",
        ),
        (
            {"owner_scope": "study_campaign", "study_id": "study", "portable": False},
            "study_campaign.*dataset_id",
        ),
        (
            {
                "owner_scope": "study_campaign",
                "study_id": "study",
                "dataset_id": "dataset",
                "portable": True,
            },
            "study_campaign.*portable",
        ),
    ],
)
def test_v3_rejects_incoherent_campaign_ownership(tmp_path: Path, ownership: dict, error: str) -> None:
    payload = _base_config(tmp_path)
    payload["ownership"] = ownership

    with pytest.raises(ConfigError, match=error):
        load_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "ownership",
    [
        {"owner_scope": "opal_demo", "portable": True},
        {
            "owner_scope": "study_campaign",
            "study_id": "example_study",
            "dataset_id": "example_candidates",
            "portable": False,
        },
    ],
)
def test_v3_accepts_explicit_campaign_ownership(tmp_path: Path, ownership: dict) -> None:
    payload = _base_config(tmp_path)
    payload["ownership"] = ownership

    cfg = load_config(_write_config(tmp_path, payload))

    assert cfg.ownership.owner_scope == ownership["owner_scope"]


@pytest.mark.parametrize("inline_key", ["plots", "plot_defaults", "plot_presets"])
def test_v3_rejects_inline_plot_configuration(tmp_path: Path, inline_key: str) -> None:
    payload = _base_config(tmp_path)
    payload[inline_key] = [] if inline_key == "plots" else {}

    with pytest.raises(ConfigError, match=inline_key):
        load_config(_write_config(tmp_path, payload))
