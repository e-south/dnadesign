"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_plan_template_expansion.py

Config expansion tests for generation.plan fixed-element matrix expansion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from dnadesign.densegen.src.config import ConfigError, load_config

MIN_TEMPLATE_CONFIG = {
    "densegen": {
        "schema_version": "2.9",
        "run": {"id": "demo", "root": "."},
        "inputs": [{"name": "background", "type": "sequence_library", "path": "inputs/background.csv"}],
        "motif_sets": {
            "up_core": {
                "consensus": "TTGACA",
                "shift": "TTTACA",
            },
            "down_core": {
                "consensus": "TATAAT",
                "shift": "TAAAAT",
            },
        },
        "output": {
            "targets": ["parquet"],
            "schema": {"bio_type": "dna", "alphabet": "dna_4"},
            "parquet": {"path": "outputs/tables/records.parquet"},
        },
        "generation": {
            "sequence_length": 90,
            "expansion": {"max_plans": 256},
            "plan": [
                {
                    "name": "sigma70",
                    "sequences": 8,
                    "sampling": {"include_inputs": ["background"]},
                    "regulator_constraints": {"groups": []},
                    "fixed_elements": {
                        "fixed_element_matrix": {
                            "name": "sigma70_core",
                            "upstream_from_set": "up_core",
                            "downstream_from_set": "down_core",
                            "pairing": {"mode": "zip"},
                            "spacer_length": [16, 18],
                            "upstream_pos": [0, 20],
                        }
                    },
                }
            ],
        },
        "solver": {"backend": "CBC", "strategy": "iterate"},
        "logging": {"log_dir": "outputs/logs"},
    }
}


def _write(cfg: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_plan_templates_zip_expands_into_deterministic_plan_names(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    names = [item.name for item in loaded.root.densegen.generation.plan]
    assert names == [
        "sigma70__up=consensus__down=consensus",
        "sigma70__up=shift__down=shift",
    ]


def test_plan_templates_cross_product_expands_all_pairs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {
        "mode": "cross_product"
    }
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    names = [item.name for item in loaded.root.densegen.generation.plan]
    assert names == [
        "sigma70__up=consensus__down=consensus",
        "sigma70__up=consensus__down=shift",
        "sigma70__up=shift__down=consensus",
        "sigma70__up=shift__down=shift",
    ]


def test_plan_templates_explicit_pairs_expands_only_requested_pairs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {
        "mode": "explicit_pairs",
        "pairs": [{"up": "shift", "down": "consensus"}],
    }
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    names = [item.name for item in loaded.root.densegen.generation.plan]
    assert names == ["sigma70__up=shift__down=consensus"]


def test_plan_templates_zip_requires_matching_keys(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["motif_sets"]["down_core"] = {"consensus": "TATAAT"}
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="matching upstream/downstream"):
        load_config(cfg_path)


def test_plan_templates_respect_expansion_cap(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["expansion"]["max_plans"] = 2
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {
        "mode": "cross_product"
    }
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="max_plans"):
        load_config(cfg_path)


def test_plan_templates_validate_geometry_at_load_time(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["sequence_length"] = 20
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["spacer_length"] = [16, 18]
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["upstream_pos"] = [0, 1]
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="geometry"):
        load_config(cfg_path)


def test_plan_templates_expansion_cap_applies_to_total_expanded_plan_count(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    second_plan = copy.deepcopy(cfg["densegen"]["generation"]["plan"][0])
    second_plan["name"] = "sigma32"
    cfg["densegen"]["generation"]["plan"].append(second_plan)
    cfg["densegen"]["generation"]["expansion"]["max_plans"] = 3
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="max_plans"):
        load_config(cfg_path)


def test_plan_templates_accept_large_targets_without_global_quota_cap(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["sequences"] = 6000
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {"mode": "zip"}
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    assert sum(int(item.quota) for item in loaded.root.densegen.generation.resolve_plan()) == 6000


def test_plan_fixed_element_matrix_distributes_non_divisible_quota_evenly(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {
        "mode": "cross_product"
    }
    cfg["densegen"]["generation"]["plan"][0]["sequences"] = 10
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    plans = list(loaded.root.densegen.generation.plan)
    assert [item.sequences for item in plans] == [3, 3, 2, 2]
    assert sum(int(item.sequences) for item in plans) == 10


def test_plan_fixed_element_matrix_uses_partial_expansion_when_quota_is_smaller(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["fixed_elements"]["fixed_element_matrix"]["pairing"] = {
        "mode": "cross_product"
    }
    cfg["densegen"]["generation"]["plan"][0]["sequences"] = 2
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    loaded = load_config(cfg_path)
    plans = list(loaded.root.densegen.generation.plan)
    assert [item.name for item in plans] == [
        "sigma70__up=consensus__down=consensus",
        "sigma70__up=consensus__down=shift",
    ]
    assert [item.sequences for item in plans] == [1, 1]
    assert sum(int(item.sequences) for item in plans) == 2


def test_generation_rejects_legacy_plan_templates_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan_templates"] = [
        {
            "base_name": "legacy",
            "quota_per_variant": 1,
            "sampling": {"include_inputs": ["background"]},
            "regulator_constraints": {"groups": []},
        }
    ]
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="plan_templates"):
        load_config(cfg_path)


def test_generation_rejects_legacy_quota_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["quota"] = 1
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="quota"):
        load_config(cfg_path)


def test_generation_rejects_legacy_total_quota_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["total_quota"] = 8
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="total_quota"):
        load_config(cfg_path)


def test_generation_rejects_legacy_target_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0].pop("sequences")
    cfg["densegen"]["generation"]["plan"][0]["target"] = {"sequences": 8}
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="target"):
        load_config(cfg_path)


def test_generation_rejects_legacy_distribution_policy_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan"][0]["distribution_policy"] = "uniform"
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="distribution_policy"):
        load_config(cfg_path)


def test_generation_rejects_legacy_max_expanded_plans_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["max_expanded_plans"] = 4
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="max_expanded_plans"):
        load_config(cfg_path)


def test_generation_rejects_legacy_max_total_quota_key(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["max_total_quota"] = 8
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="max_total_quota"):
        load_config(cfg_path)
