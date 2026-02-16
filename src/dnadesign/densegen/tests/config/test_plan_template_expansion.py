"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_plan_template_expansion.py

Config expansion tests for generation.plan_templates.

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
            "plan_templates": [
                {
                    "base_name": "sigma70",
                    "quota_per_variant": 4,
                    "sampling": {"include_inputs": ["background"]},
                    "regulator_constraints": {"groups": []},
                    "fixed_elements": {
                        "promoter_matrix": {
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
    cfg["densegen"]["generation"]["plan_templates"][0]["fixed_elements"]["promoter_matrix"]["pairing"] = {
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
    cfg["densegen"]["generation"]["plan_templates"][0]["fixed_elements"]["promoter_matrix"]["pairing"] = {
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
    with pytest.raises(ConfigError, match="matching keys"):
        load_config(cfg_path)


def test_plan_templates_respect_expansion_cap(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["plan_template_max_expanded_plans"] = 2
    cfg["densegen"]["generation"]["plan_templates"][0]["fixed_elements"]["promoter_matrix"]["pairing"] = {
        "mode": "cross_product"
    }
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="max_expanded_plans"):
        load_config(cfg_path)


def test_plan_templates_validate_geometry_at_load_time(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_TEMPLATE_CONFIG)
    cfg["densegen"]["generation"]["sequence_length"] = 20
    cfg["densegen"]["generation"]["plan_templates"][0]["fixed_elements"]["promoter_matrix"]["spacer_length"] = [16, 18]
    cfg["densegen"]["generation"]["plan_templates"][0]["fixed_elements"]["promoter_matrix"]["upstream_pos"] = [0, 1]
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="geometry"):
        load_config(cfg_path)
