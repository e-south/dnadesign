"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_campaign_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.campaign_service import expand_campaign
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def _base_config() -> dict:
    return {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [],
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }


def test_campaign_expansion_within_across(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"].update(
        {
            "regulator_categories": {
                "CatA": ["A", "B"],
                "CatB": ["C", "D"],
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cfg = load_config(config_path)
    expansion = expand_campaign(cfg=cfg, config_path=config_path, campaign_name="demo")

    assert expansion.regulator_sets == [
        ["A", "B"],
        ["C", "D"],
        ["A", "C"],
        ["A", "D"],
        ["B", "C"],
        ["B", "D"],
    ]


def test_campaign_overlap_disallowed(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"].update(
        {
            "regulator_categories": {
                "CatA": ["A", "B"],
                "CatB": ["B", "C"],
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "allow_overlap": False,
                }
            ],
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="forbids overlaps"):
        load_config(config_path)


def test_campaign_selectors_min_site_count(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    index = CatalogIndex()
    index.entries["regulondb:A1"] = CatalogEntry(
        source="regulondb",
        motif_id="A1",
        tf_name="LexA",
        kind="PFM",
        has_sites=True,
        site_count=5,
        site_total=5,
    )
    index.entries["regulondb:B1"] = CatalogEntry(
        source="regulondb",
        motif_id="B1",
        tf_name="CpxR",
        kind="PFM",
        has_sites=True,
        site_count=1,
        site_total=1,
    )
    index.save(catalog_root)

    config = _base_config()
    config["cruncher"].update(
        {
            "motif_store": {"catalog_root": ".cruncher", "pwm_source": "sites"},
            "regulator_categories": {"CatA": ["LexA", "CpxR"]},
            "campaigns": [
                {
                    "name": "filter",
                    "categories": ["CatA"],
                    "within_category": {"sizes": [1]},
                    "selectors": {"min_site_count": 2},
                }
            ],
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cfg = load_config(config_path)
    expansion = expand_campaign(cfg=cfg, config_path=config_path, campaign_name="filter")
    assert expansion.regulator_sets == [["LexA"]]
