"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_config_strictness.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dnadesign.cruncher.config.load import load_config


def _base_config() -> dict:
    return {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": ".cruncher", "pwm_source": "matrix"},
        }
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _sample_block(*, total_sweeps: int = 3, adapt_sweep_frac: float = 0.34) -> dict:
    tune = max(0, min(total_sweeps - 1, int(round(total_sweeps * adapt_sweep_frac))))
    draws = total_sweeps - tune
    return {
        "seed": 7,
        "sequence_length": 12,
        "budget": {"tune": tune, "draws": draws},
        "objective": {"bidirectional": True, "score_scale": "llr", "combine": "min"},
        "elites": {
            "k": 1,
            "select": {"diversity": 0.0, "pool_size": "auto"},
        },
        "moves": {
            "profile": "balanced",
            "overrides": {
                "block_len_range": [2, 2],
                "multi_k_range": [2, 2],
                "slide_max_shift": 1,
                "swap_len_range": [2, 2],
                "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
            },
        },
        "output": {
            "save_trace": False,
            "save_sequences": True,
            "include_tune_in_sequences": False,
            "live_metrics": False,
        },
    }


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["unknown_block"] = {"value": 1}
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(
        err.get("type") == "extra_forbidden" and err.get("loc", ())[-1] == "unknown_block" for err in exc.value.errors()
    )


def test_multi_tf_llr_rejects_removed_allow_unscaled_key(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["workspace"]["regulator_sets"] = [["lexA", "cpxR"]]
    config["cruncher"]["sample"] = _sample_block()
    config["cruncher"]["sample"]["objective"]["allow_unscaled_llr"] = True
    config_path = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="allow_unscaled_llr"):
        load_config(config_path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["catalog"]["bogus"] = "nope"
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("type") == "extra_forbidden" and err.get("loc", ())[-1] == "bogus" for err in exc.value.errors())


def test_motif_discovery_accepts_meme_prior(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["discover"] = {"tool": "meme", "meme_prior": "addone"}
    config_path = _write_config(tmp_path, config)

    load_config(config_path)


def test_motif_discovery_rejects_invalid_meme_prior(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["discover"] = {"tool": "meme", "meme_prior": "invalid"}
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError, match="meme_prior"):
        load_config(config_path)


def test_missing_cruncher_root_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"not_cruncher": {"out_dir": "runs"}})

    with pytest.raises(ValueError, match="missing root key: cruncher"):
        load_config(config_path)


@pytest.mark.parametrize("out_dir", ["__absolute__", "../runs"])
def test_out_dir_must_be_workspace_relative(tmp_path: Path, out_dir: str) -> None:
    config = _base_config()
    if out_dir == "__absolute__":
        out_dir = str(tmp_path / "runs")
    config["cruncher"]["workspace"]["out_dir"] = str(out_dir)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "workspace", "out_dir") for err in exc.value.errors())


@pytest.mark.parametrize("catalog_root", ["../catalog"])
def test_catalog_root_rejects_parent_traversal(tmp_path: Path, catalog_root: str) -> None:
    config = _base_config()
    config["cruncher"]["catalog"]["root"] = str(catalog_root)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "catalog", "root") for err in exc.value.errors())


def test_catalog_root_allows_absolute(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["catalog"]["root"] = str(tmp_path / "catalog")
    config_path = _write_config(tmp_path, config)

    cfg = load_config(config_path)
    assert cfg.catalog.root.is_absolute()


def test_catalog_pwm_window_lengths_removed_key_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["catalog"]["pwm_window_lengths"] = {"lexA": 12}
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_catalog_pwm_window_strategy_removed_key_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["catalog"]["pwm_window_strategy"] = "max_info"
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_workspace_regulator_sets_duplicate_tf_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["workspace"]["regulator_sets"] = [["lexA", "lexA"]]
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError, match="duplicate TF"):
        load_config(config_path)


def test_workspace_regulator_categories_duplicate_tf_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["workspace"]["regulator_categories"] = {"stress": ["lexA", "lexA"]}
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError, match="duplicate TF"):
        load_config(config_path)


def test_removed_motif_store_alias_property_is_not_exposed(tmp_path: Path) -> None:
    config = _base_config()
    config_path = _write_config(tmp_path, config)
    cfg = load_config(config_path)

    with pytest.raises(AttributeError):
        _ = cfg.motif_store


def test_removed_motif_discovery_alias_property_is_not_exposed(tmp_path: Path) -> None:
    config = _base_config()
    config_path = _write_config(tmp_path, config)
    cfg = load_config(config_path)

    with pytest.raises(AttributeError):
        _ = cfg.motif_discovery


@pytest.mark.parametrize("genome_cache", ["__absolute__", "../genomes"])
def test_genome_cache_must_be_workspace_relative(tmp_path: Path, genome_cache: str) -> None:
    config = _base_config()
    if genome_cache == "__absolute__":
        genome_cache = str(tmp_path / "genomes")
    config["cruncher"]["ingest"] = {"genome_cache": str(genome_cache)}
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "ingest", "genome_cache") for err in exc.value.errors())
