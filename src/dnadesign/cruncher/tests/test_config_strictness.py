"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_config_strictness.py

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
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher", "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _sample_block(
    *,
    optimizer_name: str,
    gibbs_schedule: dict | None = None,
    pt_ladder: dict | None = None,
    restarts: int = 2,
) -> dict:
    return {
        "mode": "sample",
        "rng": {"seed": 7, "deterministic": True},
        "budget": {"draws": 2, "tune": 1, "restarts": restarts},
        "init": {"kind": "random", "length": 12, "pad_with": "background"},
        "objective": {"bidirectional": True, "score_scale": "llr"},
        "elites": {"k": 1, "min_hamming": 0, "filters": {"pwm_sum_min": 0.0}},
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
        "optimizer": {"name": optimizer_name},
        "optimizers": {
            "gibbs": {
                "beta_schedule": gibbs_schedule or {"kind": "linear", "beta": [0.1, 0.2]},
                "apply_during": "tune",
            },
            "pt": {
                "beta_ladder": pt_ladder or {"kind": "geometric", "betas": [1.0, 0.5]},
                "swap_prob": 0.1,
            },
        },
        "auto_opt": {"enabled": optimizer_name == "auto"},
        "output": {"trace": {"save": False}, "save_sequences": True},
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


def test_multi_tf_llr_requires_allow_unscaled(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["regulator_sets"] = [["lexA", "cpxR"]]
    config["cruncher"]["sample"] = _sample_block(optimizer_name="gibbs")
    config_path = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="allow_unscaled_llr"):
        load_config(config_path)
    config["cruncher"]["sample"]["objective"]["allow_unscaled_llr"] = True
    config_path = _write_config(tmp_path, config)
    load_config(config_path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["motif_store"]["bogus"] = "nope"
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("type") == "extra_forbidden" and err.get("loc", ())[-1] == "bogus" for err in exc.value.errors())


def test_missing_cruncher_root_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"not_cruncher": {"out_dir": "runs"}})

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("type") == "missing" and err.get("loc") == ("cruncher",) for err in exc.value.errors())


@pytest.mark.parametrize("out_dir", ["__absolute__", "../runs"])
def test_out_dir_must_be_workspace_relative(tmp_path: Path, out_dir: str) -> None:
    config = _base_config()
    if out_dir == "__absolute__":
        out_dir = str(tmp_path / "runs")
    config["cruncher"]["out_dir"] = str(out_dir)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "out_dir") for err in exc.value.errors())


@pytest.mark.parametrize("catalog_root", ["../catalog"])
def test_catalog_root_rejects_parent_traversal(tmp_path: Path, catalog_root: str) -> None:
    config = _base_config()
    config["cruncher"]["motif_store"]["catalog_root"] = str(catalog_root)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "motif_store", "catalog_root") for err in exc.value.errors())


def test_catalog_root_allows_absolute(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["motif_store"]["catalog_root"] = str(tmp_path / "catalog")
    config_path = _write_config(tmp_path, config)

    cfg = load_config(config_path)
    assert cfg.motif_store.catalog_root.is_absolute()


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


def test_gibbs_rejects_geometric_cooling(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(
        optimizer_name="gibbs",
        gibbs_schedule={"kind": "geometric", "beta": [1.0, 0.5]},
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("gibbs" in str(err.get("msg")) for err in exc.value.errors())


def test_gibbs_global_schedule_requires_apply_all(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(optimizer_name="gibbs")
    config["cruncher"]["sample"]["optimizers"]["gibbs"]["schedule_scope"] = "global"
    config["cruncher"]["sample"]["optimizers"]["gibbs"]["apply_during"] = "tune"
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("schedule_scope" in str(err.get("msg")) for err in exc.value.errors())


def test_pt_rejects_missing_ladder_params(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(
        optimizer_name="pt",
        pt_ladder={"kind": "geometric"},
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(
        "beta_ladder requires betas or beta_min/beta_max/n_temps" in str(err.get("msg")) for err in exc.value.errors()
    )


def test_pt_requires_restarts_one(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(
        optimizer_name="pt",
        pt_ladder={"kind": "fixed", "beta": 1.0},
        restarts=4,
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("budget.restarts" in str(err.get("msg")) for err in exc.value.errors())
