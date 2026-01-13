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


def _sample_block(*, optimiser_kind: str, cooling: dict, chains: int = 2) -> dict:
    return {
        "bidirectional": True,
        "seed": 7,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "save_trace": False,
        "init": {"kind": "random", "length": 12, "pad_with": "background"},
        "draws": 2,
        "tune": 1,
        "chains": chains,
        "min_dist": 0,
        "top_k": 1,
        "moves": {
            "block_len_range": [2, 2],
            "multi_k_range": [2, 2],
            "slide_max_shift": 1,
            "swap_len_range": [2, 2],
            "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
        },
        "optimiser": {
            "kind": optimiser_kind,
            "scorer_scale": "llr",
            "cooling": cooling,
            "swap_prob": 0.1,
        },
        "save_sequences": True,
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


@pytest.mark.parametrize("catalog_root", ["__absolute__", "../catalog"])
def test_catalog_root_must_be_workspace_relative(tmp_path: Path, catalog_root: str) -> None:
    config = _base_config()
    if catalog_root == "__absolute__":
        catalog_root = str(tmp_path / "catalog")
    config["cruncher"]["motif_store"]["catalog_root"] = str(catalog_root)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("loc") == ("cruncher", "motif_store", "catalog_root") for err in exc.value.errors())


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
        optimiser_kind="gibbs",
        cooling={"kind": "geometric", "beta": [1.0, 0.5]},
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("gibbs" in str(err.get("msg")) for err in exc.value.errors())


def test_pt_requires_beta_ladder_length_match(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(
        optimiser_kind="pt",
        cooling={"kind": "geometric", "beta": [1.0, 0.5, 0.25]},
        chains=2,
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("cooling.beta length must match sample.chains" in str(err.get("msg")) for err in exc.value.errors())


def test_pt_fixed_requires_single_chain(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"] = _sample_block(
        optimiser_kind="pt",
        cooling={"kind": "fixed", "beta": 1.0},
        chains=4,
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any("fixed cooling requires chains=1" in str(err.get("msg")) for err in exc.value.errors())
