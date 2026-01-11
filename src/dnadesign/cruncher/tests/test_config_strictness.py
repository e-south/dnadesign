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
