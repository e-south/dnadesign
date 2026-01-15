from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src import cli
from dnadesign.densegen.src.config import load_config


def _demo_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runs" / "demo" / "config.yaml"


def test_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path()
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo"
    assert cli._default_config_path().resolve() == cfg_path


def test_demo_inputs_present() -> None:
    inputs_dir = _demo_config_path().parent / "inputs"
    required = [
        "tf2tfbs_mapping_cpxR_LexA.csv",
        "pwm_demo.meme",
        "pwm_demo.jaspar",
        "pwm_matrix_demo.csv",
    ]
    missing = [name for name in required if not (inputs_dir / name).exists()]
    assert not missing, f"Missing demo inputs: {missing}"
