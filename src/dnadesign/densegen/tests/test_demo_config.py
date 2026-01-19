from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src import cli
from dnadesign.densegen.src.adapters.sources.base import resolve_path
from dnadesign.densegen.src.config import load_config


def _demo_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workspaces" / "demo_meme_two_tf" / "config.yaml"


def test_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path()
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_meme_two_tf"
    assert cli._default_config_path().resolve() == cfg_path


def test_demo_artifacts_present() -> None:
    cfg_path = _demo_config_path()
    cfg = load_config(cfg_path)
    pwm_inputs = [inp for inp in cfg.root.densegen.inputs if inp.type == "pwm_meme_set"]
    assert pwm_inputs, "Demo config should include pwm_meme_set inputs."
    missing: list[str] = []
    for inp in pwm_inputs:
        for path in getattr(inp, "paths", []) or []:
            resolved = resolve_path(cfg_path, path)
            if not resolved.exists():
                missing.append(str(resolved))
    assert not missing, f"Missing demo artifacts: {missing}"
