"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_demo_config.py

Smoke tests for packaged demo configurations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.adapters.sources.base import resolve_path
from dnadesign.densegen.src.config import load_config


def _demo_config_path(workspace_id: str) -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces" / workspace_id / "config.yaml"


def test_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path("demo_meme_three_tfs")
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_meme_three_tfs"


def test_demo_artifacts_present() -> None:
    cfg_path = _demo_config_path("demo_meme_three_tfs")
    cfg = load_config(cfg_path)
    pwm_inputs = [inp for inp in cfg.root.densegen.inputs if inp.type == "pwm_artifact"]
    assert pwm_inputs, "Demo config should include pwm_artifact inputs."
    missing: list[str] = []
    for inp in pwm_inputs:
        path = getattr(inp, "path", None)
        if not path:
            missing.append(f"Missing path for input {inp.name}")
            continue
        resolved = resolve_path(cfg_path, path)
        if not resolved.exists():
            missing.append(str(resolved))
    assert not missing, f"Missing demo artifacts: {missing}"


def test_vanilla_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path("demo_binding_sites_vanilla")
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_binding_sites_vanilla"


def test_vanilla_demo_plan_has_no_constraints() -> None:
    cfg_path = _demo_config_path("demo_binding_sites_vanilla")
    cfg = load_config(cfg_path)
    plan = cfg.root.densegen.generation.plan
    assert plan
    for item in plan:
        assert list(item.regulator_constraints.groups or []) == []
        assert list(item.fixed_elements.promoter_constraints or []) == []
        side_biases = item.fixed_elements.side_biases
        if side_biases is None:
            continue
        assert list(side_biases.left or []) == []
        assert list(side_biases.right or []) == []


def test_vanilla_demo_uses_local_output_with_padding_enabled() -> None:
    cfg_path = _demo_config_path("demo_binding_sites_vanilla")
    cfg = load_config(cfg_path)
    output = cfg.root.densegen.output
    assert output.targets == ["parquet"]
    assert cfg.root.densegen.postprocess.pad.mode == "adaptive"
