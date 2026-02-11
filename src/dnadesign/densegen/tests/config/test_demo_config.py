"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_demo_config.py

Smoke tests for packaged demo configurations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

from dnadesign.densegen.src.adapters.sources.base import resolve_path
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.config.base import LATEST_SCHEMA_VERSION


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


def test_demo_default_plots_include_all_families() -> None:
    cfg_path = _demo_config_path("demo_meme_three_tfs")
    cfg = load_config(cfg_path)
    plots = cfg.root.plots
    assert plots is not None
    assert list(plots.default or []) == ["stage_a_summary", "placement_map", "run_health", "tfbs_usage"]


def test_demo_meme_three_tfs_uses_background_and_plan_specific_sigma70_spacers() -> None:
    cfg_path = _demo_config_path("demo_meme_three_tfs")
    cfg = load_config(cfg_path)

    input_names = [inp.name for inp in cfg.root.densegen.inputs]
    assert "background" in input_names
    assert "neutral_bg" not in input_names

    plan = cfg.root.densegen.generation.plan
    assert plan
    plan_by_name = {item.name: item for item in plan}

    expected_spacers = {
        "controls": (16, 18),
        "ciprofloxacin": (16, 18),
        "ethanol": (16, 20),
        "ethanol_ciprofloxacin": (16, 20),
    }
    assert set(plan_by_name) == set(expected_spacers)

    for plan_name, spacer in expected_spacers.items():
        plan_item = plan_by_name[plan_name]
        include_inputs = list(plan_item.sampling.include_inputs or [])
        assert "background" in include_inputs
        assert "neutral_bg" not in include_inputs

        pcs = list(plan_item.fixed_elements.promoter_constraints or [])
        assert len(pcs) == 1
        pc = pcs[0]
        assert pc.name == "sigma70_consensus"
        assert tuple(pc.spacer_length or ()) == spacer


def test_demo_meme_three_tfs_stage_a_sampling_targets() -> None:
    cfg_path = _demo_config_path("demo_meme_three_tfs")
    cfg = load_config(cfg_path)
    inputs = list(cfg.root.densegen.inputs)
    pwm_inputs = [inp for inp in inputs if inp.type == "pwm_artifact"]
    assert pwm_inputs
    for inp in pwm_inputs:
        sampling = inp.sampling
        assert sampling.n_sites == 250
        assert sampling.mining.budget.mode == "fixed_candidates"
        assert sampling.mining.budget.candidates == 1_000_000
        assert sampling.uniqueness.cross_regulator_core_collisions == "error"

    background_inputs = [inp for inp in inputs if inp.type == "background_pool"]
    assert len(background_inputs) == 1
    assert background_inputs[0].sampling.n_sites == 500


def test_binding_sites_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path("demo_binding_sites")
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_binding_sites"


def test_binding_sites_demo_mock_sites_lengths_are_16_to_20_bp() -> None:
    cfg_path = _demo_config_path("demo_binding_sites")
    cfg = load_config(cfg_path)
    sites_input = next(inp for inp in cfg.root.densegen.inputs if inp.type == "binding_sites")
    sites_path = resolve_path(cfg_path, sites_input.path)
    assert sites_path.exists()

    with sites_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        lengths = [len((row.get("tfbs") or "").strip()) for row in reader]
    assert lengths
    assert min(lengths) >= 16
    assert max(lengths) <= 20


def test_binding_sites_demo_plan_compares_unconstrained_and_sigma70() -> None:
    cfg_path = _demo_config_path("demo_binding_sites")
    cfg = load_config(cfg_path)
    plan = cfg.root.densegen.generation.plan
    assert plan
    plan_by_name = {item.name: item for item in plan}
    assert set(plan_by_name) == {"baseline", "baseline_sigma70"}

    baseline = plan_by_name["baseline"]
    assert list(baseline.regulator_constraints.groups or []) == []
    assert list(baseline.fixed_elements.promoter_constraints or []) == []

    sigma70 = plan_by_name["baseline_sigma70"]
    assert list(sigma70.regulator_constraints.groups or []) == []
    pcs = list(sigma70.fixed_elements.promoter_constraints or [])
    assert len(pcs) == 1
    pc = pcs[0]
    assert pc.name == "sigma70_consensus"
    assert pc.upstream == "TTGACA"
    assert pc.downstream == "TATAAT"
    assert tuple(pc.spacer_length or ()) == (16, 18)

    for item in plan:
        side_biases = item.fixed_elements.side_biases
        if side_biases is None:
            continue
        assert list(side_biases.left or []) == []
        assert list(side_biases.right or []) == []


def test_binding_sites_demo_uses_local_output_with_padding_enabled() -> None:
    cfg_path = _demo_config_path("demo_binding_sites")
    cfg = load_config(cfg_path)
    output = cfg.root.densegen.output
    assert output.targets == ["parquet"]
    assert cfg.root.densegen.postprocess.pad.mode == "adaptive"


def test_packaged_demo_configs_track_latest_schema_version() -> None:
    for workspace_id in ("demo_binding_sites", "demo_meme_three_tfs"):
        cfg = load_config(_demo_config_path(workspace_id))
        assert cfg.root.densegen.schema_version == LATEST_SCHEMA_VERSION
