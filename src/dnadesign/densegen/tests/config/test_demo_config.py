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
import json
from pathlib import Path

from dnadesign.densegen.src.adapters.sources.base import resolve_path
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.config.base import LATEST_SCHEMA_VERSION


def _demo_config_path(workspace_id: str) -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces" / workspace_id / "config.yaml"


def test_demo_sampling_baseline_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_sampling_baseline"


def test_demo_artifacts_present() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
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


def test_demo_sampling_baseline_default_plots_are_minimal() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    cfg = load_config(cfg_path)
    plots = cfg.root.plots
    assert plots is not None
    assert list(plots.default or []) == ["stage_a_summary", "placement_map"]


def test_demo_sampling_baseline_uses_background_and_plan_specific_sigma70_spacers() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    cfg = load_config(cfg_path)

    input_names = [inp.name for inp in cfg.root.densegen.inputs]
    assert "background" in input_names
    assert "neutral_bg" not in input_names

    plan = cfg.root.densegen.generation.plan
    assert plan
    plan_by_name = {item.name: item for item in plan}

    expected_spacers = {
        "ciprofloxacin": (16, 18),
        "ethanol": (16, 20),
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

    seq_constraints = cfg.root.densegen.generation.sequence_constraints
    assert seq_constraints is not None
    assert "validate_final_sequence" not in cfg.root.densegen.postprocess.model_dump(exclude_none=False)


def test_demo_sampling_baseline_stage_a_sampling_targets() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    cfg = load_config(cfg_path)
    inputs = list(cfg.root.densegen.inputs)
    pwm_inputs = [inp for inp in inputs if inp.type == "pwm_artifact"]
    assert pwm_inputs
    for inp in pwm_inputs:
        sampling = inp.sampling
        assert sampling.n_sites == 100
        assert sampling.mining.budget.mode == "fixed_candidates"
        assert sampling.mining.budget.candidates == 150_000
        assert sampling.uniqueness.cross_regulator_core_collisions == "error"

    background_inputs = [inp for inp in inputs if inp.type == "background_pool"]
    assert len(background_inputs) == 1
    assert background_inputs[0].sampling.n_sites == 200


def test_demo_sampling_baseline_uses_both_outputs_and_cbc_solver() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    cfg = load_config(cfg_path)
    output = cfg.root.densegen.output
    solver = cfg.root.densegen.solver
    plots = cfg.root.plots

    assert output.targets == ["parquet", "usr"]
    assert output.parquet is not None
    assert output.parquet.path == "outputs/tables/records.parquet"
    assert output.usr is not None
    assert output.usr.root == "outputs/usr_datasets"
    assert output.usr.dataset == "densegen/demo_sampling_baseline"
    assert solver.backend == "CBC"
    assert solver.threads is None
    assert plots is not None
    assert plots.source == "usr"


def test_tfbs_baseline_demo_config_exists_and_loads() -> None:
    cfg_path = _demo_config_path("demo_tfbs_baseline")
    assert cfg_path.exists(), f"Missing demo config: {cfg_path}"
    loaded = load_config(cfg_path)
    assert loaded.root.densegen.run.id == "demo_tfbs_baseline"


def test_tfbs_baseline_demo_mock_sites_lengths_are_16_to_20_bp() -> None:
    cfg_path = _demo_config_path("demo_tfbs_baseline")
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


def test_tfbs_baseline_demo_plan_compares_unconstrained_and_sigma70() -> None:
    cfg_path = _demo_config_path("demo_tfbs_baseline")
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


def test_tfbs_baseline_demo_uses_local_output_with_padding_enabled() -> None:
    cfg_path = _demo_config_path("demo_tfbs_baseline")
    cfg = load_config(cfg_path)
    output = cfg.root.densegen.output
    assert output.targets == ["parquet"]
    assert cfg.root.densegen.postprocess.pad.mode == "adaptive"


def test_packaged_workspace_configs_track_latest_schema_version() -> None:
    for workspace_id in (
        "demo_tfbs_baseline",
        "demo_sampling_baseline",
        "study_constitutive_sigma_panel",
        "study_stress_ethanol_cipro",
    ):
        cfg = load_config(_demo_config_path(workspace_id))
        assert cfg.root.densegen.schema_version == LATEST_SCHEMA_VERSION


def test_packaged_workspace_semantic_ids_align_to_workspace_name() -> None:
    for workspace_id in (
        "demo_tfbs_baseline",
        "demo_sampling_baseline",
        "study_constitutive_sigma_panel",
        "study_stress_ethanol_cipro",
    ):
        cfg_path = _demo_config_path(workspace_id)
        cfg = load_config(cfg_path)
        assert cfg.root.densegen.run.id == workspace_id
        usr_cfg = cfg.root.densegen.output.usr
        if usr_cfg is not None:
            assert usr_cfg.dataset.endswith(workspace_id)
            assert usr_cfg.dataset.startswith("densegen/")


def test_packaged_workspace_configs_exclude_stale_legacy_namespaces() -> None:
    for workspace_id in (
        "demo_tfbs_baseline",
        "demo_sampling_baseline",
        "study_constitutive_sigma_panel",
        "study_stress_ethanol_cipro",
    ):
        raw = _demo_config_path(workspace_id).read_text()
        assert "demo_meme_three_tfs" not in raw
        assert "neutral_bg" not in raw
        assert "validate_final_sequence" not in raw
        assert "allow_overwrite" not in raw
        assert "match_exact_coordinates" not in raw
        assert "outside_allowed_placements" not in raw


def test_packaged_motif_artifact_manifests_are_workspace_local_and_current() -> None:
    workspace_ids = (
        "demo_sampling_baseline",
        "study_stress_ethanol_cipro",
    )
    for workspace_id in workspace_ids:
        manifest_path = _demo_config_path(workspace_id).parent / "inputs" / "motif_artifacts" / "artifact_manifest.json"
        payload = json.loads(manifest_path.read_text())
        artifacts = list(payload.get("artifacts") or [])
        assert artifacts
        raw = manifest_path.read_text()
        assert "demo_meme_three_tfs" not in raw
        for artifact in artifacts:
            rel = Path(str(artifact.get("path") or ""))
            assert rel.as_posix() == rel.name
            assert not rel.is_absolute()
            assert (manifest_path.parent / rel).exists()
        catalog_root = str(payload.get("catalog_root") or "")
        config_path = str(payload.get("config_path") or "")
        assert not Path(catalog_root).is_absolute()
        assert not Path(config_path).is_absolute()


def test_gitignore_workspaces_allowlist_matches_packaged_workspace_names() -> None:
    gitignore = (Path(__file__).resolve().parents[5] / ".gitignore").read_text()
    assert "!src/dnadesign/densegen/workspaces/demo_tfbs_baseline/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_sampling_baseline/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_binding_sites/" not in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_meme_three_tfs/" not in gitignore


def test_study_constitutive_sigma_panel_focuses_on_fixed_elements() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    input_types = {inp.type for inp in cfg.root.densegen.inputs}
    assert input_types == {"background_pool"}
    plan = list(cfg.root.densegen.generation.plan or [])
    assert len(plan) >= 6

    promoter_pairs: set[tuple[str, str]] = set()
    for item in plan:
        pcs = list(item.fixed_elements.promoter_constraints or [])
        assert len(pcs) == 1
        pc = pcs[0]
        assert len(pc.upstream) == 6
        assert len(pc.downstream) == 6
        promoter_pairs.add((pc.upstream, pc.downstream))
        assert list(item.regulator_constraints.groups or []) == []

    assert len(promoter_pairs) >= 6


def test_study_constitutive_sigma_panel_uses_bounded_total_quota_templates() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    generation = cfg.root.densegen.generation
    templates = list(cfg.root.densegen.generation.plan_templates or [])
    assert templates
    for template in templates:
        assert template.quota_per_variant is None
        assert template.total_quota is not None
    assert generation.plan_template_max_expanded_plans <= 32
    assert generation.plan_template_max_total_quota <= 64
    assert generation.total_quota() <= generation.plan_template_max_total_quota


def test_study_stress_ethanol_cipro_uses_pwm_artifact_sampling() -> None:
    cfg = load_config(_demo_config_path("study_stress_ethanol_cipro"))
    input_types = [inp.type for inp in cfg.root.densegen.inputs]
    assert input_types.count("pwm_artifact") == 3
    assert input_types.count("background_pool") == 1
    plan_names = [item.name for item in cfg.root.densegen.generation.plan]
    assert plan_names == ["ethanol", "ciprofloxacin", "ethanol_ciprofloxacin"]

    pwm_inputs = [inp for inp in cfg.root.densegen.inputs if inp.type == "pwm_artifact"]
    assert len(pwm_inputs) == 3
    for inp in pwm_inputs:
        assert inp.sampling.n_sites == 250
        assert inp.sampling.mining.batch_size == 5000
        assert inp.sampling.mining.budget.mode == "fixed_candidates"
        assert inp.sampling.mining.budget.candidates == 1_000_000

    background = next(inp for inp in cfg.root.densegen.inputs if inp.type == "background_pool")
    assert background.sampling.n_sites == 500
    assert background.sampling.mining.batch_size == 20000
    assert background.sampling.mining.budget.mode == "fixed_candidates"
    assert background.sampling.mining.budget.candidates == 5_000_000

    sampling = cfg.root.densegen.generation.sampling
    assert sampling.library_size == 10
    assert cfg.root.densegen.generation.sequence_constraints is not None
    assert "validate_final_sequence" not in cfg.root.densegen.postprocess.model_dump(exclude_none=False)
