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

PACKAGED_WORKSPACE_IDS = (
    "demo_tfbs_baseline",
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)

WORKSPACE_TUTORIAL_PATHS = {
    "demo_tfbs_baseline": "src/dnadesign/densegen/docs/tutorials/demo_tfbs_baseline.md",
    "demo_sampling_baseline": "src/dnadesign/densegen/docs/tutorials/demo_sampling_baseline.md",
    "study_constitutive_sigma_panel": "src/dnadesign/densegen/docs/tutorials/study_constitutive_sigma_panel.md",
    "study_stress_ethanol_cipro": "src/dnadesign/densegen/docs/tutorials/study_stress_ethanol_cipro.md",
}

USR_WORKSPACE_IDS = (
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)


def _demo_config_path(workspace_id: str) -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces" / workspace_id / "config.yaml"


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


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


def test_demo_sampling_baseline_default_plots_cover_core_diagnostics() -> None:
    cfg_path = _demo_config_path("demo_sampling_baseline")
    cfg = load_config(cfg_path)
    plots = cfg.root.plots
    assert plots is not None
    assert list(plots.default or []) == ["stage_a_summary", "placement_map", "run_health", "tfbs_usage"]


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
        "study_constitutive_sigma_panel",
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


def test_packaged_motif_artifact_manifests_use_active_cruncher_workspaces() -> None:
    expected_config_paths = {
        "demo_sampling_baseline": "src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml",
        "study_stress_ethanol_cipro": "src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml",
        "study_constitutive_sigma_panel": "src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml",
    }
    for workspace_id, expected_config in expected_config_paths.items():
        manifest_path = _demo_config_path(workspace_id).parent / "inputs" / "motif_artifacts" / "artifact_manifest.json"
        payload = json.loads(manifest_path.read_text())
        config_path = str(payload.get("config_path") or "")
        assert "densegen_prep_three_tf" not in config_path
        assert config_path == expected_config
        assert (Path(__file__).resolve().parents[5] / config_path).exists()


def test_gitignore_workspaces_allowlist_matches_packaged_workspace_names() -> None:
    gitignore = (Path(__file__).resolve().parents[5] / ".gitignore").read_text()
    assert "!src/dnadesign/densegen/workspaces/demo_tfbs_baseline/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_sampling_baseline/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_tfbs_baseline/runbook.md" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_sampling_baseline/runbook.md" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/runbook.md" in gitignore
    assert "!src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/runbook.md" in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_binding_sites/" not in gitignore
    assert "!src/dnadesign/densegen/workspaces/demo_meme_three_tfs/" not in gitignore


def test_study_constitutive_sigma_panel_focuses_on_fixed_elements() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    input_types = [inp.type for inp in cfg.root.densegen.inputs]
    assert input_types.count("background_pool") == 1
    assert input_types.count("pwm_artifact") == 2
    plan = list(cfg.root.densegen.generation.plan or [])
    assert len(plan) == 48

    promoter_pairs: set[tuple[str, str]] = set()
    for item in plan:
        assert item.name.startswith("sigma70_")
        pcs = list(item.fixed_elements.promoter_constraints or [])
        assert len(pcs) == 1
        pc = pcs[0]
        assert len(pc.upstream) == 6
        assert len(pc.downstream) == 6
        promoter_pairs.add((pc.upstream, pc.downstream))
        assert list(item.regulator_constraints.groups or []) == []

    assert len(promoter_pairs) == 48


def test_study_constitutive_sigma_panel_uses_laci_arac_background_exclusion() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    inputs = list(cfg.root.densegen.inputs)
    pwm_inputs = [inp for inp in inputs if inp.type == "pwm_artifact"]
    assert [inp.name for inp in pwm_inputs] == ["lacI_pwm", "araC_pwm"]
    assert all(inp.sampling.trimming.window_length == 16 for inp in pwm_inputs)

    cfg_path = _demo_config_path("study_constitutive_sigma_panel")
    for inp in pwm_inputs:
        resolved = resolve_path(cfg_path, inp.path)
        assert resolved.exists()

    background = next(inp for inp in inputs if inp.type == "background_pool")
    assert background.sampling.n_sites == 1200
    assert background.sampling.mining.batch_size == 25000
    assert background.sampling.mining.budget.mode == "fixed_candidates"
    assert background.sampling.mining.budget.candidates == 8_000_000
    fimo_cfg = background.sampling.filters.fimo_exclude
    assert fimo_cfg is not None
    assert list(fimo_cfg.pwms_input) == ["lacI_pwm", "araC_pwm"]
    assert fimo_cfg.allow_zero_hit_only is True
    assert fimo_cfg.max_score_norm is None


def test_study_constitutive_sigma_panel_uses_bounded_total_quota_limits() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    generation = cfg.root.densegen.generation
    assert generation.expansion.max_plans == 64
    assert generation.total_quota() == 48


def test_study_constitutive_sigma_panel_expansion_uses_uniform_quota_units() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    plans = list(cfg.root.densegen.generation.plan or [])
    assert plans
    assert all(int(item.sequences) == 1 for item in plans)
    assert all("__sig35=" in item.name and "__sig10=" in item.name for item in plans)
    assert all("__up=" not in item.name and "__down=" not in item.name for item in plans)

    panel_plans = [item for item in plans if item.name.startswith("sigma70_panel__")]
    assert len(panel_plans) == 48


def test_study_constitutive_sigma_panel_runtime_allows_bounded_retries() -> None:
    cfg = load_config(_demo_config_path("study_constitutive_sigma_panel"))
    runtime = cfg.root.densegen.runtime
    assert runtime.max_failed_solutions >= 1
    assert runtime.max_seconds_per_plan >= 300


def test_packaged_workspace_plot_defaults_cover_primary_runtime_diagnostics() -> None:
    expected_defaults = {
        "demo_tfbs_baseline": {"stage_a_summary", "placement_map", "run_health", "tfbs_usage"},
        "demo_sampling_baseline": {"stage_a_summary", "placement_map", "run_health", "tfbs_usage"},
        "study_constitutive_sigma_panel": {"stage_a_summary", "placement_map", "run_health", "tfbs_usage"},
        "study_stress_ethanol_cipro": {"stage_a_summary", "placement_map", "run_health", "tfbs_usage"},
    }
    for workspace_id, expected in expected_defaults.items():
        cfg = load_config(_demo_config_path(workspace_id))
        plots = cfg.root.plots
        assert plots is not None
        assert set(plots.default) == expected


def test_matrix_studies_use_auto_scoped_stage_b_plot_defaults() -> None:
    for workspace_id in ("study_constitutive_sigma_panel", "study_stress_ethanol_cipro"):
        cfg = load_config(_demo_config_path(workspace_id))
        plots = cfg.root.plots
        assert plots is not None
        placement_opts = dict((plots.options or {}).get("placement_map") or {})
        tfbs_opts = dict((plots.options or {}).get("tfbs_usage") or {})
        assert placement_opts.get("scope") == "auto"
        assert int(placement_opts.get("max_plans", 0)) == 12
        assert tfbs_opts.get("scope") == "auto"
        assert int(tfbs_opts.get("max_plans", 0)) == 12


def test_study_stress_ethanol_cipro_uses_pwm_artifact_sampling() -> None:
    cfg = load_config(_demo_config_path("study_stress_ethanol_cipro"))
    output = cfg.root.densegen.output
    assert output.targets == ["parquet", "usr"]
    assert output.usr is not None
    assert output.usr.dataset == "densegen/study_stress_ethanol_cipro"
    assert output.usr.root == "outputs/usr_datasets"
    assert float(output.usr.health_event_interval_seconds) > 0

    input_types = [inp.type for inp in cfg.root.densegen.inputs]
    assert input_types.count("pwm_artifact") == 3
    assert input_types.count("background_pool") == 1
    plan_names = [item.name for item in cfg.root.densegen.generation.plan]
    assert len(plan_names) == 15
    assert len([name for name in plan_names if name.startswith("ethanol__sig35=")]) == 5
    assert len([name for name in plan_names if name.startswith("ciprofloxacin__sig35=")]) == 5
    assert len([name for name in plan_names if name.startswith("ethanol_ciprofloxacin__sig35=")]) == 5
    assert all("__sig35=" in name for name in plan_names)

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
    assert cfg.root.densegen.generation.expansion.max_plans == 64
    quotas_by_base = {
        "ethanol": [
            item.sequences for item in cfg.root.densegen.generation.plan if item.name.startswith("ethanol__sig35=")
        ],
        "ciprofloxacin": [
            item.sequences
            for item in cfg.root.densegen.generation.plan
            if item.name.startswith("ciprofloxacin__sig35=")
        ],
        "ethanol_ciprofloxacin": [
            item.sequences
            for item in cfg.root.densegen.generation.plan
            if item.name.startswith("ethanol_ciprofloxacin__sig35=")
        ],
    }
    assert set(quotas_by_base["ethanol"]) == {12}
    assert set(quotas_by_base["ciprofloxacin"]) == {12}
    assert set(quotas_by_base["ethanol_ciprofloxacin"]) == {16}
    assert cfg.root.densegen.generation.total_quota() == 200
    assert cfg.root.densegen.generation.sequence_constraints is not None
    assert "validate_final_sequence" not in cfg.root.densegen.postprocess.model_dump(exclude_none=False)


def test_packaged_workspace_stage_a_length_policy_is_range_16_20() -> None:
    workspace_ids = (
        "demo_sampling_baseline",
        "study_constitutive_sigma_panel",
        "study_stress_ethanol_cipro",
    )
    stage_a_sampling_input_types = {
        "background_pool",
        "pwm_meme",
        "pwm_meme_set",
        "pwm_jaspar",
        "pwm_matrix_csv",
        "pwm_artifact",
        "pwm_artifact_set",
    }

    for workspace_id in workspace_ids:
        cfg = load_config(_demo_config_path(workspace_id))
        for inp in cfg.root.densegen.inputs:
            if inp.type not in stage_a_sampling_input_types:
                continue
            length = inp.sampling.length
            assert length.policy == "range"
            assert tuple(length.range or ()) == (16, 20)


def test_packaged_sampling_workspace_regulator_constraints_match_packaged_pwm_artifacts() -> None:
    workspace_ids = ("demo_sampling_baseline", "study_stress_ethanol_cipro")
    for workspace_id in workspace_ids:
        cfg_path = _demo_config_path(workspace_id)
        cfg = load_config(cfg_path)
        available_motif_ids: set[str] = set()
        for inp in cfg.root.densegen.inputs:
            if inp.type != "pwm_artifact":
                continue
            artifact_path = resolve_path(cfg_path, inp.path)
            payload = json.loads(artifact_path.read_text())
            motif_id = str(payload.get("motif_id") or "").strip()
            assert motif_id, f"{workspace_id}: missing motif_id in {artifact_path}"
            available_motif_ids.add(motif_id)
        assert available_motif_ids, f"{workspace_id}: no packaged pwm_artifact motif ids found"

        for plan in cfg.root.densegen.generation.plan:
            groups = list(plan.regulator_constraints.groups or [])
            for group in groups:
                missing = sorted(set(group.members or []) - available_motif_ids)
                assert not missing, (
                    f"{workspace_id}/{plan.name}/{group.name}: regulator constraints reference missing motifs: "
                    f"{', '.join(missing)}. Available: {', '.join(sorted(available_motif_ids))}"
                )


def test_packaged_sampling_workspaces_allow_bounded_failed_solutions() -> None:
    for workspace_id in ("demo_sampling_baseline", "study_stress_ethanol_cipro"):
        cfg = load_config(_demo_config_path(workspace_id))
        assert cfg.root.densegen.runtime.max_failed_solutions >= 1
        assert cfg.root.densegen.runtime.max_failed_solutions_per_target > 0


def test_packaged_workspace_runbooks_exist_with_fast_path_and_stepwise_flow() -> None:
    workspace_root = Path(__file__).resolve().parents[2] / "workspaces"
    required_tokens = (
        "**Workspace Path**",
        "**Regulators**",
        "**Purpose**",
        "**Run This Single Command**",
        "Run this single command to do everything below:",
        "### Step-by-Step Commands",
        "set -euo pipefail",
        "dense validate-config",
        "dense run",
        "dense inspect run",
        "dense plot",
        "dense notebook generate",
        "dense campaign-reset",
    )
    for workspace_id in PACKAGED_WORKSPACE_IDS:
        runbook_path = workspace_root / workspace_id / "runbook.md"
        assert runbook_path.exists(), f"Missing packaged workspace runbook: {runbook_path}"
        content = runbook_path.read_text()
        assert content.lstrip().startswith("## "), f"{runbook_path}: runbook title must use level-2 header"
        assert "### Workspace Path" not in content, (
            f"{runbook_path}: Workspace Path should be a bold label, not a subheader"
        )
        assert "### Regulators" not in content, f"{runbook_path}: Regulators should be a bold label, not a subheader"
        assert "### Purpose" not in content, f"{runbook_path}: Purpose should be a bold label, not a subheader"
        assert "### Run This Single Command" not in content, (
            f"{runbook_path}: Run This Single Command should be a bold label, not a subheader"
        )
        for token in required_tokens:
            assert token in content, f"{runbook_path}: missing required runbook token: {token!r}"
        _assert_token_order(
            content,
            [
                "**Workspace Path**",
                "**Regulators**",
                "**Purpose**",
                "**Run This Single Command**",
                "Run this single command to do everything below:",
                "### Step-by-Step Commands",
                "set -euo pipefail",
                "dense validate-config",
                "dense run",
                "dense inspect run",
                "dense plot",
                "dense notebook generate",
                "dense campaign-reset",
            ],
            label=str(runbook_path),
        )
        if workspace_id in USR_WORKSPACE_IDS:
            assert "dense workspace init" in content, (
                f"{runbook_path}: runbook must stage a workspace for USR outputs before dense run"
            )
            assert "--output-mode both" in content, (
                f"{runbook_path}: runbook must create workspace with usr+parquet sink wiring"
            )
            assert "cruncher catalog export-densegen" in content, (
                f"{runbook_path}: runbook must include optional Cruncher artifact refresh command"
            )
            assert "### Optional artifact refresh from Cruncher" in content, (
                f"{runbook_path}: optional artifact refresh section should be explicit"
            )
        assert "### Optional workspace reset" in content, (
            f"{runbook_path}: optional workspace reset section should be explicit"
        )


def test_workspace_tutorials_link_to_workspace_runbooks_for_fast_path() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    for workspace_id, tutorial_rel_path in WORKSPACE_TUTORIAL_PATHS.items():
        tutorial_path = repo_root / tutorial_rel_path
        assert tutorial_path.exists(), f"Missing tutorial: {tutorial_path}"
        content = tutorial_path.read_text()
        runbook_rel = f"../../workspaces/{workspace_id}/runbook.md"
        assert runbook_rel in content, f"{tutorial_path}: missing runbook link {runbook_rel}"
        assert "Run this single command" in content, f"{tutorial_path}: missing single-command fast-path wording"
        if workspace_id in USR_WORKSPACE_IDS:
            assert "dense workspace init" in content, (
                f"{tutorial_path}: fast path must stage a workspace before running USR-backed flow"
            )
            assert "--output-mode both" in content, (
                f"{tutorial_path}: fast path must use workspace init with both sinks"
            )
            assert "cruncher catalog export-densegen" in content, (
                f"{tutorial_path}: tutorial must include Cruncher export refresh path"
            )


def test_usr_workspace_tutorials_reference_existing_cruncher_configs() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    expected_config_paths = {
        "demo_sampling_baseline": "src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml",
        "study_constitutive_sigma_panel": "src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml",
        "study_stress_ethanol_cipro": "src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml",
    }
    for workspace_id, expected_path in expected_config_paths.items():
        tutorial_path = repo_root / WORKSPACE_TUTORIAL_PATHS[workspace_id]
        content = tutorial_path.read_text()
        assert expected_path in content, f"{tutorial_path}: expected Cruncher config path '{expected_path}' not found"
        assert (repo_root / expected_path).exists(), (
            f"{tutorial_path}: referenced Cruncher config path does not exist: {expected_path}"
        )
        stale_path = expected_path.replace("/configs/config.yaml", "/config.yaml")
        assert stale_path not in content, f"{tutorial_path}: stale Cruncher config path detected: {stale_path}"


def test_cruncher_pwm_handoff_howto_references_existing_cruncher_configs() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    howto_path = repo_root / "src" / "dnadesign" / "densegen" / "docs" / "howto" / "cruncher_pwm_pipeline.md"
    content = howto_path.read_text()

    expected_paths = (
        "src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml",
        "src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml",
    )
    for expected_path in expected_paths:
        assert expected_path in content, f"{howto_path}: expected Cruncher config path '{expected_path}' not found"
        assert (repo_root / expected_path).exists(), (
            f"{howto_path}: referenced Cruncher config path does not exist: {expected_path}"
        )

    stale_paths = (
        "src/dnadesign/cruncher/workspaces/demo_multitf/config.yaml",
        "src/dnadesign/cruncher/workspaces/pairwise_laci_arac/config.yaml",
    )
    for stale_path in stale_paths:
        assert stale_path not in content, f"{howto_path}: stale Cruncher config path detected: {stale_path}"
