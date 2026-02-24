"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_docs_path_contracts.py

Validate public docs against the current run-artifact path contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

NEW_OPTIMIZATION_WORKSPACES = (
    "pairwise_cpxr_baer",
    "pairwise_cpxr_lexa",
    "pairwise_baer_lexa",
    "pairwise_cpxr_soxr",
    "pairwise_baer_soxr",
    "pairwise_soxr_soxs",
    "multitf_cpxr_baer_lexa",
    "multitf_baer_lexa_soxr",
    "multitf_baer_lexa_soxr_soxs",
)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_substrings_in_order(text: str, substrings: list[str], *, label: str) -> None:
    cursor = -1
    for token in substrings:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing required token: {token}"
        assert idx > cursor, f"{label}: token appears out of order: {token}"
        cursor = idx


def test_artifacts_reference_uses_analysis_subdir_contract() -> None:
    artifacts = (_package_root() / "docs" / "reference" / "artifacts.md").read_text()
    assert "analysis/reports/summary.json" in artifacts
    assert "analysis/reports/report.md" in artifacts
    assert "outputs/analysis/reports/summary.json" not in artifacts
    assert "outputs/analysis/reports/report.md" not in artifacts
    assert "outputs/analysis/summary.json" not in artifacts
    assert "outputs/analysis/report.md" not in artifacts
    assert "outputs/output/" not in artifacts


def test_readme_routes_to_demos_not_quickstart() -> None:
    readme = (_package_root() / "README.md").read_text()
    assert "docs/demos/demo_pairwise.md" in readme
    assert "docs/demos/demo_multitf.md" in readme
    assert "docs/demos/project_all_tfs.md" in readme
    assert "Quickstart" not in readme
    assert "pixi run cruncher --" not in readme


def test_root_dependencies_doc_references_existing_cruncher_workspace_config() -> None:
    repo_root = _package_root().parents[2]
    deps_doc = (repo_root / "docs" / "dependencies.md").read_text()
    expected = "src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml"
    stale = "src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml"
    assert expected in deps_doc
    assert stale not in deps_doc
    assert (repo_root / expected).exists()


def test_multitf_demo_reset_is_shell_safe_without_nomatch_globs() -> None:
    demo_doc = (_package_root() / "docs" / "demos" / "demo_multitf.md").read_text()
    assert "rm -f campaign_*.yaml campaign_*.campaign_manifest.json" not in demo_doc


def test_docs_define_baserender_public_api_boundary() -> None:
    architecture = (_package_root() / "docs" / "reference" / "architecture.md").read_text()
    assert "from dnadesign.baserender import" in architecture
    disallowed_path_marker = "dnadesign.baserender" + ".src.*"
    assert disallowed_path_marker in architecture

    analysis_guide = (_package_root() / "docs" / "guides" / "sampling_and_analysis.md").read_text()
    assert "minimal rendering primitives" in analysis_guide
    assert "dnadesign.baserender" in analysis_guide


def test_demo_docs_encode_merged_meme_oops_provenance_pattern() -> None:
    docs_root = _package_root() / "docs" / "demos"
    pairwise = (docs_root / "demo_pairwise.md").read_text()
    multitf = (docs_root / "demo_multitf.md").read_text()

    assert "fetch sites --source demo_local_meme --tf lexA --tf cpxR" in pairwise
    assert "fetch sites --source regulondb      --tf lexA --tf cpxR" in pairwise
    assert "--tool meme --meme-mod oops --source-id demo_merged_meme_oops" in pairwise

    assert "fetch sites --source demo_local_meme --tf lexA --tf cpxR" in multitf
    assert "fetch sites --source regulondb      --tf lexA --tf cpxR --tf baeR" in multitf
    assert "fetch sites --source baer_chip_exo --tf baeR" in multitf
    assert "--tool meme --meme-mod oops --source-id demo_merged_meme_oops_multitf" in multitf


def test_demo_length_study_respects_workspace_motif_width_bounds() -> None:
    workspace = _package_root() / "workspaces" / "demo_pairwise"
    config_payload = yaml.safe_load((workspace / "configs" / "config.yaml").read_text())
    assert isinstance(config_payload, dict)
    cruncher_payload = config_payload.get("cruncher")
    assert isinstance(cruncher_payload, dict)
    sample_payload = cruncher_payload.get("sample")
    assert isinstance(sample_payload, dict)
    motif_width_payload = sample_payload.get("motif_width")
    assert isinstance(motif_width_payload, dict)
    maxw = motif_width_payload.get("maxw")
    if maxw is not None:
        assert isinstance(maxw, int)

    study_payload = yaml.safe_load((workspace / "configs" / "studies" / "length_vs_score.study.yaml").read_text())
    assert isinstance(study_payload, dict)
    study_section = study_payload.get("study")
    assert isinstance(study_section, dict)
    trials = study_section.get("trials", [])
    assert isinstance(trials, list)
    trial_grids = study_section.get("trial_grids", [])
    assert isinstance(trial_grids, list) and trial_grids

    effective_maxw = maxw
    lengths: list[int] = []
    for trial in trials:
        assert isinstance(trial, dict)
        factors = trial.get("factors")
        assert isinstance(factors, dict)
        if "sample.sequence_length" in factors:
            lengths.append(int(factors["sample.sequence_length"]))
    for grid in trial_grids:
        assert isinstance(grid, dict)
        factors = grid.get("factors")
        assert isinstance(factors, dict)
        grid_lengths = factors.get("sample.sequence_length")
        if grid_lengths is None:
            continue
        assert isinstance(grid_lengths, list) and grid_lengths
        lengths.extend(int(item) for item in grid_lengths)

    assert lengths, "length_vs_score must define sequence lengths"
    for length in sorted(set(lengths)):
        assert length >= 1
        if effective_maxw is not None:
            assert isinstance(effective_maxw, int)
            assert length >= effective_maxw


def test_demo_study_specs_use_canonical_artifact_profile_without_output_root_override() -> None:
    workspace = _package_root() / "workspaces" / "demo_pairwise"
    for spec_name in ("diversity_vs_score.study.yaml", "length_vs_score.study.yaml"):
        payload = yaml.safe_load((workspace / "configs" / "studies" / spec_name).read_text())
        assert isinstance(payload, dict)
        study_section = payload.get("study")
        assert isinstance(study_section, dict)
        artifacts = study_section.get("artifacts")
        assert isinstance(artifacts, dict)
        assert artifacts.get("trial_output_profile") == "minimal"
        assert "output_root" not in artifacts


def test_demo_length_and_diversity_study_sweep_ranges() -> None:
    workspace = _package_root() / "workspaces" / "demo_pairwise"

    length_payload = yaml.safe_load((workspace / "configs" / "studies" / "length_vs_score.study.yaml").read_text())
    assert isinstance(length_payload, dict)
    length_study = length_payload.get("study")
    assert isinstance(length_study, dict)
    trial_grids = length_study.get("trial_grids")
    assert isinstance(trial_grids, list) and trial_grids
    length_values = trial_grids[0]["factors"]["sample.sequence_length"]
    assert isinstance(length_values, list)
    assert length_values[0] == 15
    assert length_values[-1] == 49
    assert 18 in length_values
    odd_lengths = [value for value in length_values if value != 18]
    assert odd_lengths == list(range(15, 50, 2))

    diversity_payload = yaml.safe_load(
        (workspace / "configs" / "studies" / "diversity_vs_score.study.yaml").read_text()
    )
    assert isinstance(diversity_payload, dict)
    diversity_study = diversity_payload.get("study")
    assert isinstance(diversity_study, dict)
    replays = diversity_study.get("replays")
    assert isinstance(replays, dict)
    mmr_sweep = replays.get("mmr_sweep")
    assert isinstance(mmr_sweep, dict)
    diversity_values = mmr_sweep.get("diversity_values")
    assert isinstance(diversity_values, list)
    assert diversity_values == [round(v * 0.05, 2) for v in range(0, 21)]
    diversity_trials = diversity_study.get("trials")
    assert isinstance(diversity_trials, list) and diversity_trials
    assert "sample.sequence_length" not in diversity_trials[0]["factors"]


def test_demo_and_workspace_docs_preserve_end_to_end_flow_order() -> None:
    docs_root = _package_root()
    flow_tokens = [
        "fetch sites",
        "discover motifs",
        'lock -c "$CONFIG"',
        'parse --force-overwrite -c "$CONFIG"',
        'sample --force-overwrite -c "$CONFIG"',
        'analyze --summary -c "$CONFIG"',
        'export sequences --latest -c "$CONFIG"',
        "catalog logos",
    ]

    pairwise = (docs_root / "docs" / "demos" / "demo_pairwise.md").read_text()
    multitf = (docs_root / "docs" / "demos" / "demo_multitf.md").read_text()
    project = (docs_root / "docs" / "demos" / "project_all_tfs.md").read_text()

    _assert_substrings_in_order(pairwise, flow_tokens, label="demo_pairwise.md")
    _assert_substrings_in_order(multitf, flow_tokens, label="demo_multitf.md")
    _assert_substrings_in_order(project, flow_tokens, label="project_all_tfs.md")


def test_demo_docs_use_workspace_reset_command_for_clean_reruns() -> None:
    docs_root = _package_root() / "docs" / "demos"
    for name in ("demo_pairwise.md", "demo_multitf.md", "project_all_tfs.md"):
        text = (docs_root / name).read_text()
        assert "cruncher workspaces reset --root . --confirm" in text
        assert "rm -rf outputs" not in text
        assert "rm -rf .cruncher/parse .cruncher/locks" not in text


def test_demo_docs_explain_workspace_studies_and_invocation() -> None:
    docs_root = _package_root() / "docs" / "demos"
    for name in ("demo_pairwise.md", "demo_multitf.md", "project_all_tfs.md"):
        text = (docs_root / name).read_text()
        assert "Run this single command to do everything in this demo:" in text
        assert "workspaces run --runbook configs/runbook.yaml" in text
        assert "Or run the same flow step by step with context below." in text
        assert "study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite" in text
        assert "study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite" in text
        assert "length_vs_score" in text
        assert "diversity_vs_score" in text


def test_demo_docs_reference_elites_csv_export_contract() -> None:
    docs_root = _package_root() / "docs" / "demos"
    for name in ("demo_pairwise.md", "demo_multitf.md"):
        text = (docs_root / name).read_text()
        assert "outputs/export/table__elites.csv" in text
        assert "outputs/export/export_manifest.json" in text
        assert "outputs/export/sequences/export_manifest.json" not in text


def test_studies_guide_shows_separate_length_and_diversity_paths() -> None:
    guide = (_package_root() / "docs" / "guides" / "studies.md").read_text()
    assert "study run --spec configs/studies/length_vs_score.study.yaml" in guide
    assert "study run --spec configs/studies/diversity_vs_score.study.yaml" in guide
    assert "study show --run outputs/studies/length_vs_score/<study_id>" in guide
    assert "study show --run outputs/studies/diversity_vs_score/<study_id>" in guide
    assert "open outputs/plots/study__length_vs_score__<study_id>__plot__sequence_length_tradeoff.pdf" in guide
    assert "open outputs/plots/study__diversity_vs_score__<study_id>__plot__mmr_diversity_tradeoff.pdf" in guide


def test_demo_docs_call_out_lock_refresh_after_provenance_changes() -> None:
    docs_root = _package_root() / "docs" / "demos"
    required_note = (
        "If you change `catalog.source_preference` or discovery `--source-id`, "
        're-run `cruncher lock -c "$CONFIG"` before parse.'
    )
    for name in ("demo_pairwise.md", "demo_multitf.md", "project_all_tfs.md"):
        text = (docs_root / name).read_text()
        assert required_note in text, f"{name}: missing lock refresh provenance note"


def test_workspaces_readme_preserves_lifecycle_and_transient_cleanup_contract() -> None:
    readme = (_package_root() / "workspaces" / "README.md").read_text()
    flow_tokens = [
        "fetch sites",
        "discover motifs",
        'lock -c "$CONFIG"',
        'parse --force-overwrite -c "$CONFIG"',
        'sample --force-overwrite -c "$CONFIG"',
        'analyze --summary -c "$CONFIG"',
        'export sequences --latest -c "$CONFIG"',
    ]
    _assert_substrings_in_order(readme, flow_tokens, label="workspaces/README.md")
    assert (
        "If you change `catalog.source_preference` or discovery `--source-id`, "
        're-run `cruncher lock -c "$CONFIG"` before parse.'
    ) in readme
    assert "cruncher workspaces reset --root src/dnadesign/cruncher/workspaces" in readme
    assert "cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --confirm" in readme


def test_docs_use_flat_plots_output_layout() -> None:
    docs_root = _package_root() / "docs"
    paths = [
        docs_root / "reference" / "cli.md",
        docs_root / "reference" / "architecture.md",
        docs_root / "internals" / "spec.md",
        docs_root / "demos" / "demo_pairwise.md",
        docs_root / "demos" / "demo_multitf.md",
        docs_root / "demos" / "project_all_tfs.md",
    ]
    for path in paths:
        text = path.read_text()
        assert "plots/analysis/" not in text, f"{path}: remove nested plots/analysis references"
        assert "plots/logos/" not in text, f"{path}: remove nested plots/logos references"


def test_docs_define_repo_shared_matplotlib_cache_contract() -> None:
    docs_root = _package_root() / "docs"
    architecture = (docs_root / "reference" / "architecture.md").read_text()
    ingestion = (docs_root / "guides" / "ingestion.md").read_text()
    internals = (docs_root / "internals" / "spec.md").read_text()

    expected = ".cache/matplotlib/cruncher"
    assert expected in architecture
    assert expected in ingestion
    assert expected in internals

    disallowed = "<catalog.root>/.mplcache"
    assert disallowed not in architecture
    assert disallowed not in internals
    assert ".mplcache/" not in ingestion


def test_docs_define_strict_ht_dataset_contracts() -> None:
    docs_root = _package_root() / "docs"
    ingestion = (docs_root / "guides" / "ingestion.md").read_text()
    cli = (docs_root / "reference" / "cli.md").read_text()
    config = (docs_root / "reference" / "config.md").read_text()
    assert "HT contracts are strict (no curated fallback)" in ingestion
    assert "`fetch sites --limit <N>` without `--dataset-id` is rejected" in ingestion
    assert "with both curated and HT enabled, `--limit` requires explicit mode" in cli
    assert "`sources datasets --dataset-source <X>` performs a strict row-level source filter" in cli
    assert "`ht_sites: true` is strict." in config


def test_cli_densegen_export_examples_use_active_workspace_names() -> None:
    cli = (_package_root() / "docs" / "reference" / "cli.md").read_text()
    assert "demo_meme_three_tfs" not in cli
    assert "densegen_prep_three_tf" not in cli
    assert "cruncher catalog export-sites --set 1 --densegen-workspace demo_tfbs_baseline <config>" in cli
    assert "cruncher catalog export-densegen --set 1 --densegen-workspace demo_sampling_baseline <config>" in cli


def test_new_optimization_workspaces_have_config_and_runbook_with_flow_order() -> None:
    workspaces_root = _package_root() / "workspaces"
    flow_tokens = [
        "fetch sites --source",
        "discover motifs --set 1 --tool meme --meme-mod oops --source-id",
        'lock -c "$CONFIG"',
        'parse --force-overwrite -c "$CONFIG"',
        'sample --force-overwrite -c "$CONFIG"',
        'analyze --summary -c "$CONFIG"',
        'export sequences --latest -c "$CONFIG"',
        "catalog logos --source",
    ]
    for workspace_name in NEW_OPTIMIZATION_WORKSPACES:
        workspace = workspaces_root / workspace_name
        assert workspace.is_dir(), f"missing workspace directory: {workspace_name}"
        config_path = workspace / "configs" / "config.yaml"
        runbook_path = workspace / "runbook.md"
        assert config_path.exists(), f"{workspace_name}: missing config.yaml"
        assert runbook_path.exists(), f"{workspace_name}: missing runbook.md"
        runbook_text = runbook_path.read_text()
        assert "workspaces reset --root . --confirm" in runbook_text
        _assert_substrings_in_order(runbook_text, flow_tokens, label=f"{workspace_name}/runbook.md")


def test_new_optimization_workspaces_use_available_sources_for_each_regulator() -> None:
    workspaces_root = _package_root() / "workspaces"
    local_meme_tfs = {"cpxR", "lexA", "soxR"}
    for workspace_name in NEW_OPTIMIZATION_WORKSPACES:
        config_path = workspaces_root / workspace_name / "configs" / "config.yaml"
        runbook_path = workspaces_root / workspace_name / "runbook.md"
        config_payload = yaml.safe_load(config_path.read_text())
        assert isinstance(config_payload, dict)
        cruncher_payload = config_payload.get("cruncher")
        assert isinstance(cruncher_payload, dict)
        workspace_payload = cruncher_payload.get("workspace")
        assert isinstance(workspace_payload, dict)
        regulator_sets = workspace_payload.get("regulator_sets")
        assert isinstance(regulator_sets, list) and len(regulator_sets) == 1
        regulators = regulator_sets[0]
        assert isinstance(regulators, list) and regulators
        regulators_set = {str(tf) for tf in regulators}

        catalog_payload = cruncher_payload.get("catalog")
        discover_payload = cruncher_payload.get("discover")
        assert isinstance(catalog_payload, dict)
        assert isinstance(discover_payload, dict)
        source_preference = catalog_payload.get("source_preference")
        assert isinstance(source_preference, list) and len(source_preference) == 1
        discover_source_id = discover_payload.get("source_id")
        assert isinstance(discover_source_id, str) and discover_source_id
        assert source_preference == [discover_source_id]

        runbook_text = runbook_path.read_text()
        assert "fetch sites --source regulondb" in runbook_text
        for tf in regulators_set:
            assert f"--tf {tf}" in runbook_text

        if regulators_set & local_meme_tfs:
            assert "fetch sites --source demo_local_meme" in runbook_text
        if "baeR" in regulators_set:
            assert "fetch sites --source baer_chip_exo --tf baeR" in runbook_text

        motif_input_root = workspaces_root / workspace_name / "inputs" / "local_motifs"
        for tf in sorted(regulators_set & local_meme_tfs):
            motif_path = motif_input_root / f"{tf}.txt"
            assert motif_path.exists(), f"{workspace_name}: missing local motif input for {tf}"


def test_new_optimization_workspaces_share_sample_and_optimizer_hyperparams() -> None:
    workspaces_root = _package_root() / "workspaces"
    baseline_workspace = NEW_OPTIMIZATION_WORKSPACES[0]
    expected_sequence_lengths = {
        "pairwise_cpxr_baer": 18,
        "pairwise_cpxr_lexa": 18,
        "pairwise_baer_lexa": 18,
        "pairwise_cpxr_soxr": 18,
        "pairwise_baer_soxr": 18,
        "pairwise_soxr_soxs": 18,
        "multitf_cpxr_baer_lexa": 18,
        "multitf_baer_lexa_soxr": 18,
        "multitf_baer_lexa_soxr_soxs": 18,
    }
    baseline_payload = yaml.safe_load((workspaces_root / baseline_workspace / "configs" / "config.yaml").read_text())
    assert isinstance(baseline_payload, dict)
    baseline_cruncher = baseline_payload.get("cruncher")
    assert isinstance(baseline_cruncher, dict)
    baseline_sample = baseline_cruncher.get("sample")
    assert isinstance(baseline_sample, dict)
    baseline_sample_common = dict(baseline_sample)
    baseline_sample_common.pop("sequence_length", None)
    for workspace_name in NEW_OPTIMIZATION_WORKSPACES[1:]:
        payload = yaml.safe_load((workspaces_root / workspace_name / "configs" / "config.yaml").read_text())
        assert isinstance(payload, dict)
        cruncher_payload = payload.get("cruncher")
        assert isinstance(cruncher_payload, dict)
        sample_payload = cruncher_payload.get("sample")
        assert isinstance(sample_payload, dict)
        assert sample_payload.get("sequence_length") == expected_sequence_lengths[workspace_name], (
            f"{workspace_name}: sample.sequence_length must match width-derived minimum."
        )
        sample_common = dict(sample_payload)
        sample_common.pop("sequence_length", None)
        assert sample_common == baseline_sample_common, (
            f"{workspace_name}: sample hyperparameter block drifted from shared profile."
        )
