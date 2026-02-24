"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_master_orchestration_contracts.py

Contracts for workspace study wiring and master portfolio orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = ROOT / "workspaces"
PORTFOLIO_WORKSPACE = WORKSPACES_ROOT / "portfolios"
MASTER_SPEC = PORTFOLIO_WORKSPACE / "configs" / "master_all_workspaces.portfolio.yaml"


def _workspace_names() -> list[str]:
    return sorted(path.name for path in WORKSPACES_ROOT.iterdir() if path.is_dir())


def _non_portfolio_workspaces() -> list[str]:
    excluded = {"portfolio", "portfolios"}
    return [name for name in _workspace_names() if name not in excluded]


def test_all_non_portfolio_workspaces_have_length_and_diversity_study_specs() -> None:
    for workspace_name in _non_portfolio_workspaces():
        length_spec = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "length_vs_score.study.yaml"
        diversity_spec = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "diversity_vs_score.study.yaml"
        assert length_spec.exists(), f"{workspace_name}: missing configs/studies/length_vs_score.study.yaml"
        assert diversity_spec.exists(), f"{workspace_name}: missing configs/studies/diversity_vs_score.study.yaml"


def test_all_non_portfolio_workspaces_do_not_keep_legacy_portfolio_ready_studies() -> None:
    for workspace_name in _non_portfolio_workspaces():
        legacy = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "portfolio_ready.study.yaml"
        assert not legacy.exists(), f"{workspace_name}: remove legacy {legacy.name}"


def test_all_non_portfolio_workspaces_runbooks_have_length_and_diversity_study_steps() -> None:
    for workspace_name in _non_portfolio_workspaces():
        runbook_path = WORKSPACES_ROOT / workspace_name / "configs" / "runbook.yaml"
        payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
        runbook = payload["runbook"]
        steps = runbook["steps"]
        by_id = {item.get("id"): item for item in steps}
        for step_id, spec_name in [
            ("study_run_length_vs_score", "configs/studies/length_vs_score.study.yaml"),
            ("study_run_diversity_vs_score", "configs/studies/diversity_vs_score.study.yaml"),
        ]:
            assert step_id in by_id, f"{workspace_name}: missing {step_id} step"
            run = by_id[step_id]["run"]
            assert run[:2] == ["study", "run"], f"{workspace_name}: {step_id} must invoke study run"
            assert run[-1] == "--force-overwrite", f"{workspace_name}: {step_id} must force_overwrite"
            assert spec_name in run, f"{workspace_name}: {step_id} must target {spec_name}"


def test_master_portfolio_spec_exists_and_covers_every_non_portfolio_workspace() -> None:
    assert MASTER_SPEC.exists(), "missing master portfolio spec"
    payload = yaml.safe_load(MASTER_SPEC.read_text(encoding="utf-8"))
    portfolio = payload["portfolio"]

    assert portfolio["schema_version"] == 3
    assert portfolio["execution"]["mode"] == "prepare_then_aggregate"
    studies = portfolio.get("studies")
    assert isinstance(studies, dict)
    assert studies.get("ensure_specs") == [
        "configs/studies/length_vs_score.study.yaml",
        "configs/studies/diversity_vs_score.study.yaml",
    ]
    sequence_length_table = studies.get("sequence_length_table")
    assert isinstance(sequence_length_table, dict)
    assert sequence_length_table.get("enabled") is True
    assert sequence_length_table.get("study_spec") == "configs/studies/length_vs_score.study.yaml"
    assert sequence_length_table.get("top_n_lengths") == 6

    expected = _non_portfolio_workspaces()
    sources = portfolio["sources"]
    seen_workspaces = sorted(Path(item["workspace"]).name for item in sources)
    assert seen_workspaces == expected

    required_prepare_steps = [
        "fetch_sites_regulondb",
        "discover_motifs",
        "render_logos",
        "lock_targets",
        "parse_run",
        "sample_run",
        "analyze_summary",
        "export_sequences_latest",
    ]
    for source in sources:
        assert "top_k" not in source
        assert source.get("study_spec") == "configs/studies/diversity_vs_score.study.yaml"
        prepare = source.get("prepare")
        assert isinstance(prepare, dict)
        assert prepare.get("runbook") == "configs/runbook.yaml"
        step_ids = prepare.get("step_ids")
        assert isinstance(step_ids, list)
        for step in required_prepare_steps:
            assert step in step_ids, f"source={source['id']}: missing prepare step {step}"
