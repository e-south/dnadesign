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
    names: list[str] = []
    for path in WORKSPACES_ROOT.iterdir():
        if not path.is_dir():
            continue
        config_path = path / "configs" / "config.yaml"
        if config_path.exists() or path.name == "portfolios":
            names.append(path.name)
    return sorted(names)


def _load_workspace_config(workspace_name: str) -> dict:
    payload = yaml.safe_load((WORKSPACES_ROOT / workspace_name / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    cruncher = payload.get("cruncher")
    assert isinstance(cruncher, dict)
    return cruncher


def _is_occurrence_aware_workspace(workspace_name: str) -> bool:
    cfg = _load_workspace_config(workspace_name)
    sample = cfg.get("sample")
    if not isinstance(sample, dict):
        return False
    objective = sample.get("objective")
    if not isinstance(objective, dict):
        return False
    multiplicity = objective.get("multiplicity")
    if not isinstance(multiplicity, dict):
        return False
    return bool(multiplicity.get("enabled"))


def _is_motif_ingest_workspace(workspace_name: str) -> bool:
    cfg = _load_workspace_config(workspace_name)
    discover = cfg.get("discover")
    assert isinstance(discover, dict)
    return discover.get("enabled") is False


def _non_portfolio_workspaces() -> list[str]:
    excluded = {"archived", "portfolio", "portfolios"}
    eligible: list[str] = []
    for name in _workspace_names():
        if name in excluded:
            continue
        if not (WORKSPACES_ROOT / name / "configs" / "config.yaml").is_file():
            continue
        eligible.append(name)
    return eligible


def _study_orchestration_workspaces() -> list[str]:
    return [name for name in _non_portfolio_workspaces() if not _is_occurrence_aware_workspace(name)]


def test_representative_hit_workspaces_have_length_and_diversity_study_specs() -> None:
    for workspace_name in _study_orchestration_workspaces():
        length_spec = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "length_vs_score.study.yaml"
        diversity_spec = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "diversity_vs_score.study.yaml"
        assert length_spec.exists(), f"{workspace_name}: missing configs/studies/length_vs_score.study.yaml"
        assert diversity_spec.exists(), f"{workspace_name}: missing configs/studies/diversity_vs_score.study.yaml"


def test_all_non_portfolio_workspaces_do_not_keep_legacy_portfolio_ready_studies() -> None:
    for workspace_name in _non_portfolio_workspaces():
        legacy = WORKSPACES_ROOT / workspace_name / "configs" / "studies" / "portfolio_ready.study.yaml"
        assert not legacy.exists(), f"{workspace_name}: remove legacy {legacy.name}"


def test_representative_hit_workspaces_runbooks_have_length_and_diversity_study_steps() -> None:
    for workspace_name in _study_orchestration_workspaces():
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
    assert studies.get("enabled") is False
    assert studies.get("ensure_specs") == [
        "configs/studies/length_vs_score.study.yaml",
        "configs/studies/diversity_vs_score.study.yaml",
    ]
    sequence_length_table = studies.get("sequence_length_table")
    assert isinstance(sequence_length_table, dict)
    assert sequence_length_table.get("enabled") is False
    assert sequence_length_table.get("study_spec") == "configs/studies/length_vs_score.study.yaml"
    assert sequence_length_table.get("top_n_lengths") == 6

    expected = _study_orchestration_workspaces()
    sources = portfolio["sources"]
    seen_workspaces = sorted(Path(item["workspace"]).name for item in sources)
    assert seen_workspaces == expected

    required_prepare_steps = [
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
        if _is_motif_ingest_workspace(str(source["id"])):
            assert any(str(step).startswith("fetch_motifs") for step in step_ids), (
                f"source={source['id']}: motif-ingest workspace must fetch motifs during prepare"
            )
            assert "export_meme" in step_ids, f"source={source['id']}: motif-ingest workspace must export MEME"
        else:
            assert "fetch_sites_regulondb" in step_ids, (
                f"source={source['id']}: discovery workspace must fetch RegulonDB sites during prepare"
            )
            assert "discover_motifs" in step_ids, f"source={source['id']}: discovery workspace must discover motifs"
