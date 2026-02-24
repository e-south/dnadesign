"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_portfolio_docs_contracts.py

Docs contracts for Portfolio aggregation flow and CLI references.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_docs_index_lists_portfolio_guide() -> None:
    docs_index = _read("docs/index.md")
    assert "guides/portfolio_aggregation.md" in docs_index


def test_cli_reference_lists_portfolio_commands() -> None:
    cli_ref = _read("docs/reference/cli.md")
    assert "Portfolio workflows" in cli_ref
    assert "cruncher portfolio run" in cli_ref
    assert "cruncher portfolio show" in cli_ref
    assert "--prepare-ready {prompt|skip|rerun}" in cli_ref
    assert "--spec` must point to a `.portfolio.yaml` file path" in cli_ref
    assert "outputs/export/portfolios/<portfolio_name>/<portfolio_id>" in cli_ref


def test_portfolio_guide_describes_export_then_aggregate_sequence() -> None:
    guide = _read("docs/guides/portfolio_aggregation.md")
    assert "cruncher portfolio run --spec" in guide
    assert "table__handoff_windows_long" in guide
    assert "table__handoff_elites_summary" in guide
    assert "table__source_summary" in guide
    assert "table__handoff_sequence_length" in guide
    assert "run_dir: outputs" in guide
    assert "prepare_then_aggregate" in guide
    assert "--prepare-ready skip" in guide
    assert "aggregate_only" in guide
    assert "ensure_specs" in guide
    assert "configs/studies/length_vs_score.study.yaml" in guide
    assert "configs/studies/diversity_vs_score.study.yaml" in guide
    assert "outputs/export/portfolios/<portfolio_name>/<portfolio_id>" in guide
    assert "Passing `configs/` (directory only) fails fast." in guide
    assert (
        "step_ids: [fetch_sites_regulondb, discover_motifs, render_logos, lock_targets, "
        "parse_run, sample_run, analyze_summary, export_sequences_latest]"
    ) in guide


def test_workspaces_readme_lists_portfolio_workspace_template() -> None:
    readme = _read("workspaces/README.md")
    assert "portfolios/" in readme
    assert "run_dir: outputs" in readme
    assert "workspaces run --runbook configs/runbook.yaml" in readme
    assert "master_all_workspaces.portfolio.yaml" in readme


def test_portfolio_workspace_template_spec_exists_and_uses_prepare_mode_with_explicit_run_dirs() -> None:
    runbook_path = ROOT / "workspaces" / "portfolios" / "configs" / "runbook.yaml"
    assert runbook_path.exists()
    runbook_payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    steps = runbook_payload["runbook"]["steps"]
    assert len(steps) == 1
    for step in steps:
        command = list(step.get("run") or [])
        assert "--prepare-ready" in command
        idx = command.index("--prepare-ready")
        assert idx + 1 < len(command)
        assert command[idx + 1] == "skip"

    spec_path = ROOT / "workspaces" / "portfolios" / "configs" / "master_all_workspaces.portfolio.yaml"
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = payload["portfolio"]
    assert spec["name"] == "master_all_workspaces"
    assert spec["schema_version"] == 3
    assert spec["execution"]["mode"] == "prepare_then_aggregate"
    studies = spec.get("studies")
    assert isinstance(studies, dict)
    assert studies.get("ensure_specs") == [
        "configs/studies/length_vs_score.study.yaml",
        "configs/studies/diversity_vs_score.study.yaml",
    ]
    sequence_length_table = studies.get("sequence_length_table")
    assert isinstance(sequence_length_table, dict)
    assert sequence_length_table.get("enabled") is True
    assert sequence_length_table.get("top_n_lengths") == 6
    assert len(spec["sources"]) >= 1
    for source in spec["sources"]:
        assert source["run_dir"] == "outputs"
        assert "top_k" not in source
        assert source["study_spec"] == "configs/studies/diversity_vs_score.study.yaml"
        prepare = source.get("prepare")
        assert isinstance(prepare, dict)
        assert prepare.get("runbook") == "configs/runbook.yaml"
        step_ids = prepare.get("step_ids")
        assert isinstance(step_ids, list)
        for step in ["sample_run", "analyze_summary", "export_sequences_latest"]:
            assert step in step_ids
