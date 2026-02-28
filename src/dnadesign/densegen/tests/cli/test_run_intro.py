"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_run_intro.py

Unit tests for DenseGen notebook run details rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from dnadesign.densegen.src.cli.run_intro import (
    PlanContract,
    PlanOutcome,
    RunContractSummary,
    RunDetailsPathsContext,
    RunOutcomeSummary,
    extract_contract,
    parse_plan_name,
    render_intro_md,
)


def _sample_contract() -> RunContractSummary:
    return RunContractSummary(
        sequence_length_bp=100,
        plans=(
            PlanContract(name="baseline", quota=50, acceptance_detail=None),
            PlanContract(
                name="baseline_sigma70",
                quota=50,
                acceptance_detail="promoter upstream/downstream motifs with a 16–18 bp spacer",
            ),
        ),
        total_quota=100,
        global_acceptance_detail=None,
        inputs_used=("Basic",),
        max_accepted_per_library=10,
        round_robin=True,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_attempt_timeout_seconds=10,
        no_progress_seconds_before_resample=10,
        max_consecutive_no_progress_resamples=60,
    )


def _sample_outcome() -> RunOutcomeSummary:
    return RunOutcomeSummary(
        available=True,
        message="",
        generated_total=100,
        quota_total=100,
        progress_pct=100.0,
        per_plan=(
            PlanOutcome(
                name="baseline",
                generated=50,
                quota=50,
                stall_events=0,
                total_resamples=0,
                failed_solutions=0,
            ),
            PlanOutcome(
                name="baseline_sigma70",
                generated=50,
                quota=50,
                stall_events=3,
                total_resamples=6,
                failed_solutions=0,
            ),
        ),
        stall_events=3,
        total_resamples=6,
        failed_solutions=0,
    )


def _summary_bullets(rendered: str) -> list[str]:
    lines = rendered.splitlines()
    bullets: list[str] = []
    for line in lines[1:]:
        if line.startswith("### "):
            break
        if line.startswith("- "):
            bullets.append(line)
    return bullets


def test_render_intro_omits_top_summary_bullets() -> None:
    rendered = render_intro_md(_sample_contract(), _sample_outcome(), style="didactic")
    assert rendered.startswith("## Run details")
    bullets = _summary_bullets(rendered)
    assert len(bullets) == 0


def test_render_intro_has_required_details_sections_in_order() -> None:
    rendered = render_intro_md(_sample_contract(), _sample_outcome(), style="didactic")
    expected = [
        "### Definitions",
        "### Scope and quotas (2 plans)",
        "### Acceptance and inputs",
        "### Constraint literals",
        "### Execution policy",
        "### Outcome and pressure",
        "### Pressure by plan",
        "### Sources and freshness",
    ]
    positions = [rendered.index(token) for token in expected]
    assert positions == sorted(positions)


def test_render_intro_missing_manifest_uses_explicit_outcome_and_pressure_lines() -> None:
    missing_outcome = RunOutcomeSummary(
        available=False,
        message="manifest not found",
        generated_total=None,
        quota_total=None,
        progress_pct=None,
        per_plan=tuple(),
        stall_events=0,
        total_resamples=0,
        failed_solutions=0,
    )
    rendered = render_intro_md(_sample_contract(), missing_outcome, style="didactic")
    assert "Outcome: manifest not found. [manifest]" in rendered
    assert "Pressure: manifest not found. [manifest]" in rendered


def test_render_intro_groups_large_plan_sets_and_nests_raw_plan_list() -> None:
    plans = tuple(
        PlanContract(
            name=f"sigma70_panel__sig35={letter}__sig10={digit}",
            quota=1,
            acceptance_detail="promoter upstream/downstream motifs with a 16–18 bp spacer",
        )
        for letter in ("a", "b", "c")
        for digit in ("A", "B", "C")
    )
    outcome = RunOutcomeSummary(
        available=True,
        message="",
        generated_total=9,
        quota_total=9,
        progress_pct=100.0,
        per_plan=tuple(
            PlanOutcome(
                name=plan.name,
                generated=1,
                quota=1,
                stall_events=0,
                total_resamples=0,
                failed_solutions=0,
            )
            for plan in plans
        ),
        stall_events=0,
        total_resamples=0,
        failed_solutions=0,
    )
    contract = RunContractSummary(
        sequence_length_bp=60,
        plans=plans,
        total_quota=9,
        global_acceptance_detail="1 forbid_kmers rule",
        inputs_used=("Background",),
        max_accepted_per_library=2,
        round_robin=True,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_attempt_timeout_seconds=10,
        no_progress_seconds_before_resample=10,
        max_consecutive_no_progress_resamples=25,
    )
    rendered = render_intro_md(contract, outcome, style="didactic")
    assert "### Scope and quotas (9 plans)" in rendered
    assert "### All plans (raw list, 9 plans)" in rendered
    assert "| Group | Plans | Quota | Progress | Status |" in rendered
    assert "| Plan | Variants | Quota | Generated | Progress | Raw plan id |" in rendered


def test_extract_contract_renders_sigma_name_negative_selection_and_expansion_details() -> None:
    payload = {
        "densegen": {
            "inputs": [
                {
                    "name": "background",
                    "type": "background_pool",
                    "sampling": {
                        "filters": {
                            "fimo_exclude": {
                                "pwms_input": ["lacI_pwm", "araC_pwm"],
                                "allow_zero_hit_only": True,
                            }
                        }
                    },
                }
            ],
            "motif_sets": {
                "sigma70_upstream_35": {"a": "TTGACA", "b": "CTGACA"},
                "sigma70_downstream_10": {"A": "TATAAT", "B": "TATAGT"},
            },
            "generation": {
                "sequence_length": 60,
                "plan": [
                    {
                        "name": "sigma70_panel__sig35=a__sig10=A",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["background"]},
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream": "TTGACA",
                                    "downstream": "TATAAT",
                                    "upstream_variant_id": "a",
                                    "downstream_variant_id": "A",
                                    "spacer_length": [16, 18],
                                    "upstream_pos": [15, 30],
                                }
                            ]
                        },
                    },
                    {
                        "name": "sigma70_panel__sig35=b__sig10=B",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["background"]},
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream": "CTGACA",
                                    "downstream": "TATAGT",
                                    "upstream_variant_id": "b",
                                    "downstream_variant_id": "B",
                                    "spacer_length": [16, 18],
                                    "upstream_pos": [15, 30],
                                }
                            ]
                        },
                    },
                    {
                        "name": "sigma70_panel__sig35=a__sig10=B",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["background"]},
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream": "TTGACA",
                                    "downstream": "TATAGT",
                                    "upstream_variant_id": "a",
                                    "downstream_variant_id": "B",
                                    "spacer_length": [16, 18],
                                    "upstream_pos": [15, 30],
                                }
                            ]
                        },
                    },
                    {
                        "name": "sigma70_panel__sig35=b__sig10=A",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["background"]},
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream": "CTGACA",
                                    "downstream": "TATAAT",
                                    "upstream_variant_id": "b",
                                    "downstream_variant_id": "A",
                                    "spacer_length": [16, 18],
                                    "upstream_pos": [15, 30],
                                }
                            ]
                        },
                    },
                ],
                "sequence_constraints": {
                    "forbid_kmers": [{"strands": "both", "include_reverse_complements": True}],
                    "allowlist": [{"kind": "fixed_element_instance"}],
                },
            },
            "solver": {
                "backend": "CBC",
                "strategy": "iterate",
                "solver_attempt_timeout_seconds": 10,
            },
            "runtime": {
                "round_robin": True,
                "max_accepted_per_library": 2,
                "no_progress_seconds_before_resample": 10,
                "max_consecutive_no_progress_resamples": 25,
            },
        }
    }

    contract = extract_contract(payload)
    outcome = RunOutcomeSummary(
        available=False,
        message="run outcomes are unavailable for this test.",
        generated_total=None,
        quota_total=None,
        progress_pct=None,
        per_plan=tuple(),
        stall_events=0,
        total_resamples=0,
        failed_solutions=0,
    )
    rendered = render_intro_md(contract, outcome, style="didactic")
    assert "σ70 promoter upstream/downstream motifs with a 16–18 bp spacer" in rendered
    assert "Background excludes FIMO hits for LacI, AraC (zero-hit only)" in rendered
    assert "Literal set 1:" in rendered
    assert "Upstream set `sigma70_upstream_35` literals:" in rendered
    assert "`a=TTGACA`" in rendered
    assert "`b=CTGACA`" in rendered
    assert "Downstream set `sigma70_downstream_10` literals:" in rendered
    assert "`A=TATAAT`" in rendered
    assert "`B=TATAGT`" in rendered


def test_parse_plan_name_extracts_base_and_variant_map() -> None:
    parsed = parse_plan_name("sigma70_panel__sig35=a__sig10=B")
    assert parsed.base == "sigma70_panel"
    assert parsed.variants == {"sig35": "a", "sig10": "B"}


def test_sources_and_freshness_marks_manifest_newer_than_notebook(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "run_manifest.json"
    notebook_path = tmp_path / "densegen_run_overview.py"
    config_path.write_text("densegen: {}\n")
    notebook_path.write_text("# notebook\n")
    manifest_path.write_text("{}\n")
    notebook_mtime = notebook_path.stat().st_mtime
    os.utime(manifest_path, (notebook_mtime + 5.0, notebook_mtime + 5.0))
    paths = RunDetailsPathsContext(
        config_path=config_path,
        manifest_path=manifest_path,
        notebook_path=notebook_path,
    )
    rendered = render_intro_md(_sample_contract(), _sample_outcome(), style="didactic", paths_context=paths)
    assert "Sources and freshness" in rendered
    assert "Config:" in rendered
    assert "Manifest:" in rendered
    assert "Notebook:" in rendered
    assert "- Run root:" in rendered
    assert "- Config:" in rendered
    assert "- Records path:" in rendered
    assert "- Manifest:" in rendered
    assert "- Notebook:" in rendered
    assert "`[config]`" in rendered
    assert "`[manifest]`" in rendered
    assert "Manifest is newer than this notebook file. Regenerate the notebook to update the narrative." in rendered
    assert manifest_path.stat().st_mtime >= notebook_mtime
