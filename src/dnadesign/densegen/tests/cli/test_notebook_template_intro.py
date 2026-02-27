"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_notebook_template_intro.py

Tests for DenseGen notebook title and intro narrative scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from dnadesign.densegen.src.cli.notebook_template import (
    NotebookTemplateContext,
    _build_workspace_intro,
    _format_workspace_heading,
)


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_sampling_baseline
                root: "."
              inputs:
                - name: lexA_pwm
                  type: binding_sites
                  path: inputs/lexA.csv
                - name: cpxR_pwm
                  type: binding_sites
                  path: inputs/cpxR.csv
              generation:
                sequence_length: 100
                plan:
                  - name: ethanol
                    sequences: 6
                    sampling:
                      include_inputs: [lexA_pwm]
                    regulator_constraints:
                      groups: []
                  - name: ciprofloxacin
                    sequences: 6
                    sampling:
                      include_inputs: [cpxR_pwm]
                    regulator_constraints:
                      groups: []
              runtime:
                round_robin: true
                max_accepted_per_library: 10
                no_progress_seconds_before_resample: 60
                max_consecutive_no_progress_resamples: 6
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
            """
        ).strip()
        + "\n"
    )


def _write_fixed_promoter_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_tfbs_baseline
                root: "."
              inputs:
                - name: basic_sites
                  type: binding_sites
                  path: inputs/sites.csv
              generation:
                sequence_length: 100
                plan:
                  - name: baseline
                    sequences: 50
                    sampling:
                      include_inputs: [basic_sites]
                    regulator_constraints:
                      groups: []
                  - name: baseline_sigma70
                    sequences: 50
                    sampling:
                      include_inputs: [basic_sites]
                    fixed_elements:
                      promoter_constraints:
                        - name: sigma70_consensus
                          upstream: TTGACA
                          downstream: TATAAT
                          spacer_length: [16, 18]
                    regulator_constraints:
                      groups: []
              runtime:
                round_robin: true
                max_accepted_per_library: 10
                no_progress_seconds_before_resample: 10
                max_consecutive_no_progress_resamples: 60
              solver:
                backend: CBC
                strategy: iterate
                solver_attempt_timeout_seconds: 10
              logging:
                log_dir: outputs/logs
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
            """
        ).strip()
        + "\n"
    )


def _write_expansion_background_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: study_constitutive_sigma_panel
                root: "."
              motif_sets:
                sigma70_upstream_35:
                  a: TTGACA
                  b: CTGACA
                sigma70_downstream_10:
                  A: TATAAT
                  B: TATAGT
              inputs:
                - name: lacI_pwm
                  type: pwm_matrix_csv
                  path: inputs/lacI_pwm.csv
                  motif_id: lacI
                  sampling:
                    strategy: stochastic
                    n_sites: 2
                    mining:
                      batch_size: 10
                      budget:
                        mode: fixed_candidates
                        candidates: 20
                    length:
                      policy: exact
                - name: araC_pwm
                  type: pwm_matrix_csv
                  path: inputs/araC_pwm.csv
                  motif_id: araC
                  sampling:
                    strategy: stochastic
                    n_sites: 2
                    mining:
                      batch_size: 10
                      budget:
                        mode: fixed_candidates
                        candidates: 20
                    length:
                      policy: exact
                - name: background
                  type: background_pool
                  sampling:
                    n_sites: 20
                    mining:
                      batch_size: 100
                      budget:
                        mode: fixed_candidates
                        candidates: 200
                    length:
                      policy: range
                      range: [16, 20]
                    uniqueness:
                      key: sequence
                    gc:
                      min: 0.4
                      max: 0.6
                    filters:
                      fimo_exclude:
                        pwms_input: [lacI_pwm, araC_pwm]
                        allow_zero_hit_only: true
              generation:
                sequence_length: 60
                plan:
                  - name: sigma70_panel
                    sequences: 4
                    expanded_name_template: "{base}__sig35={up}__sig10={down}"
                    sampling:
                      include_inputs: [background]
                    fixed_elements:
                      fixed_element_matrix:
                        name: sigma70_core
                        upstream_from_set: sigma70_upstream_35
                        downstream_from_set: sigma70_downstream_10
                        pairing:
                          mode: cross_product
                        spacer_length: [16, 18]
                        upstream_pos: [15, 30]
                    regulator_constraints:
                      groups: []
              runtime:
                round_robin: true
                max_accepted_per_library: 2
                no_progress_seconds_before_resample: 10
                max_consecutive_no_progress_resamples: 25
              solver:
                backend: CBC
                strategy: iterate
                solver_attempt_timeout_seconds: 10
              logging:
                log_dir: outputs/logs
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
            """
        ).strip()
        + "\n"
    )


def _make_context(tmp_path: Path) -> NotebookTemplateContext:
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    return NotebookTemplateContext(
        run_root=tmp_path,
        cfg_path=cfg_path,
        records_path=tmp_path / "outputs" / "tables" / "records.parquet",
        output_source="parquet",
        usr_root=None,
        usr_dataset=None,
    )


def _make_fixed_context(tmp_path: Path) -> NotebookTemplateContext:
    cfg_path = tmp_path / "config.yaml"
    _write_fixed_promoter_config(cfg_path)
    return NotebookTemplateContext(
        run_root=tmp_path,
        cfg_path=cfg_path,
        records_path=tmp_path / "outputs" / "tables" / "records.parquet",
        output_source="parquet",
        usr_root=None,
        usr_dataset=None,
    )


def _make_expansion_context(tmp_path: Path) -> NotebookTemplateContext:
    cfg_path = tmp_path / "config.yaml"
    _write_expansion_background_config(cfg_path)
    return NotebookTemplateContext(
        run_root=tmp_path,
        cfg_path=cfg_path,
        records_path=tmp_path / "outputs" / "tables" / "records.parquet",
        output_source="parquet",
        usr_root=None,
        usr_dataset=None,
    )


def _summary_bullets(intro: str) -> list[str]:
    lines = intro.splitlines()
    bullets: list[str] = []
    for line in lines[1:]:
        if line.startswith("### "):
            break
        if line.startswith("- "):
            bullets.append(line)
    return bullets


def test_format_workspace_heading_converts_workspace_id_to_readable_title() -> None:
    assert _format_workspace_heading("study_constitutive_sigma_panel") == "Study Constitutive Sigma Panel"


def test_format_workspace_heading_preserves_known_acronyms() -> None:
    assert _format_workspace_heading("demo_tfbs_usr_dna") == "Demo TFBS USR DNA"


def test_build_workspace_intro_reports_missing_manifest_with_source_tagged_summary(tmp_path: Path) -> None:
    intro = _build_workspace_intro(_make_context(tmp_path))
    assert "## Run details" in intro
    bullets = _summary_bullets(intro)
    assert len(bullets) == 0
    assert "Outcome: manifest not found. [manifest]" in intro
    assert "Pressure: manifest not found. [manifest]" in intro
    expected = [
        "### Definitions",
        "### Scope and quotas (2 plans)",
        "### Acceptance and inputs",
        "### Constraint literals",
        "### Execution policy",
        "### Outcome and pressure",
        "### Sources and freshness",
    ]
    positions = [intro.index(token) for token in expected]
    assert positions == sorted(positions)
    assert "### Scope and quotas (2 plans)" in intro
    assert "### Acceptance and inputs" in intro
    assert "### Execution policy" in intro
    assert "### Outcome and pressure" in intro
    assert "### Sources and freshness" in intro


def test_build_workspace_intro_reports_run_outcomes_when_manifest_exists(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 9,
                "total_quota": 12,
                "quota_progress_pct": 75.0,
                "items": [
                    {
                        "plan_name": "ethanol",
                        "generated": 6,
                        "quota": 6,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                    {
                        "plan_name": "ciprofloxacin",
                        "generated": 3,
                        "quota": 6,
                        "stall_events": 2,
                        "total_resamples": 1,
                        "failed_solutions": 4,
                    },
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    assert "Outcome: 9/12 generated (75.0%). [manifest]" in intro
    assert "Plans at quota: 1/2. [manifest]" in intro
    assert "Pressure: stall events=2; resamples=1; failed solves=4. [manifest]" in intro


def test_build_workspace_intro_omits_zero_pressure_counters_by_default(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 12,
                "total_quota": 12,
                "quota_progress_pct": 100.0,
                "items": [
                    {
                        "plan_name": "ethanol",
                        "generated": 6,
                        "quota": 6,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                    {
                        "plan_name": "ciprofloxacin",
                        "generated": 6,
                        "quota": 6,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    assert "Pressure: none recorded. [manifest]" in intro


def test_build_workspace_intro_describes_fixed_promoter_constraints_with_en_dash(tmp_path: Path) -> None:
    intro = _build_workspace_intro(_make_fixed_context(tmp_path))
    assert "16–18 bp spacer" in intro
    assert "σ70 promoter upstream/downstream motifs" in intro


def test_build_workspace_intro_includes_background_negative_selection_and_expansion_summary(tmp_path: Path) -> None:
    intro = _build_workspace_intro(_make_expansion_context(tmp_path))
    assert "Background excludes FIMO hits for LacI, AraC (zero-hit only)" in intro
    assert "sigma70 panel expands σ70 via exhaustive cross-product" in intro
    assert "Literal set 1:" in intro
    assert "Upstream set `sigma70_upstream_35` literals:" in intro
    assert "`a=TTGACA`" in intro
    assert "`b=CTGACA`" in intro
    assert "Downstream set `sigma70_downstream_10` literals:" in intro
    assert "`A=TATAAT`" in intro
    assert "`B=TATAGT`" in intro


def test_build_workspace_intro_keeps_plan_order_from_config(tmp_path: Path) -> None:
    context = _make_fixed_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 100,
                "total_quota": 100,
                "quota_progress_pct": 100.0,
                "items": [
                    {
                        "plan_name": "baseline_sigma70",
                        "generated": 50,
                        "quota": 50,
                        "stall_events": 3,
                        "total_resamples": 6,
                        "failed_solutions": 0,
                    },
                    {
                        "plan_name": "baseline",
                        "generated": 50,
                        "quota": 50,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    baseline_idx = intro.index("| baseline | 1 | 50 | 50/50 | all complete |")
    sigma_idx = intro.index("| baseline_sigma70 (σ70 promoter upstream/downstream motifs with a 16–18 bp spacer) |")
    assert baseline_idx < sigma_idx


def test_build_workspace_intro_formats_percent_with_single_decimal(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 1,
                "total_quota": 3,
                "items": [
                    {
                        "plan_name": "ethanol",
                        "generated": 1,
                        "quota": 1,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                    {
                        "plan_name": "ciprofloxacin",
                        "generated": 0,
                        "quota": 2,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    assert "Outcome: 1/3 generated (33.3%). [manifest]" in intro
    assert "Plans at quota: 1/2. [manifest]" in intro


def test_build_workspace_intro_uses_grouped_plan_table_for_small_plan_sets(tmp_path: Path) -> None:
    context = _make_fixed_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 100,
                "total_quota": 100,
                "quota_progress_pct": 100.0,
                "items": [
                    {
                        "plan_name": "baseline",
                        "generated": 50,
                        "quota": 50,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    },
                    {
                        "plan_name": "baseline_sigma70",
                        "generated": 50,
                        "quota": 50,
                        "stall_events": 3,
                        "total_resamples": 6,
                        "failed_solutions": 0,
                    },
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    assert "| Group | Plans | Quota | Progress | Status |" in intro
    assert "### All plans (raw list" not in intro


def test_build_workspace_intro_manifest_without_quota_shows_na_progress(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    manifest_path = tmp_path / "outputs" / "meta" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "total_generated": 5,
                "items": [
                    {
                        "plan_name": "ethanol",
                        "generated": 5,
                        "stall_events": 0,
                        "total_resamples": 0,
                        "failed_solutions": 0,
                    }
                ],
            }
        )
    )
    intro = _build_workspace_intro(context)
    assert "Outcome: 5/0 generated (n/a). [manifest]" in intro
    assert "Plans at quota: 0/1. [manifest]" in intro
