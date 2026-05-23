"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_study_workspace_contract.py

Study-owned Construct workspace contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.construct.src.contracts.config import (
    AnnotationPairMidpointSelectorConfig,
    load_job_config,
)

WORKSPACE = Path("src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10")


def _project_config_path(project_id: str) -> Path:
    payload = yaml.safe_load((WORKSPACE / "construct.workspace.yaml").read_text(encoding="utf-8"))
    for project in payload["workspace"]["projects"]:
        if project["id"] == project_id:
            return WORKSPACE / project["artifacts"]["config"]["path"]
    raise AssertionError(f"Missing Construct workspace project: {project_id}")


def test_study_workspace_exposes_reference_core60_before_template_contexts() -> None:
    core60_config, _ = load_job_config(_project_config_path("reference_core60"))
    assert core60_config.job.mode == "normalize_anchor"
    assert core60_config.job.input.source.dataset == "usr_promoter_references"
    assert core60_config.job.output.target.dataset == "construct_prom_eth_cip_reference_core60"
    assert core60_config.job.normalize_anchor is not None
    assert core60_config.job.normalize_anchor.target_length == 60
    assert core60_config.job.normalize_anchor.product_kind == "analysis_window"
    selectors = core60_config.job.normalize_anchor.focal_selector.selectors
    assert len(selectors) == 1
    assert isinstance(selectors[0], AnnotationPairMidpointSelectorConfig)
    assert selectors[0].first.role_hint == "sigma70_minus35"
    assert selectors[0].second.role_hint == "sigma70_minus10"
    assert core60_config.job.normalize_anchor.under_length_policy is not None
    assert core60_config.job.normalize_anchor.under_length_policy.placement_ref.startswith("replace:")
    assert core60_config.job.normalize_anchor.feature_retention_policy.fail_if_loses_roles == [
        "sigma70_minus35",
        "sigma70_minus10",
    ]
    assert core60_config.job.normalize_anchor.output_sequence_view.create is True
    assert core60_config.job.normalize_anchor.output_sequence_view.recommended_pooling == "core60_mean"


def test_study_workspace_exposes_forward_and_reverse_complement_context_views() -> None:
    context_config, _ = load_job_config(_project_config_path("reference_core60_contexts"))
    assert context_config.job.mode == "realize_template"
    assert context_config.job.input.source.dataset == "construct_prom_eth_cip_reference_core60"
    assert context_config.job.output.target.dataset == "construct_prom_eth_cip_reference_contexts"
    assert [
        (variant.product_kind, variant.orientation, variant.recommended_pooling)
        for variant in context_config.job.output_variants
    ] == [
        ("realized_context", "forward", "anchor_mean"),
        ("realized_context", "reverse_complement", "anchor_mean"),
    ]


def test_study_workspace_exposes_shared_forward_and_reverse_complement_context_handoff() -> None:
    context_config, _ = load_job_config(_project_config_path("forward_anchor_window"))
    assert context_config.job.mode == "realize_template"
    assert context_config.job.input.source.dataset == "usr_prom_eth_cip_anchor"
    assert context_config.job.output.target.dataset == "construct_prom_eth_cip_context"
    assert context_config.job.output.on_conflict == "ignore"
    assert [
        (variant.product_kind, variant.orientation, variant.recommended_pooling)
        for variant in context_config.job.output_variants
    ] == [
        ("realized_context", "forward", "anchor_mean"),
        ("realized_context", "reverse_complement", "anchor_mean"),
    ]


def test_native_regulondb_core60_uses_existing_study_anchor_and_context_handoff() -> None:
    payload = yaml.safe_load((WORKSPACE / "construct.workspace.yaml").read_text(encoding="utf-8"))
    project_outputs = {
        project["contract"].get("output_dataset")
        for project in payload["workspace"]["projects"]
        if isinstance(project.get("contract"), dict)
    }

    assert "construct_prom_eth_cip_native_tf_contexts" not in project_outputs

    runbook = (WORKSPACE / "runbook.md").read_text(encoding="utf-8")
    assert "usr_regulondb_native_promoter_core60" in runbook
    assert "--src usr_regulondb_native_promoter_core60" in runbook
    assert "--dest usr_prom_eth_cip_anchor" in runbook
    assert "--project forward_anchor_window" in runbook
    assert "--namespace regulondb" in runbook
    assert "--namespace derived" in runbook
    assert "--dest-join derived__parent_id" in runbook
    assert "--dest-join construct__anchor_id" in runbook
