"""RegulonDB Construct workspace contract tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.construct.src.contracts.config import SequenceOffsetSelectorConfig, load_job_config

WORKSPACE = Path("src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel")


def _project_config_path(project_id: str) -> Path:
    payload = yaml.safe_load((WORKSPACE / "construct.workspace.yaml").read_text(encoding="utf-8"))
    for project in payload["workspace"]["projects"]:
        if project["id"] == project_id:
            return WORKSPACE / project["artifacts"]["config"]["path"]
    raise AssertionError(f"Missing Construct workspace project: {project_id}")


def test_regulondb_workspace_derives_core60_from_tss_upstream_offset() -> None:
    core60_config, _ = load_job_config(_project_config_path("native_tss_upstream_core60"))

    assert core60_config.job.mode == "normalize_anchor"
    assert core60_config.job.input.source.dataset == "usr_regulondb_native_promoters"
    assert core60_config.job.output.target.dataset == "usr_regulondb_native_promoter_core60"
    assert core60_config.job.normalize_anchor is not None
    assert core60_config.job.normalize_anchor.target_length == 60
    assert core60_config.job.normalize_anchor.over_length_policy.window_anchor == "upstream_of_focal"
    assert core60_config.job.normalize_anchor.over_length_policy.require_focal_inside is False

    selectors = core60_config.job.normalize_anchor.focal_selector.selectors
    assert len(selectors) == 1
    assert isinstance(selectors[0], SequenceOffsetSelectorConfig)
    assert selectors[0].offset_0 == 60
    assert selectors[0].label == "tss_offset_0"
    assert core60_config.job.normalize_anchor.output_sequence_view.create is True
    assert core60_config.job.normalize_anchor.output_sequence_view.recommended_pooling == "core60_mean"
