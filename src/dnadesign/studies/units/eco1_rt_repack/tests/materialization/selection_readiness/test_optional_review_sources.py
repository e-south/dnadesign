"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_optional_review_sources.py

Optional review-source contract tests for Eco1 selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    write_inputs,
)


def test_missing_optional_model_review_sources_make_manifest_degraded(tmp_path: Path) -> None:
    class_root = tmp_path / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = tmp_path / "outputs/thread"
    write_inputs(class_root, source_root)
    for path in (
        class_root / "review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
        class_root
        / "review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        class_root / "biohub_esmc/sae_feature_window_summary.parquet",
    ):
        path.unlink()

    result = materialize_selection_readiness(
        repo_root=tmp_path,
        output_root=class_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "materialized_degraded"
    assert manifest["missing_optional_review_sources"] == ["llr_300m", "llr_6b", "sae_window"]
    assert set(manifest["optional_review_sources"]) == {"llr_300m", "llr_6b", "sae_window"}
    assert all(
        source["panel_selection_role"] == "review_annotation_not_selector"
        for source in manifest["optional_review_sources"].values()
    )
    assert manifest["handoff_readiness"]["candidate_handoff_materialized"] is False
