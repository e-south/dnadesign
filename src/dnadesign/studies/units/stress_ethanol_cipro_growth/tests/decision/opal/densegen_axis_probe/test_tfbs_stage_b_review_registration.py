"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_review_registration.py

Regression tests for TFBS stage b review registration studies units stress.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .helpers import _write_stage_b_review_fixture
from .probe_modules import probe_module

build_tfbs_stage_b_realized_label_review = probe_module(
    "tfbs.stage_b.review.materialization"
).build_tfbs_stage_b_realized_label_review


def test_stage_b_realized_review_does_not_register_failed_budget_review(tmp_path: Path) -> None:
    stage_b_root = tmp_path / "stage_b"
    manifest_path = _write_stage_b_review_fixture(stage_b_root / "manifests")
    pd.DataFrame({"id": ["c"], "pred__score_selected": [0.8]}).to_csv(
        stage_b_root
        / "manifests"
        / "campaigns"
        / "lexA_present_positive"
        / "outputs"
        / "rounds"
        / "round_1"
        / "selection"
        / "selection_top_k.csv",
        index=False,
    )
    visual_index_path = stage_b_root / "notebooks" / "collection_visuals" / "collection_visual_manifest.json"
    visual_index_path.parent.mkdir(parents=True)
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v1",
                "generated_at": "2026-06-01T00:00:00+00:00",
                "collection_id": "fixture",
                "output_dir": str(visual_index_path.parent),
                "comparison_set_count": 0,
                "comparison_sets": [],
                "visual_count": 0,
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )

    result = build_tfbs_stage_b_realized_label_review(manifest_path)

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    refreshed = json.loads(visual_index_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL_SELECTION_BUDGET"
    assert summary["notebook_visual_registration"]["realized_label_review"] == {
        "status": "SKIPPED_REVIEW_NOT_PASS",
        "review_status": "FAIL_SELECTION_BUDGET",
        "collection_visual_index_path": None,
        "registered_visual_count": 0,
    }
    assert refreshed["comparison_set_count"] == 0
    assert refreshed["visual_count"] == 0
    assert refreshed["visuals"] == []
