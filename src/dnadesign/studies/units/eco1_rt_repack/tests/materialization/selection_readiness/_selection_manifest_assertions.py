"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_selection_manifest_assertions.py

Manifest-level assertions for Eco1 RT selection-readiness materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import _materialization_assertions as materialization_assertions
from ._selection_manifest_artifact_assertions import (
    assert_selection_artifact_rows,
    assert_selection_handoff_manifest,
)
from ._selection_manifest_summary_assertions import (
    assert_local_structure_manifest,
    assert_selection_summary_and_trace,
)


def assert_materialized_selection_manifest(
    *,
    result: Any,
    manifest: dict[str, Any],
    triage: list[dict],
    panel: list[dict],
) -> None:
    assert manifest["path_policy"] == "paths_relative_to_selection_manifest"
    assert all(not Path(value).is_absolute() for value in manifest["source_tables"].values())
    assert all(not Path(value).is_absolute() for value in manifest["artifacts"].values())
    assert manifest["gate_counts"]["hard_gate_status"] == dict(
        sorted(Counter(str(row["hard_gate_status"]) for row in triage).items())
    )
    assert manifest["gate_counts"]["local_structure_gate_status"] == {"passed": len(triage)}
    assert manifest["gate_counts"]["sae_window_status"] == {"wt_like_not_used_for_selection": len(triage)}
    assert_selection_summary_and_trace(result=result, manifest=manifest, triage=triage, panel=panel)
    assert_local_structure_manifest(manifest)
    assert_selection_handoff_manifest(manifest, panel=panel)
    assert_selection_artifact_rows(result=result, manifest=manifest, triage=triage)
    materialization_assertions.assert_selection_plot_contract(
        result=result,
        manifest=manifest,
    )
