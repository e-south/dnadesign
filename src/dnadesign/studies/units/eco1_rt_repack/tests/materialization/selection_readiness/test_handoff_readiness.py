"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_handoff_readiness.py

RT-only handoff-readiness tests for Eco1 RT selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    build_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    PRIMARY_POLICY,
    panel_rows,
)


def test_handoff_readiness_uses_thread_root_candidate_handoff(tmp_path) -> None:
    selection_root = tmp_path / "outputs/thread/generation_policies_v3/selection"
    thread_handoff_path = tmp_path / "outputs/thread/candidate_handoff.yaml"
    selection_root.mkdir(parents=True)
    (selection_root / "candidate_handoff.yaml").write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")

    readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=panel_rows([PRIMARY_POLICY] * SELECTED_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )
    assert readiness["candidate_handoff_path"] == "../../candidate_handoff.yaml"
    assert readiness["candidate_handoff_file_present"] is False
    assert readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text("handoff_kind: rt_only_candidate_handoff\n", encoding="utf-8")
    invalid = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=panel_rows([PRIMARY_POLICY] * SELECTED_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )
    assert invalid["candidate_handoff_file_present"] is True
    assert invalid["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")
    materialized = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=panel_rows([PRIMARY_POLICY] * SELECTED_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )
    assert materialized["candidate_handoff_materialized"] is True
