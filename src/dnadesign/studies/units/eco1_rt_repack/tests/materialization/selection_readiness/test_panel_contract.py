"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_panel_contract.py

Panel coverage contract tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    build_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    validate_exact_panel_coverage,
)


def _expected_classes() -> list[str]:
    return [spec.design_class_id for spec in ALL_SPECS]


def _panel_rows(classes: list[str]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": design_class_id,
        }
        for index, design_class_id in enumerate(classes, start=1)
    ]


def test_valid_exact_six_class_panel_passes() -> None:
    validate_exact_panel_coverage(
        _panel_rows(_expected_classes()),
        expected_design_classes=_expected_classes(),
    )


def test_missing_design_class_fails() -> None:
    missing_class = _expected_classes()[2]
    rows = _panel_rows([design_class_id for design_class_id in _expected_classes() if design_class_id != missing_class])

    with pytest.raises(ValueError, match=missing_class):
        validate_exact_panel_coverage(rows, expected_design_classes=_expected_classes())


def test_duplicate_design_class_fails() -> None:
    classes = _expected_classes()
    rows = _panel_rows([*classes, classes[0]])

    with pytest.raises(ValueError, match=classes[0]):
        validate_exact_panel_coverage(rows, expected_design_classes=classes)


def test_unexpected_design_class_fails() -> None:
    unexpected = "eco1_rt_unexpected_mask_class_v1"
    rows = _panel_rows([*_expected_classes(), unexpected])

    with pytest.raises(ValueError, match=unexpected):
        validate_exact_panel_coverage(rows, expected_design_classes=_expected_classes())


def test_panel_count_other_than_six_fails() -> None:
    rows = _panel_rows(_expected_classes()[:-1])

    with pytest.raises(ValueError, match="Selected rows: 5"):
        validate_exact_panel_coverage(rows, expected_design_classes=_expected_classes())


def test_handoff_readiness_uses_thread_root_candidate_handoff(tmp_path) -> None:
    selection_root = tmp_path / "outputs/thread/design_classes/selection"
    thread_handoff_path = tmp_path / "outputs/thread/candidate_handoff.yaml"
    selection_root.mkdir(parents=True)
    (selection_root / "candidate_handoff.yaml").write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")

    readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=_panel_rows(_expected_classes()),
        candidate_handoff_path=thread_handoff_path,
    )

    assert readiness["candidate_handoff_path"] == "../../candidate_handoff.yaml"
    assert readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text("handoff_kind: rt_only_candidate_handoff\n", encoding="utf-8")

    assert (
        build_handoff_readiness(
            selection_root=selection_root,
            panel_rows=_panel_rows(_expected_classes()),
            candidate_handoff_path=thread_handoff_path,
        )["candidate_handoff_materialized"]
        is True
    )
