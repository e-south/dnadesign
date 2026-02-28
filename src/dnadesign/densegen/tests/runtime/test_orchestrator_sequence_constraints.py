"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_orchestrator_sequence_constraints.py

Unit coverage for Stage-B sequence-constraint evaluation in the orchestrator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.pipeline import stage_b_runtime_checks
from dnadesign.densegen.src.core.pipeline.stage_b_runtime_checks import _evaluate_sequence_constraints
from dnadesign.densegen.src.core.sequence_constraints.engine import compile_sequence_constraints


def _constraint_block() -> dict:
    return {
        "forbid_kmers": [
            {
                "name": "sigma_core",
                "patterns_from_motif_sets": ["sigma_hexamers"],
                "include_reverse_complements": True,
                "strands": "both",
            }
        ],
        "allowlist": [
            {
                "kind": "fixed_element_instance",
                "selector": {
                    "fixed_element": "promoter",
                    "component": ["upstream", "downstream"],
                },
            }
        ],
    }


def _motif_sets() -> dict[str, dict[str, str]]:
    return {"sigma_hexamers": {"upstream": "TTGACA", "downstream": "TATAAT"}}


def _fixed_elements_dump() -> dict:
    return {
        "promoter_constraints": [
            {
                "name": "sigma70",
                "upstream": "TTGACA",
                "downstream": "TATAAT",
                "spacer_length": [2, 2],
                "upstream_pos": [0, 10],
            }
        ],
        "side_biases": {"left": [], "right": []},
    }


def test_evaluate_sequence_constraints_defaults_when_rules_missing() -> None:
    result = _evaluate_sequence_constraints(
        final_seq="ACGTACGT",
        compiled_sequence_constraints=None,
        fixed_elements_dump={},
        source_label="demo",
        plan_name="baseline",
        sampling_library_index=1,
        sampling_library_hash="hash",
    )
    assert result.rejection_detail is None
    assert result.rejection_event_payload is None
    assert result.error is None
    assert result.sequence_validation == {"validation_passed": True, "violations": []}
    assert result.promoter_detail == {"placements": []}


def test_evaluate_sequence_constraints_reports_violations() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints=_constraint_block(),
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    result = _evaluate_sequence_constraints(
        final_seq="TTGACAGGTATAATTTGACA",
        compiled_sequence_constraints=compiled,
        fixed_elements_dump=_fixed_elements_dump(),
        source_label="demo",
        plan_name="baseline",
        sampling_library_index=2,
        sampling_library_hash="hash-2",
    )
    assert result.rejection_detail is not None
    assert result.error is None
    assert "violations" in result.rejection_detail
    assert result.rejection_event_payload is not None
    assert "violations" in result.rejection_event_payload
    assert result.sequence_validation["validation_passed"] is False


def test_evaluate_sequence_constraints_demotes_validation_value_error_to_rejection() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints=_constraint_block(),
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    broken_fixed_elements = {
        "promoter_constraints": [
            {
                "name": "sigma70",
                "upstream": "TTGACA",
                "downstream": "TATAAT",
                "spacer_length": [2, 2],
                "upstream_pos": [50, 60],
            }
        ],
        "side_biases": {"left": [], "right": []},
    }
    result = _evaluate_sequence_constraints(
        final_seq="TTGACAGGTATAAT",
        compiled_sequence_constraints=compiled,
        fixed_elements_dump=broken_fixed_elements,
        source_label="demo",
        plan_name="baseline",
        sampling_library_index=3,
        sampling_library_hash="hash-3",
    )
    assert result.rejection_detail is not None
    assert "error" in result.rejection_detail
    assert result.rejection_event_payload is not None
    assert "error" in result.rejection_event_payload
    assert result.error is not None
    assert isinstance(result.error, ValueError)


def test_evaluate_sequence_constraints_resolves_promoter_detail_without_forbid_rules() -> None:
    result = _evaluate_sequence_constraints(
        final_seq="TTGACAGGTATAAT",
        compiled_sequence_constraints=None,
        fixed_elements_dump=_fixed_elements_dump(),
        source_label="demo",
        plan_name="baseline",
        sampling_library_index=4,
        sampling_library_hash="hash-4",
    )
    assert result.rejection_detail is None
    assert result.rejection_event_payload is None
    assert result.sequence_validation == {"validation_passed": True, "violations": []}
    placements = result.promoter_detail["placements"]
    assert len(placements) == 1
    assert placements[0]["upstream_start"] == 0
    assert placements[0]["downstream_start"] == 8


def test_evaluate_sequence_constraints_without_forbid_rules_demotes_promoter_missing_to_rejection() -> None:
    result = _evaluate_sequence_constraints(
        final_seq="ACGTACGTACGTACGT",
        compiled_sequence_constraints=None,
        fixed_elements_dump=_fixed_elements_dump(),
        source_label="demo",
        plan_name="baseline",
        sampling_library_index=5,
        sampling_library_hash="hash-5",
    )
    assert result.rejection_detail is not None
    assert "error" in result.rejection_detail
    assert result.rejection_event_payload is not None
    assert "error" in result.rejection_event_payload
    assert result.error is not None
    assert isinstance(result.error, ValueError)


def test_evaluate_sequence_constraints_raises_on_non_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_type_error(**_kwargs):
        raise TypeError("unexpected")

    monkeypatch.setattr(stage_b_runtime_checks, "validate_sequence_constraints", _raise_type_error)
    with pytest.raises(RuntimeError, match=r"\[demo/baseline\] sequence constraint evaluation failed"):
        _evaluate_sequence_constraints(
            final_seq="TTGACAGGTATAAT",
            compiled_sequence_constraints=None,
            fixed_elements_dump=_fixed_elements_dump(),
            source_label="demo",
            plan_name="baseline",
            sampling_library_index=6,
            sampling_library_hash="hash-6",
        )
