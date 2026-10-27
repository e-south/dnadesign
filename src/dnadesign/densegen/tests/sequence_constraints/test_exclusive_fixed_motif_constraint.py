"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/sequence_constraints/test_exclusive_fixed_motif_constraint.py

Tests for global motif exclusion with exact fixed-placement allowlist semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.sequence_constraints.engine import (
    compile_sequence_constraints,
    validate_sequence_constraints,
)


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


def _motif_sets() -> dict[str, dict[str, str]]:
    return {"sigma_hexamers": {"upstream": "TTGACA", "downstream": "TATAAT"}}


def test_sequence_constraints_allow_exact_promoter_placements() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints=_constraint_block(),
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    result = validate_sequence_constraints(
        sequence="TTGACAGGTATAAT",
        compiled=compiled,
        fixed_elements_dump=_fixed_elements_dump(),
    )
    assert result.validation_passed is True
    assert result.violations == []
    placements = result.promoter_detail["placements"]
    assert placements[0]["upstream_start"] == 0
    assert placements[0]["downstream_start"] == 8
    assert placements[0]["spacer_length"] == 2


def test_sequence_constraints_reject_extra_occurrence_outside_allowed_coordinates() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints=_constraint_block(),
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    result = validate_sequence_constraints(
        sequence="TTGACAGGTATAATTTGACA",
        compiled=compiled,
        fixed_elements_dump=_fixed_elements_dump(),
    )
    assert result.validation_passed is False
    assert len(result.violations) >= 1
    assert result.violations[0]["constraint"] == "sigma_core"
    assert result.violations[0]["pattern"] in {"TTGACA", "TGTCAA", "TATAAT", "ATTATA"}


def test_sequence_constraints_resolve_promoter_detail_without_forbid_rules() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints={},
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    result = validate_sequence_constraints(
        sequence="TTGACAGGTATAAT",
        compiled=compiled,
        fixed_elements_dump=_fixed_elements_dump(),
    )
    assert result.validation_passed is True
    assert result.violations == []
    placements = result.promoter_detail["placements"]
    assert len(placements) == 1
    assert placements[0]["upstream_start"] == 0
    assert placements[0]["downstream_start"] == 8
    assert placements[0]["spacer_length"] == 2


def test_sequence_constraints_without_forbid_rules_still_reject_missing_promoter_placement() -> None:
    compiled = compile_sequence_constraints(
        sequence_constraints={},
        motif_sets=_motif_sets(),
        fixed_elements_dump=_fixed_elements_dump(),
    )
    with pytest.raises(ValueError, match="could not resolve promoter placement"):
        validate_sequence_constraints(
            sequence="ACGTACGTACGTACGT",
            compiled=compiled,
            fixed_elements_dump=_fixed_elements_dump(),
        )
