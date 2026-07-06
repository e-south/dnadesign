"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_review_axes_contract.py

Review-axis contract tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    _parse_mutations,
)


def test_parse_mutations_accepts_list_and_scalar_string() -> None:
    assert [mutation.position for mutation in _parse_mutations(["A7G", "L21V"], candidate_id="candidate_ok")] == [
        7,
        21,
    ]
    assert _parse_mutations("A7G", candidate_id="candidate_ok")[0].alt_aa == "G"


def test_parse_mutations_accepts_serialized_list() -> None:
    mutations = _parse_mutations("['A7G', 'L21V']", candidate_id="candidate_ok")

    assert [(mutation.wt_aa, mutation.position, mutation.alt_aa) for mutation in mutations] == [
        ("A", 7, "G"),
        ("L", 21, "V"),
    ]


def test_parse_mutations_rejects_malformed_tokens() -> None:
    with pytest.raises(ValueError, match="candidate_bad.*token 2.*bad-token"):
        _parse_mutations(["A7G", "bad-token"], candidate_id="candidate_bad")


def test_parse_mutations_rejects_malformed_serialized_lists() -> None:
    with pytest.raises(ValueError, match="candidate_bad"):
        _parse_mutations("['A7G', bad-token]", candidate_id="candidate_bad")
