"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_communication_movie_targets.py

Explicit Eco1 communication-movie target contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.cli import (
    build_parser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.catalog import (  # noqa: E501
    MOVIE_TARGET_PROPOSAL_BACKBONES,
    MOVIE_TARGET_PROTECTED_EVIDENCE,
    MOVIE_TARGET_SELECTED_ELECTROSTATICS,
    validated_movie_targets,
)


def test_communication_movie_targets_are_repeatable_and_explicit() -> None:
    args = build_parser().parse_args(
        [
            "--render-communication-movie",
            MOVIE_TARGET_PROTECTED_EVIDENCE,
            "--render-communication-movie",
            MOVIE_TARGET_PROPOSAL_BACKBONES,
        ]
    )

    assert args.render_communication_movie == [
        MOVIE_TARGET_PROTECTED_EVIDENCE,
        MOVIE_TARGET_PROPOSAL_BACKBONES,
    ]


def test_retired_broad_communication_render_flag_fails_fast() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--render-communication-chimerax"])


def test_movie_targets_use_semantic_names() -> None:
    assert {
        MOVIE_TARGET_PROTECTED_EVIDENCE,
        MOVIE_TARGET_PROPOSAL_BACKBONES,
        MOVIE_TARGET_SELECTED_ELECTROSTATICS,
    } == {
        "protected-evidence",
        "proposal-backbones",
        "selected-electrostatics",
    }


def test_movie_target_contract_rejects_unknown_and_duplicate_values() -> None:
    with pytest.raises(ValueError, match="Unknown communication movie target"):
        validated_movie_targets(("all-movies",))
    with pytest.raises(ValueError, match="Duplicate communication movie target"):
        validated_movie_targets((MOVIE_TARGET_PROTECTED_EVIDENCE, MOVIE_TARGET_PROTECTED_EVIDENCE))
