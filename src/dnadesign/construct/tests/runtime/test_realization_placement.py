"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_realization_placement.py

Unit contracts for construct placement and guard realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dnadesign.construct.src.contracts.config import PartConfig
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.realization.placement import (
    PlacementSite,
    planned_placements,
    resolved_placement_sites,
    template_context_sequence,
    template_match_offsets,
    validate_placement_guards,
    validate_placements,
)


@dataclass(frozen=True)
class _Template:
    id: str = "template"
    sequence: str = "AAAACCCCGGGGTTTT"
    circular: bool = False


def _part(
    *,
    name: str = "anchor",
    kind: str = "replace",
    locator: dict[str, Any] | None = None,
    guards: dict[str, Any] | None = None,
) -> PartConfig:
    return PartConfig.model_validate(
        {
            "name": name,
            "role": "anchor",
            "sequence": {
                "source": "input_field",
                "field": "sequence",
            },
            "placement": {
                "kind": kind,
                "orientation": "forward",
                "locator": locator
                or {
                    "kind": "coordinates",
                    "start": 4,
                    "end": 8,
                },
                "guards": guards,
            },
        }
    )


def test_flank_locator_requires_unique_forward_flank_matches() -> None:
    part = _part(
        locator={
            "kind": "flanks",
            "upstream_sequence": "AAAA",
            "downstream_sequence": "CCCC",
        },
        guards=None,
    )

    with pytest.raises(ValidationError, match="upstream_sequence.*found 2"):
        resolved_placement_sites(_Template(sequence="AAAAGGGGAAAACCCC"), [part])


def test_resolved_placement_sites_rejects_duplicate_part_names() -> None:
    left = _part(name="anchor")
    right = _part(name="anchor")

    with pytest.raises(ValidationError, match="Duplicate part name 'anchor'"):
        resolved_placement_sites(_Template(), [left, right])


def test_validate_placements_rejects_same_start_with_different_end() -> None:
    left = _part(name="left")
    right = _part(name="right")

    with pytest.raises(ValidationError, match="shares template start 4"):
        validate_placements(
            16,
            [left, right],
            resolved_sites={
                "left": PlacementSite(4, 8, "coordinates", None, None),
                "right": PlacementSite(4, 9, "coordinates", None, None),
            },
        )


def test_validate_placement_guards_rejects_wrong_replaced_sequence() -> None:
    part = _part(guards={"replaced_sequence": "TTTT"})

    with pytest.raises(ValidationError, match="template interval \\[4, 8\\)"):
        validate_placement_guards(
            template=_Template(),
            part=part,
            site=PlacementSite(4, 8, "coordinates", None, None),
        )


def test_validate_placement_guards_rejects_non_unique_guard_match() -> None:
    part = _part(guards={"replaced_sequence": "CCCC", "require_unique_forward_matches": True})

    with pytest.raises(ValidationError, match="unique forward-strand match.*found 2"):
        validate_placement_guards(
            template=_Template(sequence="AAAACCCCGGGGCCCC"),
            part=part,
            site=PlacementSite(4, 8, "coordinates", None, None),
        )


def test_planned_placements_exposes_guard_mode_and_observed_context() -> None:
    part = _part(
        guards={
            "replaced_sequence": "CCCC",
            "upstream_sequence": "AAAA",
            "downstream_sequence": "GGGG",
            "replaced_span_bp": 4,
            "require_unique_forward_matches": True,
        }
    )
    template = _Template()
    sites = resolved_placement_sites(template, [part])

    validate_placement_guards(template=template, part=part, site=sites["anchor"])
    planned = planned_placements([part], template=template, resolved_sites=sites)

    assert len(planned) == 1
    assert planned[0].guard_mode == "replaced_sequence_and_context"
    assert planned[0].template_sequence == "CCCC"
    assert planned[0].guard_replaced_span_bp == 4
    assert planned[0].observed_guard_upstream_sequence == "AAAA"
    assert planned[0].observed_guard_downstream_sequence == "GGGG"


def test_template_match_offsets_can_search_across_circular_origin() -> None:
    assert template_match_offsets("CCAAAAGG", "GGCC", circular=False) == []
    assert template_match_offsets("CCAAAAGG", "GGCC", circular=True) == [6]


def test_template_match_offsets_rejects_circular_match_longer_than_template() -> None:
    with pytest.raises(ValidationError, match="must not exceed template length"):
        template_match_offsets("AT", "TAT", circular=True)


def test_template_context_sequence_rejects_unknown_direction() -> None:
    with pytest.raises(ValidationError, match="direction must be 'upstream' or 'downstream'"):
        template_context_sequence(
            "AAAACCCC",
            anchor=4,
            length=2,
            circular=False,
            direction="sideways",
        )


def test_template_context_sequence_rejects_empty_circular_template() -> None:
    with pytest.raises(ValidationError, match="template sequence cannot be empty"):
        template_context_sequence(
            "",
            anchor=0,
            length=2,
            circular=True,
            direction="upstream",
        )
