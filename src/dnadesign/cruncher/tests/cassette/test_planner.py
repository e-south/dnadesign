"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cassette/test_planner.py

Planner tests for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from dnadesign.cruncher.app.cassette_workflow import validate_cassette_spec


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _write_workspace(
    tmp_path: Path,
    *,
    cassette_payload: dict[str, Any],
    catalog_entries: list[dict[str, Any]],
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "demo.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump({"nickases": {"schema_version": 1, "entries": catalog_entries}}),
        encoding="utf-8",
    )
    spec_path.write_text(yaml.safe_dump({"cassette": cassette_payload}), encoding="utf-8")
    return spec_path


def _legacy_catalog_entries() -> list[dict[str, Any]]:
    return [
        {
            "id": "nb_left",
            "recognition_sequence": "AACGA",
            "nicked_site_strand": "forward",
            "cut_offset": 2,
        },
        {
            "id": "nb_right",
            "recognition_sequence": "AACGA",
            "nicked_site_strand": "reverse",
            "cut_offset": 3,
        },
    ]


def _v1_cassette_payload(*, right_window_start: int = 11, right_window_end: int = 13) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "demo_hairpin",
        "topology": {
            "stem5p_arm": "AACGAT",
            "loop": "TT",
            "stem3p_arm_mode": "derive_reverse_complement",
        },
        "duplex_context": {"upstream": "", "downstream": ""},
        "nicking": {
            "designated_strand": "primary_strand",
            "left": {"nickase": "nb_left", "nick_window": {"start": 0, "end": 3}},
            "right": {
                "nickase": "nb_right",
                "nick_window": {"start": right_window_start, "end": right_window_end},
            },
        },
        "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        "output": {"run_dir": "outputs/cassettes", "write_render_contract": True},
    }


def _v2_catalog_entries() -> list[dict[str, Any]]:
    return [
        {
            "id": "Nt.demo",
            "specificity_id": "Demo",
            "motif_top_5to3": "AACGA",
            "top_cut_offset": 2,
            "source": "demo",
        },
        {
            "id": "Nb.demo",
            "specificity_id": "Demo",
            "motif_top_5to3": "AACGA",
            "bottom_cut_offset": 2,
            "source": "demo",
        },
        {
            "id": "Nt.alt",
            "specificity_id": "Alt",
            "motif_top_5to3": "AACGA",
            "top_cut_offset": 2,
            "source": "demo",
        },
        {
            "id": "Nt.extra",
            "specificity_id": "Extra",
            "motif_top_5to3": "GATTT",
            "top_cut_offset": 2,
            "source": "demo",
        },
        {
            "id": "Nb.cross",
            "specificity_id": "Cross",
            "motif_top_5to3": "TATCG",
            "bottom_cut_offset": 3,
            "source": "demo",
        },
        {
            "id": "Nt.start",
            "specificity_id": "Terminal",
            "motif_top_5to3": "AACGA",
            "top_cut_offset": 0,
            "source": "demo",
        },
        {
            "id": "Nb.end",
            "specificity_id": "Terminal",
            "motif_top_5to3": "AACGA",
            "bottom_cut_offset": 0,
            "source": "demo",
        },
    ]


def _v2_cassette_payload() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": "demo_hairpin",
        "topology": {
            "stem5p_arm": "AACGAT",
            "loop": "TT",
            "stem3p_arm_mode": "derived_reverse_complement",
        },
        "construct_context": {"left_flank": "", "right_flank": ""},
        "nicking": {
            "target_strand": "primary",
            "left": {"nickase": "Nt.demo", "nick_window": {"start": 2, "end": 2}},
            "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 12}},
            "require_exactly_two_intended_nicks": True,
            "bounded_segment_length": {"min": 10, "max": 10},
        },
        "site_policy": {
            "forbid_additional_designated_strand_nicks": False,
            "scan_scope": "requested_variants",
        },
        "hairpin_validation": {
            "require_topological_hairpin": True,
            "require_energetic_hairpin": False,
        },
        "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        "output": {"run_dir": "outputs/cassettes", "write_render_contract": True},
    }


def test_validate_cassette_spec_returns_satisfied_candidate_for_v1(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_v1_cassette_payload(),
        catalog_entries=_legacy_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "satisfied"
    assert report.metadata.spec_schema_version == 1
    assert report.metadata.coordinate_semantics == "legacy_v1"
    assert report.candidate is not None
    assert report.candidate.cassette_sequence == "AACGATTTATCGTT"
    assert report.candidate.stem3p_arm == "ATCGTT"
    assert report.candidate.intended_left_nick.boundary == 2
    assert report.candidate.intended_right_nick.boundary == 12
    assert report.candidate.bounded_nicked_segment.length_nt == 10
    assert report.render_contract is not None


def test_validate_cassette_spec_reports_unsatisfied_window_miss_for_v1(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_v1_cassette_payload(right_window_start=13, right_window_end=13),
        catalog_entries=_legacy_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["RIGHT_WINDOW_NO_MATCH"]


def test_validate_cassette_spec_rejects_v1_window_end_at_cassette_length(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_v1_cassette_payload(right_window_end=14),
        catalog_entries=_legacy_catalog_entries(),
    )

    with pytest.raises(ValueError, match="exceeds cassette length"):
        validate_cassette_spec(spec_path)


def test_validate_cassette_spec_returns_satisfied_candidate_for_v2(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_v2_cassette_payload(),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "satisfied"
    assert report.metadata.spec_schema_version == 2
    assert report.metadata.coordinate_semantics == "boundary_inclusive_v2"
    assert report.metadata.left_flank_length == 0
    assert report.metadata.right_flank_length == 0
    assert report.candidate is not None
    assert report.candidate.target_strand == "primary"
    assert report.candidate.intended_left_site.orientation == "forward"
    assert report.candidate.intended_right_site.orientation == "reverse"
    assert report.candidate.intended_left_nick.boundary == 2
    assert report.candidate.intended_right_nick.boundary == 12
    assert report.candidate.bounded_nicked_segment.start_boundary == 2
    assert report.candidate.bounded_nicked_segment.end_boundary == 12
    assert report.candidate.bounded_nicked_segment.length_nt == 10


def test_validate_cassette_spec_v2_allows_terminal_boundaries(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "nicking": {
                    "left": {"nickase": "Nt.start", "nick_window": {"start": 0, "end": 0}},
                    "right": {"nickase": "Nb.end", "nick_window": {"start": 14, "end": 14}},
                    "bounded_segment_length": {"min": 14, "max": 14},
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.intended_left_nick.boundary == 0
    assert report.candidate.intended_right_nick.boundary == 14


def test_validate_cassette_spec_v2_reports_target_strand_mismatch(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "nicking": {
                    "right": {"nickase": "Nt.alt", "nick_window": {"start": 12, "end": 12}},
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["TARGET_STRAND_MISMATCH"]


def test_validate_cassette_spec_v2_reports_bounded_segment_length_out_of_range(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "nicking": {
                    "bounded_segment_length": {"min": 11, "max": 11},
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["BOUNDED_SEGMENT_LENGTH_OUT_OF_RANGE"]


def test_validate_cassette_spec_v2_reports_extra_designated_strand_nicks_under_catalog_scan(
    tmp_path: Path,
) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "site_policy": {
                    "forbid_additional_designated_strand_nicks": True,
                    "scan_scope": "catalog",
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["EXTRA_DESIGNATED_STRAND_NICKS_FOUND"]


def test_validate_cassette_spec_v2_reports_site_crossing_stem_boundary(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "nicking": {
                    "target_strand": "complement",
                    "left": {"nickase": "Nb.demo", "nick_window": {"start": 2, "end": 2}},
                    "right": {"nickase": "Nb.cross", "nick_window": {"start": 10, "end": 10}},
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["RIGHT_SITE_NOT_IN_RIGHT_STEM"]


def test_validate_cassette_spec_v2_reports_unsat_by_mirror_symmetry(tmp_path: Path) -> None:
    spec_path = _write_workspace(
        tmp_path,
        cassette_payload=_deep_merge(
            _v2_cassette_payload(),
            {
                "nicking": {
                    "right": {"nickase": "Nt.demo", "nick_window": {"start": 12, "end": 12}},
                }
            },
        ),
        catalog_entries=_v2_catalog_entries(),
    )

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["UNSAT_BY_MIRROR_SYMMETRY"]
