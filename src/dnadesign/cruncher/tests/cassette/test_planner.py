"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cassette/test_planner.py

Planner tests for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.cassette_workflow import validate_cassette_spec


def _write_workspace(
    tmp_path: Path,
    *,
    right_window_start: int = 11,
    right_window_end: int = 13,
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "demo.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
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
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "cassette": {
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
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def test_validate_cassette_spec_returns_satisfied_candidate(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    report = validate_cassette_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.cassette_sequence == "AACGATTTATCGTT"
    assert report.candidate.left_nick.nick_coordinate == 2
    assert report.candidate.right_nick.nick_coordinate == 12
    assert report.candidate.bounded_segment.length == 10
    assert report.render_contract is not None


def test_validate_cassette_spec_reports_unsatisfied_window_miss(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, right_window_start=13, right_window_end=13)

    report = validate_cassette_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["missing_right_nick"]


def test_validate_cassette_spec_rejects_window_end_at_cassette_length(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, right_window_end=14)

    with pytest.raises(ValueError, match="exceeds cassette length"):
        validate_cassette_spec(spec_path)
