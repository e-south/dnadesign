"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/integrations/test_dense_arrays_playback.py

Verify DenseGen-to-dense-arrays playback translation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dense_arrays.playback import reconstruct_playback
from dense_arrays.realized import Orientation

from dnadesign.densegen.src.integrations.dense_arrays.playback import (
    realized_array_from_densegen_record,
)


def test_reverse_placement_uses_realized_reverse_complement() -> None:
    record = {
        "id": "record-1",
        "sequence": "AAACCGT",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "sequence": "ACGG",
                "offset": 3,
                "offset_raw": 3,
                "end": 7,
                "orientation": "rev",
                "tfbs_id": "tfbs-1",
                "regulator": "TF_A",
            }
        ],
    }

    realized = realized_array_from_densegen_record(
        record,
        source_ref="fixture.parquet",
    )

    assert realized.placements[0].orientation is Orientation.REVERSE
    assert realized.placements[0].sequence == "CCGT"
    assert realized.placements[0].metadata["library_sequence"] == "ACGG"
    assert reconstruct_playback(realized).steps[0].placement_sequence == "CCGT"


def test_fixed_element_recovers_sequence_consistent_raw_coordinate() -> None:
    record = {
        "id": "record-2",
        "sequence": "AAACCCGGG",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "fixed_element",
                "sequence": "CCC",
                "offset": 4,
                "offset_raw": 3,
                "pad_left": 1,
                "end": 7,
                "constraint_name": "anchor",
                "placement_index": 0,
                "role": "upstream",
            }
        ],
    }

    realized = realized_array_from_densegen_record(
        record,
        source_ref="fixture.parquet",
    )

    assert realized.placements[0].start == 3
    assert realized.placements[0].metadata["coordinate_source"] == "offset_raw"
    plan = reconstruct_playback(realized)
    assert any(notice.code == "coordinate_recovered" for notice in plan.notices)
