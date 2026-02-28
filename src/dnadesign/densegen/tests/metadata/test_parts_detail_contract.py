"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/metadata/test_parts_detail_contract.py

Fail-fast contract tests for DenseGen TFBS parts detail metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.metadata_schema import validate_metadata
from dnadesign.densegen.tests.meta_fixtures import output_meta


def test_validate_metadata_rejects_tfbs_entry_missing_required_keys() -> None:
    meta = output_meta(library_hash="demo_hash", library_index=1)
    meta["used_tfbs_detail"] = [
        {
            "part_kind": "tfbs",
            "part_index": 0,
            "sequence": "AAAA",
            "core_sequence": "AAAA",
            "orientation": "fwd",
            "offset": 0,
            "offset_raw": 0,
            "pad_left": 0,
            "length": 4,
            "end": 4,
            "source": "demo",
            "motif_id": "motif_1",
            "tfbs_id": "tfbs_1",
        }
    ]
    with pytest.raises(ValueError, match="regulator"):
        validate_metadata(meta)


def test_validate_metadata_rejects_pwm_tfbs_missing_lineage_fields() -> None:
    meta = output_meta(library_hash="demo_hash", library_index=1)
    meta["input_mode"] = "pwm_sampled"
    meta["input_pwm_ids"] = ["MOTIF_A"]
    meta["used_tfbs_detail"] = [
        {
            "part_kind": "tfbs",
            "part_index": 0,
            "regulator": "TF1",
            "sequence": "AAAA",
            "core_sequence": "AAAA",
            "orientation": "fwd",
            "offset": 0,
            "offset_raw": 0,
            "pad_left": 0,
            "length": 4,
            "end": 4,
            "source": "demo",
            "motif_id": "motif_1",
            "tfbs_id": "tfbs_1",
            "score_best_hit_raw": 7.5,
        }
    ]
    with pytest.raises(ValueError, match="score_theoretical_max"):
        validate_metadata(meta)
