"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_ingest_duplicate_policy.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.transforms_y import get_transform_y
from dnadesign.opal.src.runtime.ingest import run_ingest
from dnadesign.opal.src.transforms_y import sfxi_vec8_from_table_v1  # noqa: F401 (registers)


def _csv_with_duplicate_sequence():
    return pd.DataFrame(
        {
            "sequence": ["AAA", "AAA"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 1.0],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )


def _ingest_ctx():
    reg = PluginRegistryView("model", "objective", "selection", "transform_x", "sfxi_vec8_from_table_v1")
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    tfn = get_transform_y("sfxi_vec8_from_table_v1")
    return rctx.for_plugin(category="transform_y", name="sfxi_vec8_from_table_v1", plugin=tfn)


def test_ingest_duplicate_policy_error():
    records_df = pd.DataFrame(
        {
            "id": ["x"],
            "sequence": ["AAA"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
        }
    )
    csv_df = _csv_with_duplicate_sequence()
    with pytest.raises(OpalError):
        run_ingest(
            records_df,
            csv_df,
            transform_name="sfxi_vec8_from_table_v1",
            transform_params={"sequence_column": "sequence"},
            y_expected_length=8,
            y_column_name="Y",
            duplicate_policy="error",
            ctx=_ingest_ctx(),
        )


def test_ingest_duplicate_policy_keep_last():
    records_df = pd.DataFrame(
        {
            "id": ["x"],
            "sequence": ["AAA"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
        }
    )
    csv_df = _csv_with_duplicate_sequence()
    labels, preview = run_ingest(
        records_df,
        csv_df,
        transform_name="sfxi_vec8_from_table_v1",
        transform_params={"sequence_column": "sequence"},
        y_expected_length=8,
        y_column_name="Y",
        duplicate_policy="keep_last",
        ctx=_ingest_ctx(),
    )
    assert len(labels) == 1
    assert preview.duplicates_found == 2
    assert preview.duplicates_dropped == 1


def test_ingest_transform_error_is_user_friendly():
    records_df = pd.DataFrame(
        {
            "id": ["x"],
            "sequence": ["AAA"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
        }
    )
    # Missing required columns for sfxi_vec8_from_table_v1
    csv_df = pd.DataFrame({"sequence": ["AAA"], "v00": [0.0]})
    with pytest.raises(OpalError) as exc:
        run_ingest(
            records_df,
            csv_df,
            transform_name="sfxi_vec8_from_table_v1",
            transform_params={"sequence_column": "sequence"},
            y_expected_length=8,
            y_column_name="Y",
            duplicate_policy="error",
            ctx=_ingest_ctx(),
        )
    msg = str(exc.value)
    assert "Y transform 'sfxi_vec8_from_table_v1' failed" in msg
    assert "Input columns" in msg
