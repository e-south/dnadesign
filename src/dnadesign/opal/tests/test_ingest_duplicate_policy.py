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

from dnadesign.opal.src.ingest import run_ingest
from dnadesign.opal.src.transforms_y import sfxi_vec8_from_table_v1  # noqa: F401 (registers)
from dnadesign.opal.src.utils import OpalError


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
        }
    )


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
    )
    assert len(labels) == 1
    assert preview.duplicates_found == 2
    assert preview.duplicates_dropped == 1
