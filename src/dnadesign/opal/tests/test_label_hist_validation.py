"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_label_hist_validation.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.data_access import RecordsStore


def _store(tmp_path):
    return RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )


def test_label_hist_validation_rejects_malformed_entries(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [[{"y": [1.0, 2.0]}]],  # missing r
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        store.validate_label_hist(df, require=True)


def test_label_hist_validation_allows_missing_when_not_required(tmp_path):
    df = pd.DataFrame({"id": ["a"], "sequence": ["AAA"], "bio_type": ["dna"], "alphabet": ["dna_4"]})
    store = _store(tmp_path)
    store.validate_label_hist(df, require=False)
