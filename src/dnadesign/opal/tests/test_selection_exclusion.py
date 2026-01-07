"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_selection_exclusion.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd

from dnadesign.opal.src.storage.data_access import RecordsStore


def test_labeled_id_set_leq_round_basic(tmp_path):
    # Minimal frame with label_hist column
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAA", "CCC"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1], [0.2]],
            "opal__demo__label_hist": [[{"r": 0, "y": [0, 0, 0, 0, 0, 0, 0, 0]}], None],
        }
    )
    p = tmp_path / "records.parquet"
    df.to_parquet(p, index=False)
    store = RecordsStore(
        kind="local",
        records_path=p,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )
    s = store.labeled_id_set_leq_round(df, as_of_round=0)
    assert "a" in s and "b" not in s
