"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_binding_sites_duplicates.py

Binding-sites duplicate handling tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.adapters.sources.binding_sites import BindingSitesDataSource


def test_binding_sites_duplicates_allowed(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF1,AAA\nTF2,CCC\n")
    ds = BindingSitesDataSource(path=str(csv_path), cfg_path=tmp_path)
    with caplog.at_level("WARNING"):
        entries, df, _summaries = ds.load_data()
    assert len(entries) == 3
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert "source" in df.columns
    assert set(df["source"].tolist()) == {str(csv_path)}
    assert "duplicate regulator/binding-site pairs" in caplog.text.lower()
