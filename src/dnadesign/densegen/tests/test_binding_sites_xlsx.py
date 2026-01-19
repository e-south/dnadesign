from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.adapters.sources.binding_sites import BindingSitesDataSource


def test_binding_sites_xlsx_loads(tmp_path: Path) -> None:
    df = pd.DataFrame({"tf": ["TF1", "TF2"], "tfbs": ["ACGT", "TTAA"]})
    path = tmp_path / "sites.xlsx"
    df.to_excel(path, index=False)

    ds = BindingSitesDataSource(path=str(path), cfg_path=tmp_path)
    entries, meta = ds.load_data()

    assert len(entries) == 2
    assert meta is not None
    assert entries[0][0] == "TF1"
    assert entries[0][1] == "ACGT"
