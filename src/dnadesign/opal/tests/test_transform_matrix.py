"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_transform_matrix.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.data_access import RecordsStore
from dnadesign.opal.src.registries.transforms_x import get_transform_x
from dnadesign.opal.src.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers identity)
from dnadesign.opal.src.utils import OpalError


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


def test_transform_matrix_preserves_order(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "bio_type": ["dna", "dna", "dna"],
            "sequence": ["BBB", "AAA", "CCC"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[2.0], [1.0], [3.0]],
        }
    )
    store = _store(tmp_path)
    reg = PluginRegistryView("model", "objective", "selection", "identity", "transform_y")
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    tx = get_transform_x("identity", {})
    tctx = rctx.for_plugin(category="transform_x", name="identity", plugin=tx)
    X, order = store.transform_matrix(df, ["b", "a"], ctx=tctx)
    assert order == ["b", "a"]
    assert X.tolist() == [[2.0], [1.0]]


def test_transform_matrix_rejects_duplicate_ids(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAA", "BBB"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[1.0], [2.0]],
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        reg = PluginRegistryView("model", "objective", "selection", "identity", "transform_y")
        rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
        tx = get_transform_x("identity", {})
        tctx = rctx.for_plugin(category="transform_x", name="identity", plugin=tx)
        store.transform_matrix(df, ["a", "a"], ctx=tctx)
