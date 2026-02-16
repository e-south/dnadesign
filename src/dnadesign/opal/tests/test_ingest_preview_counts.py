# ABOUTME: Validates ingest preview counts for id/sequence presence.
# ABOUTME: Ensures preview statistics reflect actual non-null counts.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_ingest_preview_counts.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src.cli.formatting.renderers.ingest import render_ingest_preview_human
from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.registries.transforms_y import get_transform_y, register_transform_y
from dnadesign.opal.src.runtime.ingest import IngestPreview, run_ingest
from dnadesign.opal.src.transforms_y import scalar_from_table_v1  # noqa: F401 (registers)


def _ingest_ctx() -> RoundCtx:
    reg = PluginRegistryView(
        model="model",
        objective="objective",
        selection="selection",
        transform_x="transform_x",
        transform_y="scalar_from_table_v1",
    )
    rctx = RoundCtx(core={"core/round_index": 0, "core/campaign_slug": "demo"}, registry=reg)
    fn = get_transform_y("scalar_from_table_v1")
    return rctx.for_plugin(category="transform_y", name="scalar_from_table_v1", plugin=fn)


def _ingest_ctx_for(name: str) -> RoundCtx:
    reg = PluginRegistryView(
        model="model",
        objective="objective",
        selection="selection",
        transform_x="transform_x",
        transform_y=name,
    )
    rctx = RoundCtx(core={"core/round_index": 0, "core/campaign_slug": "demo"}, registry=reg)
    fn = get_transform_y(name)
    return rctx.for_plugin(category="transform_y", name=name, plugin=fn)


@register_transform_y("table_with_optional_id_v1")
def _table_with_optional_id_v1(csv_df: pd.DataFrame, params: dict, ctx=None) -> pd.DataFrame:
    _ = params, ctx
    return pd.DataFrame(
        {
            "id": csv_df["id"],
            "sequence": csv_df["sequence"],
            "y": [[float(v)] for v in csv_df["y"].to_numpy(dtype=float)],
        }
    )


def test_ingest_preview_counts_rows_with_id_and_sequence(tmp_path: Path) -> None:
    records_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
        }
    )
    csv_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "y": [0.1, 0.2],
        }
    )
    labels_df, preview = run_ingest(
        records_df,
        csv_df,
        transform_name="scalar_from_table_v1",
        transform_params={"sequence_column": "sequence", "y_column": "y", "id_column": "id"},
        y_expected_length=1,
        y_column_name="Y",
        duplicate_policy="error",
        ctx=_ingest_ctx(),
    )
    assert len(labels_df) == 2
    assert preview.rows_with_id == 2
    assert preview.rows_with_sequence == 2


def test_ingest_resolves_missing_ids_by_sequence(tmp_path: Path) -> None:
    records_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
        }
    )
    csv_df = pd.DataFrame(
        {
            "id": [None, "b"],
            "sequence": ["AAA", "BBB"],
            "y": [0.1, 0.2],
        }
    )
    labels_df, preview = run_ingest(
        records_df,
        csv_df,
        transform_name="table_with_optional_id_v1",
        transform_params={},
        y_expected_length=1,
        y_column_name="Y",
        duplicate_policy="error",
        ctx=_ingest_ctx_for("table_with_optional_id_v1"),
    )
    assert labels_df["id"].astype(str).tolist() == ["a", "b"]
    assert preview.rows_with_sequence == 2


def test_ingest_preview_renders_unresolved_id_instead_of_nan() -> None:
    preview = IngestPreview(
        total_rows_in_csv=2,
        rows_with_id=0,
        rows_with_sequence=2,
        resolved_ids_by_sequence=1,
        unknown_sequences=1,
        y_expected_length=1,
        y_length_ok=2,
        y_length_bad=0,
        y_column_name="Y",
        duplicate_policy="error",
        duplicate_key="id",
        duplicates_found=0,
        duplicates_dropped=0,
        warnings=[],
    )
    sample_rows = [
        {"id": np.nan, "sequence": "AAA", "y": [0.1]},
        {"id": "known", "sequence": "BBB", "y": [0.2]},
    ]
    rendered = render_ingest_preview_human(preview, sample_rows, transform_name="scalar_from_table_v1")
    assert "id=<unresolved>" in rendered
    assert "id=nan" not in rendered
