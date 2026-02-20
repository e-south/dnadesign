"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_ingest_missing_x_drop.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.registries.transforms_y import register_transform_y
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


@register_transform_y("test_reorder_reset")
def _test_reorder_reset(df_tidy: pd.DataFrame, params: dict, *, ctx=None) -> pd.DataFrame:
    _unused = (params, ctx)
    out = df_tidy.sort_values("sequence").reset_index(drop=True)
    return pd.DataFrame(
        {
            "sequence": out["sequence"],
            "y": out["y_val"].map(lambda v: [float(v)]),
        }
    )


def test_ingest_y_drops_unknown_missing_x_by_sequence(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=False)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="test_reorder_reset",
        transforms_y_params={},
        y_expected_length=1,
    )

    csv_path = workdir / "labels.parquet"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "DDD", "CCC"],
            "y_val": [0.2, 0.9, 0.1],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.3, 0.4], None],
        }
    )
    df.to_parquet(csv_path, index=False)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--unknown-sequences",
            "create",
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout

    out_df = pd.read_parquet(records)
    sequences = set(out_df["sequence"].astype(str).tolist())
    assert "DDD" in sequences
    assert "CCC" not in sequences
