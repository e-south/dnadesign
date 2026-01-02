from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_records


def test_notebook_generate_smoke(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    out_path = workdir / "notebooks" / "opal_demo_analysis.py"
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--out",
            str(out_path),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out_path.exists()

    txt = out_path.read_text()
    assert "marimo.App" in txt
    assert "CampaignAnalysis.from_config_path" in txt
    assert "opal" in txt.lower()

    # Optional import check if marimo is installed
    if importlib.util.find_spec("marimo") is not None:
        spec = importlib.util.spec_from_file_location("opal_campaign_nb", out_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "app")
