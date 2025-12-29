"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_cli_job_v2.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.baserender.src.cli import app

runner = CliRunner()


def _write_parquet(path):
    seqs = ["ACGTTT"]
    anns = [[{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}]]
    table = pa.table(
        {
            "sequence": seqs,
            "densegen__used_tfbs_detail": anns,
        }
    )
    pq.write_table(table, path)


def _write_job(path, data):
    import yaml

    path.write_text(yaml.safe_dump(data))


def test_cli_job_validate_exits_0_on_valid_job(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path)
    job = {
        "version": 2,
        "input": {
            "path": str(data_path),
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
        },
        "output": {"images": {"dir": str(tmp_path / "images"), "fmt": "png"}},
    }
    job_path = tmp_path / "job.yml"
    _write_job(job_path, job)
    result = runner.invoke(app, ["job", "validate", str(job_path)])
    assert result.exit_code == 0


def test_cli_job_lint_reports_redundant_style_keys(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path)
    job = {
        "version": 2,
        "input": {
            "path": str(data_path),
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
        },
        "style": {"overrides": {"track_spacing": 35}},
        "output": {"images": {"dir": str(tmp_path / "images"), "fmt": "png"}},
    }
    job_path = tmp_path / "job.yml"
    _write_job(job_path, job)
    result = runner.invoke(app, ["job", "lint", str(job_path)])
    assert result.exit_code == 0
    assert "track_spacing" in result.stdout


def test_cli_job_normalize_outputs_version_2_job(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path)
    job = {
        "version": 2,
        "input": {
            "path": str(data_path),
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
        },
        "output": {"images": {"dir": str(tmp_path / "images"), "fmt": "png"}},
    }
    job_path = tmp_path / "job.yml"
    _write_job(job_path, job)
    out_path = tmp_path / "out.yml"
    result = runner.invoke(app, ["job", "normalize", str(job_path), "--out", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    assert "version: 2" in out_path.read_text()
