"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_job_v2_schema.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.baserender.src.config.job_v2 import load_job_v2
from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.job_runner import run_job


def _write_parquet(path, *, include_id: bool = False):
    seqs = ["ACGTTT"]
    anns = [[{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}]]
    data = {
        "sequence": seqs,
        "densegen__used_tfbs_detail": anns,
    }
    if include_id:
        data["id"] = ["rec_1"]
    table = pa.table(data)
    pq.write_table(table, path)


def _minimal_job(path, *, include_output_video: bool = False, include_images: bool = True):
    output = {}
    if include_output_video:
        output["video"] = {"path": str(path.parent / "out.mp4"), "fmt": "mp4"}
    if include_images:
        output["images"] = {"dir": str(path.parent / "images"), "fmt": "png"}
    return {
        "version": 2,
        "input": {
            "path": str(path),
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
        },
        "output": output,
    }


def test_job_v2_requires_version_2(tmp_path):
    job = {
        "input": {
            "path": "x.parquet",
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
        },
        "output": {"images": {"dir": "images", "fmt": "png"}},
    }
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(SchemaError):
        load_job_v2(path)


def test_job_v2_rejects_unknown_top_level_keys(tmp_path):
    job = _minimal_job(tmp_path / "data.parquet")
    job["unexpected"] = True
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(SchemaError):
        load_job_v2(path)


def test_job_v2_columns_id_requires_presence(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path, include_id=False)
    job = _minimal_job(data_path)
    job["input"]["columns"]["id"] = "id"
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    cfg = load_job_v2(path)
    with pytest.raises(SchemaError):
        run_job(cfg)


def test_job_v2_overlay_column_missing_errors(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path, include_id=True)
    sel_path = tmp_path / "sel.csv"
    sel_path.write_text("id,other\nrec_1,foo\n")
    job = _minimal_job(data_path)
    job["input"]["columns"]["id"] = "id"
    job["selection"] = {
        "path": str(sel_path),
        "match_on": "id",
        "column": "id",
        "overlay_column": "details",
    }
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    cfg = load_job_v2(path)
    with pytest.raises(SchemaError):
        run_job(cfg)


def test_job_v2_annotations_policy_on_missing_kmer_supported(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path, include_id=True)
    job = _minimal_job(data_path)
    job["input"]["annotations"] = {"on_missing_kmer": "skip_entry"}
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    cfg = load_job_v2(path)
    assert cfg.input.annotations.on_missing_kmer == "skip_entry"


def test_job_v2_sample_conflicts_with_limit_errors(tmp_path):
    data_path = tmp_path / "data.parquet"
    _write_parquet(data_path, include_id=True)
    job = _minimal_job(data_path)
    job["input"]["limit"] = 10
    job["input"]["sample"] = {"mode": "first_n", "n": 5}
    path = tmp_path / "job.yml"
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(SchemaError):
        load_job_v2(path)
