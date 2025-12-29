"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_output_paths.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.baserender.src.presets.loader import _baserender_root, load_job


def _write_job(path, *, video_path=None, images_dir=None):
    data = {
        "input": {
            "path": "inputs/records.parquet",
            "format": "parquet",
            "columns": {
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
            "alphabet": "DNA",
        },
        "output": {
            "video": {
                "fmt": "mp4",
                "fps": 2,
            }
        },
    }
    if video_path is not None:
        data["output"]["video"]["path"] = video_path
    if images_dir is not None:
        data["output"]["images"] = {"dir": images_dir, "fmt": "png"}
    path.write_text(yaml.safe_dump(data))


def test_images_dir_results_prefix_resolves_to_root_results(tmp_path):
    job_path = tmp_path / "job_a.yml"
    _write_job(job_path, images_dir="results/shared/images")
    cfg = load_job(job_path)
    root = _baserender_root()
    assert cfg.images is not None
    assert cfg.images.dir == root / "results" / "shared" / "images"


def test_images_dir_leaf_resolves_under_results_job(tmp_path):
    job_path = tmp_path / "job_b.yml"
    _write_job(job_path, images_dir="images")
    cfg = load_job(job_path)
    root = _baserender_root()
    assert cfg.images is not None
    assert cfg.images.dir == root / "results" / "job_b" / "images"


def test_video_path_leaf_resolves_under_results_job(tmp_path):
    job_path = tmp_path / "job_c.yml"
    _write_job(job_path, video_path="video.mp4")
    cfg = load_job(job_path)
    root = _baserender_root()
    assert cfg.video.out_path == root / "results" / "job_c" / "video.mp4"
