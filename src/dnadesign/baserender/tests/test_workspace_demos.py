"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_workspace_demos.py

Tests for curated baserender workspace demos and their self-contained IO assets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from dnadesign.baserender.src.api import run_cruncher_showcase_job
from dnadesign.baserender.src.config import load_cruncher_showcase_job


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_under(path: Path, root: Path) -> bool:
    path_r = path.resolve()
    root_r = root.resolve()
    return path_r == root_r or root_r in path_r.parents


def test_curated_workspace_demos_are_self_contained() -> None:
    root = _pkg_root()
    demos = ("demo_densegen_render", "demo_cruncher_render")

    for name in demos:
        ws = root / "workspaces" / name
        job_path = ws / "job.yaml"
        assert ws.exists(), f"missing workspace: {ws}"
        assert (ws / "inputs").exists(), f"missing inputs dir for {name}"
        assert job_path.exists(), f"missing job.yaml for {name}"

        raw = yaml.safe_load(job_path.read_text())
        assert raw["render"]["style"]["overrides"], f"{name} must define style overrides"

        job = load_cruncher_showcase_job(job_path, caller_root=root)
        assert _is_under(job.input.path, ws / "inputs"), f"{name} input path must be within workspace inputs/"

        if name == "demo_cruncher_render":
            cols = job.input.adapter.columns
            assert job.input.adapter.kind == "generic_features"
            assert cols["features"] == "features"
            assert cols["effects"] == "effects"
            assert cols["display"] == "display"
            import pyarrow.parquet as pq

            rows = pq.read_table(ws / "inputs" / "elites_showcase_records.parquet").to_pylist()
            assert len(rows) == 2
            for row in rows:
                assert {"id", "sequence", "features", "effects", "display"} <= set(row.keys())
                assert isinstance(row["features"], list)
                assert isinstance(row["effects"], list)
                assert isinstance(row["display"], dict)


def test_docs_cruncher_example_uses_local_examples_data() -> None:
    root = _pkg_root()
    job = load_cruncher_showcase_job(root / "docs" / "examples" / "cruncher_job.yaml", caller_root=root)
    docs_data = root / "docs" / "examples" / "data"

    assert _is_under(job.input.path, docs_data)
    if job.input.adapter.kind == "cruncher_best_window":
        cols = job.input.adapter.columns
        assert _is_under(Path(str(cols["hits_path"])), docs_data)
        assert _is_under(Path(str(cols["config_path"])), docs_data)


def test_curated_workspace_demos_run_in_isolated_copy(tmp_path: Path) -> None:
    root = _pkg_root()
    copied_root = tmp_path / "workspaces"
    copied_root.mkdir(parents=True, exist_ok=True)

    expected_fmt = {
        "demo_densegen_render": ".png",
        "demo_cruncher_render": ".png",
    }

    for name, suffix in expected_fmt.items():
        src_ws = root / "workspaces" / name
        dst_ws = copied_root / name
        shutil.copytree(src_ws, dst_ws)

        report = run_cruncher_showcase_job(dst_ws / "job.yaml", caller_root=tmp_path)
        images_dir = Path(report.outputs["images_dir"])
        assert images_dir.exists()
        assert any(p.suffix.lower() == suffix for p in images_dir.iterdir())
