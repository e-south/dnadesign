"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_workspace_demos.py

Tests for curated baserender workspace demos and their self-contained IO assets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from dnadesign.baserender import cruncher_showcase_style_overrides
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
            assert raw["render"]["style"]["overrides"] == cruncher_showcase_style_overrides()
            cols = job.input.adapter.columns
            assert job.input.adapter.kind == "generic_features"
            assert cols["features"] == "features"
            assert cols["effects"] == "effects"
            assert cols["display"] == "display"
            plugins = [spec.name for spec in job.pipeline.plugins]
            assert plugins == []
            assert not (ws / "inputs" / "motif_library.json").exists()
            import pyarrow.parquet as pq

            rows = pq.read_table(ws / "inputs" / "elites_showcase_records.parquet").to_pylist()
            assert len(rows) == 2
            for row in rows:
                assert {"id", "sequence", "features", "effects", "display"} <= set(row.keys())
                assert isinstance(row["features"], list)
                assert isinstance(row["effects"], list)
                assert isinstance(row["display"], dict)
                assert str(row["display"].get("overlay_text", "")).startswith("Elite #")
                feature_ids = set()
                for feature in row["features"]:
                    feature_ids.add(str(feature["id"]))
                    assert re.match(r".+:best_window:[^:]+:\d+$", str(feature["id"]))
                    assert set((feature.get("attrs") or {}).keys()) == {"tf"}
                    assert int(feature["span"]["end"]) > int(feature["span"]["start"])
                    assert len(str(feature["label"])) == int(feature["span"]["end"]) - int(feature["span"]["start"])
                for effect in row["effects"]:
                    target = (effect.get("target") or {}).get("feature_id")
                    assert target in feature_ids
                    assert effect.get("kind") == "motif_logo"
                    assert "matrix" in (effect.get("params") or {})

        if name == "demo_densegen_render":
            cols = job.input.adapter.columns
            assert job.input.adapter.kind == "densegen_tfbs"
            assert set(cols.keys()) == {"sequence", "annotations", "id"}
            assert "overlay_text" not in cols
            import pyarrow.parquet as pq

            rows = pq.read_table(ws / "inputs" / "input.parquet").to_pylist()
            assert len(rows) == 1
            row = rows[0]
            assert set(row.keys()) == {"id", "sequence", "densegen__used_tfbs_detail"}
            ann = row["densegen__used_tfbs_detail"]
            assert isinstance(ann, list) and ann
            assert any(str(entry.get("orientation")) == "rev" for entry in ann)
            assert any("motif_id" in entry for entry in ann)


def test_docs_cruncher_example_uses_local_examples_data() -> None:
    root = _pkg_root()
    job = load_cruncher_showcase_job(root / "docs" / "examples" / "cruncher_job.yaml", caller_root=root)
    docs_data = root / "docs" / "examples" / "data"

    assert _is_under(job.input.path, docs_data)
    if job.input.adapter.kind == "cruncher_best_window":
        cols = job.input.adapter.columns
        assert _is_under(Path(str(cols["hits_path"])), docs_data)
        assert _is_under(Path(str(cols["config_path"])), docs_data)


def test_docs_densegen_example_matches_notebook_contract() -> None:
    root = _pkg_root()
    docs_data = root / "docs" / "examples" / "data"
    job = load_cruncher_showcase_job(root / "docs" / "examples" / "densegen_job.yaml", caller_root=root)

    cols = job.input.adapter.columns
    assert job.input.adapter.kind == "densegen_tfbs"
    assert set(cols.keys()) == {"sequence", "annotations", "id"}
    assert "overlay_text" not in cols
    assert _is_under(job.input.path, docs_data)

    import pyarrow.parquet as pq

    rows = pq.read_table(docs_data / "densegen_demo.parquet").to_pylist()
    assert len(rows) >= 1
    row = rows[0]
    assert set(row.keys()) == {"id", "sequence", "densegen__used_tfbs_detail"}
    ann = row["densegen__used_tfbs_detail"]
    assert isinstance(ann, list) and ann
    assert any("motif_id" in entry for entry in ann)


def test_curated_workspace_demos_run_in_isolated_copy(tmp_path: Path) -> None:
    root = _pkg_root()
    copied_root = tmp_path / "workspaces"
    copied_root.mkdir(parents=True, exist_ok=True)

    expected_fmt = {
        "demo_densegen_render": ".png",
        "demo_cruncher_render": ".pdf",
    }

    for name, suffix in expected_fmt.items():
        src_ws = root / "workspaces" / name
        dst_ws = copied_root / name
        shutil.copytree(src_ws, dst_ws)

        report = run_cruncher_showcase_job(dst_ws / "job.yaml", caller_root=tmp_path)
        images_dir = Path(report.outputs["images_dir"])
        assert images_dir.exists()
        assert any(p.suffix.lower() == suffix for p in images_dir.iterdir())
