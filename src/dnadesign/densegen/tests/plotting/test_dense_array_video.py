"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_dense_array_video.py

Dense-array video showcase plotting tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.viz import plotting as plotting_module
from dnadesign.densegen.src.viz.plotting import run_plots_from_config


def _write_video_config(
    path: Path,
    *,
    video_yaml: str | None = None,
    default_plots_yaml: str = "[]",
) -> None:
    video_block = (
        textwrap.dedent(video_yaml).strip()
        if video_yaml is not None
        else textwrap.dedent(
            """
            enabled: true
            mode: all_plans_round_robin_single_video
            sampling:
              stride: 1
              max_source_rows: 100
              max_snapshots: 20
            playback:
              target_duration_sec: 5
              fps: 8
            """
        ).strip()
    )
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: plan_a
                    sequences: 2
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
                  - name: plan_b
                    sequences: 2
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              format: png
              default: PLACEHOLDER_DEFAULT
              options: {}
              video:
            PLACEHOLDER_VIDEO
            """
        )
        .strip()
        .replace("PLACEHOLDER_DEFAULT", default_plots_yaml)
        .replace("PLACEHOLDER_VIDEO", textwrap.indent(video_block, "    "))
        + "\n"
    )


def _annotations(regulator: str, seq: str, offset: int) -> list[dict]:
    return [
        {
            "regulator": regulator,
            "sequence": seq,
            "orientation": "fwd",
            "offset": int(offset),
        }
    ]


def _base_records_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["rec_b_1", "rec_a_1", "rec_b_2", "rec_a_2"],
            "sequence": ["ACGTACGTAA", "TTTTACGTAA", "ACGTCCCCAA", "TTTTGGGGAA"],
            "densegen__plan": ["plan_b", "plan_a", "plan_b", "plan_a"],
            "densegen__used_tfbs_detail": [
                _annotations("tfB", "ACGT", 0),
                _annotations("tfA", "TTTT", 0),
                _annotations("tfB", "CCCC", 4),
                _annotations("tfA", "GGGG", 4),
            ],
        }
    )


def _patch_records_loader(monkeypatch, records_df: pd.DataFrame, records_path: Path) -> None:
    def _fake_load_records_from_config(*args, **kwargs):
        del args, kwargs
        return records_df.copy(), f"parquet:{records_path}"

    monkeypatch.setattr(plotting_module, "load_records_from_config", _fake_load_records_from_config)


def _patch_video_job_capture(monkeypatch, captured: dict[str, object]) -> None:
    def _fake_run_job(job_mapping: dict[str, object], *, kind: str, caller_root: Path):
        captured["kind"] = kind
        captured["caller_root"] = caller_root
        captured["job_mapping"] = job_mapping
        captured["adapter_columns"] = dict(job_mapping["input"]["adapter"]["columns"])
        selection_path = Path(str(job_mapping["selection"]["path"]))
        with selection_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        captured["selection_ids"] = [str(row["id"]) for row in rows]
        captured["records_df"] = pd.read_parquet(Path(str(job_mapping["input"]["path"])))
        out_path = Path(str(job_mapping["outputs"][0]["path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr("dnadesign.densegen.src.viz.dense_array_video.run_job", _fake_run_job)


def test_dense_array_video_round_robin_single_output(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    _patch_records_loader(monkeypatch, _base_records_df(), records_path)

    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")

    assert captured["kind"] == "sequence_rows_v3"
    assert captured["selection_ids"] == ["rec_a_1", "rec_b_1", "rec_a_2", "rec_b_2"]

    video_path = run_root / "outputs" / "plots" / "stage_b" / "all_plans" / "showcase.mp4"
    assert video_path.exists()

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    entries = [item for item in payload.get("plots", []) if item.get("name") == "dense_array_video_showcase"]
    assert len(entries) == 1
    assert entries[0]["path"] == "stage_b/all_plans/showcase.mp4"


def test_dense_array_video_runs_when_enabled_without_only(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path, default_plots_yaml='["dense_array_video_showcase"]')
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    _patch_records_loader(monkeypatch, _base_records_df(), records_path)
    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    assert captured["selection_ids"] == ["rec_a_1", "rec_b_1", "rec_a_2", "rec_b_2"]
    assert (run_root / "outputs" / "plots" / "stage_b" / "all_plans" / "showcase.mp4").exists()


def test_dense_array_video_single_plan_mode_filters_rows(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(
        cfg_path,
        video_yaml="""
enabled: true
mode: single_plan_single_video
single_plan_name: plan_b
sampling:
  stride: 1
  max_source_rows: 100
  max_snapshots: 20
playback:
  target_duration_sec: 5
  fps: 8
""",
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    _patch_records_loader(monkeypatch, _base_records_df(), records_path)
    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")

    assert captured["selection_ids"] == ["rec_b_1", "rec_b_2"]
    assert (run_root / "outputs" / "plots" / "stage_b" / "plan_b" / "showcase.mp4").exists()


def test_dense_array_video_caps_snapshots_by_frame_budget(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(
        cfg_path,
        video_yaml="""
enabled: true
mode: all_plans_round_robin_single_video
sampling:
  stride: 1
  max_source_rows: 1000
  max_snapshots: 500
playback:
  target_duration_sec: 3
  fps: 8
""",
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    rows: list[dict[str, object]] = []
    for idx in range(120):
        plan = "plan_a" if idx % 2 == 0 else "plan_b"
        rows.append(
            {
                "id": f"rec_{idx:03d}",
                "sequence": "ACGTACGTAA",
                "densegen__plan": plan,
                "densegen__used_tfbs_detail": _annotations("tfX", "ACGT", 0),
            }
        )
    records_df = pd.DataFrame(rows)
    _patch_records_loader(monkeypatch, records_df, records_path)
    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")

    assert len(captured["selection_ids"]) <= 24


def test_dense_array_video_uses_notebook_style_overlay_titles(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    records_df = _base_records_df().copy()
    records_df.loc[1, "id"] = "abcdefghijklmnopqrstu"
    _patch_records_loader(monkeypatch, records_df, records_path)

    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")

    assert captured["adapter_columns"]["overlay_text"] == "densegen__video_overlay_text"
    assert captured["job_mapping"]["render"]["style"]["overrides"]["overlay_align"] == "center"
    assert captured["job_mapping"]["render"]["style"]["overrides"]["font_size_label"] == 15
    rendered_df = captured["records_df"]
    assert rendered_df["densegen__video_overlay_text"].tolist() == [
        "TFBS arrangement snapshot for sequence abcdefgh...rstu",
        "TFBS arrangement snapshot for sequence rec_b_1",
        "TFBS arrangement snapshot for sequence rec_a_2",
        "TFBS arrangement snapshot for sequence rec_b_2",
    ]


def test_dense_array_video_rejects_duplicate_ids(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    bad_df = _base_records_df().copy()
    bad_df.loc[1, "id"] = "rec_b_1"
    _patch_records_loader(monkeypatch, bad_df, records_path)
    _patch_video_job_capture(monkeypatch, {})

    loaded = load_config(cfg_path)
    try:
        run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")
    except RuntimeError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate-id validation failure for dense_array_video_showcase.")


def test_dense_array_video_single_plan_mode_requires_rows(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(
        cfg_path,
        video_yaml="""
enabled: true
mode: single_plan_single_video
single_plan_name: missing_plan
sampling:
  stride: 1
  max_source_rows: 100
  max_snapshots: 20
playback:
  target_duration_sec: 5
  fps: 8
""",
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    _patch_records_loader(monkeypatch, _base_records_df(), records_path)
    _patch_video_job_capture(monkeypatch, {})

    loaded = load_config(cfg_path)
    try:
        run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")
    except RuntimeError as exc:
        assert "selected no rows" in str(exc).lower()
    else:
        raise AssertionError("Expected single-plan empty-selection failure for dense_array_video_showcase.")


def test_dense_array_video_rejects_null_required_values(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    bad_df = _base_records_df().copy()
    bad_df.loc[0, "densegen__plan"] = None
    _patch_records_loader(monkeypatch, bad_df, records_path)
    _patch_video_job_capture(monkeypatch, {})

    loaded = load_config(cfg_path)
    try:
        run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")
    except RuntimeError as exc:
        assert "null" in str(exc).lower()
    else:
        raise AssertionError("Expected null-value validation failure for dense_array_video_showcase.")


def test_dense_array_video_rejects_invalid_annotation_string(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    bad_df = _base_records_df().copy()
    bad_df.loc[0, "densegen__used_tfbs_detail"] = "not-json"
    _patch_records_loader(monkeypatch, bad_df, records_path)
    _patch_video_job_capture(monkeypatch, {})

    loaded = load_config(cfg_path)
    try:
        run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")
    except RuntimeError as exc:
        assert "json" in str(exc).lower()
    else:
        raise AssertionError("Expected annotation JSON validation failure for dense_array_video_showcase.")


def test_dense_array_video_single_plan_output_path_is_stage_b_scoped(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_video_config(
        cfg_path,
        video_yaml="""
enabled: true
mode: single_plan_single_video
single_plan_name: ..
sampling:
  stride: 1
  max_source_rows: 100
  max_snapshots: 20
playback:
  target_duration_sec: 5
  fps: 8
""",
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"placeholder")

    records_df = _base_records_df().copy()
    records_df["densegen__plan"] = ".."
    _patch_records_loader(monkeypatch, records_df, records_path)
    captured: dict[str, object] = {}
    _patch_video_job_capture(monkeypatch, captured)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path, only="dense_array_video_showcase")

    out_path = Path(str(captured["job_mapping"]["outputs"][0]["path"]))
    expected_prefix = run_root / "outputs" / "plots" / "stage_b"
    assert out_path.resolve().is_relative_to(expected_prefix.resolve())
