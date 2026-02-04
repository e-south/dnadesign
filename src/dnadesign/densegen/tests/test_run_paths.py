from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.core.run_paths import has_existing_run_outputs, run_meta_root, run_outputs_root


def test_existing_outputs_ignores_pre_run_dirs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outputs_root = run_outputs_root(run_root)
    (outputs_root / "logs").mkdir(parents=True)
    (outputs_root / "meta").mkdir(parents=True)
    (outputs_root / "pools").mkdir(parents=True)
    (outputs_root / "libraries").mkdir(parents=True)
    (outputs_root / "tables").mkdir(parents=True)
    (outputs_root / "plots").mkdir(parents=True)
    (outputs_root / "report").mkdir(parents=True)
    assert not has_existing_run_outputs(run_root)

    (outputs_root / "pools" / "pool_manifest.json").write_text("{}")
    (outputs_root / "libraries" / "library_builds.parquet").write_text("stub")
    assert not has_existing_run_outputs(run_root)


def test_existing_outputs_detects_root_files(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outputs_root = run_outputs_root(run_root)
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True)
    (tables_root / "dense_arrays.parquet").write_text("stub")
    assert has_existing_run_outputs(run_root)


def test_existing_outputs_detects_meta_run_state(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    meta_root = run_meta_root(run_root)
    meta_root.mkdir(parents=True)
    (meta_root / "run_state.json").write_text("{}")
    assert has_existing_run_outputs(run_root)


def test_existing_outputs_ignores_events_log(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    meta_root = run_meta_root(run_root)
    meta_root.mkdir(parents=True)
    (meta_root / "events.jsonl").write_text("{}")
    assert not has_existing_run_outputs(run_root)
