"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/reader_sources.py

Loads Reader SFXI vec8 records for stress OPAL batch0 staging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import READER_EVIDENCE_PLOT_RECORD_IDS, READER_VEC8_RECORD_ID, READER_VEC8_REQUIRED_COLUMNS
from .contracts import MeasuredReaderVec8Error, ReaderVec8Source


def load_reader_vec8_records(
    reader_root: Path,
    *,
    year: int = 2026,
) -> tuple[pd.DataFrame, tuple[ReaderVec8Source, ...]]:
    """Load latest reader `sfxi_vec8/vec8` rows from SFXI experiment outputs."""

    root = Path(reader_root).expanduser().resolve()
    experiment_root = root / "experiments" / str(year)
    if not experiment_root.exists():
        raise MeasuredReaderVec8Error(f"Reader experiment year directory not found: {experiment_root}")

    frames: list[pd.DataFrame] = []
    sources: list[ReaderVec8Source] = []
    for config_path in sorted(experiment_root.glob("*sfxi*/config.yaml")):
        records_path = config_path.parent / "outputs" / "manifests" / "records.json"
        latest_records = _latest_records(records_path)
        output_record = latest_records.get(READER_VEC8_RECORD_ID)
        if output_record is None:
            continue
        if not isinstance(output_record, dict) or not output_record.get("path"):
            raise MeasuredReaderVec8Error(
                f"Reader latest {READER_VEC8_RECORD_ID!r} record is malformed: {records_path}"
            )
        table_path = _resolve_record_path(config_path.parent, output_record)
        frame = _load_vec8_table(table_path, experiment_id=config_path.parent.name)
        frame.insert(0, "reader_vec8_table_path", str(table_path))
        frame.insert(0, "reader_config_path", str(config_path))
        frame.insert(0, "reader_experiment_id", config_path.parent.name)
        frame.insert(0, "reader_source_row_index", range(len(frame)))
        frames.append(frame)
        sources.append(
            ReaderVec8Source(
                experiment_id=config_path.parent.name,
                config_path=config_path,
                table_path=table_path,
                row_count=len(frame),
                records_path=records_path,
                plot_files_by_record_id=_plot_files_by_record_id(latest_records),
            )
        )

    if not frames:
        raise MeasuredReaderVec8Error(f"No reader {READER_VEC8_RECORD_ID!r} rows found under {experiment_root}.")
    return pd.concat(frames, ignore_index=True), tuple(sources)


def _latest_records(records_json: Path) -> dict:
    if not records_json.exists():
        return {}
    try:
        payload = json.loads(records_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MeasuredReaderVec8Error(f"Could not parse reader records manifest: {records_json}") from exc
    latest = payload.get("latest")
    if not isinstance(latest, dict):
        raise MeasuredReaderVec8Error(f"Reader records manifest missing latest object: {records_json}")
    return latest


def _plot_files_by_record_id(latest_records: dict) -> dict[str, tuple[str, ...]]:
    files_by_record_id: dict[str, tuple[str, ...]] = {}
    for record_id in READER_EVIDENCE_PLOT_RECORD_IDS:
        record = latest_records.get(record_id)
        if not isinstance(record, dict):
            continue
        raw_files = record.get("files")
        if not isinstance(raw_files, list):
            continue
        files_by_record_id[record_id] = tuple(str(path) for path in raw_files if str(path).strip())
    return files_by_record_id


def _resolve_record_path(experiment_dir: Path, record: dict) -> Path:
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else experiment_dir / "outputs" / raw
    if not path.exists():
        raise MeasuredReaderVec8Error(f"Reader vec8 table not found: {path}")
    return path.resolve()


def _load_vec8_table(table_path: Path, *, experiment_id: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(table_path)
    except Exception as exc:
        raise MeasuredReaderVec8Error(f"Could not read reader vec8 table for {experiment_id}: {table_path}") from exc
    missing = [column for column in READER_VEC8_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MeasuredReaderVec8Error(f"Reader vec8 table {table_path} missing required columns: {missing}")
    return frame.reset_index(drop=True)
