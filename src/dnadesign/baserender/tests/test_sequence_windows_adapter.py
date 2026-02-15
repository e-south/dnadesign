"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sequence_windows_adapter.py

Tests for the sequence_windows_v1 adapter and schema wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.baserender.src.adapters.sequence_windows_v1 import SequenceWindowsV1Adapter
from dnadesign.baserender.src.config import load_cruncher_showcase_job

from .conftest import write_job, write_parquet


def _matrix(length: int) -> list[list[float]]:
    return [[0.7, 0.1, 0.1, 0.1] for _ in range(length)]


def test_sequence_windows_adapter_maps_windows_and_motifs_into_record() -> None:
    adapter = SequenceWindowsV1Adapter(
        columns={
            "id": "id",
            "sequence": "sequence",
            "regulator_windows": "regulator_windows",
            "motifs": "motifs",
            "display": "display",
        },
        policies={},
        alphabet="DNA",
    )
    row = {
        "id": "elite_showcase_1",
        "sequence": {"sense_5to3": "CTGCATATATTTACAG", "alphabet": "DNA"},
        "regulator_windows": [
            {
                "window_id": "lexA.best_hit",
                "tf": "lexA",
                "span": {"start": 0, "end": 15},
                "strand": "fwd",
                "window_seq_5to3": "CTGCATATATTTACA",
                "score": 12.3,
            },
            {
                "window_id": "cpxR.best_hit",
                "tf": "cpxR",
                "span": {"start": 3, "end": 14},
                "strand": "rev",
                "window_seq_5to3": "CGTATATAAAT",
                "score": 10.0,
            },
        ],
        "motifs": [
            {"tf": "lexA", "matrix": _matrix(15), "motif_ref": {"source": "motif_library", "motif_id": "lexA_1"}},
            {"tf": "cpxR", "matrix": _matrix(11), "motif_ref": {"source": "motif_library", "motif_id": "cpxR_1"}},
        ],
        "display": {"title": "id=elite_showcase_1 rank=1"},
    }

    record = adapter.apply(row, row_index=0)
    assert record.id == "elite_showcase_1"
    assert record.sequence == "CTGCATATATTTACAG"
    assert len(record.features) == 2
    assert all(feature.kind == "regulator_window" for feature in record.features)
    assert {effect.kind for effect in record.effects} == {"motif_logo"}
    assert record.display.overlay_text == "id=elite_showcase_1 rank=1"
    assert record.display.tag_labels["tf:lexA"] == "lexA"
    assert record.display.tag_labels["tf:cpxR"] == "cpxR"


def test_sequence_windows_adapter_kind_is_accepted_by_job_schema(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "elite_showcase_1",
                "sequence": {"sense_5to3": "CTGCATATATTTACAG", "alphabet": "DNA"},
                "regulator_windows": [],
                "motifs": [],
            }
        ],
    )
    payload = {
        "version": 3,
        "results_root": str(tmp_path / "outputs"),
        "input": {
            "kind": "parquet",
            "path": str(parquet),
            "adapter": {
                "kind": "sequence_windows_v1",
                "columns": {
                    "id": "id",
                    "sequence": "sequence",
                    "regulator_windows": "regulator_windows",
                    "motifs": "motifs",
                },
                "policies": {},
            },
            "alphabet": "DNA",
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)
    job = load_cruncher_showcase_job(job_path)
    assert job.input.adapter.kind == "sequence_windows_v1"
