from __future__ import annotations

from pathlib import Path

import pandas as pd

SEQ60 = "A" * 60


def write_tfbs_stage_b_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    candidate_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "densegen.parquet"
    pd.DataFrame({"id": [f"id-{idx}" for idx in range(6)], "sequence": [SEQ60] * 6}).to_parquet(
        candidate_path,
        index=False,
    )
    pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(6)] + ["sidecar-only"],
            "densegen__used_tfbs_detail": [
                _detail("LexA", "BaeR", "background"),
                _detail("BaeR", "LexA", "background"),
                _detail("background", "LexA", "BaeR"),
                _detail("LexA", "background", "BaeR"),
                _detail("BaeR", "background", "LexA"),
                _detail("background", "BaeR", "LexA"),
                _detail("CpxR", "BaeR", "background"),
            ],
        }
    ).to_parquet(sidecar_path, index=False)
    return candidate_path, sidecar_path


def write_tfbs_count_fixed_stage_b_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    candidate_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "densegen.parquet"
    ids = [f"id-{idx}" for idx in range(9)]
    pd.DataFrame({"id": ids, "sequence": [SEQ60] * len(ids)}).to_parquet(candidate_path, index=False)
    pd.DataFrame(
        {
            "id": [*ids, "sidecar-only"],
            "densegen__used_tfbs_detail": [
                _detail("LexA", "BaeR", "background"),
                _detail("BaeR", "LexA", "background"),
                _detail("background", "LexA", "BaeR"),
                _detail("LexA", "background", "BaeR"),
                _detail("BaeR", "background", "LexA"),
                _detail("background", "BaeR", "LexA"),
                _detail("LexA", "LexA", "background"),
                _detail("background", "BaeR", "CpxR"),
                _detail("background", "background", "background"),
                _detail("CpxR", "BaeR", "background"),
            ],
        }
    ).to_parquet(sidecar_path, index=False)
    return candidate_path, sidecar_path


def _detail(slot0: str, slot1: str, slot2: str) -> list[dict[str, object]]:
    return [
        _tfbs(slot0, 10),
        _tfbs(slot1, 21),
        _tfbs(slot2, 32),
        _fixed("upstream_sigma70_core", 0, variant_id="f"),
        _fixed("downstream_sigma70_core", 22, sequence="TATAAT"),
    ]


def _tfbs(regulator: str, offset_raw: int) -> dict[str, object]:
    return {
        "part_kind": "tfbs",
        "regulator": regulator,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
    }


def _fixed(
    role: str,
    offset_raw: int,
    *,
    variant_id: str | None = None,
    sequence: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "part_kind": "fixed_element",
        "role": role,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
        "spacer_length": 16,
    }
    if variant_id is not None:
        row["variant_id"] = variant_id
    if sequence is not None:
        row["sequence"] = sequence
    return row
