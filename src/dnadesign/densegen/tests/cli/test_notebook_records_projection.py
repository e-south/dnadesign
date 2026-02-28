"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_notebook_records_projection.py

Tests for notebook records projection and column curation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.densegen.src.cli.notebook_records_projection import (
    build_records_preview_table,
)


def test_records_preview_projection_curates_columns_with_global_parts_detail() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "seq-1",
                "sequence": "TTTACGTACGTACGTACGTACGTAC",
                "source": "densegen",
                "densegen__run_id": "demo-run",
                "densegen__run_config_path": "config.yaml",
                "densegen__plan": "baseline_sigma70",
                "densegen__input_name": "basic_sites",
                "densegen__length": 26,
                "densegen__compression_ratio": 0.41,
                "densegen__gc_total": 0.46,
                "densegen__gc_core": 0.44,
                "densegen__sampling_library_hash": "lib-hash-1",
                "densegen__sampling_library_index": 2,
                "densegen__used_tf_counts": [{"tf": "TF_A", "count": 1}],
                "densegen__required_regulators": ["TF_A"],
                "densegen__pad_used": True,
                "densegen__pad_bases": 3,
                "densegen__pad_end": "5prime",
                "densegen__used_tfbs_detail": [
                    {
                        "part_kind": "tfbs",
                        "part_index": 0,
                        "regulator": "TF_A",
                        "sequence": "ACGT",
                        "core_sequence": "ACGT",
                        "orientation": "fwd",
                        "offset": 3,
                        "offset_raw": 0,
                        "pad_left": 3,
                        "length": 4,
                        "end": 7,
                        "source": "demo",
                        "motif_id": "motif_1",
                        "tfbs_id": "tfbs_1",
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "upstream",
                        "constraint_name": "sigma70_consensus",
                        "sequence": "TTGACA",
                        "offset": 10,
                        "length": 6,
                    },
                    {
                        "part_kind": "fixed_element",
                        "role": "downstream",
                        "constraint_name": "sigma70_consensus",
                        "sequence": "TATAAT",
                        "offset": 33,
                        "length": 6,
                    },
                ],
                "densegen__pad_literal": "TTT",
                "densegen__random_seed": 42,
                "densegen__sampling_pool_strategy": "subsample",
            }
        ]
    )

    projected = build_records_preview_table(df)

    assert "densegen__run_config_path" not in projected.columns
    assert "densegen__used_tfbs_detail" not in projected.columns
    assert "densegen__parts_detail" in projected.columns
    assert "densegen__pad_literal" in projected.columns

    row = projected.iloc[0]
    assert row["densegen__pad_literal"] == "TTT"

    parts_detail = row["densegen__parts_detail"]
    assert isinstance(parts_detail, list)
    assert any(entry.get("part_kind") == "tfbs" for entry in parts_detail)
    assert any(entry.get("part_kind") == "fixed_element" and entry.get("role") == "upstream" for entry in parts_detail)
    assert any(
        entry.get("part_kind") == "fixed_element" and entry.get("role") == "downstream" for entry in parts_detail
    )


def test_records_preview_projection_keeps_declared_pad_literal() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "seq-2",
                "sequence": "AACCGG",
                "densegen__plan": "baseline",
                "densegen__input_name": "basic_sites",
                "densegen__pad_used": True,
                "densegen__pad_bases": 2,
                "densegen__pad_end": "3prime",
                "densegen__pad_literal": "GG",
                "densegen__used_tfbs_detail": [],
            }
        ]
    )

    projected = build_records_preview_table(df)
    assert projected.loc[0, "densegen__pad_literal"] == "GG"
