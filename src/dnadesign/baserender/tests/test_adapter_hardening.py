"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_adapter_hardening.py

Adapter hardening tests for strict error typing and schema validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.baserender.src.adapters import build_adapter
from dnadesign.baserender.src.adapters.cruncher_best_window import CruncherBestWindowAdapter
from dnadesign.baserender.src.config import AdapterCfg
from dnadesign.baserender.src.core import SchemaError


def test_build_adapter_unknown_kind_raises_schema_error() -> None:
    cfg = AdapterCfg(kind="unknown_kind", columns={}, policies={})
    with pytest.raises(SchemaError, match="Unsupported adapter kind"):
        build_adapter(cfg, alphabet="DNA")


def test_cruncher_adapter_requires_hits_columns_even_when_file_is_empty(tmp_path: Path) -> None:
    hits_path = tmp_path / "hits.parquet"
    hits_table = pa.Table.from_arrays([pa.array([], type=pa.string())], names=["elite_id"])
    pq.write_table(hits_table, hits_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cruncher": {
                    "pwms_info": {
                        "lexA": {"pwm_matrix": [[0.25, 0.25, 0.25, 0.25]]},
                    }
                }
            }
        )
    )

    with pytest.raises(SchemaError, match="hits parquet missing required columns"):
        CruncherBestWindowAdapter.from_config(
            columns={
                "hits_path": str(hits_path),
                "config_path": str(config_path),
            },
            policies={},
            alphabet="DNA",
        )


def test_cruncher_adapter_applies_sampling_window_to_pwm_matrix(tmp_path: Path) -> None:
    hits_path = tmp_path / "hits.parquet"
    hits_table = pa.Table.from_pylist(
        [
            {
                "elite_id": "e1",
                "tf": "cpxR",
                "best_start": 2,
                "best_strand": "+",
                "best_window_seq": "A" * 16,
                "best_core_seq": "A" * 8,
            }
        ]
    )
    pq.write_table(hits_table, hits_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cruncher": {
                    "sample": {
                        "sequence_length": 16,
                        "motif_width": {
                            "maxw": 16,
                            "strategy": "max_info",
                        },
                    },
                    "pwms_info": {
                        "cpxR": {"pwm_matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(21)]},
                    },
                }
            }
        )
    )

    adapter = CruncherBestWindowAdapter.from_config(
        columns={
            "sequence": "sequence",
            "id": "id",
            "hits_path": str(hits_path),
            "config_path": str(config_path),
        },
        policies={},
        alphabet="DNA",
    )
    record = adapter.apply({"id": "e1", "sequence": "A" * 40}, row_index=0)

    assert len(record.effects) == 1
    matrix = record.effects[0].params.get("matrix")
    assert isinstance(matrix, list)
    assert len(matrix) == 16
