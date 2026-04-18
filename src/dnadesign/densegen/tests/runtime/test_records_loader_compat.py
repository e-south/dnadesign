"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_records_loader_compat.py

Compatibility tests for loading legacy DenseGen records artifacts.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.densegen.src.adapters.outputs.loader import load_records_from_config
from dnadesign.densegen.src.adapters.outputs.parquet import _build_schema
from dnadesign.densegen.src.config import load_config


def _write_minimal_config(path: Path) -> None:
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
                    sequences: 1
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
              default: []
              options: {}
            """
        ).strip()
        + "\n"
    )


def test_load_records_from_config_accepts_legacy_used_tfbs_detail_schema(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_minimal_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    expected_schema = _build_schema("densegen", pa)
    legacy_used_tfbs_detail_type = pa.list_(
        pa.struct(
            [
                pa.field("part_kind", pa.string()),
                pa.field("role", pa.string()),
                pa.field("constraint_name", pa.string()),
                pa.field("sequence", pa.string()),
                pa.field("variant_id", pa.string()),
                pa.field("spacer_length", pa.int64()),
                pa.field("placement_index", pa.int64()),
                pa.field("tf", pa.string()),
                pa.field("tfbs", pa.string()),
                pa.field("motif_id", pa.string()),
                pa.field("tfbs_id", pa.string()),
                pa.field("orientation", pa.string()),
                pa.field("offset", pa.int64()),
                pa.field("offset_raw", pa.int64()),
                pa.field("length", pa.int64()),
                pa.field("end", pa.int64()),
                pa.field("pad_left", pa.int64()),
                pa.field("site_id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("stage_a_best_hit_score", pa.float64()),
                pa.field("stage_a_rank_within_regulator", pa.int64()),
                pa.field("stage_a_tier", pa.int64()),
                pa.field("stage_a_fimo_start", pa.int64()),
                pa.field("stage_a_fimo_stop", pa.int64()),
                pa.field("stage_a_fimo_strand", pa.string()),
                pa.field("stage_a_selection_rank", pa.int64()),
                pa.field("stage_a_selection_score_norm", pa.float64()),
                pa.field("stage_a_tfbs_core", pa.string()),
            ]
        )
    )
    legacy_schema = pa.schema(
        [
            pa.field(field.name, legacy_used_tfbs_detail_type) if field.name == "densegen__used_tfbs_detail" else field
            for field in expected_schema
        ]
    )
    rows = [
        {
            "id": "row1",
            "sequence": "ACGTACGTAA",
            "bio_type": "dna",
            "alphabet": "dna_4",
            "source": "demo",
            "densegen__schema_version": "2.9",
            "densegen__created_at": "2026-04-13T00:00:00Z",
            "densegen__run_id": "demo",
            "densegen__length": 10,
            "densegen__plan": "plan_a",
            "densegen__input_name": "demo_input",
            "densegen__input_mode": "binding_sites",
            "densegen__input_pwm_ids": [],
            "densegen__used_tfbs": ["ACGT"],
            "densegen__used_tfbs_detail": [
                {
                    "part_kind": "tfbs",
                    "sequence": "ACGT",
                    "tf": "lexA",
                    "tfbs": "ACGT",
                    "orientation": "fwd",
                    "offset": 0,
                    "offset_raw": 0,
                    "length": 4,
                    "end": 4,
                    "pad_left": 0,
                    "source": "demo_input",
                    "stage_a_tfbs_core": "ACGT",
                    "stage_a_selection_rank": 3,
                }
            ],
            "densegen__used_tf_counts": [{"tf": "lexA", "count": 1}],
            "densegen__library_unique_tf_count": 1,
            "densegen__library_unique_tfbs_count": 1,
            "densegen__covers_all_tfs_in_solution": True,
            "densegen__required_regulators": ["lexA"],
            "densegen__min_count_by_regulator": [{"tf": "lexA", "min_count": 1}],
            "densegen__compression_ratio": 1.0,
            "densegen__sampling_library_hash": "hash1",
            "densegen__sampling_library_index": 0,
            "densegen__pad_used": False,
            "densegen__pad_bases": 0,
            "densegen__pad_end": None,
            "densegen__pad_literal": None,
            "densegen__sequence_validation": {"validation_passed": True, "violations": []},
            "densegen__gc_total": 0.5,
            "densegen__gc_core": 0.5,
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=legacy_schema), records_path)

    loaded = load_config(cfg_path)
    records_df, source_label = load_records_from_config(
        loaded.root,
        cfg_path,
        columns=["id", "densegen__used_tfbs_detail"],
    )

    assert source_label == f"parquet:{records_path.resolve()}"
    detail = list(records_df.iloc[0]["densegen__used_tfbs_detail"])
    assert detail[0]["regulator"] == "lexA"
    assert detail[0]["sequence"] == "ACGT"
    assert detail[0]["core_sequence"] == "ACGT"
    assert detail[0]["rank_among_selected"] == 3
    assert detail[0]["part_index"] == 0


def test_load_records_from_config_recovers_missing_plan_and_input_from_source(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_minimal_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    records_path = run_root / "outputs" / "tables" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _build_schema("densegen", pa)
    rows = [
        {
            "id": "row1",
            "sequence": "ACGTACGTAA",
            "bio_type": "dna",
            "alphabet": "dna_4",
            "source": "plan_pool__ethanol_ciprofloxacin__sig35_f",
            "densegen__schema_version": "2.9",
            "densegen__created_at": "2026-04-13T00:00:00Z",
            "densegen__run_id": "demo",
            "densegen__length": 10,
            "densegen__plan": None,
            "densegen__input_name": None,
            "densegen__used_tfbs_detail": None,
            "densegen__used_tf_counts": [],
            "densegen__library_unique_tf_count": 0,
            "densegen__library_unique_tfbs_count": 0,
            "densegen__covers_all_tfs_in_solution": True,
            "densegen__required_regulators": [],
            "densegen__min_count_by_regulator": [],
            "densegen__compression_ratio": None,
            "densegen__sampling_library_hash": None,
            "densegen__sampling_library_index": None,
            "densegen__pad_used": False,
            "densegen__pad_bases": 0,
            "densegen__pad_end": None,
            "densegen__pad_literal": None,
            "densegen__sequence_validation": {"validation_passed": True, "violations": []},
            "densegen__gc_total": 0.5,
            "densegen__gc_core": 0.5,
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), records_path)

    loaded = load_config(cfg_path)
    records_df, source_label = load_records_from_config(
        loaded.root,
        cfg_path,
        columns=["id", "densegen__plan", "densegen__input_name"],
    )

    assert source_label == f"parquet:{records_path.resolve()}"
    assert "source" not in records_df.columns
    assert records_df.loc[0, "densegen__plan"] == "ethanol_ciprofloxacin__sig35=f"
    assert records_df.loc[0, "densegen__input_name"] == "plan_pool__ethanol_ciprofloxacin__sig35_f"
    assert bool(records_df.loc[0, "densegen__metadata_inferred_from_source"]) is True
