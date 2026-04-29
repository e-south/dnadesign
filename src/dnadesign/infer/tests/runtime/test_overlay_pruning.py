"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_overlay_pruning.py

Stale row-overlay pruning tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.infer.src.contracts import infer_usr_column_name
from dnadesign.infer.src.features.overlay_pruning import prune_stale_infer_overlay_columns
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces


def _dataset_with_sequence(tmp_path: Path) -> tuple[Dataset, str]:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "anchor_views")
    dataset.init(source="test", notes="overlay pruning test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    return dataset, add_result.ids[0]


def _write_mixed_stale_infer_overlay(dataset: Dataset, sequence_id: str) -> tuple[str, str, str]:
    stale_payload = infer_usr_column_name(
        model_id="evo2_20b",
        job_id="anchor_only_20b_features",
        out_id="intermediate_embedding__block23_mlp_out__seq_mean",
    )
    stale_metadata = infer_usr_column_name(
        model_id="evo2_20b",
        job_id="anchor_only_20b_features",
        out_id="metadata__feature_request_digest",
    )
    retained_payload = infer_usr_column_name(
        model_id="evo2_7b",
        job_id="anchor_only_7b_features",
        out_id="output_layer_mean__seq_mean",
    )
    register_test_namespace(
        dataset.root,
        namespace="infer",
        columns_spec=",".join(
            [
                f"{stale_payload}:list<float64>",
                f"{stale_metadata}:string",
                f"{retained_payload}:list<float64>",
            ]
        ),
        overwrite=True,
    )
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": [sequence_id],
                stale_payload: [[20.0, 21.0]],
                stale_metadata: ["old-20b-digest"],
                retained_payload: [[7.0, 8.0]],
            }
        ),
        key="id",
        actor={"tool": "test", "run_id": "stale-infer-fixture"},
    )
    return stale_payload, stale_metadata, retained_payload


def test_prune_stale_infer_overlay_columns_removes_explicit_prefix_only(tmp_path: Path) -> None:
    dataset, sequence_id = _dataset_with_sequence(tmp_path)
    stale_payload, stale_metadata, retained_payload = _write_mixed_stale_infer_overlay(dataset, sequence_id)

    dry_run = prune_stale_infer_overlay_columns(
        dataset_root=dataset.root,
        dataset_id=dataset.name,
        column_prefixes=("infer__evo2_20b__",),
        reason="collapsed 20B lane no longer supported for this study",
        write=False,
    )

    assert dry_run.parts_scanned == 1
    assert dry_run.parts_with_columns == 1
    assert stale_payload in dry_run.removed_columns
    assert stale_metadata in dry_run.removed_columns
    assert retained_payload not in dry_run.removed_columns
    before_schema = pq.ParquetFile(sorted((dataset.dir / "_derived" / "infer").glob("part-*.parquet"))[0]).schema_arrow
    assert stale_payload in before_schema.names

    written = prune_stale_infer_overlay_columns(
        dataset_root=dataset.root,
        dataset_id=dataset.name,
        column_prefixes=("infer__evo2_20b__",),
        reason="collapsed 20B lane no longer supported for this study",
        write=True,
    )

    assert written.files_rewritten == 1
    part_path = sorted((dataset.dir / "_derived" / "infer").glob("part-*.parquet"))[0]
    schema_names = pq.ParquetFile(part_path).schema_arrow.names
    assert stale_payload not in schema_names
    assert stale_metadata not in schema_names
    assert retained_payload in schema_names
    events = [json.loads(line) for line in dataset.events_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["action"] == "infer_stale_overlay_column_prune"
    assert events[-1]["metrics"]["columns_removed"] == 2


def test_prune_stale_infer_overlay_columns_refuses_join_column(tmp_path: Path) -> None:
    dataset, _ = _dataset_with_sequence(tmp_path)

    try:
        prune_stale_infer_overlay_columns(
            dataset_root=dataset.root,
            dataset_id=dataset.name,
            column_names=("id",),
            write=True,
        )
    except ValueError as error:
        assert "join column" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("stale-column pruning should refuse the join column")
