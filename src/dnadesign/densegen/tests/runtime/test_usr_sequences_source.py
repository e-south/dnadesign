"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_usr_sequences_source.py

USR sequence source behavior tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.densegen.src.adapters.sources.usr_sequences import USRSequencesDataSource
from dnadesign.usr import Dataset
from dnadesign.usr.src.registry import USR_STATE_COLUMNS, USR_STATE_NAMESPACE, register_namespace


def test_usr_sequences_source_streams_without_full_parquet_read(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    dataset_name = "densegen/demo_stream"
    register_namespace(
        root,
        namespace=USR_STATE_NAMESPACE,
        columns=USR_STATE_COLUMNS,
        owner="usr",
        description="Reserved record-state overlay (tests).",
        overwrite=False,
    )
    ds = Dataset(root, dataset_name)
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT"},
            {"sequence": "TTTT"},
            {"sequence": "GGGG"},
        ],
        default_bio_type="dna",
        default_alphabet="dna_4",
        source="test",
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen: {}\n", encoding="utf-8")
    source = USRSequencesDataSource(
        dataset=dataset_name,
        cfg_path=cfg_path,
        root=str(root),
        limit=1,
    )

    def _forbid_read_table(*_args, **_kwargs):
        raise AssertionError("source should stream sequences instead of calling pq.read_table")

    monkeypatch.setattr(pq, "read_table", _forbid_read_table)

    seqs, _meta, _extra = source.load_data(rng=None, outputs_root=None, run_id=None)
    assert seqs == ["ACGT"]
