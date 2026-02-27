"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_ingest_module.py

Layout contract tests for Dataset ingest decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_ingest_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_ingest")


def test_dataset_ingest_methods_delegate_to_ingest_module() -> None:
    prepare_source = inspect.getsource(Dataset._prepare_import_rows)
    write_source = inspect.getsource(Dataset._write_import_df)
    import_rows_source = inspect.getsource(Dataset.import_rows)
    add_sequences_source = inspect.getsource(Dataset.add_sequences)
    import_csv_source = inspect.getsource(Dataset.import_csv)
    import_jsonl_source = inspect.getsource(Dataset.import_jsonl)

    assert "prepare_import_rows_dataset(" in prepare_source
    assert "write_import_df_dataset(" in write_source
    assert "import_rows_dataset(" in import_rows_source
    assert "add_sequences_dataset(" in add_sequences_source
    assert "import_csv_dataset(" in import_csv_source
    assert "import_jsonl_dataset(" in import_jsonl_source
