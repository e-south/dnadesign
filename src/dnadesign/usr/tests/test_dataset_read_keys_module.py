"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_read_keys_module.py

Layout and behavior tests for Dataset read-key helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import pyarrow as pa
import pytest

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.errors import SchemaError


def test_dataset_read_keys_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_read_keys")
    assert hasattr(module, "key_list_from_batch")


def test_dataset_key_list_method_delegates_to_module() -> None:
    source = inspect.getsource(Dataset._key_list_from_batch)
    assert "key_list_from_batch(" in source


def test_key_list_from_batch_returns_id_values() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_read_keys")
    batch = pa.record_batch(
        [
            pa.array(["id-1", "id-2"]),
            pa.array(["ACGT", "TGCA"]),
            pa.array(["dna_4", "dna_4"]),
        ],
        names=["id", "sequence", "alphabet"],
    )
    assert module.key_list_from_batch(batch, "id") == ["id-1", "id-2"]


def test_key_list_from_batch_normalizes_sequence_ci() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_read_keys")
    batch = pa.record_batch(
        [
            pa.array(["id-1", "id-2"]),
            pa.array([" acgt ", "tgca"]),
            pa.array(["dna_4", "dna_4"]),
        ],
        names=["id", "sequence", "alphabet"],
    )
    assert module.key_list_from_batch(batch, "sequence_ci") == ["ACGT", "TGCA"]


def test_key_list_from_batch_rejects_non_dna4_sequence_ci() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_read_keys")
    batch = pa.record_batch(
        [
            pa.array(["id-1"]),
            pa.array(["ACGU"]),
            pa.array(["rna_4"]),
        ],
        names=["id", "sequence", "alphabet"],
    )
    with pytest.raises(SchemaError, match="sequence_ci is only valid for dna_4 datasets."):
        module.key_list_from_batch(batch, "sequence_ci")


def test_key_list_from_batch_rejects_unsupported_key() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_read_keys")
    batch = pa.record_batch(
        [pa.array(["id-1"])],
        names=["id"],
    )
    with pytest.raises(SchemaError, match="Unsupported join key 'bad_key'."):
        module.key_list_from_batch(batch, "bad_key")
