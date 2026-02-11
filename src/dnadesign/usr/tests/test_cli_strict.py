"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_cli_strict.py

CLI strictness tests for assertive error behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.usr.src.cli import cmd_cell
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.errors import SequencesError
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _make_dataset(tmp_path: Path) -> Dataset:
    ensure_registry(tmp_path)
    ds = Dataset(tmp_path, "ns/demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit",
            }
        ]
    )
    return ds


def test_cmd_cell_missing_column_raises(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    args = SimpleNamespace(
        root=tmp_path,
        path=None,
        target="ns/demo",
        glob=None,
        row=0,
        col="missing_col",
    )
    with pytest.raises(SequencesError, match="missing_col"):
        cmd_cell(args)
