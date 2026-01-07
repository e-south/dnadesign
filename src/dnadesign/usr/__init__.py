"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/__init__.py

Public entry point for the USR package.

This module re-exports the stable Python API:

    from dnadesign.usr import Dataset, NamespaceError, ValidationError, ...

Default layout (editable install):
    src/dnadesign/usr/
      ├─ src/                # package code
      ├─ datasets/          # <-- dataset root used by the CLI by default
      │    └─ <dataset_name>/
      │         ├─ records.parquet
      │         ├─ meta.md
      │         └─ _snapshots/
      └─ demo_material/     # example CSVs for README walkthrough

You can override the root on the CLI via --root, or when using the Python API:
    from pathlib import Path
    root = Path(__file__).resolve().parent / "datasets"
    ds = Dataset(root, "mock_dataset")
    ds.init(source="example")

See README.md for a CLI walkthrough and the console script entrypoint ("usr")
defined in pyproject.toml.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# re-exported API
from .src.dataset import Dataset  # noqa: F401
from .src.errors import (  # noqa: F401
    AlphabetError,
    DuplicateIDError,
    EmbeddingDimensionError,  # legacy, kept for compatibility
    NamespaceError,
    SchemaError,
    SequencesError,
    ValidationError,
)
from .src.normalize import compute_id, normalize_sequence  # noqa: F401
from .src.schema import ARROW_SCHEMA, REQUIRED_COLUMNS  # noqa: F401
from .src.version import __version__  # noqa: F401

__all__ = [
    "Dataset",
    "AlphabetError",
    "DuplicateIDError",
    "EmbeddingDimensionError",
    "NamespaceError",
    "SchemaError",
    "SequencesError",
    "ValidationError",
    "compute_id",
    "normalize_sequence",
    "ARROW_SCHEMA",
    "REQUIRED_COLUMNS",
    "__version__",
]
