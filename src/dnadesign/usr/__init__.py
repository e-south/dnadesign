"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/__init__.py

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
