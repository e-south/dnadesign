"""
DenseGen output sinks package.
"""

from .base import DEFAULT_NAMESPACE, SinkBase, USRSink
from .factory import build_sinks, resolve_bio_alphabet
from .loader import load_records_from_config
from .parquet import ParquetSink
from .record import OutputRecord

__all__ = [
    "DEFAULT_NAMESPACE",
    "SinkBase",
    "USRSink",
    "ParquetSink",
    "build_sinks",
    "resolve_bio_alphabet",
    "load_records_from_config",
    "OutputRecord",
]
