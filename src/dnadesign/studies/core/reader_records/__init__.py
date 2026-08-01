"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/__init__.py

Public package surface for source-closed Reader record handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contracts import (
    READER_CATALOG_SCHEMA_VERSION,
    READER_CLI_SCHEMA,
    READER_RECORD_SCHEMA_VERSION,
    ReaderArtifactFile,
    ReaderDataframeRecordRef,
    ReaderRecordExpectation,
    ReaderRecordSet,
    ReaderResolvedRecord,
)
from .resolver import resolve_digest_verified_dataframe_record, resolve_digest_verified_records
from .validation import ReaderDataframeRecordError, ReaderRecordError

__all__ = [
    "READER_CATALOG_SCHEMA_VERSION",
    "READER_CLI_SCHEMA",
    "READER_RECORD_SCHEMA_VERSION",
    "ReaderArtifactFile",
    "ReaderDataframeRecordError",
    "ReaderDataframeRecordRef",
    "ReaderRecordError",
    "ReaderRecordExpectation",
    "ReaderRecordSet",
    "ReaderResolvedRecord",
    "resolve_digest_verified_dataframe_record",
    "resolve_digest_verified_records",
]
