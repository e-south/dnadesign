"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/__init__.py

Generic fold-check request and report contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.foldcheck.hashes import sequence_hash
from dnadesign.thread.foldcheck.models import FoldCheckIssue, FoldCheckSequenceRecord
from dnadesign.thread.foldcheck.report import (
    FOLDCHECK_REPORT_SCHEMA_ID,
    validate_foldcheck_report,
    write_foldcheck_report,
)
from dnadesign.thread.foldcheck.request import (
    FOLDCHECK_REQUEST_SCHEMA_ID,
    build_foldcheck_request_manifest,
    request_hash,
    write_foldcheck_fasta,
)

__all__ = [
    "FOLDCHECK_REPORT_SCHEMA_ID",
    "FOLDCHECK_REQUEST_SCHEMA_ID",
    "FoldCheckIssue",
    "FoldCheckSequenceRecord",
    "build_foldcheck_request_manifest",
    "request_hash",
    "sequence_hash",
    "validate_foldcheck_report",
    "write_foldcheck_fasta",
    "write_foldcheck_report",
]
