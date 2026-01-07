"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/errors.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List, Optional


class SequencesError(Exception):
    """Base class for USR-related errors (operational or validation)."""

    pass


class ValidationError(SequencesError):
    """Base class for data validation problems."""

    pass


@dataclass(frozen=True)
class DuplicateGroup:
    """
    One duplicate cluster to report back to the user.
    - id        : canonical USR id (sha1(bio_type|sequence_norm))
    - count     : how many incoming rows hit this duplicate group
    - rows      : 1-based input row indices (helps users edit their file)
    - sequence  : the (case-preserving) sequence payload to display
    """

    id: str
    count: int
    rows: List[int]
    sequence: str


class DuplicateIDError(ValidationError):
    """
    Raised when duplicates are detected either:
      - EXACT: same canonical id (byte-for-byte, same case)
      - CASEFOLD: same biological letters ignoring case (e.g., 'acgt' vs 'ACGT')

    Attributes (all optional — callers can still raise with a simple message):
      - groups            : top exact-duplicate groups (List[DuplicateGroup])
      - casefold_groups   : top case-insensitive duplicate groups
      - hint              : short, user-facing remediation guidance
    """

    def __init__(
        self,
        message: str,
        *,
        groups: Optional[List[DuplicateGroup]] = None,
        casefold_groups: Optional[List[DuplicateGroup]] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(message)
        self.groups: List[DuplicateGroup] = groups or []
        self.casefold_groups: List[DuplicateGroup] = casefold_groups or []
        self.hint: Optional[str] = hint


class AlphabetError(ValidationError):
    pass


class SchemaError(ValidationError):
    pass


class NamespaceError(ValidationError):
    pass


# Legacy (kept for compatibility)
class EmbeddingDimensionError(ValidationError):
    pass


class RemoteConfigError(SequencesError):
    """Bad or missing remote configuration."""

    pass


class RemoteUnavailableError(SequencesError):
    """SSH/SFTP not reachable or failing commands."""

    pass


class TransferError(SequencesError):
    """Rsync/transfer errors."""

    pass


class VerificationError(SequencesError):
    """Checksum/size/shape mismatches after transfer."""

    pass


class UserAbort(SequencesError):
    """User declined an overwrite."""

    pass
