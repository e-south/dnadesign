"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/errors.py

Narrow, typed exceptions used across the USR package. This keeps try/except
blocks readable and allows callers to distinguish data/validation issues from
I/O or operational failures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

class SequencesError(Exception):
    """Base class for USR-related errors (operational or validation)."""
    pass


class ValidationError(SequencesError):
    """Base class for data validation problems."""
    pass


class DuplicateIDError(ValidationError):
    pass


class AlphabetError(ValidationError):
    pass


class SchemaError(ValidationError):
    pass


class NamespaceError(ValidationError):
    pass


# Legacy (kept for compatibility)
class EmbeddingDimensionError(ValidationError):
    pass


# ----- Remote/Sync specific -----

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
