"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_snapshot.py

Close Reader record and authoring attestation over one stable source snapshot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from dnadesign.studies.core.reader_records import (
    ReaderRecordError,
    ReaderRecordExpectation,
    ReaderRecordSet,
)

from .reader_config_attestation import ReaderResponseConfigAttestation


def resolve_matching_reader_snapshot(
    initial: ReaderRecordSet,
    *,
    config_attestation: ReaderResponseConfigAttestation,
    expected_records: Mapping[str, ReaderRecordExpectation],
    reader_command: Sequence[str] | None,
    resolver: Callable[..., ReaderRecordSet],
) -> ReaderRecordSet:
    """Resolve again and reject record or config drift across attestation."""

    confirmed = resolver(
        initial.config_path,
        reader_root=initial.reader_root,
        experiment_id=initial.experiment_id,
        protocol_id=initial.protocol_id,
        expected_records=expected_records,
        reader_command=reader_command,
    )
    if confirmed != initial:
        raise ReaderRecordError("Reader catalog or exact response record identity changed during config attestation")
    try:
        config_sha256 = hashlib.sha256(confirmed.config_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReaderRecordError(f"could not re-read attested Reader config: {exc}") from exc
    if config_sha256 != config_attestation.config_sha256:
        raise ReaderRecordError("Reader config bytes changed after config attestation")
    return confirmed


__all__ = ["resolve_matching_reader_snapshot"]
