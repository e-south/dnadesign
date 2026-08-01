"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/verification.py

Join canonical Reader diagnostic evidence to one study-owned candidate.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dnadesign.studies.core.reader_records import ReaderRecordError
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_records import (
    load_reader_response_display_record,
    load_reader_response_records,
)

from .binding_verification import resolve_reader_study_binding
from .contracts import (
    ReaderPromoterEvidenceError,
    VerifiedReaderPromoterEvidenceSource,
)


def verify_reader_promoter_evidence_source(
    *,
    reader_root: Path,
    experiment_root: Path,
    projection_path: Path,
    bindings_bundle: Path,
    reader_command: Sequence[str] | None = None,
) -> VerifiedReaderPromoterEvidenceSource:
    """Resolve one display row without importing Reader or creating a lifecycle."""

    try:
        records = load_reader_response_records(
            reader_root=reader_root,
            experiment_root=experiment_root,
            projection_path=projection_path,
            reader_command=reader_command,
        )
        display = load_reader_response_display_record(records, reader_command=reader_command)
    except ReaderRecordError as exc:
        raise ReaderPromoterEvidenceError(f"Canonical Reader response-window evidence did not verify: {exc}") from exc
    selected_binding, binding_source = resolve_reader_study_binding(
        display.design_id,
        bindings_bundle=bindings_bundle,
    )
    if selected_binding["reader_design_id"] != display.design_id:
        raise ReaderPromoterEvidenceError("Study candidate binding disagrees with the diagnostic design pin.")
    return VerifiedReaderPromoterEvidenceSource(
        records=records,
        display=display,
        selected_binding=selected_binding,
        binding_source=binding_source,
    )


__all__ = ["verify_reader_promoter_evidence_source"]
