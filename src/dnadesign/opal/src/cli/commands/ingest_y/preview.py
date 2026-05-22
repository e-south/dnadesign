"""
Preview warning and nudge helpers for `opal ingest-y`.
"""

from __future__ import annotations

from ....runtime.ingest import IngestPreview
from ....storage.label_sources import SharedObservedLabelSource


def rewrite_preview_warnings(
    preview: IngestPreview,
    *,
    label_source: object,
    unknown_sequences_policy: str,
) -> None:
    if not preview.unknown_sequences:
        return
    warnings = list(preview.warnings or [])
    if isinstance(label_source, SharedObservedLabelSource):
        warnings = [
            warning
            for warning in warnings
            if "new rows will be created" not in warning.lower()
            and "created for new sequences" not in warning.lower()
            and "create deterministic ids" not in warning.lower()
        ]
        warnings.append(
            "shared usr_sidecar label sources use a fixed candidate universe; unknown labels cannot create records."
        )
    if unknown_sequences_policy in {"drop", "error"}:
        warnings = [
            warning
            for warning in warnings
            if "sequences not found" not in warning.lower() and "new rows will be created" not in warning.lower()
        ]
    if unknown_sequences_policy == "drop":
        warnings.append(
            f"{int(preview.unknown_sequences)} unknown sequences will be dropped (--unknown-sequences drop)."
        )
    elif unknown_sequences_policy == "error":
        warnings.append(
            f"{int(preview.unknown_sequences)} unknown sequences will abort ingest (--unknown-sequences error)."
        )
    preview.warnings = warnings


def build_ingest_nudges(
    preview: IngestPreview,
    *,
    unknown_sequences_policy: str,
    unknown_count_after_policy: int,
    required_cols: list[str],
    dropped_missing_x: int,
) -> list[str]:
    nudges = list(preview.warnings or [])
    total_unknown = int(preview.unknown_sequences or 0)
    if total_unknown:
        if unknown_sequences_policy == "drop":
            nudges = _without_unknown_create_warnings(nudges)
            nudges.append(f"Dropping {total_unknown} unknown sequences (--unknown-sequences drop).")
        elif dropped_missing_x:
            nudges = _without_unknown_create_warnings(nudges)
            nudges.append(f"Dropping {dropped_missing_x} unknown sequences missing X data.")
        if unknown_count_after_policy > 0 and required_cols and unknown_sequences_policy != "drop":
            nudges.append(
                "New sequences will be created; required columns for new rows: " + ", ".join(required_cols) + "."
            )
    return nudges


def _without_unknown_create_warnings(warnings: list[str]) -> list[str]:
    return [
        warning
        for warning in warnings
        if "sequences not found" not in warning.lower() and "new rows will be created" not in warning.lower()
    ]
