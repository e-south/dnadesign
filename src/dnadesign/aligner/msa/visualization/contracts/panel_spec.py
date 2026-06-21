"""Display-only panel specification for generic MSA visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PANEL_SPEC_SCHEMA_ID = "dnadesign.aligner.msa.visualization.panel_spec"


@dataclass(frozen=True)
class MsaPanelSpec:
    """Options for display-only MSA overview panels."""

    overview_enabled: bool = False
    consensus_histogram_enabled: bool = False
    max_display_rows: int = 8
    display_gap_trim_threshold: float | None = None
    note: str = "Display sidecar only; not a conservation denominator."

    @property
    def has_panels(self) -> bool:
        return self.overview_enabled or self.consensus_histogram_enabled


def load_panel_spec(path: Path | None) -> MsaPanelSpec:
    """Load and validate an optional panel specification."""

    if path is None:
        return MsaPanelSpec()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("panel spec YAML must be a mapping")
    if payload.get("schema_id") != _PANEL_SPEC_SCHEMA_ID:
        raise ValueError(f"panel spec schema_id must be {_PANEL_SPEC_SCHEMA_ID}")

    overview = _mapping(payload.get("overview"), "overview")
    histogram = _mapping(payload.get("consensus_histogram"), "consensus_histogram")
    display = _mapping(payload.get("display_columns"), "display_columns")

    max_display_rows = _positive_int(overview.get("max_display_rows", 8), "overview.max_display_rows")
    trim_threshold = display.get("high_gap_trim_threshold")
    if trim_threshold is not None:
        if isinstance(trim_threshold, bool) or not isinstance(trim_threshold, int | float):
            raise ValueError("display_columns.high_gap_trim_threshold must be a number between 0 and 1")
        trim_threshold = float(trim_threshold)
        if trim_threshold < 0.0 or trim_threshold > 1.0:
            raise ValueError("display_columns.high_gap_trim_threshold must be between 0 and 1")

    note = payload.get("sidecar_note", "Display sidecar only; not a conservation denominator.")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("panel spec sidecar_note must be a non-empty string")

    return MsaPanelSpec(
        overview_enabled=_bool(overview.get("enabled", False), "overview.enabled"),
        consensus_histogram_enabled=_bool(histogram.get("enabled", False), "consensus_histogram.enabled"),
        max_display_rows=max_display_rows,
        display_gap_trim_threshold=trim_threshold,
        note=note,
    )


def _mapping(value: object, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"panel spec {context} must be a mapping")
    return value


def _bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"panel spec {context} must be a boolean")
    return value


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"panel spec {context} must be a positive integer")
    return value
