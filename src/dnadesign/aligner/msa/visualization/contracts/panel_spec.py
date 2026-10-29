"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/contracts/panel_spec.py

Display-only panel specification for generic MSA visualizations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PANEL_SPEC_SCHEMA_ID = "dnadesign.aligner.msa.visualization.panel_spec"


@dataclass(frozen=True)
class MsaOverviewProfileSpec:
    """Profile-specific display metadata for whole-alignment overview rows."""

    group: str = ""
    target_label: str = ""
    label_template: str = "{record_id}"
    label_max_chars: int = 72
    source_manifest_path: Path | None = None


@dataclass(frozen=True)
class MsaPanelSpec:
    """Options for display-only MSA overview panels."""

    overview_enabled: bool = False
    consensus_histogram_enabled: bool = False
    overview_row_source: str = "exemplar_rows"
    max_display_rows: int | None = 8
    display_gap_trim_threshold: float | None = None
    profile_specs: dict[str, MsaOverviewProfileSpec] | None = None
    note: str = "Display sidecar only; not a conservation denominator."

    @property
    def has_panels(self) -> bool:
        return self.overview_enabled or self.consensus_histogram_enabled

    def overview_profile_spec(self, profile_id: str) -> MsaOverviewProfileSpec:
        """Return profile-specific overview metadata, if declared."""

        return (self.profile_specs or {}).get(profile_id, MsaOverviewProfileSpec())


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

    overview_row_source = _row_source(overview.get("row_source", "exemplar_rows"), "overview.row_source")
    max_display_rows = _max_display_rows(
        overview.get("max_display_rows", "all" if overview_row_source == "all_records" else 8),
        "overview.max_display_rows",
    )
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
        overview_row_source=overview_row_source,
        max_display_rows=max_display_rows,
        display_gap_trim_threshold=trim_threshold,
        profile_specs=_profile_specs(payload.get("profiles"), config_path=path),
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


def _max_display_rows(value: object, context: str) -> int | None:
    if isinstance(value, str) and value == "all":
        return None
    return _positive_int(value, context)


def _row_source(value: object, context: str) -> str:
    if value not in {"exemplar_rows", "all_records"}:
        raise ValueError(f"panel spec {context} must be 'exemplar_rows' or 'all_records'")
    return str(value)


def _profile_specs(value: object, *, config_path: Path) -> dict[str, MsaOverviewProfileSpec]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("panel spec profiles must be a mapping")
    specs: dict[str, MsaOverviewProfileSpec] = {}
    for profile_id, raw_spec in value.items():
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("panel spec profile ids must be non-empty strings")
        spec = _mapping(raw_spec, f"profiles.{profile_id}")
        source_manifest = spec.get("source_manifest_path")
        specs[profile_id] = MsaOverviewProfileSpec(
            group=_optional_string(spec.get("group"), f"profiles.{profile_id}.group"),
            target_label=_optional_string(spec.get("target_label"), f"profiles.{profile_id}.target_label"),
            label_template=_optional_string(
                spec.get("label_template", "{record_id}"),
                f"profiles.{profile_id}.label_template",
            )
            or "{record_id}",
            label_max_chars=_positive_int(
                spec.get("label_max_chars", 72),
                f"profiles.{profile_id}.label_max_chars",
            ),
            source_manifest_path=(
                _configured_path(source_manifest, config_path=config_path) if source_manifest is not None else None
            ),
        )
    return specs


def _optional_string(value: object, context: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"panel spec {context} must be a non-empty string when provided")
    return value.strip()


def _configured_path(value: object, *, config_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("panel spec source_manifest_path must be a non-empty string")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config_path.parent / candidate
