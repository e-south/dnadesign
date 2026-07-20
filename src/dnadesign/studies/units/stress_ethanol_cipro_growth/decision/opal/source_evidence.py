"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/source_evidence.py

Study-owned locations for immutable OPAL source evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

SFXI_ROUND0_SOURCE_EVIDENCE_ROOT = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/opal_sfxi_round0"
)
RESPONSE_WINDOW_ROUND0_SOURCE_EVIDENCE_ROOT = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/opal_response_window_round0"
)


def sfxi_round0_source_evidence_dir(repo_root: str | Path, *, source_slug: str) -> Path:
    """Resolve one source-run directory under the study-owned evidence root."""

    slug = str(source_slug).strip()
    if not slug or Path(slug).name != slug or slug in {".", ".."}:
        raise ValueError("source_slug must be one non-empty path segment")
    return Path(repo_root) / SFXI_ROUND0_SOURCE_EVIDENCE_ROOT / slug


def response_window_round0_source_evidence_root(repo_root: str | Path) -> Path:
    """Resolve frozen response-window prediction evidence outside campaign routing."""

    return Path(repo_root) / RESPONSE_WINDOW_ROUND0_SOURCE_EVIDENCE_ROOT


__all__ = [
    "RESPONSE_WINDOW_ROUND0_SOURCE_EVIDENCE_ROOT",
    "SFXI_ROUND0_SOURCE_EVIDENCE_ROOT",
    "response_window_round0_source_evidence_root",
    "sfxi_round0_source_evidence_dir",
]
