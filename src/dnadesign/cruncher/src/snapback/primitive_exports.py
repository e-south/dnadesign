"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/primitive_exports.py

Public primitive export helpers for released-product snapback solve bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.cruncher.snapback.released_artifacts import released_solve_report_json_path


class SnapbackPrimitiveExportError(ValueError):
    """Raised when a public Snapback primitive export cannot be read safely."""


@dataclass(frozen=True)
class SnapbackCapPrimitive:
    rank: int
    primitive_id: str
    sequence: str
    hit_kind: str
    nickase_variant_id: str
    release_variant_id: str
    source_report: str


def load_released_solve_cap_primitives(run_dir: str | Path) -> list[SnapbackCapPrimitive]:
    """Load cap-segment primitive options from a released solve bundle."""

    run_path = Path(run_dir).expanduser().resolve()
    report_path = released_solve_report_json_path(run_path)
    if not report_path.is_file():
        raise SnapbackPrimitiveExportError(f"Released Snapback solve report not found: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapbackPrimitiveExportError(f"Released Snapback solve report is invalid JSON: {report_path}") from exc
    if not isinstance(payload, dict) or payload.get("workflow") != "snapback_released_solve":
        raise SnapbackPrimitiveExportError(
            f"Released Snapback solve report must declare workflow=snapback_released_solve: {report_path}"
        )
    hits = payload.get("hits")
    if not isinstance(hits, list):
        raise SnapbackPrimitiveExportError(f"Released Snapback solve report hits must be a list: {report_path}")

    primitives: list[SnapbackCapPrimitive] = []
    for index, raw_hit in enumerate(hits, start=1):
        if not isinstance(raw_hit, dict):
            raise SnapbackPrimitiveExportError(f"Released Snapback hit #{index} must be a mapping: {report_path}")
        rank = _positive_int(raw_hit.get("rank"), label=f"hits[{index}].rank", report_path=report_path)
        target_hit = _mapping(raw_hit.get("target_search_hit"), label=f"hits[{index}].target_search_hit")
        final_candidate = _mapping(
            target_hit.get("final_candidate"), label=f"hits[{index}].target_search_hit.final_candidate"
        )
        sequence = _dna_sequence(
            final_candidate.get("designed_sequence"),
            label=f"hits[{index}].target_search_hit.final_candidate.designed_sequence",
            report_path=report_path,
        )
        primitives.append(
            SnapbackCapPrimitive(
                rank=rank,
                primitive_id=f"snapback-rank-{rank:02d}",
                sequence=sequence,
                hit_kind=_not_blank(raw_hit.get("hit_kind"), label=f"hits[{index}].hit_kind"),
                nickase_variant_id=_not_blank(
                    raw_hit.get("nickase_variant_id"), label=f"hits[{index}].nickase_variant_id"
                ),
                release_variant_id=_not_blank(
                    raw_hit.get("release_variant_id"), label=f"hits[{index}].release_variant_id"
                ),
                source_report=report_path.as_posix(),
            )
        )
    return sorted(primitives, key=lambda primitive: primitive.rank)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SnapbackPrimitiveExportError(f"Released Snapback field {label} must be a mapping.")
    return value


def _positive_int(value: Any, *, label: str, report_path: Path) -> int:
    if isinstance(value, bool):
        raise SnapbackPrimitiveExportError(f"Released Snapback field {label} must be a positive integer: {report_path}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SnapbackPrimitiveExportError(
            f"Released Snapback field {label} must be a positive integer: {report_path}"
        ) from exc
    if parsed < 1:
        raise SnapbackPrimitiveExportError(f"Released Snapback field {label} must be >= 1: {report_path}")
    return parsed


def _not_blank(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SnapbackPrimitiveExportError(f"Released Snapback field {label} cannot be empty.")
    return text


def _dna_sequence(value: Any, *, label: str, report_path: Path) -> str:
    text = _not_blank(value, label=label).upper()
    invalid = sorted(set(text) - {"A", "C", "G", "T"})
    if invalid:
        raise SnapbackPrimitiveExportError(
            f"Released Snapback field {label} contains non-DNA bases {''.join(invalid)}: {report_path}"
        )
    return text


__all__ = [
    "SnapbackCapPrimitive",
    "SnapbackPrimitiveExportError",
    "load_released_solve_cap_primitives",
]
