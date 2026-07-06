"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/handoff_readiness.py

RT-only handoff readiness fields for Eco1 selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.candidate_handoff import (
    validate_candidate_handoff_content,
)

from .constants import CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME


def build_handoff_readiness(
    *,
    selection_root: Path,
    panel_rows: Sequence[Mapping[str, object]],
    candidate_handoff_path: Path,
) -> dict[str, object]:
    """Return manifest fields for the RT-only handoff boundary."""

    return normalize_handoff_readiness(
        selection_root=selection_root,
        raw={
            "panel_selected": bool(panel_rows),
            "candidate_handoff_path": _relative_path(selection_root, candidate_handoff_path),
            "candidate_handoff_sequence_csv_path": CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME,
            "construct_subject_created": False,
        },
    )


def normalize_handoff_readiness(
    *,
    selection_root: Path,
    raw: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize manifest readiness fields and recompute file-presence booleans."""

    values = dict(raw or {})
    handoff_path = str(values.get("candidate_handoff_path") or "candidate_handoff.yaml")
    sequence_csv_path = str(
        values.get("candidate_handoff_sequence_csv_path") or CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME
    )
    resolved_handoff_path = selection_root / handoff_path
    handoff_file_present = resolved_handoff_path.exists()
    handoff_valid = handoff_file_present and not validate_candidate_handoff_content(resolved_handoff_path)
    return {
        "handoff_kind": str(values.get("handoff_kind") or "rt_only_candidate_handoff"),
        "panel_selected": bool(values.get("panel_selected")),
        "candidate_handoff_path": handoff_path,
        "candidate_handoff_sequence_csv_path": sequence_csv_path,
        "candidate_handoff_sequence_csv_materialized": (selection_root / sequence_csv_path).exists(),
        "candidate_handoff_file_present": handoff_file_present,
        "candidate_handoff_materialized": handoff_valid,
        "construct_subject_created": bool(values.get("construct_subject_created")),
    }


def _relative_path(root: Path, target: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return os.path.relpath(target, start=root)
