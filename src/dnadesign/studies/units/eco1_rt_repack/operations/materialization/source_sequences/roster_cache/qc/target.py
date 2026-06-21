"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/qc/target.py

Target-sequence authority loading for Eco1 roster-cache QC.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ConservationSourceContract,
)


def load_target_sequence_from_contract(repo_root: Path, contract: ConservationSourceContract) -> str:
    """Load and verify the ec86kit reference sequence declared by the source contract."""

    target_payload = _require_mapping(contract.sources.get("target_sequence"), "target_sequence")
    source_ref = _require_text(target_payload, "source_ref")
    path = _resolve_source_ref(repo_root, source_ref)
    sequence = _load_reference_sequence_from_manifest(path)
    observed_hash = "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    if observed_hash != contract.target_sequence_hash:
        raise ValueError(
            f"target_sequence.source_ref hash mismatch: expected {contract.target_sequence_hash}, "
            f"observed {observed_hash}"
        )
    expected_length = target_payload.get("reference_sequence_length")
    if expected_length != len(sequence):
        raise ValueError("target_sequence.reference_sequence_length does not match source_ref sequence length")
    return sequence


def _load_reference_sequence_from_manifest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ec86kit manifest must be a JSON object")
    for step in payload.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        config = step.get("config")
        if isinstance(config, Mapping) and isinstance(config.get("sequence"), str):
            sequence = config["sequence"].strip().upper()
            if sequence:
                return sequence
    raise ValueError(f"ec86kit manifest does not declare a reference sequence: {path}")


def _resolve_source_ref(repo_root: Path, source_ref: str) -> Path:
    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (repo_root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
