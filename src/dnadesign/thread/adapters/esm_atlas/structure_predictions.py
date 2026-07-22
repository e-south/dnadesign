"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/structure_predictions.py

Structure-prediction provenance rows for Atlas on-demand folds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.thread.adapters.esm_atlas.hashes import raw_response_hash
from dnadesign.thread.structure_predictions import file_sha256_uri, text_sha256_uri

_DEFAULT_MODEL_NAME = "esm_atlas_structure_prediction"
_DEFAULT_MODEL_FAMILY = "esmfold_family"
_PDB_KEYS = ("pdb", "pdb_string", "structure_pdb")
_NESTED_STRUCTURE_PDB_KEYS = ("pdb", "pdb_string", "pdb_text")


def build_atlas_structure_prediction_row(
    *,
    candidate_id: str,
    sequence_hash: str,
    response: Mapping[str, Any],
    atlas_request_hash: str,
    source_request_hash: str,
    prediction_set_id: str,
    output_root: Path,
    atlas_api_base_url: str,
    atlas_api_version: str,
) -> dict[str, object] | None:
    """Write an Atlas structure payload when present and return a registry row."""

    folded_on_demand = bool(response.get("folded_on_demand", False))
    pdb_text = _extract_pdb_text(response)
    if not folded_on_demand and pdb_text is None:
        return None

    prediction_id = _prediction_id(
        prediction_set_id=prediction_set_id,
        candidate_id=candidate_id,
        sequence_hash=sequence_hash,
        atlas_request_hash=atlas_request_hash,
    )
    base_row = {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "prediction_id": prediction_id,
        "prediction_set_id": prediction_set_id,
        "backend_kind": "esm_atlas",
        "model_family": _DEFAULT_MODEL_FAMILY,
        "model_name": _DEFAULT_MODEL_NAME,
        "model_version": atlas_api_version,
        "runtime_or_endpoint": atlas_api_base_url,
        "parameters_hash": text_sha256_uri(
            json.dumps(
                {
                    "adapter": "thread.adapters.esm_atlas",
                    "atlas_api_base_url": atlas_api_base_url,
                    "atlas_api_version": atlas_api_version,
                    "fold_on_miss": True,
                    "structure_payload_source": "atlas_response",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "request_hash": atlas_request_hash,
        "source_request_hash": source_request_hash,
        "raw_response_hash": raw_response_hash(response),
        "structure_hash": "",
        "structure_source_uri": f"atlas://{response.get('protein_hash', '')}",
        "local_structure_path": "",
        "plddt": _optional_float(response, "mean_plddt", "plddt"),
        "ptm": _optional_float(response, "ptm", "pTM"),
        "pae_summary_hash": _optional_payload_hash(response, "pae", "pae_summary", "predicted_aligned_error"),
        "status": "accepted",
        "failure_reason": "",
    }
    if pdb_text is None:
        base_row.update({"status": "errored", "failure_reason": "atlas_structure_payload_missing"})
        return base_row

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{_safe_file_stem(candidate_id)}.{prediction_id}.pdb"
    path.write_text(_normalize_pdb_text(pdb_text), encoding="utf-8")
    base_row.update({"structure_hash": file_sha256_uri(path), "local_structure_path": str(path)})
    return base_row


def _extract_pdb_text(response: Mapping[str, Any]) -> str | None:
    for key in _PDB_KEYS:
        value = response.get(key)
        if _looks_like_pdb(value):
            return str(value)
    structure = response.get("structure")
    if isinstance(structure, Mapping):
        for key in _NESTED_STRUCTURE_PDB_KEYS:
            value = structure.get(key)
            if _looks_like_pdb(value):
                return str(value)
    return None


def _looks_like_pdb(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and ("ATOM" in text or "HETATM" in text)


def _normalize_pdb_text(value: str) -> str:
    return value.rstrip() + "\n"


def _prediction_id(
    *,
    prediction_set_id: str,
    candidate_id: str,
    sequence_hash: str,
    atlas_request_hash: str,
) -> str:
    digest = text_sha256_uri(f"{prediction_set_id}|{candidate_id}|{sequence_hash}|{atlas_request_hash}").removeprefix(
        "sha256:"
    )[:16]
    return f"{_safe_file_stem(prediction_set_id)}_{_safe_file_stem(candidate_id)}_{digest}"


def _safe_file_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "prediction"


def _optional_float(response: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = response.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _optional_payload_hash(response: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = response.get(key)
        if value is not None:
            return text_sha256_uri(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
    return ""
