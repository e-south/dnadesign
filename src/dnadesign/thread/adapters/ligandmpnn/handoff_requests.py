"""Versioned JSON request admission for the LigandMPNN command boundary."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

from .context_inventory import LigandMpnnContextPolymer
from .context_probe import LigandMpnnContextProbeRequest
from .models import (
    LigandMpnnContextInventoryReference,
    LigandMpnnPackingConfig,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnResidueAlphabet,
    LigandMpnnUpstreamPin,
)
from .scoring import LigandMpnnScoreMode, LigandMpnnScoreRequest

_UPSTREAM_FIELDS = {
    "commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "packing_checkpoint_path",
    "packing_checkpoint_sha256",
}
_PACKING_FIELDS = {"enabled", "number_of_packs_per_design", "repack_everything", "use_ligand_context"}
_COMMON_FIELDS = {
    "request_id",
    "pdb_path",
    "pdb_sha256",
    "output_dir",
    "upstream",
    "context_inventory",
    "fixed_residues",
    "redesigned_residues",
    "seeds",
    "batch_size",
    "number_of_batches",
    "use_atom_context",
    "use_side_chain_context",
}
_SCORE_FIELDS = _COMMON_FIELDS | {"mode", "use_sequence"}
_DESIGN_FIELDS = _COMMON_FIELDS | {"temperature", "residue_alphabets", "packing"}
_CONTEXT_FIELDS = {
    "request_id",
    "pdb_path",
    "pdb_sha256",
    "output_path",
    "upstream",
    "minimum_nucleotide_atoms",
    "required_polymer_types",
    "chains",
    "parse_all_atoms",
    "parse_atoms_with_zero_occupancy",
}


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def document_digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def exact_fields(payload: object, expected: set[str], label: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")
    return payload


def _document(payload: object, kind: str, names: set[str]) -> dict:
    row = exact_fields(payload, {"schema_id", "schema_version", *names}, kind)
    if (
        row["schema_id"] != f"thread.ligandmpnn.{kind}"
        or type(row["schema_version"]) is not int
        or row["schema_version"] != 1
    ):
        raise ValueError(f"unsupported {kind} schema")
    return {key: value for key, value in row.items() if key not in {"schema_id", "schema_version"}}


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("request paths must be nonempty strings")
    return Path(value)


def _rows(value: object) -> list:
    if not isinstance(value, list):
        raise ValueError("request arrays must be JSON lists")
    return value


def upstream_from_document(payload: object) -> LigandMpnnUpstreamPin:
    row = dict(exact_fields(payload, _UPSTREAM_FIELDS, "upstream"))
    for key in ("checkpoint_path", "packing_checkpoint_path"):
        row[key] = _path(row[key])
    return LigandMpnnUpstreamPin(**row)


def _residue(payload: object) -> LigandMpnnResidue:
    row = exact_fields(payload, {"chain_id", "residue_number", "insertion_code"}, "residue")
    return LigandMpnnResidue(**row)


def _common(row: dict) -> dict:
    row["upstream"] = upstream_from_document(row["upstream"])
    for key in ("pdb_path", "output_dir"):
        row[key] = _path(row[key])
    reference = exact_fields(row["context_inventory"], {"path", "sha256"}, "context_inventory")
    digest = reference["sha256"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("context inventory sha256 must be a SHA256 URI")
    row["context_inventory"] = LigandMpnnContextInventoryReference(_path(reference["path"]), digest[7:])
    for key in ("fixed_residues", "redesigned_residues"):
        row[key] = tuple(_residue(value) for value in _rows(row[key]))
    row["seeds"] = tuple(_rows(row["seeds"]))
    return row


def score_request_from_document(payload: object) -> LigandMpnnScoreRequest:
    row = _common(_document(payload, "score_command_request", _SCORE_FIELDS))
    row["mode"] = LigandMpnnScoreMode(row["mode"])
    return LigandMpnnScoreRequest(**row)


def design_request_from_document(payload: object) -> LigandMpnnRequest:
    row = _common(_document(payload, "design_command_request", _DESIGN_FIELDS))
    row["packing"] = LigandMpnnPackingConfig(**exact_fields(row["packing"], _PACKING_FIELDS, "packing"))
    alphabets = []
    for value in _rows(row["residue_alphabets"]):
        value = exact_fields(value, {"residue", "allowed_amino_acids"}, "residue alphabet")
        alphabets.append(
            LigandMpnnResidueAlphabet(_residue(value["residue"]), tuple(_rows(value["allowed_amino_acids"])))
        )
    row["residue_alphabets"] = tuple(alphabets)
    return LigandMpnnRequest(**row)


def context_request_from_document(payload: object) -> LigandMpnnContextProbeRequest:
    row = _document(payload, "context_command_request", _CONTEXT_FIELDS)
    row["upstream"] = upstream_from_document(row["upstream"])
    row["pdb_path"] = _path(row["pdb_path"])
    row["output_path"] = _path(row["output_path"])
    row["required_polymer_types"] = tuple(
        LigandMpnnContextPolymer(value) for value in _rows(row["required_polymer_types"])
    )
    row["chains"] = tuple(_rows(row["chains"]))
    return LigandMpnnContextProbeRequest(**row)


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, LigandMpnnContextInventoryReference):
        return value.to_dict()
    if isinstance(value, LigandMpnnUpstreamPin):
        return {name: _json_value(getattr(value, name)) for name in sorted(_UPSTREAM_FIELDS)}
    if isinstance(value, LigandMpnnResidue):
        return {
            "chain_id": value.chain_id,
            "residue_number": value.residue_number,
            "insertion_code": value.insertion_code,
        }
    if isinstance(value, LigandMpnnResidueAlphabet):
        return {"residue": _json_value(value.residue), "allowed_amino_acids": list(value.allowed_amino_acids)}
    if isinstance(value, LigandMpnnPackingConfig):
        return {name: _json_value(getattr(value, name)) for name in sorted(_PACKING_FIELDS)}
    return value


def score_request_document(request: LigandMpnnScoreRequest) -> dict:
    return {
        "schema_id": "thread.ligandmpnn.score_command_request",
        "schema_version": 1,
        **{name: _json_value(getattr(request, name)) for name in sorted(_SCORE_FIELDS)},
    }
