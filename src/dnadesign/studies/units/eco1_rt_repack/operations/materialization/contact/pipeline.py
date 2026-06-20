"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/pipeline.py

Materialize Eco1 RT retained-context contact evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONTRACT_ROOT = _DOCS_ROOT / "operations/contract"
_PROFILE = _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"
_NUMBERING_POLICY = _DOCS_ROOT / "workbench/provenance/residue-numbering-policy.yaml"
_DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact"
_DEFAULT_CREATED_AT = "2026-06-19T00:00:00Z"
_CONTACT_PROFILE_COLUMNS = (
    "canonical_position",
    "retained_context_id",
    "nearest_context_atom_distance_angstrom",
    "contact_threshold_angstrom",
    "passes_contact_mask",
    "source_hash",
    "wt_aa",
    "structure_chain_id",
    "structure_residue_id",
    "nearest_dna_distance_angstrom",
    "nearest_rna_distance_angstrom",
    "nearest_dna_chain",
    "nearest_rna_chain",
    "nearest_context_molecule_type",
    "nearest_context_chain_id",
    "mapping_status",
    "contact_policy_id",
)
_AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class MaterializedContactArtifacts:
    """Paths emitted by one Eco1 contact-evidence materialization pass."""

    contact_profile_path: Path


def materialize_contact_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedContactArtifacts:
    """Materialize retained-context distance evidence into a thread-shaped artifact."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_output_root(root, output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    profile = _load_yaml(root / _PROFILE)
    numbering_policy = _load_yaml(root / _NUMBERING_POLICY)
    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    residue_map_path = out_root / "residue_map.parquet"
    if not backbone_bundle_path.exists():
        raise FileNotFoundError(backbone_bundle_path)
    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)

    backbone_bundle = _load_yaml(backbone_bundle_path)
    retained_context = _retained_context_inventory(backbone_bundle)
    _require_retained_nucleic_acid_context(retained_context)

    distance_profile_path = _resolve_source_ref(root, _require_text(numbering_policy, "source_distance_profile_ref"))
    distance_profile_sha256 = _require_text(numbering_policy, "source_distance_profile_sha256")
    _require_hash(distance_profile_path, distance_profile_sha256)
    distance_profile_source_hash = "sha256:" + distance_profile_sha256.removeprefix("sha256:")
    contact_threshold = _contact_threshold_angstrom(profile)
    contact_policy_id = f"retained_nucleic_acid_contact_le_{_format_threshold_id(contact_threshold)}_v1"

    residue_map = pq.read_table(residue_map_path)
    residue_rows = residue_map.to_pylist()
    distance_rows = _load_distance_rows(
        distance_profile_path,
        rt_chain_id=_require_text(backbone_bundle, "rt_chain_id"),
        retained_context_chains=set(retained_context),
    )
    contact_rows = _build_contact_rows(
        residue_rows=residue_rows,
        distance_rows=distance_rows,
        contact_threshold=contact_threshold,
        source_hash=distance_profile_source_hash,
        contact_policy_id=contact_policy_id,
    )

    contact_profile_path = out_root / "contact_profile.parquet"
    _write_contact_profile(
        contact_profile_path,
        rows=contact_rows,
        contact_threshold=contact_threshold,
        source_hash=distance_profile_source_hash,
        backbone_bundle_path=backbone_bundle_path,
        residue_map_path=residue_map_path,
        numbering_policy=numbering_policy,
        created_at=created_at,
    )
    return MaterializedContactArtifacts(contact_profile_path=contact_profile_path)


def _retained_context_inventory(backbone_bundle: Mapping[str, Any]) -> dict[str, str]:
    chain_inventory = backbone_bundle.get("chain_inventory")
    if not isinstance(chain_inventory, list):
        raise ValueError("backbone_bundle.yaml must declare chain_inventory before contact materialization")

    retained: dict[str, str] = {}
    for item in chain_inventory:
        if not isinstance(item, Mapping):
            continue
        if item.get("retention") != "retained" or item.get("thread_role") != "retained_context":
            continue
        chain_id = str(item.get("chain_id", "")).strip()
        molecule_type = str(item.get("molecule_type", "")).strip()
        if chain_id:
            retained[chain_id] = molecule_type
    return retained


def _require_retained_nucleic_acid_context(retained_context: Mapping[str, str]) -> None:
    molecule_types = set(retained_context.values())
    if "dna" not in molecule_types or "rna" not in molecule_types:
        raise ValueError("contact materialization requires retained DNA and RNA context chains")


def _load_distance_rows(
    path: Path,
    *,
    rt_chain_id: str,
    retained_context_chains: set[str],
) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    loaded: dict[int, dict[str, Any]] = {}
    for row in rows:
        position = int(_require_row_text(row, "aa_pos"))
        prot_chain = _require_row_text(row, "prot_chain")
        if prot_chain != rt_chain_id:
            raise ValueError(f"distance profile contains protein chain {prot_chain!r}, expected {rt_chain_id!r}")
        dna_chain = _require_row_text(row, "DNA_chain")
        rna_chain = _require_row_text(row, "RNA_chain")
        if dna_chain not in retained_context_chains or rna_chain not in retained_context_chains:
            raise ValueError(
                f"distance profile row {position} references context chains {dna_chain!r}/{rna_chain!r} "
                "outside retained backbone context"
            )
        residue_name = _require_row_text(row, "prot_resname").upper()
        wt_aa = _AA3_TO_AA1.get(residue_name)
        if wt_aa is None:
            raise ValueError(f"distance profile row {position} has unsupported residue name {residue_name!r}")
        loaded[position] = {
            "wt_aa": wt_aa,
            "structure_chain_id": prot_chain,
            "structure_residue_id": int(_require_row_text(row, "prot_resnum")),
            "nearest_dna_distance_angstrom": float(_require_row_text(row, "min_to_DNA_A")),
            "nearest_rna_distance_angstrom": float(_require_row_text(row, "min_to_RNA_A")),
            "nearest_dna_chain": dna_chain,
            "nearest_rna_chain": rna_chain,
        }
    return loaded


def _build_contact_rows(
    *,
    residue_rows: list[dict[str, Any]],
    distance_rows: Mapping[int, Mapping[str, Any]],
    contact_threshold: float,
    source_hash: str,
    contact_policy_id: str,
) -> list[dict[str, Any]]:
    contact_rows: list[dict[str, Any]] = []
    for residue in residue_rows:
        position = int(residue["canonical_position"])
        if residue.get("mapping_status") != "mapped":
            contact_rows.append(
                _unresolved_contact_row(
                    residue=residue,
                    contact_threshold=contact_threshold,
                    source_hash=source_hash,
                    contact_policy_id=contact_policy_id,
                )
            )
            continue

        distance = distance_rows.get(position)
        if distance is None:
            raise ValueError(f"mapped canonical position {position} is missing from contact distance profile")
        _validate_distance_row_matches_residue(position=position, residue=residue, distance=distance)

        dna_distance = float(distance["nearest_dna_distance_angstrom"])
        rna_distance = float(distance["nearest_rna_distance_angstrom"])
        if dna_distance <= rna_distance:
            nearest_distance = dna_distance
            nearest_molecule_type = "dna"
            nearest_chain = str(distance["nearest_dna_chain"])
        else:
            nearest_distance = rna_distance
            nearest_molecule_type = "rna"
            nearest_chain = str(distance["nearest_rna_chain"])
        contact_rows.append(
            {
                "canonical_position": position,
                "retained_context_id": "retained_nucleic_acid_context",
                "nearest_context_atom_distance_angstrom": nearest_distance,
                "contact_threshold_angstrom": contact_threshold,
                "passes_contact_mask": nearest_distance <= contact_threshold,
                "source_hash": source_hash,
                "wt_aa": residue["wt_aa"],
                "structure_chain_id": residue["structure_chain_id"],
                "structure_residue_id": int(residue["structure_residue_id"]),
                "nearest_dna_distance_angstrom": dna_distance,
                "nearest_rna_distance_angstrom": rna_distance,
                "nearest_dna_chain": str(distance["nearest_dna_chain"]),
                "nearest_rna_chain": str(distance["nearest_rna_chain"]),
                "nearest_context_molecule_type": nearest_molecule_type,
                "nearest_context_chain_id": nearest_chain,
                "mapping_status": "mapped",
                "contact_policy_id": contact_policy_id,
            }
        )
    return contact_rows


def _unresolved_contact_row(
    *,
    residue: Mapping[str, Any],
    contact_threshold: float,
    source_hash: str,
    contact_policy_id: str,
) -> dict[str, Any]:
    return {
        "canonical_position": int(residue["canonical_position"]),
        "retained_context_id": "retained_nucleic_acid_context",
        "nearest_context_atom_distance_angstrom": None,
        "contact_threshold_angstrom": contact_threshold,
        "passes_contact_mask": False,
        "source_hash": source_hash,
        "wt_aa": str(residue["wt_aa"]),
        "structure_chain_id": "",
        "structure_residue_id": None,
        "nearest_dna_distance_angstrom": None,
        "nearest_rna_distance_angstrom": None,
        "nearest_dna_chain": "",
        "nearest_rna_chain": "",
        "nearest_context_molecule_type": "",
        "nearest_context_chain_id": "",
        "mapping_status": "unresolved_structure",
        "contact_policy_id": contact_policy_id,
    }


def _validate_distance_row_matches_residue(
    *,
    position: int,
    residue: Mapping[str, Any],
    distance: Mapping[str, Any],
) -> None:
    expected = {
        "wt_aa": residue.get("wt_aa"),
        "structure_chain_id": residue.get("structure_chain_id"),
        "structure_residue_id": residue.get("structure_residue_id"),
    }
    observed = {
        "wt_aa": distance.get("wt_aa"),
        "structure_chain_id": distance.get("structure_chain_id"),
        "structure_residue_id": distance.get("structure_residue_id"),
    }
    if observed != expected:
        raise ValueError(f"distance profile row for canonical position {position} does not match residue_map.parquet")


def _write_contact_profile(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    contact_threshold: float,
    source_hash: str,
    backbone_bundle_path: Path,
    residue_map_path: Path,
    numbering_policy: Mapping[str, Any],
    created_at: str,
) -> None:
    schema = pa.schema(
        [
            pa.field("canonical_position", pa.int32(), nullable=False),
            pa.field("retained_context_id", pa.string(), nullable=False),
            pa.field("nearest_context_atom_distance_angstrom", pa.float64(), nullable=True),
            pa.field("contact_threshold_angstrom", pa.float64(), nullable=False),
            pa.field("passes_contact_mask", pa.bool_(), nullable=False),
            pa.field("source_hash", pa.string(), nullable=False),
            pa.field("wt_aa", pa.string(), nullable=False),
            pa.field("structure_chain_id", pa.string(), nullable=False),
            pa.field("structure_residue_id", pa.int32(), nullable=True),
            pa.field("nearest_dna_distance_angstrom", pa.float64(), nullable=True),
            pa.field("nearest_rna_distance_angstrom", pa.float64(), nullable=True),
            pa.field("nearest_dna_chain", pa.string(), nullable=False),
            pa.field("nearest_rna_chain", pa.string(), nullable=False),
            pa.field("nearest_context_molecule_type", pa.string(), nullable=False),
            pa.field("nearest_context_chain_id", pa.string(), nullable=False),
            pa.field("mapping_status", pa.string(), nullable=False),
            pa.field("contact_policy_id", pa.string(), nullable=False),
        ],
        metadata={
            b"schema_id": b"thread.contact_profile",
            b"schema_version": b"1",
            b"artifact_id": b"eco1_rt_conservative_v1.contact_profile",
            b"status": b"materialized",
            b"created_by": _CREATED_BY.encode("utf-8"),
            b"created_at": created_at.encode("utf-8"),
            b"reference_sequence_hash": _require_text(numbering_policy, "reference_sequence_hash").encode("utf-8"),
            b"selected_structure_source_id": _require_text(numbering_policy, "selected_structure_source_id").encode(
                "utf-8"
            ),
            b"source_hash": source_hash.encode("utf-8"),
            b"contact_threshold_angstrom": str(contact_threshold).encode("utf-8"),
            b"upstream_artifact_hashes": json.dumps(
                {
                    "backbone_bundle": "sha256:" + _sha256(backbone_bundle_path),
                    "residue_map": "sha256:" + _sha256(residue_map_path),
                    "ec86kit_distance_profile": source_hash,
                },
                sort_keys=True,
            ).encode("utf-8"),
        },
    )
    missing_columns = sorted(set(_CONTACT_PROFILE_COLUMNS) - {field.name for field in schema})
    if missing_columns:
        raise ValueError(f"contact profile schema is missing required columns: {missing_columns}")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _contact_threshold_angstrom(profile: Mapping[str, Any]) -> float:
    conservative_policy = profile.get("conservative_policy")
    if not isinstance(conservative_policy, Mapping):
        raise ValueError("profile must declare conservative_policy before contact materialization")
    value = conservative_policy.get("substrate_contact_threshold_angstrom")
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0:
        raise ValueError("substrate_contact_threshold_angstrom must be a positive number")
    return float(value)


def _format_threshold_id(value: float) -> str:
    if value.is_integer():
        return f"{int(value)}A"
    return str(value).replace(".", "p") + "A"


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    resolved = output_root or repo_root / _DEFAULT_OUTPUT_ROOT
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _resolve_source_ref(repo_root: Path, source_ref: str) -> Path:
    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (repo_root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _require_hash(path: Path, expected_sha256: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    expected = expected_sha256.removeprefix("sha256:")
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: {observed} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_row_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"distance profile row is missing {field!r}")
    return value.strip()


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 RT retained-context contact profile.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_contact_profile(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps({"contact_profile_path": str(result.contact_profile_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
