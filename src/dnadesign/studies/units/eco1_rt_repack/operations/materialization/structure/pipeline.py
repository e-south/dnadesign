"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/pipeline.py

Materialize Eco1 RT structure authority into thread-shaped primitives.

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

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_STRUCTURE_SOURCES = _DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
_NUMBERING_POLICY = _DOCS_ROOT / "workbench/provenance/residue-numbering-policy.yaml"
_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure"
_DEFAULT_CREATED_AT = "2026-06-19T00:00:00Z"
_RESIDUE_MAP_COLUMNS = (
    "canonical_position",
    "wt_aa",
    "structure_chain_id",
    "structure_residue_id",
    "pdb_insertion_code",
    "cds_codon_index",
    "design_position",
    "mapping_status",
    "mapping_issue",
    "is_designable_initially",
    "unresolved_policy",
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
class MaterializedStructureArtifacts:
    """Paths emitted by one Eco1 structure-authority materialization pass."""

    backbone_bundle_path: Path
    residue_map_path: Path


def materialize_structure_authority(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedStructureArtifacts:
    """Materialize selected Eco1 structure authority into runtime artifacts."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = (output_root or root / _DEFAULT_OUTPUT_ROOT).expanduser()
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    structure_sources = _load_yaml(root / _STRUCTURE_SOURCES)
    numbering_policy = _load_yaml(root / _NUMBERING_POLICY)
    selected_source = _require_mapping(structure_sources.get("selected_source"), "selected_source")

    manifest_path = _resolve_source_ref(root, _require_text(selected_source, "ec86kit_manifest_ref"))
    model_path = _resolve_source_ref(root, _require_text(selected_source, "ec86kit_model_ref"))
    aa_map_path = _resolve_source_ref(root, _require_text(numbering_policy, "source_map_ref"))
    distance_profile_path = _resolve_source_ref(root, _require_text(numbering_policy, "source_distance_profile_ref"))

    _require_hash(manifest_path, _require_text(selected_source, "ec86kit_manifest_sha256"))
    _require_hash(model_path, _require_text(selected_source, "ec86kit_model_sha256"))
    _require_hash(aa_map_path, _require_text(numbering_policy, "source_map_sha256"))
    _require_hash(distance_profile_path, _require_text(numbering_policy, "source_distance_profile_sha256"))

    reference_sequence = _load_reference_sequence_from_manifest(manifest_path)
    reference_sequence_hash = "sha256:" + hashlib.sha256(reference_sequence.encode("utf-8")).hexdigest()
    if reference_sequence_hash != _require_text(selected_source, "reference_sequence_hash"):
        raise ValueError("ec86kit manifest sequence hash does not match selected structure authority")
    if reference_sequence_hash != _require_text(numbering_policy, "reference_sequence_hash"):
        raise ValueError("ec86kit manifest sequence hash does not match selected numbering policy")

    residue_map_rows = _build_residue_map_rows(
        aa_map_path=aa_map_path,
        reference_sequence=reference_sequence,
        selected_source=selected_source,
        numbering_policy=numbering_policy,
    )
    residue_map_path = out_root / "residue_map.parquet"
    _write_residue_map(
        residue_map_path,
        rows=residue_map_rows,
        selected_source=selected_source,
        numbering_policy=numbering_policy,
        upstream_hashes=_upstream_artifact_hashes(root, selected_source, numbering_policy),
        created_at=created_at,
    )

    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    backbone_bundle = _build_backbone_bundle(
        repo_root=root,
        selected_source=selected_source,
        numbering_policy=numbering_policy,
        reference_sequence=reference_sequence,
        created_at=created_at,
        residue_map_path=residue_map_path,
    )
    backbone_bundle_path.write_text(yaml.safe_dump(backbone_bundle, sort_keys=False), encoding="utf-8")

    return MaterializedStructureArtifacts(
        backbone_bundle_path=backbone_bundle_path,
        residue_map_path=residue_map_path,
    )


def _build_residue_map_rows(
    *,
    aa_map_path: Path,
    reference_sequence: str,
    selected_source: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rt_chain_id = _require_text(selected_source, "rt_chain_id")
    expected_length = int(_require_mapping(numbering_policy.get("coverage"), "coverage")["reference_sequence_length"])
    if expected_length != len(reference_sequence):
        raise ValueError("numbering policy reference_sequence_length does not match manifest sequence length")

    mapped_by_position = _load_aa_map(aa_map_path, rt_chain_id=rt_chain_id)
    unresolved_positions = set(
        _as_int_list(
            _require_mapping(numbering_policy.get("coverage"), "coverage").get("unresolved_canonical_positions")
        )
    )
    expected_mapped_count = int(_require_mapping(numbering_policy.get("coverage"), "coverage")["mapped_residue_count"])
    if len(mapped_by_position) != expected_mapped_count:
        raise ValueError("source aa map mapped residue count does not match numbering policy coverage")

    rows: list[dict[str, Any]] = []
    for position, wt_aa in enumerate(reference_sequence, start=1):
        mapped = mapped_by_position.get(position)
        if mapped is None:
            if position not in unresolved_positions:
                raise ValueError(f"canonical position {position} is absent from aa map but not listed unresolved")
            rows.append(
                {
                    "canonical_position": position,
                    "wt_aa": wt_aa,
                    "structure_chain_id": "",
                    "structure_residue_id": None,
                    "pdb_insertion_code": "",
                    "cds_codon_index": position,
                    "design_position": position,
                    "mapping_status": "unresolved_structure",
                    "mapping_issue": "not_resolved_in_structure_authority",
                    "is_designable_initially": False,
                    "unresolved_policy": "fixed",
                }
            )
            continue

        if mapped["wt_aa"] != wt_aa:
            raise ValueError(
                f"aa map residue mismatch at canonical position {position}: {mapped['wt_aa']} != reference {wt_aa}"
            )
        rows.append(
            {
                "canonical_position": position,
                "wt_aa": wt_aa,
                "structure_chain_id": mapped["structure_chain_id"],
                "structure_residue_id": mapped["structure_residue_id"],
                "pdb_insertion_code": "",
                "cds_codon_index": position,
                "design_position": position,
                "mapping_status": "mapped",
                "mapping_issue": "",
                "is_designable_initially": False,
                "unresolved_policy": "",
            }
        )
    return rows


def _load_aa_map(aa_map_path: Path, *, rt_chain_id: str) -> dict[int, dict[str, Any]]:
    with aa_map_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapped: dict[int, dict[str, Any]] = {}
    for row in rows:
        position = int(str(row["aa_pos"]).strip())
        chain_id = str(row["prot_chain"]).strip()
        if chain_id != rt_chain_id:
            raise ValueError(f"source aa map contains chain {chain_id!r}, expected {rt_chain_id!r}")
        residue_name = str(row["prot_resname"]).strip().upper()
        wt_aa = _AA3_TO_AA1.get(residue_name)
        if wt_aa is None:
            raise ValueError(f"source aa map contains unsupported residue name {residue_name!r}")
        mapped[position] = {
            "wt_aa": wt_aa,
            "structure_chain_id": chain_id,
            "structure_residue_id": int(str(row["prot_resnum"]).strip()),
        }
    return mapped


def _write_residue_map(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    selected_source: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> None:
    schema = pa.schema(
        [
            pa.field("canonical_position", pa.int32(), nullable=False),
            pa.field("wt_aa", pa.string(), nullable=False),
            pa.field("structure_chain_id", pa.string(), nullable=False),
            pa.field("structure_residue_id", pa.int32(), nullable=True),
            pa.field("pdb_insertion_code", pa.string(), nullable=False),
            pa.field("cds_codon_index", pa.int32(), nullable=False),
            pa.field("design_position", pa.int32(), nullable=False),
            pa.field("mapping_status", pa.string(), nullable=False),
            pa.field("mapping_issue", pa.string(), nullable=False),
            pa.field("is_designable_initially", pa.bool_(), nullable=False),
            pa.field("unresolved_policy", pa.string(), nullable=False),
        ],
        metadata={
            b"schema_id": b"thread.residue_map",
            b"schema_version": b"1",
            b"artifact_id": b"eco1_rt_conservative_v1.residue_map",
            b"status": b"materialized",
            b"created_by": _CREATED_BY.encode("utf-8"),
            b"created_at": created_at.encode("utf-8"),
            b"selected_structure_source_id": _require_text(selected_source, "source_id").encode("utf-8"),
            b"reference_sequence_hash": _require_text(selected_source, "reference_sequence_hash").encode("utf-8"),
            b"residue_numbering_origin": _require_text(numbering_policy, "residue_numbering_origin").encode("utf-8"),
            b"upstream_artifact_hashes": json.dumps(dict(upstream_hashes), sort_keys=True).encode("utf-8"),
        },
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _build_backbone_bundle(
    *,
    repo_root: Path,
    selected_source: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
    reference_sequence: str,
    created_at: str,
    residue_map_path: Path,
) -> dict[str, Any]:
    return {
        "schema_id": "thread.backbone_bundle",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.backbone_bundle",
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "structure_source_id": _require_text(selected_source, "source_id"),
        "source_ref": _require_text(selected_source, "source_ref"),
        "source_format": _require_text(selected_source, "structure_format"),
        "source_hash": "sha256:" + _require_text(selected_source, "ec86kit_model_sha256"),
        "source_path": _require_text(selected_source, "ec86kit_model_ref"),
        "reference_sequence_authority": _require_text(selected_source, "reference_sequence_authority"),
        "reference_sequence_hash": _require_text(selected_source, "reference_sequence_hash"),
        "reference_sequence_length": len(reference_sequence),
        "rt_chain_id": _require_text(selected_source, "rt_chain_id"),
        "selected_protomer": int(selected_source["selected_protomer"]),
        "retained_context_policy": _require_text(selected_source, "retained_context_policy"),
        "residue_numbering_origin": _require_text(numbering_policy, "residue_numbering_origin"),
        "chain_inventory": _chain_inventory(selected_source),
        "coverage": numbering_policy["coverage"],
        "upstream_artifact_hashes": _upstream_artifact_hashes(repo_root, selected_source, numbering_policy),
        "paired_artifacts": {
            "residue_map": {
                "path": _repo_relative_path(repo_root, residue_map_path),
                "sha256": "sha256:" + _sha256(residue_map_path),
            }
        },
    }


def _upstream_artifact_hashes(
    repo_root: Path,
    selected_source: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "structure_sources_yaml": "sha256:" + _sha256(repo_root / _STRUCTURE_SOURCES),
        "residue_numbering_policy_yaml": "sha256:" + _sha256(repo_root / _NUMBERING_POLICY),
        "ec86kit_manifest": "sha256:" + _require_text(selected_source, "ec86kit_manifest_sha256"),
        "ec86kit_model": "sha256:" + _require_text(selected_source, "ec86kit_model_sha256"),
        "ec86kit_aa_map": "sha256:" + _require_text(numbering_policy, "source_map_sha256"),
        "ec86kit_distance_profile": "sha256:" + _require_text(numbering_policy, "source_distance_profile_sha256"),
    }


def _chain_inventory(selected_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rt_chain_id = _require_text(selected_source, "rt_chain_id")
    retained_context_chains = set(_as_string_list(selected_source.get("retained_context_chains")))
    expected_context = {"D": "dna", "E": "rna", "F": "rna"}
    missing_context = sorted(set(expected_context) - retained_context_chains)
    if missing_context:
        raise ValueError(f"selected source is missing retained context chains {missing_context}")
    inventory = [
        {
            "chain_id": rt_chain_id,
            "molecule_type": "protein",
            "biological_role": "Eco1 reverse transcriptase",
            "thread_role": "design_backbone",
            "retention": "retained",
            "designable": False,
        }
    ]
    for chain_id in ("D", "E", "F"):
        molecule_type = expected_context[chain_id]
        inventory.append(
            {
                "chain_id": chain_id,
                "molecule_type": molecule_type,
                "biological_role": "msDNA context" if molecule_type == "dna" else "msrRNA context",
                "thread_role": "retained_context",
                "retention": "retained",
                "designable": False,
            }
        )
    for chain_id in _as_string_list(selected_source.get("removed_context_chains")):
        inventory.append(
            {
                "chain_id": chain_id,
                "molecule_type": "unknown",
                "biological_role": "excluded context from ec86kit protomer selection",
                "thread_role": "excluded_context",
                "retention": "removed",
                "designable": False,
            }
        )
    return inventory


def _load_reference_sequence_from_manifest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ec86kit manifest must be a JSON object")
    for step in payload.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        config = step.get("config")
        if isinstance(config, Mapping) and isinstance(config.get("sequence"), str):
            sequence = config["sequence"].strip()
            if sequence:
                return sequence
    raise ValueError("ec86kit manifest does not declare a reference sequence")


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


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 RT structure authority artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_structure_authority(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "backbone_bundle_path": str(result.backbone_bundle_path),
                "residue_map_path": str(result.residue_map_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
