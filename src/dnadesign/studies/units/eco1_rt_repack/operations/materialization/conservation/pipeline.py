"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/pipeline.py

Materialize Eco1 RT conservation evidence from declared aligned protein MSAs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONSERVATION_SOURCES = _DOCS_ROOT / "workbench/provenance/conservation-sources.yaml"
_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_DEFAULT_ALIGNMENT_ROOT = _DEFAULT_OUTPUT_ROOT / "conservation_alignments"
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation"
_DEFAULT_CREATED_AT = "2026-06-20T00:00:00Z"
_CONSERVATION_PROFILE_COLUMNS = (
    "canonical_position",
    "profile_id",
    "wt_aa",
    "msa_column",
    "non_gap_count",
    "wt_count",
    "wt_frequency",
    "plurality_aa",
    "wt_is_plurality",
    "conservation_threshold",
    "min_non_gap_count",
    "passes_conservation_mask",
    "source_hash",
    "target_sequence_hash",
    "mapping_status",
    "evidence_status",
)


@dataclass(frozen=True)
class MaterializedConservationArtifacts:
    """Paths emitted by one Eco1 conservation-profile materialization pass."""

    conservation_profile_path: Path


def materialize_conservation_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    alignment_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedConservationArtifacts:
    """Materialize Tao-style per-position conservation evidence from aligned FASTA files."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    aln_root = (
        _resolve_path(root, alignment_root) if alignment_root is not None else out_root / "conservation_alignments"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sources = _load_yaml(root / _CONSERVATION_SOURCES)
    residue_map_path = out_root / "residue_map.parquet"
    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    target_sequence = "".join(_require_text(row, "wt_aa") for row in residue_rows)
    target_hash = "sha256:" + hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()
    if target_hash != _require_nested_text(sources, ("target_sequence", "reference_sequence_hash")):
        raise ValueError("residue_map.parquet target sequence hash does not match conservation-sources.yaml")

    profile_ids = _required_profile_ids(sources)
    source_groups = _source_groups_by_id(sources)
    threshold = _require_number(_require_mapping(sources.get("source_method"), "source_method"), "threshold")
    target_row_id = _require_nested_text(sources, ("alignment_policy", "target_row_id"))
    rows: list[dict[str, Any]] = []
    upstream_hashes: dict[str, str] = {
        "conservation_sources_yaml": "sha256:" + _sha256(root / _CONSERVATION_SOURCES),
        "residue_map": "sha256:" + _sha256(residue_map_path),
    }
    for profile_id in profile_ids:
        source_group = _require_mapping(source_groups.get(profile_id), f"source group {profile_id}")
        alignment_path = aln_root / f"{profile_id}.aligned.fasta"
        source_hash = "sha256:" + _sha256(alignment_path)
        upstream_hashes[f"{profile_id}_aligned_fasta"] = source_hash
        records = _load_aligned_fasta(alignment_path)
        target_aligned = records.get(target_row_id)
        if target_aligned is None:
            raise ValueError(f"aligned FASTA for {profile_id} is missing target row {target_row_id!r}")
        column_by_position = _map_target_columns(target_aligned=target_aligned, target_sequence=target_sequence)
        rows.extend(
            _score_profile_rows(
                profile_id=profile_id,
                records=records,
                column_by_position=column_by_position,
                residue_rows=residue_rows,
                threshold=threshold,
                min_non_gap_count=int(source_group["min_non_gap_count"]),
                source_hash=source_hash,
                target_sequence_hash=target_hash,
            )
        )

    conservation_profile_path = out_root / "conservation_profile.parquet"
    _write_conservation_profile(
        conservation_profile_path,
        rows=rows,
        profile_ids=profile_ids,
        target_sequence_hash=target_hash,
        upstream_hashes=upstream_hashes,
        created_at=created_at,
    )
    return MaterializedConservationArtifacts(conservation_profile_path=conservation_profile_path)


def _score_profile_rows(
    *,
    profile_id: str,
    records: Mapping[str, str],
    column_by_position: Mapping[int, int],
    residue_rows: Sequence[Mapping[str, Any]],
    threshold: float,
    min_non_gap_count: int,
    source_hash: str,
    target_sequence_hash: str,
) -> list[dict[str, Any]]:
    profile_rows: list[dict[str, Any]] = []
    sequences = list(records.values())
    for residue in residue_rows:
        position = int(residue["canonical_position"])
        wt_aa = _require_text(residue, "wt_aa")
        msa_column = int(column_by_position[position])
        column_chars = [sequence[msa_column - 1] for sequence in sequences]
        amino_acids = [char for char in column_chars if char != "-"]
        counts = Counter(amino_acids)
        non_gap_count = len(amino_acids)
        plurality_aa = _plurality_aa(counts)
        wt_count = counts.get(wt_aa, 0)
        wt_frequency = wt_count / non_gap_count if non_gap_count else 0.0
        wt_is_plurality = wt_aa == plurality_aa
        mapping_status = _require_text(residue, "mapping_status")
        evidence_usable = mapping_status == "mapped" and non_gap_count >= min_non_gap_count
        passes = evidence_usable and wt_is_plurality and wt_frequency >= threshold
        profile_rows.append(
            {
                "canonical_position": position,
                "profile_id": profile_id,
                "wt_aa": wt_aa,
                "msa_column": msa_column,
                "non_gap_count": non_gap_count,
                "wt_count": wt_count,
                "wt_frequency": wt_frequency,
                "plurality_aa": plurality_aa,
                "wt_is_plurality": wt_is_plurality,
                "conservation_threshold": threshold,
                "min_non_gap_count": min_non_gap_count,
                "passes_conservation_mask": passes,
                "source_hash": source_hash,
                "target_sequence_hash": target_sequence_hash,
                "mapping_status": mapping_status,
                "evidence_status": _evidence_status(mapping_status, non_gap_count, min_non_gap_count),
            }
        )
    return profile_rows


def _evidence_status(mapping_status: str, non_gap_count: int, min_non_gap_count: int) -> str:
    if mapping_status != "mapped":
        return "not_used_unresolved_structure"
    if non_gap_count < min_non_gap_count:
        return "low_non_gap_support"
    return "usable"


def _plurality_aa(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _load_aligned_fasta(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: dict[str, str] = {}
    current_id: str | None = None
    current_chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records[current_id] = "".join(current_chunks).upper()
            current_id = line[1:].split()[0]
            if current_id in records:
                raise ValueError(f"duplicate FASTA record id {current_id!r} in {path}")
            current_chunks = []
        elif current_id is None:
            raise ValueError(f"FASTA sequence data appears before a record id in {path}")
        else:
            current_chunks.append(line)
    if current_id is not None:
        records[current_id] = "".join(current_chunks).upper()
    if not records:
        raise ValueError(f"aligned FASTA is empty: {path}")
    alignment_lengths = {len(sequence) for sequence in records.values()}
    if len(alignment_lengths) != 1:
        raise ValueError(f"aligned FASTA records must have one alignment length: {path}")
    return records


def _map_target_columns(*, target_aligned: str, target_sequence: str) -> dict[int, int]:
    column_by_position: dict[int, int] = {}
    position = 0
    for column_index, aligned_aa in enumerate(target_aligned, start=1):
        if aligned_aa == "-":
            continue
        position += 1
        if position > len(target_sequence) or aligned_aa != target_sequence[position - 1]:
            raise ValueError("aligned FASTA target row does not match the ec86kit target row")
        column_by_position[position] = column_index
    if position != len(target_sequence):
        raise ValueError("aligned FASTA target row does not cover the full ec86kit target sequence")
    return column_by_position


def _write_conservation_profile(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    profile_ids: list[str],
    target_sequence_hash: str,
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> None:
    schema = pa.schema(
        [
            pa.field("canonical_position", pa.int32(), nullable=False),
            pa.field("profile_id", pa.string(), nullable=False),
            pa.field("wt_aa", pa.string(), nullable=False),
            pa.field("msa_column", pa.int32(), nullable=False),
            pa.field("non_gap_count", pa.int32(), nullable=False),
            pa.field("wt_count", pa.int32(), nullable=False),
            pa.field("wt_frequency", pa.float64(), nullable=False),
            pa.field("plurality_aa", pa.string(), nullable=False),
            pa.field("wt_is_plurality", pa.bool_(), nullable=False),
            pa.field("conservation_threshold", pa.float64(), nullable=False),
            pa.field("min_non_gap_count", pa.int32(), nullable=False),
            pa.field("passes_conservation_mask", pa.bool_(), nullable=False),
            pa.field("source_hash", pa.string(), nullable=False),
            pa.field("target_sequence_hash", pa.string(), nullable=False),
            pa.field("mapping_status", pa.string(), nullable=False),
            pa.field("evidence_status", pa.string(), nullable=False),
        ],
        metadata={
            b"schema_id": b"thread.conservation_profile",
            b"schema_version": b"1",
            b"artifact_id": b"eco1_rt_conservative_v1.conservation_profile",
            b"status": b"materialized",
            b"created_by": _CREATED_BY.encode("utf-8"),
            b"created_at": created_at.encode("utf-8"),
            b"profile_ids": json.dumps(profile_ids).encode("utf-8"),
            b"target_sequence_hash": target_sequence_hash.encode("utf-8"),
            b"source_contract_hash": upstream_hashes["conservation_sources_yaml"].encode("utf-8"),
            b"upstream_artifact_hashes": json.dumps(dict(upstream_hashes), sort_keys=True).encode("utf-8"),
        },
    )
    missing_columns = sorted(set(_CONSERVATION_PROFILE_COLUMNS) - {field.name for field in schema})
    if missing_columns:
        raise ValueError(f"conservation profile schema is missing required columns: {missing_columns}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _required_profile_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    profile_ids = acceptance.get("required_profile_ids")
    if not isinstance(profile_ids, list) or not all(isinstance(item, str) and item for item in profile_ids):
        raise ValueError("phase1_acceptance.required_profile_ids must be a non-empty list of strings")
    return list(profile_ids)


def _source_groups_by_id(sources: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = sources.get("source_groups")
    if not isinstance(groups, list):
        raise ValueError("conservation-sources.yaml must declare source_groups")
    grouped: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        mapping = _require_mapping(group, "source group")
        grouped[_require_text(mapping, "profile_id")] = mapping
    return grouped


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_nested_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    current: Any = payload
    for field in fields:
        current = _require_mapping(current, ".".join(fields)).get(field)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{'.'.join(fields)} must be a non-empty string")
    return current.strip()


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_number(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 RT conservation profile from aligned FASTA files.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=_DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_conservation_profile(
        repo_root=args.repo_root,
        output_root=args.output_root,
        alignment_root=args.alignment_root,
        created_at=args.created_at,
    )
    print(json.dumps({"conservation_profile_path": str(result.conservation_profile_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
