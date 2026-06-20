"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/pipeline.py

Materialize Eco1 RT conservation source FASTA bundles from explicit local caches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.manifest import (
    write_index_manifest,
    write_profile_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_BUNDLE_ROOT,
    DEFAULT_SOURCE_CACHE_ROOT,
    DOCS_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.providers import (
    ProviderCache,
    load_provider_caches,
)

_DOCS_ROOT = DOCS_ROOT
_CONSERVATION_SOURCES = CONSERVATION_SOURCES
_DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_DEFAULT_SOURCE_CACHE_ROOT = DEFAULT_SOURCE_CACHE_ROOT
_DEFAULT_SOURCE_BUNDLE_ROOT = DEFAULT_SOURCE_BUNDLE_ROOT
_DEFAULT_CREATED_AT = DEFAULT_CREATED_AT
_RECORD_STATUSES = {"included", "excluded"}


@dataclass(frozen=True)
class MaterializedSourceSequenceBundles:
    """Paths emitted by one Eco1 source-sequence bundle materialization pass."""

    fasta_paths: dict[str, Path]
    manifest_paths: dict[str, Path]
    bundle_manifest_path: Path


def materialize_source_sequence_bundles(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_cache_root: Path | None = None,
    bundle_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedSourceSequenceBundles:
    """Materialize unaligned source FASTA bundles from explicit provider caches."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    cache_root = _resolve_path(root, source_cache_root or _DEFAULT_SOURCE_CACHE_ROOT)
    source_root = _resolve_path(root, bundle_root or _DEFAULT_SOURCE_BUNDLE_ROOT)
    source_root.mkdir(parents=True, exist_ok=True)

    sources = _load_yaml(root / _CONSERVATION_SOURCES)
    residue_map_path = out_root / "residue_map.parquet"
    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)

    target_sequence = _load_target_sequence(residue_map_path)
    target_sequence_hash = "sha256:" + hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()
    if target_sequence_hash != _require_nested_text(sources, ("target_sequence", "reference_sequence_hash")):
        raise ValueError("residue_map.parquet target sequence hash does not match conservation-sources.yaml")

    target_row_id = _require_nested_text(sources, ("alignment_policy", "target_row_id"))
    profile_ids = _required_profile_ids(sources)
    provider_ids = _required_provider_ids(sources)
    source_groups = _source_groups_by_id(sources)
    _validate_profile_provider_contract(profile_ids=profile_ids, provider_ids=provider_ids, source_groups=source_groups)

    records_payload = _load_yaml(cache_root / "source_records.yaml")
    source_records = _load_source_record_rows(records_payload, valid_profile_ids=set(profile_ids))
    provider_caches = load_provider_caches(cache_root / "provider_caches", provider_ids)

    base_upstream_hashes = {
        "conservation_sources_yaml": "sha256:" + _sha256(root / _CONSERVATION_SOURCES),
        "residue_map": "sha256:" + _sha256(residue_map_path),
        "source_records_yaml": "sha256:" + _sha256(cache_root / "source_records.yaml"),
    }
    provider_hashes = {f"provider_cache_{provider_id}": cache.sha256 for provider_id, cache in provider_caches.items()}

    fasta_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    for profile_id in profile_ids:
        profile_records = [row for row in source_records if row["profile_id"] == profile_id]
        if not profile_records:
            raise ValueError(f"source_records.yaml has no rows for profile_id {profile_id!r}")
        fasta_path, manifest_path = _write_profile_bundle(
            source_root=source_root,
            profile_id=profile_id,
            target_row_id=target_row_id,
            target_sequence=target_sequence,
            target_sequence_hash=target_sequence_hash,
            records=profile_records,
            provider_caches=provider_caches,
            provider_ids=provider_ids,
            upstream_hashes={**base_upstream_hashes, **provider_hashes},
            created_at=created_at,
        )
        fasta_paths[profile_id] = fasta_path
        manifest_paths[profile_id] = manifest_path

    bundle_manifest_path = source_root / "source_sequence_bundle_manifest.yaml"
    write_index_manifest(
        bundle_manifest_path,
        profile_ids=profile_ids,
        profile_manifests=manifest_paths,
        target_row_id=target_row_id,
        target_sequence_hash=target_sequence_hash,
        upstream_hashes={**base_upstream_hashes, **provider_hashes},
        created_at=created_at,
    )
    return MaterializedSourceSequenceBundles(
        fasta_paths=fasta_paths,
        manifest_paths=manifest_paths,
        bundle_manifest_path=bundle_manifest_path,
    )


def _write_profile_bundle(
    *,
    source_root: Path,
    profile_id: str,
    target_row_id: str,
    target_sequence: str,
    target_sequence_hash: str,
    records: Sequence[Mapping[str, str]],
    provider_caches: Mapping[str, ProviderCache],
    provider_ids: Sequence[str],
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> tuple[Path, Path]:
    fasta_records: dict[str, str] = {target_row_id: target_sequence}
    included_records: list[dict[str, str]] = []
    excluded_records: list[dict[str, str]] = []
    declared_provider_ids = set(provider_ids)

    for record in records:
        record_id = _require_text(record, "record_id")
        if record_id == target_row_id:
            raise ValueError("target row is inserted by the materializer; do not provide it in source_records.yaml")
        provider_id = _require_text(record, "provider_id")
        if provider_id not in declared_provider_ids:
            raise ValueError(f"provider_id {provider_id!r} is not declared in conservation-sources.yaml")
        status = _require_text(record, "status")
        if status not in _RECORD_STATUSES:
            raise ValueError(f"source record {record_id!r} has unsupported status {status!r}")

        if status == "excluded":
            excluded_records.append(_excluded_record(record))
            continue

        accession = _require_text(record, "accession")
        provider_cache = provider_caches[provider_id]
        sequence = provider_cache.records.get(accession)
        if sequence is None:
            raise ValueError(f"missing provider FASTA record {accession!r} for provider {provider_id!r}")
        if record_id in fasta_records:
            raise ValueError(f"duplicate output FASTA record id {record_id!r} in profile {profile_id!r}")
        fasta_records[record_id] = sequence
        included_records.append(
            {
                "record_id": record_id,
                "provider_id": provider_id,
                "accession": accession,
                "sequence_sha256": "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            }
        )

    if not included_records:
        raise ValueError(f"profile {profile_id!r} has no included source records")

    fasta_path = source_root / f"{profile_id}.source.fasta"
    _write_fasta_records(fasta_path, fasta_records)
    fasta_sha256 = "sha256:" + _sha256(fasta_path)
    manifest_path = source_root / f"{profile_id}.source_manifest.yaml"
    write_profile_manifest(
        manifest_path,
        profile_id=profile_id,
        fasta_path=fasta_path,
        fasta_sha256=fasta_sha256,
        target_row_id=target_row_id,
        target_sequence_hash=target_sequence_hash,
        included_records=included_records,
        excluded_records=excluded_records,
        upstream_hashes=upstream_hashes,
        created_at=created_at,
    )
    return fasta_path, manifest_path


def _excluded_record(record: Mapping[str, str]) -> dict[str, str]:
    exclusion_reason = _require_text(record, "exclusion_reason")
    return {
        "record_id": _require_text(record, "record_id"),
        "provider_id": _require_text(record, "provider_id"),
        "accession": _require_text(record, "accession"),
        "exclusion_reason": exclusion_reason,
    }


def _load_source_record_rows(payload: Mapping[str, Any], *, valid_profile_ids: set[str]) -> list[dict[str, str]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("source_records.yaml must declare records as a list")
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"source_records.yaml records[{index}] must be a mapping")
        profile_id = _require_text(record, "profile_id")
        if profile_id not in valid_profile_ids:
            raise ValueError(f"profile_id {profile_id!r} is not declared in conservation-sources.yaml")
        rows.append({str(key): str(value) for key, value in record.items()})
    return rows


def _load_target_sequence(residue_map_path: Path) -> str:
    rows = pq.read_table(residue_map_path).to_pylist()
    return "".join(_require_text(row, "wt_aa") for row in rows)


def _validate_profile_provider_contract(
    *,
    profile_ids: Sequence[str],
    provider_ids: Sequence[str],
    source_groups: Mapping[str, Mapping[str, Any]],
) -> None:
    for profile_id in profile_ids:
        group = _require_mapping(source_groups.get(profile_id), f"source group {profile_id}")
        group_provider_ids = group.get("provider_ids")
        if group_provider_ids != list(provider_ids):
            raise ValueError(f"source group {profile_id!r} provider_ids must match phase1_acceptance")


def _required_profile_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    profile_ids = acceptance.get("required_profile_ids")
    if not isinstance(profile_ids, list) or not all(isinstance(item, str) and item for item in profile_ids):
        raise ValueError("phase1_acceptance.required_profile_ids must be a non-empty list of strings")
    return list(profile_ids)


def _required_provider_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    provider_ids = acceptance.get("required_provider_ids")
    if not isinstance(provider_ids, list) or not all(isinstance(item, str) and item for item in provider_ids):
        raise ValueError("phase1_acceptance.required_provider_ids must be a non-empty list of strings")
    return list(provider_ids)


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


def _write_fasta_records(path: Path, records: Mapping[str, str]) -> None:
    if not records:
        raise ValueError("Cannot write empty FASTA records")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record_id, sequence in records.items():
        if not record_id:
            raise ValueError("FASTA record id must be non-empty")
        lines.append(f">{record_id}")
        lines.append(sequence.upper())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 RT conservation source-sequence bundles.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-cache-root", type=Path, default=_DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=_DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_source_sequence_bundles(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_cache_root=args.source_cache_root,
        bundle_root=args.bundle_root,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "bundle_manifest_path": str(result.bundle_manifest_path),
                "fasta_paths": {key: str(value) for key, value in result.fasta_paths.items()},
                "manifest_paths": {key: str(value) for key, value in result.manifest_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
