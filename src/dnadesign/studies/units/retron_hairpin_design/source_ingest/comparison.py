"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/comparison.py

Compare normalized MSD-region records to existing materialized sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml
from Bio import SeqIO

from .genbank_utils import sha256_text, variant_id_for_existing_row
from .models import (
    MsdRegionComparisonReport,
    MsdRegionDiscrepancy,
    NormalizedMsdRegionRecord,
)
from .record_normalization import PRIMITIVE_ROLES


def compare_records_to_existing_sources(
    records: Sequence[NormalizedMsdRegionRecord],
    *,
    existing_roots: Sequence[str | Path],
    cap_source_path: str | Path | None = None,
) -> MsdRegionComparisonReport:
    """Compare normalized MSD records to existing materialized/catalog sources."""

    by_variant = {record.variant_id: record for record in records}
    discrepancies: list[MsdRegionDiscrepancy] = []
    comparison_count = 0
    for root in existing_roots:
        for item in _iter_existing_sequence_rows(Path(root).expanduser().resolve()):
            record = by_variant.get(item["variant_id"])
            if record is None:
                continue
            comparison_count += 1
            discrepancies.extend(_compare_existing_sequence(record, item))
    if cap_source_path is not None:
        cap_comparisons, cap_discrepancies = _compare_cap_sources(by_variant, Path(cap_source_path).expanduser())
        comparison_count += cap_comparisons
        discrepancies.extend(cap_discrepancies)
    return MsdRegionComparisonReport(comparison_count=comparison_count, discrepancies=tuple(discrepancies))


def _iter_existing_sequence_rows(root: Path) -> Iterable[dict[str, object]]:
    for index_path in (
        root / "manifest/indexes/sequence_index.tsv",
        root / "materialized/manifest/indexes/sequence_index.tsv",
    ):
        if not index_path.exists():
            continue
        bundle_root = index_path.parents[2]
        display_by_variant_key = _review_display_map(root)
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                variant_id = variant_id_for_existing_row(row, display_by_variant_key)
                if variant_id is None:
                    continue
                genbank_path = bundle_root / str(row.get("genbank", ""))
                features_path = bundle_root / str(row.get("features_csv") or "")
                if not features_path.exists() and genbank_path.exists():
                    features_path = genbank_path.parent / "features.csv"
                yield {
                    "variant_id": variant_id,
                    "index_path": index_path,
                    "genbank_path": genbank_path,
                    "features_path": features_path,
                }


def _compare_existing_sequence(
    record: NormalizedMsdRegionRecord,
    item: Mapping[str, object],
) -> list[MsdRegionDiscrepancy]:
    discrepancies: list[MsdRegionDiscrepancy] = []
    genbank_path = Path(str(item["genbank_path"]))
    if genbank_path.exists():
        parsed = list(SeqIO.parse(genbank_path, "genbank"))
        if parsed:
            existing_sequence = str(parsed[0].seq).upper()
            if existing_sequence != record.msd_sequence_5to3:
                discrepancies.append(
                    MsdRegionDiscrepancy(
                        kind="sequence_mismatch",
                        variant_id=record.variant_id,
                        compared_path=genbank_path.as_posix(),
                        details={
                            "canonical_sha256": record.msd_sequence_sha256,
                            "existing_sha256": sha256_text(existing_sequence),
                            "canonical_length_nt": record.sequence_length_nt,
                            "existing_length_nt": len(existing_sequence),
                        },
                    )
                )
    features_path = Path(str(item["features_path"]))
    if features_path.exists():
        mismatches = _feature_mismatches(record, features_path)
        if mismatches:
            discrepancies.append(
                MsdRegionDiscrepancy(
                    kind="annotation_mismatch",
                    variant_id=record.variant_id,
                    compared_path=features_path.as_posix(),
                    details={"mismatches": mismatches},
                )
            )
    return discrepancies


def _feature_mismatches(record: NormalizedMsdRegionRecord, features_path: Path) -> list[dict[str, object]]:
    expected = {
        feature.role: {
            "start_0": feature.display_start_0,
            "end_0": feature.display_end_0,
            "sequence": feature.sequence_5to3.upper(),
        }
        for feature in record.features
        if feature.role in PRIMITIVE_ROLES
    }
    observed: dict[str, dict[str, object]] = {}
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            role = str(row.get("role") or "").strip()
            if role in PRIMITIVE_ROLES:
                observed[role] = {
                    "start_0": int(row["start_0"]),
                    "end_0": int(row["end_0"]),
                    "sequence": str(row.get("sequence") or "").upper(),
                }
    mismatches: list[dict[str, object]] = []
    for role, expected_payload in expected.items():
        observed_payload = observed.get(role)
        if observed_payload is not None and observed_payload != expected_payload:
            mismatches.append({"role": role, "expected": expected_payload, "observed": observed_payload})
    return mismatches


def _compare_cap_sources(
    by_variant: Mapping[str, NormalizedMsdRegionRecord],
    path: Path,
) -> tuple[int, list[MsdRegionDiscrepancy]]:
    if not path.exists():
        return 0, []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or {}
    cap_map = {
        "retron26": "C26",
        "retron43": "C43",
        "retron172": "C172",
        "retron173": "C173",
        "retron174": "C174",
        "retron175": "C175",
        "retron176": "C176",
    }
    comparisons = 0
    discrepancies: list[MsdRegionDiscrepancy] = []
    for variant_id, cap_id in cap_map.items():
        record = by_variant.get(variant_id)
        source = sources.get(cap_id) if isinstance(sources, dict) else None
        if record is None or not isinstance(source, dict):
            continue
        comparisons += 1
        details = _cap_source_details(record, source)
        if details:
            discrepancies.append(
                MsdRegionDiscrepancy(
                    kind="cap_source_mismatch",
                    variant_id=variant_id,
                    compared_path=f"{path.as_posix()}#sources.{cap_id}",
                    details=details,
                )
            )
    return comparisons, discrepancies


def _cap_source_details(record: NormalizedMsdRegionRecord, source: Mapping[str, object]) -> dict[str, object]:
    details: dict[str, object] = {}
    cap_sequence = str(source.get("sequence_5to3") or "").upper()
    observed_cap = record.primitive("snapback_foldback_geometry").sequence_5to3.upper()
    if cap_sequence and cap_sequence != observed_cap:
        details["cap_sequence"] = {"expected": observed_cap, "existing": cap_sequence}
    full_sequence = str(source.get("full_msd_sequence_5to3") or "").upper()
    if full_sequence and full_sequence != record.msd_sequence_5to3:
        details["full_msd_sequence"] = {
            "expected_sha256": record.msd_sequence_sha256,
            "existing_sha256": sha256_text(full_sequence),
        }
    return details


def _review_display_map(root: Path) -> dict[str, str]:
    path = root / "reviews/review_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sequence_montage", {}).get("review_variant_ids", {})
    return dict(mapping) if isinstance(mapping, dict) else {}


__all__ = ["compare_records_to_existing_sources"]
