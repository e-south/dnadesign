"""
Retron-hairpin structure thumbnail manifest for the Reader SPOP composite.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .identifiers import assay_subject_key_for_display_id, display_id_for_assay_subject, variant_sort_key
from .paths import DEFAULT_HAIRPIN_OUTPUT_DIR, relative_path, resolve_repo_root

STRUCTURE_THUMBNAIL_MANIFEST_TABLE = "retron_structure_thumbnail_manifest.parquet"


@dataclass(frozen=True, slots=True)
class RetronStructureThumbnailRow:
    assay_subject_key: str
    display_variant_id: str
    hairpin_variant_id: str | None
    construct_id: str | None
    source_precedent_id: str | None
    sequence_sha256: str | None
    sequence_length_nt: int | None
    folding_status: str
    structure_status: str
    structure_png_path: str
    composition_png_path: str
    source_bundle_path: str
    review_manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_retron_structure_thumbnail_manifest(
    *,
    repo_root: Path | None = None,
    assay_subject_keys: Sequence[str],
    hairpin_output_dir: Path | None = None,
) -> tuple[RetronStructureThumbnailRow, ...]:
    """Build a plot-facing thumbnail manifest from retron-hairpin outputs."""

    root = resolve_repo_root(repo_root)
    hairpin_root = root / (hairpin_output_dir or DEFAULT_HAIRPIN_OUTPUT_DIR)
    review_manifest_path = hairpin_root / "reviews/review_manifest.json"
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    display_by_hairpin_variant = dict(review_manifest["sequence_montage"]["review_variant_ids"])
    source_precedent_by_hairpin_variant = dict(
        review_manifest.get("benchling_genbank_import", {}).get("source_precedent_ids", {})
    )
    handoff_by_variant = _load_tsv_by_key(
        hairpin_root / "reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv",
        key="variant_id",
    )
    materialized_root = hairpin_root / "materialized"
    sequence_by_construct = _load_tsv_by_key(
        materialized_root / "manifest/indexes/sequence_index.tsv",
        key="construct_id",
    )
    hairpin_variant_by_assay_subject = {
        assay_subject_key_for_display_id(display_id): hairpin_variant_id
        for hairpin_variant_id, display_id in display_by_hairpin_variant.items()
    }
    rows: list[RetronStructureThumbnailRow] = []
    for assay_subject_key in sorted(set(assay_subject_keys), key=variant_sort_key):
        hairpin_variant_id = hairpin_variant_by_assay_subject.get(assay_subject_key)
        if hairpin_variant_id is None:
            rows.append(
                RetronStructureThumbnailRow(
                    assay_subject_key=assay_subject_key,
                    display_variant_id=display_id_for_assay_subject(assay_subject_key),
                    hairpin_variant_id=None,
                    construct_id=None,
                    source_precedent_id=None,
                    sequence_sha256=None,
                    sequence_length_nt=None,
                    folding_status="missing",
                    structure_status="missing_hairpin_materialization",
                    structure_png_path="",
                    composition_png_path="",
                    source_bundle_path="",
                    review_manifest_path=relative_path(review_manifest_path, root),
                )
            )
            continue
        handoff = handoff_by_variant[hairpin_variant_id]
        sequence = sequence_by_construct[handoff["construct_id"]]
        structure_path = materialized_root / sequence["secondary_structure_native_png"]
        composition_path = materialized_root / sequence["composition_overview_png"]
        source_bundle_path = materialized_root / sequence["artifact_bundle"]
        rows.append(
            RetronStructureThumbnailRow(
                assay_subject_key=assay_subject_key,
                display_variant_id=display_by_hairpin_variant[hairpin_variant_id],
                hairpin_variant_id=hairpin_variant_id,
                construct_id=handoff["construct_id"],
                source_precedent_id=source_precedent_by_hairpin_variant.get(
                    hairpin_variant_id,
                    display_by_hairpin_variant[hairpin_variant_id],
                ),
                sequence_sha256=sequence["sequence_sha256"],
                sequence_length_nt=int(sequence["sequence_length"]),
                folding_status=sequence["folding_status"],
                structure_status="available" if structure_path.exists() else "missing_thumbnail_path",
                structure_png_path=relative_path(structure_path, root),
                composition_png_path=relative_path(composition_path, root),
                source_bundle_path=relative_path(source_bundle_path, root),
                review_manifest_path=relative_path(review_manifest_path, root),
            )
        )
    return tuple(rows)


def write_retron_structure_thumbnail_manifest(
    rows: Sequence[RetronStructureThumbnailRow],
    *,
    output_dir: Path,
) -> str:
    """Write the thumbnail manifest table and return its path."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    path = resolved_output_dir / STRUCTURE_THUMBNAIL_MANIFEST_TABLE
    pq.write_table(_thumbnail_table(rows), path)
    return path.as_posix()


def _load_tsv_by_key(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle, delimiter="\t")}


def _thumbnail_table(rows: Sequence[RetronStructureThumbnailRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_dict() for row in rows], schema=_thumbnail_schema())


def _thumbnail_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("assay_subject_key", pa.string()),
            pa.field("display_variant_id", pa.string()),
            pa.field("hairpin_variant_id", pa.string()),
            pa.field("construct_id", pa.string()),
            pa.field("source_precedent_id", pa.string()),
            pa.field("sequence_sha256", pa.string()),
            pa.field("sequence_length_nt", pa.int64()),
            pa.field("folding_status", pa.string()),
            pa.field("structure_status", pa.string()),
            pa.field("structure_png_path", pa.string()),
            pa.field("composition_png_path", pa.string()),
            pa.field("source_bundle_path", pa.string()),
            pa.field("review_manifest_path", pa.string()),
        ]
    )
