"""
Export a USR GenBank annotation projection for BaseRender.

This helper is USR-owned on purpose: it reads datasets through the canonical
Dataset API and writes a plain file-contract parquet that BaseRender can render
without importing USR internals or inspecting overlay directories.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.contracts import SchemaError
from dnadesign.usr.src.storage.parquet import PARQUET_COMPRESSION

DEFAULT_DATASET = "usr_promoter_references"
DEFAULT_WORKSPACE = "usr_promoter_references_genbank"
EXPORT_COLUMNS = (
    "id",
    "sequence",
    "usr_label__primary",
    "seq_annot__source_file",
    "seq_annot__features",
    "derived__product_kind",
)


@dataclass(frozen=True)
class ExportResult:
    dataset: str
    output_path: str
    rows_seen: int
    rows_written: int
    skipped_without_genbank_annotations: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_usr_root() -> Path:
    return _repo_root() / "src" / "dnadesign" / "usr" / "datasets"


def _default_output_path() -> Path:
    return (
        _repo_root()
        / "src"
        / "dnadesign"
        / "baserender"
        / "workspaces"
        / DEFAULT_WORKSPACE
        / "inputs"
        / "input.parquet"
    )


def project_genbank_baserender_rows(rows: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    projected: list[dict[str, object]] = []
    skipped_without_annotations = 0
    for row in rows:
        features = row.get("seq_annot__features")
        if features is None or features == []:
            skipped_without_annotations += 1
            continue
        if not isinstance(features, list):
            raise SchemaError("seq_annot__features must be a list")
        for idx, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                raise SchemaError(f"seq_annot__features[{idx}] must be a mapping")
        projected.append({column: row.get(column) for column in EXPORT_COLUMNS})
    return projected, skipped_without_annotations


def export_genbank_baserender_projection(
    *,
    usr_root: Path,
    dataset_name: str,
    output_path: Path,
    batch_size: int = 65_536,
) -> ExportResult:
    dataset = Dataset(usr_root, dataset_name)
    if not dataset.records_path.exists():
        raise FileNotFoundError(f"USR dataset does not exist: {dataset.dir}")

    rows_seen = 0
    projected_rows: list[dict[str, object]] = []
    skipped_without_annotations = 0
    for batch in dataset.scan(
        columns=list(EXPORT_COLUMNS),
        include_overlays=("usr_label", "seq_annot", "derived"),
        batch_size=batch_size,
    ):
        rows = batch.to_pylist()
        rows_seen += len(rows)
        projected, skipped = project_genbank_baserender_rows(rows)
        projected_rows.extend(projected)
        skipped_without_annotations += skipped

    if not projected_rows:
        raise SchemaError(f"Dataset '{dataset.name}' has no rows with seq_annot__features to render.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(projected_rows)
    pq.write_table(table, output_path, compression=PARQUET_COMPRESSION)
    return ExportResult(
        dataset=dataset.name,
        output_path=str(output_path),
        rows_seen=rows_seen,
        rows_written=len(projected_rows),
        skipped_without_genbank_annotations=skipped_without_annotations,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a USR seq_annot GenBank projection for a BaseRender workspace."
    )
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=_default_output_path())
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_genbank_baserender_projection(
        usr_root=args.usr_root,
        dataset_name=args.dataset,
        output_path=args.out,
        batch_size=int(args.batch_size),
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
