"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/scripts/project_regulondb_functional_annotations.py

Materializes BioCyc regulator GO annotations for RegulonDB promoter USR data.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dnadesign.usr.src.regulondb.functional_annotations import (
    build_regulondb_go_projection,
    write_regulondb_go_projection,
)

DEFAULT_DATASET = "usr_regulondb_native_promoters"
DEFAULT_TERMS_SOURCE_ID = "biocyc_29_6_smarttable_regulator_go_terms"
DEFAULT_COVERAGE_SOURCE_ID = "biocyc_29_6_smarttable_regulator_go_coverage"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    usr_root = Path(args.usr_root)
    dataset_root = usr_root / args.dataset
    terms_path, coverage_path = _resolve_biocyc_source_paths(
        data_root=Path(args.data_root) if args.data_root else None,
        terms_source_id=args.terms_source_id,
        coverage_source_id=args.coverage_source_id,
    )
    projection = build_regulondb_go_projection(
        dataset_root=dataset_root,
        terms_path=terms_path,
        coverage_path=coverage_path,
        min_covered_regulator_fraction=float(args.min_covered_regulator_fraction),
    )
    if args.write:
        write_regulondb_go_projection(projection, dataset_root=dataset_root)
    payload = {
        "ok": True,
        "written": bool(args.write),
        "dataset_root": str(dataset_root),
        "terms_source_id": args.terms_source_id,
        "coverage_source_id": args.coverage_source_id,
        "summary": projection.summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project BioCyc SmartTable regulator GO terms onto a RegulonDB USR dataset."
    )
    parser.add_argument(
        "--usr-root",
        default="src/dnadesign/usr/datasets",
        help="USR dataset root containing usr_regulondb_native_promoters.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--data-root",
        help="dnadesign-data checkout root. Defaults to dnadesign-data's public default_data_root().",
    )
    parser.add_argument("--terms-source-id", default=DEFAULT_TERMS_SOURCE_ID)
    parser.add_argument("--coverage-source-id", default=DEFAULT_COVERAGE_SOURCE_ID)
    parser.add_argument(
        "--min-covered-regulator-fraction",
        type=float,
        default=0.95,
        help="Fail if fewer interacting regulators have at least one BioCyc GO term.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write relation sidecars. Without this flag, only the projection contract is checked.",
    )
    return parser


def _resolve_biocyc_source_paths(
    *,
    data_root: Path | None,
    terms_source_id: str,
    coverage_source_id: str,
) -> tuple[Path, Path]:
    try:
        from dnadesign_data.catalog.sources import resolve_source_record
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "BioCyc GO projection requires the public dnadesign-data catalog API. "
            "Install dnadesign-data or run with PYTHONPATH pointing at its src/ directory."
        ) from exc
    terms_record = _resolve_record(resolve_source_record, terms_source_id, data_root)
    coverage_record = _resolve_record(resolve_source_record, coverage_source_id, data_root)
    _require_biocyc_record(terms_record, terms_source_id)
    _require_biocyc_record(coverage_record, coverage_source_id)
    return Path(str(terms_record["absolute_path"])), Path(str(coverage_record["absolute_path"]))


def _resolve_record(resolve_source_record: Any, source_id: str, data_root: Path | None) -> dict[str, object]:
    return resolve_source_record(source_id, root=data_root)


def _require_biocyc_record(record: dict[str, object], source_id: str) -> None:
    if record.get("source") != "biocyc":
        raise ValueError(f"Source {source_id!r} must resolve to BioCyc, got {record.get('source')!r}")
    if record.get("file_format") != "tsv":
        raise ValueError(f"Source {source_id!r} must resolve to a TSV file")


if __name__ == "__main__":
    raise SystemExit(main())
