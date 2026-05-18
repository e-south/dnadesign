"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/compiler/catalog_bundle.py

Retron MSD catalog-bundle writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.contracts.sequence import MsdDesignCatalogV1

from ..outputs.layout import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    CATALOG_FILENAME,
    MANIFEST_DIRNAME,
    REFERENCE_DIRNAME,
    REFERENCE_INDEX_FILENAME,
    VARIANT_DIRNAME,
)
from ..outputs.manifests import (
    reference_bundle_filename,
    reference_index_row,
    write_bundle_manifest,
    write_bundle_readme,
    write_reference_index,
)
from ..outputs.output_guards import guard_catalog_output_layout
from .exceptions import RetronMsdCompilerError


def write_msd_design_catalog(catalog: MsdDesignCatalogV1, *, out_dir: str | Path) -> Path:
    return _write_msd_design_catalog(catalog, out_dir=out_dir, extra_allowed_top_level=set())


def _write_msd_design_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    out_dir: str | Path,
    extra_allowed_top_level: set[str],
) -> Path:
    root = Path(out_dir).expanduser().resolve()
    reference_filenames = [reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    guard_catalog_output_layout(
        root,
        expected_reference_filenames=set(reference_filenames),
        extra_allowed_top_level=extra_allowed_top_level,
    )
    root.mkdir(parents=True, exist_ok=True)
    references_dir = root / REFERENCE_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)

    reference_rows: list[dict[str, object]] = []
    for record, reference_filename in zip(catalog.records, reference_filenames, strict=True):
        reference_path = references_dir / reference_filename
        reference_path.write_text(record.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        reference_rows.append(reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = root / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    write_reference_index(root / REFERENCE_INDEX_FILENAME, reference_rows)
    write_bundle_manifest(root / BUNDLE_MANIFEST_FILENAME, catalog=catalog, reference_rows=reference_rows)
    write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=catalog)
    return catalog_path


def write_materialized_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    root: Path,
    bundle_manifest_dir: Path,
    catalog_dir: Path,
    indexes_dir: Path,
) -> Path:
    reference_filenames = [reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    references_dir = catalog_dir / REFERENCE_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)
    reference_rows: list[dict[str, object]] = []
    for record, reference_filename in zip(catalog.records, reference_filenames, strict=True):
        reference_path = references_dir / reference_filename
        reference_path.write_text(record.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        reference_rows.append(reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = catalog_dir / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    write_reference_index(indexes_dir / REFERENCE_INDEX_FILENAME, reference_rows)
    write_bundle_manifest(
        bundle_manifest_dir / BUNDLE_MANIFEST_FILENAME,
        catalog=catalog,
        reference_rows=reference_rows,
        catalog_path=catalog_path.relative_to(root).as_posix(),
        reference_index_path=(indexes_dir / REFERENCE_INDEX_FILENAME).relative_to(root).as_posix(),
        references_dir=references_dir.relative_to(root).as_posix(),
        grouped_dirs=[MANIFEST_DIRNAME, VARIANT_DIRNAME],
        top_level_files=[BUNDLE_README_FILENAME],
    )
    return catalog_path


__all__ = ["write_materialized_catalog", "write_msd_design_catalog"]
