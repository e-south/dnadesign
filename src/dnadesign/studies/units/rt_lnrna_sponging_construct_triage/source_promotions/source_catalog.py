"""
dnadesign-data source catalog boundary for RT-lnRNA source promotions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from .contracts import SourcePromotionContractError

CRAWFORD_REFERENCE_SOURCE_ID = "crawford_2025_retron_ncrna_ml_eco1_lnrna_msd_designs_tsv"
CRAWFORD_ABUNDANCE_SOURCE_ID = "crawford_2025_retron_ncrna_ml_eco1_ncrna_abundance_observations_tsv"
KHAN_ABUNDANCE_SOURCE_ID = "khan_2024_retron_census_abundance_prior_overlay_tsv"
KHAN_SEQUENCE_AUTHORITY_SOURCE_ID = "khan_2024_retron_census_rt_lnrna_sequence_authority_tsv"

SourceRecordResolver = Callable[[str, Path], Mapping[str, object]]


def resolve_source_table_path(
    *,
    source_id: str,
    data_root: Path,
    source_record_resolver: SourceRecordResolver | None = None,
) -> Path:
    resolver = source_record_resolver or _public_source_record_resolver()
    root = Path(data_root).resolve()
    try:
        record = resolver(source_id, root)
    except SourcePromotionContractError:
        raise
    except Exception as exc:
        raise SourcePromotionContractError(
            f"dnadesign-data source catalog failed to resolve {source_id!r}: {exc}"
        ) from exc
    if not isinstance(record, Mapping):
        raise SourcePromotionContractError(
            f"dnadesign-data source catalog returned a non-mapping record for {source_id!r}."
        )
    if not bool(record.get("available")):
        raise SourcePromotionContractError(f"dnadesign-data source {source_id!r} is not available.")
    raw_path = str(record.get("absolute_path") or "").strip()
    if not raw_path:
        raise SourcePromotionContractError(f"dnadesign-data source {source_id!r} did not provide an absolute_path.")
    path = Path(raw_path)
    if not path.is_absolute():
        raise SourcePromotionContractError(
            f"dnadesign-data source {source_id!r} returned a non-absolute path: {raw_path}"
        )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourcePromotionContractError(
            f"dnadesign-data source {source_id!r} resolved outside data_root: {path}"
        ) from exc
    if not path.is_file():
        raise SourcePromotionContractError(f"dnadesign-data source {source_id!r} is missing at {path}.")
    return path


def _public_source_record_resolver() -> SourceRecordResolver:
    try:
        from dnadesign_data.catalog.sources import resolve_source_record
    except ModuleNotFoundError as exc:
        raise SourcePromotionContractError(
            "RT-lnRNA source promotion requires the dnadesign_data package public source catalog. "
            "Install sibling dnadesign-data or pass source_record_resolver for a controlled fixture."
        ) from exc
    return resolve_source_record


__all__ = [
    "CRAWFORD_ABUNDANCE_SOURCE_ID",
    "CRAWFORD_REFERENCE_SOURCE_ID",
    "KHAN_ABUNDANCE_SOURCE_ID",
    "KHAN_SEQUENCE_AUTHORITY_SOURCE_ID",
    "SourceRecordResolver",
    "resolve_source_table_path",
]
