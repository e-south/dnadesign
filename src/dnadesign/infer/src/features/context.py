"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/context.py

Sequence-context resolution for Evo2 promoter feature extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..errors import CapabilityError
from .contracts import PromoterFeatureBundleConfig, SequenceContextRecord

_CONSTRUCT_REQUIRED_COLUMNS = (
    "construct__context_id",
    "construct__template_id",
    "construct__anchor_start",
    "construct__anchor_end",
)


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _ordered_usr_rows(ds, *, ids: List[str]) -> List[Dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    wanted = set(str(value) for value in ids)
    for batch in ds.scan(include_overlays=True):
        payload = batch.to_pydict()
        for row_index in range(batch.num_rows):
            row = {name: payload[name][row_index] for name in payload}
            row_id = str(row["id"])
            if row_id not in wanted:
                continue
            found[row_id] = row
            if len(found) == len(wanted):
                return [found[str(row_id)] for row_id in ids]
    missing = [str(row_id) for row_id in ids if str(row_id) not in found]
    preview = ", ".join(missing[:5])
    raise CapabilityError(
        f"Unable to resolve infer feature-bundle context rows from USR overlays. Missing ids: {preview}."
    )


def _build_anchor_only_contexts(
    *,
    seqs: List[str],
    ids: Optional[List[str]],
    records: Optional[List[Dict[str, Any]]],
    bundle: PromoterFeatureBundleConfig,
) -> List[SequenceContextRecord]:
    contexts: list[SequenceContextRecord] = []
    for index, sequence in enumerate(seqs):
        row_id = str(ids[index]) if ids is not None else str(records[index].get("id", index) if records else index)
        row = records[index] if records is not None else {}
        contexts.append(
            SequenceContextRecord(
                sequence_id=row_id,
                anchor_id=row_id,
                context_id=f"{bundle.context.kind}:{row_id}",
                context_kind=bundle.context.kind,
                template_id=None,
                resolved_sequence=sequence,
                resolved_length=len(sequence),
                anchor_start=0,
                anchor_end=len(sequence),
                anchor_orientation=str(row.get("construct__anchor_orientation") or "forward"),
                construct_version=str(row.get("construct__spec_id")) if row.get("construct__spec_id") else None,
                is_wildtype=_bool_or_none(row.get("is_wildtype")),
            )
        )
    return contexts


def _build_templated_contexts(
    *,
    seqs: List[str],
    rows: List[Dict[str, Any]],
    bundle: PromoterFeatureBundleConfig,
) -> List[SequenceContextRecord]:
    contexts: list[SequenceContextRecord] = []
    for index, sequence in enumerate(seqs):
        row = rows[index]
        missing = [column for column in _CONSTRUCT_REQUIRED_COLUMNS if row.get(column) in {None, ""}]
        if missing:
            rendered = ", ".join(missing)
            raise CapabilityError(
                "Templated infer feature-bundle contexts require construct metadata columns: "
                f"{rendered}. Run construct first or use context.kind='anchor_only'."
            )
        anchor_start = int(row["construct__anchor_start"])
        anchor_end = int(row["construct__anchor_end"])
        if anchor_start < 0 or anchor_end <= anchor_start or anchor_end > len(sequence):
            raise CapabilityError(
                "Construct anchor span is invalid for infer feature-bundle pooling: "
                f"id={row.get('id')} start={anchor_start} end={anchor_end} length={len(sequence)}"
            )
        contexts.append(
            SequenceContextRecord(
                sequence_id=str(row["id"]),
                anchor_id=str(row.get("construct__anchor_id") or row.get("construct__input_id") or row["id"]),
                context_id=str(row["construct__context_id"]),
                context_kind=bundle.context.kind,
                template_id=(
                    str(row.get("construct__template_id"))
                    if row.get("construct__template_id") not in {None, ""}
                    else bundle.context.template_id
                ),
                resolved_sequence=sequence,
                resolved_length=int(row.get("construct__resolved_length") or len(sequence)),
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                anchor_orientation=(
                    str(row.get("construct__anchor_orientation"))
                    if row.get("construct__anchor_orientation") not in {None, ""}
                    else None
                ),
                construct_version=(
                    str(row.get("construct__spec_id")) if row.get("construct__spec_id") not in {None, ""} else None
                ),
                is_wildtype=_bool_or_none(row.get("is_wildtype")),
            )
        )
    return contexts


def resolve_sequence_contexts(
    *,
    seqs: List[str],
    source: str,
    ids: Optional[List[str]],
    records: Optional[List[Dict[str, Any]]],
    ds,
    bundle: PromoterFeatureBundleConfig,
) -> List[SequenceContextRecord]:
    if bundle.context.kind == "anchor_only":
        return _build_anchor_only_contexts(seqs=seqs, ids=ids, records=records, bundle=bundle)

    if source == "usr":
        if ids is None or ds is None:
            raise CapabilityError("USR-backed templated feature-bundle jobs require ids and dataset handle.")
        rows = _ordered_usr_rows(ds, ids=ids)
        return _build_templated_contexts(seqs=seqs, rows=rows, bundle=bundle)

    if source not in {"records", "pt_file"}:
        raise CapabilityError(
            "Templated feature-bundle jobs require ingest.source=usr, records, or pt_file so construct metadata "
            "is available alongside the resolved sequence."
        )
    if records is None:
        raise CapabilityError("Templated feature-bundle jobs require record payloads with construct metadata.")
    return _build_templated_contexts(seqs=seqs, rows=list(records), bundle=bundle)
