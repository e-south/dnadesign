"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/io/parquet.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..contracts import SchemaError, SkipRecord, ensure
from ..model import Annotation, Guide, SeqRecord

# Local, minimal DNA utilities
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _find_all(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    i = haystack.find(needle, 0)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


def _ensure_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pa, pq
    except Exception as e:
        raise SchemaError(
            "Reading parquet requires 'pyarrow' to be installed and importable."
        ) from e


def _normalize_id_value(val: object) -> str:
    """
    Canonicalize the Parquet id column to a hyphenated string.
    - utf8/large_utf8 → str(val)
    - binary(16)/fixed_size_binary(16) → UUID hyphenated string
    - anything else → explicit error (assertive; no fallbacks)
    """
    if val is None:
        raise SchemaError(
            "id column contains null values; selection requires non-null ids."
        )
    if isinstance(val, (bytes, bytearray, memoryview)):
        b = bytes(val)
        if len(b) != 16:
            raise SchemaError(
                f"id column is binary with {len(b)} bytes; expected 16 (UUID)."
            )
        return str(uuid.UUID(bytes=b))
    return str(val)


def _prepare_isin_values(id_type, raw_ids: Sequence[str]):
    """
    Build a Python list of scalars suitable for ds.field(...).isin(values),
    with strict typing based on the Parquet id column type.
    """
    pa, _ = _ensure_pyarrow()
    from pyarrow import types as pat

    ids_clean = [str(x).strip() for x in raw_ids if str(x).strip() != ""]
    if pat.is_string(id_type) or pat.is_large_string(id_type):
        return ids_clean
    if (
        (
            pat.is_fixed_size_binary(id_type)
            and getattr(id_type, "byte_width", None) == 16
        )
        or pat.is_binary(id_type)
        or pat.is_large_binary(id_type)
    ):
        try:
            return [uuid.UUID(x).bytes for x in ids_clean]
        except Exception as e:
            raise SchemaError(
                "selection.match_on=id: CSV ids must be valid UUID strings when the Parquet id column is binary(16)."
            ) from e
    raise SchemaError(f"Unsupported Parquet id column type for selection: {id_type!r}")


def _ensure_cols(table, cols: Sequence[str]) -> None:
    names = set(table.column_names)
    missing = [c for c in cols if c not in names]
    ensure(
        not missing,
        f"Missing required columns in Parquet: {missing}. Present: {sorted(names)}",
        SchemaError,
    )


def _parse_ann_list(
    obj: Any,
    *,
    sequence: str,
    record_id: str | None = None,
    policy: Optional[Mapping[str, object]] = None,
) -> Sequence[Annotation]:
    if obj is None:
        return []
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception as e:
            raise SchemaError(
                "annotations column is a string but not valid JSON"
            ) from e
    if not isinstance(obj, (list, tuple)):
        raise SchemaError("annotations column must be a list of dicts")
    seq_u = sequence.upper()
    anns: list[Annotation] = []
    pol = dict(policy or {})
    ambiguous = str(pol.get("ambiguous", "error")).lower()
    offset_mode = str(pol.get("offset_mode", "auto")).lower()
    zero_unspec = bool(pol.get("zero_as_unspecified", True))
    if ambiguous not in {"error", "first", "last", "drop"}:
        raise SchemaError(f"Unknown ambiguous policy: {ambiguous!r}")
    if offset_mode not in {"auto", "zero_based", "one_based"}:
        raise SchemaError(f"Unknown offset_mode: {offset_mode!r}")

    for it in obj:
        if not isinstance(it, dict):
            raise SchemaError("annotation entries must be dicts")
        try:
            # Read raw fields
            offset_val = it.get("offset", None)
            offset: Optional[int]
            if offset_val is None:
                offset = None
            else:
                o = int(offset_val)
                if o == 0 and zero_unspec:
                    offset = None
                else:
                    offset = o
            orientation = str(it["orientation"])
            tf = str(it.get("tf", ""))
            tfbs = str(it["tfbs"]).upper()
        except KeyError as ke:
            raise SchemaError(f"annotation dict missing key: {ke}") from ke
        strand = {"fwd": "fwd", "rev": "rev"}.get(orientation)
        if strand is None:
            raise SchemaError(f"Unknown orientation value: {orientation!r}")

        # The label is always the TF's reading orientation.
        # To locate the site in the forward sequence, search for:
        #   - 'tfbs' when on 'fwd'
        #   - revcomp('tfbs') when on 'rev'
        query = tfbs if strand == "fwd" else _revcomp(tfbs)
        hits = _find_all(seq_u, query)

        # Assertive: letters must be present in the sequence.
        if not hits:
            rec = f" (record '{record_id}')" if record_id else ""
            on_missing = str(pol.get("on_missing_kmer", "error")).lower()
            if on_missing == "skip_entry":
                # Skip just this annotation; keep the rest of the record.
                continue
            raise SchemaError(
                "Annotation not found in sequence"
                f"{rec}: tf={tf!r}, strand={strand}, tfbs={tfbs!r}"
            )

        # If unique, that's the left-most start (forward coordinates).
        if len(hits) == 1:
            start = hits[0]
        else:
            # Ambiguity handling is explicit and policy‑driven.
            if offset is None:
                if ambiguous == "error":
                    rec = f" (record '{record_id}')" if record_id else ""
                    raise SchemaError(
                        "Ambiguous annotation (k-mer occurs multiple times) and no offset "
                        f"was provided{rec}: tf={tf!r}, strand={strand}, tfbs={tfbs!r}, hits={hits}"
                    )
                elif ambiguous == "first":
                    start = hits[0]
                elif ambiguous == "last":
                    start = hits[-1]
                elif ambiguous == "drop":
                    # New behavior: skip the entire record if any annotation is ambiguous.
                    raise SkipRecord(
                        "Ambiguous annotation with policy=drop — skipping record."
                    )
                else:  # defensive (should be unreachable due to earlier validation)
                    raise SchemaError(f"Unsupported ambiguous policy: {ambiguous!r}")
            else:
                # Try to resolve via offset; if still ambiguous, apply policy.
                candidates = set()
                if offset_mode in {"auto", "zero_based"}:
                    candidates |= {p for p in hits if p == offset}
                if offset_mode in {"auto", "one_based"}:
                    candidates |= {p for p in hits if p == (offset - 1)}
                if len(candidates) == 1:
                    start = next(iter(candidates))
                else:
                    if ambiguous == "error":
                        rec = f" (record '{record_id}')" if record_id else ""
                        raise SchemaError(
                            "Ambiguous annotation: provided offset does not resolve a unique match"
                            f"{rec} (offset={offset}, hits={hits}, offset_mode={offset_mode})."
                        )
                    elif ambiguous == "first":
                        start = hits[0]
                    elif ambiguous == "last":
                        start = hits[-1]
                    elif ambiguous == "drop":
                        # New behavior: skip the entire record.
                        raise SkipRecord(
                            "Ambiguous annotation after offset with policy=drop — skipping record."
                        )

        anns.append(
            Annotation(
                start=start,
                length=len(tfbs),
                strand=strand,
                label=tfbs,
                tag=f"tf:{tf}",
            )
        )
    return anns


def read_parquet_records(
    path: Path,
    *,
    sequence_col: str = "sequence",
    annotations_col: str = "densegen__used_tfbs_detail",
    id_col: Optional[str] = "id",
    details_col: Optional[str] = "details",
    alphabet: str = "DNA",
    ann_policy: Optional[Mapping[str, object]] = None,
) -> Iterable[SeqRecord]:
    _pa, pq = _ensure_pyarrow()
    path = Path(path)
    table = pq.read_table(path)
    _ensure_cols(table, [sequence_col, annotations_col] + ([id_col] if id_col else []))
    has_details = bool(details_col) and (str(details_col) in table.column_names)
    # Row-level gating options
    pol = dict(ann_policy or {})
    min_per = int(pol.get("min_per_record", 0))
    if bool(pol.get("require_non_empty", False)) and min_per < 1:
        min_per = 1
    require_cols = [str(c) for c in (pol.get("require_non_null_cols") or [])]

    # Iterate rows
    for row in table.to_pylist():
        # --- Require specific columns to be non-null/blank (optional)
        bad = False
        for col in require_cols:
            v = row.get(col, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                bad = True
                break
        if bad:
            continue

        # --- Sequence must be present and non-empty
        seq_raw = row.get(sequence_col, None)
        if seq_raw is None:
            continue
        seq = str(seq_raw)
        if seq.strip() == "":
            continue

        # --- If caller requires non-empty annotations and raw is missing, drop early
        ann_raw = row.get(annotations_col, None)
        if min_per >= 1 and ann_raw is None:
            continue
        rid_raw = row.get(id_col) if id_col else None
        rid = (
            _normalize_id_value(rid_raw)
            if rid_raw is not None
            else f"row_{hash(seq) & 0xffff}"
        )
        # Optional details overlay (top-left). Use only if present & non-blank.
        details_raw = row.get(details_col) if has_details else None
        details_text = str(details_raw).strip() if details_raw is not None else ""
        guides: list[Guide] = []
        if details_text:
            if id_col and rid_raw is not None:
                overlay = f"{details_text}  id={rid}"
            else:
                overlay = details_text
            guides.append(Guide(kind="overlay_label", start=0, end=0, label=overlay))
        try:
            anns = _parse_ann_list(
                ann_raw,
                sequence=seq,
                record_id=rid,
                policy=ann_policy,
            )
        except SkipRecord:
            # Do not yield this record at all
            continue
        # --- Enforce minimum annotations per record (after parsing)
        if min_per > 0 and len(anns) < min_per:
            continue
        yield SeqRecord(
            id=rid,
            alphabet=alphabet,
            sequence=seq,
            annotations=anns,
            guides=tuple(guides),
        ).validate()


def read_parquet_records_by_ids(
    path: Path,
    *,
    ids: Sequence[str],
    sequence_col: str = "sequence",
    annotations_col: str = "densegen__used_tfbs_detail",
    id_col: Optional[str] = "id",
    details_col: Optional[str] = "details",
    alphabet: str = "DNA",
    ann_policy: Optional[Mapping[str, object]] = None,
) -> Iterable[SeqRecord]:
    """
    Fast path for selection by id: read only the rows whose id is in `ids`
    using pyarrow.dataset filters. Falls back to full-table scan if dataset
    API is unavailable.
    """
    pa, pq = _ensure_pyarrow()
    from pyarrow import dataset as ds  # assertive: require dataset module

    path = Path(path)
    # Determine id column type from the Parquet schema (assertive)
    schema = pq.read_schema(path)
    if not id_col:
        raise SchemaError("Selection by id requires an 'id' column name.")
    if id_col not in schema.names:
        raise SchemaError(f"Selection by id: column '{id_col}' not found in Parquet.")
    id_type = schema.field(id_col).type
    isin_values = _prepare_isin_values(id_type, ids)
    has_details = bool(details_col) and (str(details_col) in schema.names)

    ds_ = ds.dataset(str(path), format="parquet")
    pol = dict(ann_policy or {})
    require_cols = [str(c) for c in (pol.get("require_non_null_cols") or [])]
    cols = list(
        {sequence_col, annotations_col}
        | ({id_col} if id_col else set())
        | ({details_col} if has_details else set())
        | set(require_cols)
    )
    filt = ds.field(id_col).isin(isin_values)
    table = ds_.to_table(columns=cols, filter=filt)

    # Reuse parsing/gating logic
    min_per = int(pol.get("min_per_record", 0))
    if bool(pol.get("require_non_empty", False)) and min_per < 1:
        min_per = 1

    for row in table.to_pylist():
        bad = False
        for col in require_cols:
            v = row.get(col, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                bad = True
                break
        if bad:
            continue
        seq_raw = row.get(sequence_col, None)
        if seq_raw is None:
            continue
        seq = str(seq_raw)
        if seq.strip() == "":
            continue
        ann_raw = row.get(annotations_col, None)
        if min_per >= 1 and ann_raw is None:
            continue
        rid_raw = row.get(id_col) if id_col else None
        rid = (
            _normalize_id_value(rid_raw)
            if rid_raw is not None
            else f"row_{hash(seq) & 0xffff}"
        )
        details_raw = row.get(details_col) if has_details else None
        details_text = str(details_raw).strip() if details_raw is not None else ""
        guides: list[Guide] = []
        if details_text:
            overlay = (
                f"{details_text}  id={rid}" if rid_raw is not None else details_text
            )
            guides.append(Guide(kind="overlay_label", start=0, end=0, label=overlay))
        try:
            anns = _parse_ann_list(
                ann_raw, sequence=seq, record_id=rid, policy=ann_policy
            )
        except SkipRecord:
            continue
        if min_per > 0 and len(anns) < min_per:
            continue
        yield SeqRecord(
            id=rid,
            alphabet=alphabet,
            sequence=seq,
            annotations=anns,
            guides=tuple(guides),
        ).validate()


def resolve_present_ids(
    path: Path,
    *,
    id_col: str,
    ids: Sequence[str],
) -> set[str]:
    """
    Return the set of CSV ids that actually exist in the Parquet (canonicalized).
    This does NOT parse annotations or apply gating/policies.
    """
    pa, pq = _ensure_pyarrow()
    from pyarrow import dataset as ds

    path = Path(path)
    schema = pq.read_schema(path)
    if id_col not in schema.names:
        raise SchemaError(f"Selection by id: column '{id_col}' not found in Parquet.")
    id_type = schema.field(id_col).type
    isin_values = _prepare_isin_values(id_type, ids)
    ds_ = ds.dataset(str(path), format="parquet")
    tbl = ds_.to_table(columns=[id_col], filter=ds.field(id_col).isin(isin_values))
    present: set[str] = set()
    col = tbl.column(id_col)
    for v in col.to_pylist():
        present.add(_normalize_id_value(v))
    return present
