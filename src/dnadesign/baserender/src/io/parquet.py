"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/io/parquet.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..contracts import SchemaError, SkipRecord, ensure
from ..model import Annotation, SeqRecord

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
    alphabet: str = "DNA",
    ann_policy: Optional[Mapping[str, object]] = None,
) -> Iterable[SeqRecord]:
    _pa, pq = _ensure_pyarrow()
    path = Path(path)
    table = pq.read_table(path)
    _ensure_cols(table, [sequence_col, annotations_col] + ([id_col] if id_col else []))
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
        rid = str(rid_raw) if rid_raw is not None else f"row_{hash(seq) & 0xffff}"
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
            id=rid, alphabet=alphabet, sequence=seq, annotations=anns
        ).validate()
