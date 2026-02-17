"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/densegen_tfbs.py

DenseGen TFBS adapter converting DenseGen annotation rows into Record v1 features.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..core import Record, SchemaError, SkipRecord, Span
from ..core.record import Display, Feature, revcomp

_DENSEGEN_POLICY_DEFAULTS: dict[str, object] = {
    "ambiguous": "error",
    "offset_mode": "auto",
    "zero_as_unspecified": False,
    "on_missing_kmer": "error",
    "require_non_empty": False,
    "min_per_record": 0,
    "require_non_null_cols": (),
    "on_invalid_row": "skip",
}


def _find_all(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    i = haystack.find(needle, 0)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


@dataclass(frozen=True)
class DensegenTfbsAdapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def __post_init__(self) -> None:
        merged = dict(_DENSEGEN_POLICY_DEFAULTS)
        merged.update(dict(self.policies or {}))
        object.__setattr__(self, "policies", merged)

    def _parse_annotations(self, obj: Any, *, sequence: str, record_id: str) -> list[Feature]:
        if obj is None:
            return []
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception as exc:
                raise SchemaError("DenseGen annotations column is a string but not valid JSON") from exc
        if not isinstance(obj, (list, tuple)):
            raise SchemaError("DenseGen annotations column must be a list of dicts")

        ambiguous = str(self.policies["ambiguous"]).lower()
        offset_mode = str(self.policies["offset_mode"]).lower()
        zero_unspec = bool(self.policies["zero_as_unspecified"])
        on_missing = str(self.policies["on_missing_kmer"]).lower()

        if ambiguous not in {"error", "first", "last", "drop"}:
            raise SchemaError(f"Unknown densegen_tfbs policy ambiguous={ambiguous!r}")
        if offset_mode not in {"auto", "zero_based", "one_based"}:
            raise SchemaError(f"Unknown densegen_tfbs policy offset_mode={offset_mode!r}")
        if on_missing not in {"error", "skip_entry"}:
            raise SchemaError(f"Unknown densegen_tfbs policy on_missing_kmer={on_missing!r}")

        seq_u = sequence.upper()
        features: list[Feature] = []

        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                raise SchemaError("DenseGen annotation entries must be dicts")

            tf_raw = item.get("tf")
            tf = str(tf_raw or "").strip()
            if tf == "":
                raise SchemaError("DenseGen annotation dict missing non-empty key: tf")

            tfbs_raw = item.get("tfbs")
            tfbs = str(tfbs_raw or "").strip().upper()
            if tfbs == "":
                raise SchemaError("DenseGen annotation dict missing non-empty key: tfbs")

            orientation = str(item.get("orientation", "")).strip().lower()
            strand = {"fwd": "fwd", "rev": "rev"}.get(orientation)
            if strand is None:
                raise SchemaError(f"Unknown densegen orientation value: {orientation!r}")

            offset_val = item.get("offset")
            if offset_val is None:
                offset: int | None = None
            else:
                raw = int(offset_val)
                offset = None if (raw == 0 and zero_unspec) else raw

            query = tfbs if strand == "fwd" else revcomp(tfbs)
            hits = _find_all(seq_u, query)

            if not hits:
                if on_missing == "skip_entry":
                    continue
                raise SchemaError(
                    f"DenseGen kmer not found in sequence (record={record_id}, tf={tf}, strand={strand}, tfbs={tfbs})"
                )

            if len(hits) == 1:
                start = hits[0]
            else:
                if offset is None:
                    if ambiguous == "error":
                        raise SchemaError(
                            "Ambiguous DenseGen annotation with no offset "
                            f"(record={record_id}, tf={tf}, strand={strand}, tfbs={tfbs}, hits={hits})"
                        )
                    if ambiguous == "drop":
                        raise SkipRecord("densegen_tfbs ambiguous=drop triggered")
                    start = hits[0] if ambiguous == "first" else hits[-1]
                else:
                    candidates = set()
                    if offset_mode in {"auto", "zero_based"}:
                        candidates |= {h for h in hits if h == offset}
                    if offset_mode in {"auto", "one_based"}:
                        candidates |= {h for h in hits if h == (offset - 1)}

                    if len(candidates) == 1:
                        start = next(iter(candidates))
                    else:
                        if ambiguous == "error":
                            raise SchemaError(
                                "Ambiguous DenseGen annotation: offset did not resolve unique hit "
                                f"(record={record_id}, offset={offset}, hits={hits}, offset_mode={offset_mode})"
                            )
                        if ambiguous == "drop":
                            raise SkipRecord("densegen_tfbs ambiguous=drop triggered after offset")
                        start = hits[0] if ambiguous == "first" else hits[-1]

            feature = Feature(
                id=f"{record_id}:tf:{tf}:{idx}",
                kind="kmer",
                span=Span(start=start, end=start + len(tfbs), strand=strand),
                label=tfbs,
                tags=(f"tf:{tf}",),
                attrs={"tf": tf, "source": "densegen_tfbs"},
                render={"priority": 10},
            )
            features.append(feature)

        return features

    def apply(self, row: dict, *, row_index: int) -> Record:
        sequence_col = str(self.columns.get("sequence"))
        ann_col = str(self.columns.get("annotations"))
        id_col = self.columns.get("id")
        overlay_text_col = self.columns.get("overlay_text")

        sequence_raw = row.get(sequence_col)
        if sequence_raw is None or str(sequence_raw).strip() == "":
            on_invalid = str(self.policies["on_invalid_row"]).lower()
            if on_invalid == "skip":
                raise SkipRecord("densegen_tfbs missing sequence")
            raise SchemaError(f"DenseGen row missing sequence column '{sequence_col}'")
        sequence = str(sequence_raw)

        if id_col is not None:
            rid_raw = row.get(str(id_col))
            if rid_raw is None or str(rid_raw).strip() == "":
                on_invalid = str(self.policies["on_invalid_row"]).lower()
                if on_invalid == "skip":
                    raise SkipRecord("densegen_tfbs null id")
                raise SchemaError(f"DenseGen row missing id column '{id_col}'")
            record_id = str(rid_raw)
        else:
            record_id = f"row_{row_index}"

        # Optional gating for required non-null columns.
        req_cols = self.policies["require_non_null_cols"]
        if not isinstance(req_cols, (list, tuple)):
            raise SchemaError("densegen_tfbs policies.require_non_null_cols must be a list")
        for col in req_cols:
            val = row.get(str(col))
            if val is None or (isinstance(val, str) and val.strip() == ""):
                on_invalid = str(self.policies["on_invalid_row"]).lower()
                if on_invalid == "skip":
                    raise SkipRecord(f"densegen_tfbs missing required_non_null col={col}")
                raise SchemaError(f"DenseGen row has blank required column '{col}'")

        features = self._parse_annotations(row.get(ann_col), sequence=sequence, record_id=record_id)

        min_required = int(self.policies["min_per_record"])
        require_non_empty = bool(self.policies["require_non_empty"])
        if require_non_empty and min_required < 1:
            min_required = 1
        if min_required > 0 and len(features) < min_required:
            on_invalid = str(self.policies["on_invalid_row"]).lower()
            if on_invalid == "skip":
                raise SkipRecord("densegen_tfbs below min_per_record")
            raise SchemaError(f"DenseGen row produced {len(features)} features < min_per_record={min_required}")

        tag_labels: dict[str, str] = {}
        for feat in features:
            for tag in feat.tags:
                if tag.startswith("tf:") and tag not in tag_labels:
                    tag_labels[tag] = tag[3:]

        overlay_text = None
        if overlay_text_col is not None:
            overlay_text_raw = row.get(str(overlay_text_col))
            if overlay_text_raw is not None and str(overlay_text_raw).strip() != "":
                overlay_text = str(overlay_text_raw).strip()

        record = Record(
            id=record_id,
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=(),
            display=Display(overlay_text=overlay_text, tag_labels=tag_labels),
            meta={"row_index": row_index, "adapter": "densegen_tfbs"},
        )
        return record.validate()
