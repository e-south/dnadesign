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

from ..core import Effect, Record, SchemaError, SkipRecord, Span
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


def _promoter_display_name(name: str) -> str:
    raw = str(name).strip()
    if raw == "":
        return "promoter"
    normalized = raw.replace("_", "").replace("-", "").lower()
    if normalized.startswith("sigma"):
        suffix = normalized[len("sigma") :]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return f"σ{digits}"
        return "σ"
    return raw.replace("_", " ")


def _promoter_component_label(name: str, component: str) -> str:
    display_name = _promoter_display_name(name)
    if str(component).strip().lower() == "upstream":
        return f"{display_name} -35 site"
    if str(component).strip().lower() == "downstream":
        return f"{display_name} -10 site"
    return display_name


def _append_variant_suffix(label: str, variant_id: str | None) -> str:
    variant = str(variant_id or "").strip()
    if variant == "":
        return label
    return f"{label} ({variant})"


def _title_case_first(value: str) -> str:
    raw = str(value).strip()
    if raw == "":
        return raw
    return raw[:1].upper() + raw[1:]


@dataclass(frozen=True)
class DensegenTfbsAdapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def __post_init__(self) -> None:
        merged = dict(_DENSEGEN_POLICY_DEFAULTS)
        merged.update(dict(self.policies or {}))
        object.__setattr__(self, "policies", merged)

    def _parse_annotations(
        self,
        obj: Any,
        *,
        sequence: str,
        record_id: str,
    ) -> tuple[list[Feature], list[dict[str, Any]]]:
        if obj is None:
            return ([], [])
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
        fixed_entries: list[dict[str, Any]] = []

        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                raise SchemaError("DenseGen annotation entries must be dicts")

            part_kind = str(item.get("part_kind") or "tfbs").strip().lower()
            if part_kind == "fixed_element":
                fixed_entries.append(dict(item))
                continue
            if part_kind != "tfbs":
                raise SchemaError(f"Unknown densegen part_kind value: {part_kind!r}")

            regulator_raw = item.get("regulator")
            regulator = str(regulator_raw or "").strip()
            if regulator == "":
                raise SchemaError("DenseGen annotation dict missing non-empty key: regulator")

            sequence_raw = item.get("sequence")
            sequence_literal = str(sequence_raw or "").strip().upper()
            if sequence_literal == "":
                raise SchemaError("DenseGen annotation dict missing non-empty key: sequence")

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

            query = sequence_literal if strand == "fwd" else revcomp(sequence_literal)
            hits = _find_all(seq_u, query)

            if not hits:
                if on_missing == "skip_entry":
                    continue
                raise SchemaError(
                    "DenseGen kmer not found in sequence "
                    f"(record={record_id}, regulator={regulator}, strand={strand}, sequence={sequence_literal})"
                )

            if len(hits) == 1:
                start = hits[0]
            else:
                if offset is None:
                    if ambiguous == "error":
                        raise SchemaError(
                            "Ambiguous DenseGen annotation with no offset "
                            "(record={record_id}, regulator={regulator}, "
                            "strand={strand}, sequence={sequence_literal}, hits={hits})".format(
                                record_id=record_id,
                                regulator=regulator,
                                strand=strand,
                                sequence_literal=sequence_literal,
                                hits=hits,
                            )
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
                id=f"{record_id}:tf:{regulator}:{idx}",
                kind="kmer",
                span=Span(start=start, end=start + len(sequence_literal), strand=strand),
                label=sequence_literal,
                tags=(f"tf:{regulator}",),
                attrs={"tf": regulator, "source": "densegen_tfbs"},
                render={"priority": 10},
            )
            features.append(feature)

        return features, fixed_entries

    def _parse_promoter_entries_from_annotations(
        self,
        entries: list[dict[str, Any]],
        *,
        sequence: str,
        record_id: str,
    ) -> tuple[list[Feature], list[Effect], dict[str, str]]:
        if not entries:
            return ([], [], {})
        seq_u = sequence.upper()
        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for idx, entry in enumerate(entries):
            role = str(entry.get("role") or "").strip().lower()
            if role not in {"upstream", "downstream"}:
                raise SchemaError(f"DenseGen fixed-element entry has unknown role: {role!r}")
            name = str(entry.get("constraint_name") or "promoter").strip() or "promoter"
            placement_index_raw = entry.get("placement_index")
            placement_index = int(placement_index_raw) if placement_index_raw is not None else 0
            sequence_literal = str(entry.get("sequence") or "").strip().upper()
            if sequence_literal == "":
                raise SchemaError("DenseGen fixed-element entry is missing non-empty sequence")
            start_raw = entry.get("offset_raw")
            if start_raw is None:
                start_raw = entry.get("offset")
            if start_raw is None:
                raise SchemaError("DenseGen fixed-element entry is missing offset or offset_raw")
            start = int(start_raw)
            end = start + len(sequence_literal)
            if start < 0 or end > len(seq_u):
                raise SchemaError(
                    "DenseGen fixed-element placement is out of sequence bounds "
                    f"(record={record_id}, start={start}, end={end}, n={len(seq_u)})"
                )
            if seq_u[start:end] != sequence_literal:
                raise SchemaError(
                    "DenseGen fixed-element placement does not match sequence "
                    f"(record={record_id}, name={name}, role={role}, expected={sequence_literal})"
                )
            length_raw = entry.get("length")
            if length_raw is not None and int(length_raw) != len(sequence_literal):
                raise SchemaError(
                    "DenseGen fixed-element length does not match sequence literal length "
                    f"(record={record_id}, name={name}, role={role}, "
                    f"length={int(length_raw)}, expected={len(sequence_literal)})"
                )
            grouped.setdefault((name, placement_index), {})[role] = {
                "sequence": sequence_literal,
                "start": start,
                "end": end,
                "variant_id": str(entry.get("variant_id") or "").strip(),
                "spacer_length": entry.get("spacer_length"),
                "entry_index": int(idx),
            }

        features: list[Feature] = []
        effects: list[Effect] = []
        labels: dict[str, str] = {}
        for (name, placement_index), components in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
            if "upstream" not in components or "downstream" not in components:
                raise SchemaError(
                    "DenseGen fixed-element entries must include both upstream and downstream components "
                    f"(record={record_id}, name={name}, placement_index={placement_index})"
                )
            upstream = components["upstream"]
            downstream = components["downstream"]
            spacer_bp = int(downstream["start"]) - int(upstream["end"])
            if spacer_bp < 0:
                raise SchemaError(
                    "DenseGen fixed-element downstream offset must be >= upstream end "
                    f"(record={record_id}, name={name}, upstream_end={upstream['end']}, "
                    f"downstream_start={downstream['start']})"
                )
            upstream_spacer = upstream.get("spacer_length")
            if upstream_spacer is not None and int(upstream_spacer) != spacer_bp:
                raise SchemaError(
                    "DenseGen fixed-element upstream spacer_length does not match resolved span "
                    f"(record={record_id}, name={name}, spacer_length={int(upstream_spacer)}, resolved={spacer_bp})"
                )
            downstream_spacer = downstream.get("spacer_length")
            if downstream_spacer is not None and int(downstream_spacer) != spacer_bp:
                raise SchemaError(
                    "DenseGen fixed-element downstream spacer_length does not match resolved span "
                    f"(record={record_id}, name={name}, spacer_length={int(downstream_spacer)}, resolved={spacer_bp})"
                )

            upstream_tag = f"promoter:{name}:upstream"
            downstream_tag = f"promoter:{name}:downstream"
            labels.setdefault(
                upstream_tag,
                _append_variant_suffix(_promoter_component_label(name, "upstream"), upstream.get("variant_id")),
            )
            labels.setdefault(
                downstream_tag,
                _append_variant_suffix(_promoter_component_label(name, "downstream"), downstream.get("variant_id")),
            )

            upstream_feature_id = f"{record_id}:promoter:{name}:{placement_index}:upstream"
            downstream_feature_id = f"{record_id}:promoter:{name}:{placement_index}:downstream"
            track = int(placement_index)
            upstream_attrs = {"name": name, "component": "upstream", "source": "densegen_promoter"}
            if upstream.get("variant_id"):
                upstream_attrs["variant_id"] = upstream.get("variant_id")
            downstream_attrs = {"name": name, "component": "downstream", "source": "densegen_promoter"}
            if downstream.get("variant_id"):
                downstream_attrs["variant_id"] = downstream.get("variant_id")
            features.append(
                Feature(
                    id=upstream_feature_id,
                    kind="kmer",
                    span=Span(start=int(upstream["start"]), end=int(upstream["end"]), strand="fwd"),
                    label=str(upstream["sequence"]),
                    tags=(upstream_tag,),
                    attrs=upstream_attrs,
                    render={"priority": 8, "track": track},
                )
            )
            features.append(
                Feature(
                    id=downstream_feature_id,
                    kind="kmer",
                    span=Span(start=int(downstream["start"]), end=int(downstream["end"]), strand="fwd"),
                    label=str(downstream["sequence"]),
                    tags=(downstream_tag,),
                    attrs=downstream_attrs,
                    render={"priority": 8, "track": track},
                )
            )
            effects.append(
                Effect(
                    kind="span_link",
                    target={"from_feature_id": upstream_feature_id, "to_feature_id": downstream_feature_id},
                    params={"label": f"{spacer_bp} bp", "lane": "top"},
                    render={"priority": 8, "track": track},
                )
            )
        return features, effects, labels

    def _parse_promoter_detail(
        self,
        obj: Any,
        *,
        sequence: str,
        record_id: str,
    ) -> tuple[list[Feature], list[Effect], dict[str, str]]:
        if obj is None:
            return ([], [], {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception as exc:
                raise SchemaError("DenseGen promoter detail column is a string but not valid JSON") from exc
        if not isinstance(obj, dict):
            raise SchemaError("DenseGen promoter detail column must be a dict")

        placements = obj.get("placements", [])
        if hasattr(placements, "tolist"):
            placements = placements.tolist()
        if not isinstance(placements, (list, tuple)):
            raise SchemaError("DenseGen promoter detail placements must be a list")

        seq_u = sequence.upper()
        promoter_features: list[Feature] = []
        promoter_effects: list[Effect] = []
        promoter_labels: dict[str, str] = {}

        for placement_index, placement in enumerate(placements):
            if not isinstance(placement, dict):
                raise SchemaError("DenseGen promoter detail placements must contain dicts")

            name = str(placement.get("name") or "promoter").strip() or "promoter"
            upstream_seq = str(placement.get("upstream_seq") or "").strip().upper()
            downstream_seq = str(placement.get("downstream_seq") or "").strip().upper()
            if upstream_seq == "" or downstream_seq == "":
                raise SchemaError("DenseGen promoter placement must include non-empty upstream_seq/downstream_seq")

            upstream_start_raw = placement.get("upstream_start")
            downstream_start_raw = placement.get("downstream_start")
            if upstream_start_raw is None or downstream_start_raw is None:
                raise SchemaError("DenseGen promoter placement must include upstream_start/downstream_start")

            upstream_start = int(upstream_start_raw)
            downstream_start = int(downstream_start_raw)
            upstream_end = upstream_start + len(upstream_seq)
            downstream_end = downstream_start + len(downstream_seq)
            if upstream_start < 0 or upstream_end > len(seq_u):
                raise SchemaError(
                    "DenseGen promoter upstream placement is out of sequence bounds "
                    f"(record={record_id}, start={upstream_start}, end={upstream_end}, n={len(seq_u)})"
                )
            if downstream_start < 0 or downstream_end > len(seq_u):
                raise SchemaError(
                    "DenseGen promoter downstream placement is out of sequence bounds "
                    f"(record={record_id}, start={downstream_start}, end={downstream_end}, n={len(seq_u)})"
                )
            if seq_u[upstream_start:upstream_end] != upstream_seq:
                raise SchemaError(
                    "DenseGen promoter upstream placement does not match sequence "
                    f"(record={record_id}, name={name}, expected={upstream_seq})"
                )
            if seq_u[downstream_start:downstream_end] != downstream_seq:
                raise SchemaError(
                    "DenseGen promoter downstream placement does not match sequence "
                    f"(record={record_id}, name={name}, expected={downstream_seq})"
                )

            spacer_bp = downstream_start - upstream_end
            if spacer_bp < 0:
                raise SchemaError(
                    "DenseGen promoter downstream_start must be >= upstream_end "
                    f"(record={record_id}, name={name}, upstream_end={upstream_end}, "
                    f"downstream_start={downstream_start})"
                )
            spacer_length_raw = placement.get("spacer_length")
            if spacer_length_raw is not None:
                try:
                    spacer_length = int(spacer_length_raw)
                except Exception as exc:
                    raise SchemaError("DenseGen promoter placement spacer_length must be an integer") from exc
                if spacer_length != spacer_bp:
                    raise SchemaError(
                        "DenseGen promoter spacer_length does not match resolved span "
                        f"(record={record_id}, name={name}, spacer_length={spacer_length}, resolved={spacer_bp})"
                    )

            variant_ids = placement.get("variant_ids")
            if hasattr(variant_ids, "as_py"):
                variant_ids = variant_ids.as_py()
            if not isinstance(variant_ids, Mapping):
                variant_ids = {}
            upstream_variant_id = str(variant_ids.get("up_id") or "").strip()
            downstream_variant_id = str(variant_ids.get("down_id") or "").strip()

            upstream_tag = f"promoter:{name}:upstream"
            downstream_tag = f"promoter:{name}:downstream"
            promoter_labels.setdefault(
                upstream_tag,
                _append_variant_suffix(_promoter_component_label(name, "upstream"), upstream_variant_id),
            )
            promoter_labels.setdefault(
                downstream_tag,
                _append_variant_suffix(_promoter_component_label(name, "downstream"), downstream_variant_id),
            )
            placement_track = int(placement_index)
            upstream_feature_id = f"{record_id}:promoter:{name}:{placement_index}:upstream"
            downstream_feature_id = f"{record_id}:promoter:{name}:{placement_index}:downstream"
            upstream_attrs = {
                "name": name,
                "component": "upstream",
                "source": "densegen_promoter",
            }
            if upstream_variant_id:
                upstream_attrs["variant_id"] = upstream_variant_id
            downstream_attrs = {
                "name": name,
                "component": "downstream",
                "source": "densegen_promoter",
            }
            if downstream_variant_id:
                downstream_attrs["variant_id"] = downstream_variant_id
            promoter_features.append(
                Feature(
                    id=upstream_feature_id,
                    kind="kmer",
                    span=Span(start=upstream_start, end=upstream_end, strand="fwd"),
                    label=upstream_seq,
                    tags=(upstream_tag,),
                    attrs=upstream_attrs,
                    render={"priority": 8, "track": placement_track},
                )
            )
            promoter_features.append(
                Feature(
                    id=downstream_feature_id,
                    kind="kmer",
                    span=Span(start=downstream_start, end=downstream_end, strand="fwd"),
                    label=downstream_seq,
                    tags=(downstream_tag,),
                    attrs=downstream_attrs,
                    render={"priority": 8, "track": placement_track},
                )
            )
            promoter_effects.append(
                Effect(
                    kind="span_link",
                    target={
                        "from_feature_id": upstream_feature_id,
                        "to_feature_id": downstream_feature_id,
                    },
                    params={
                        "label": f"{spacer_bp} bp",
                        "lane": "top",
                    },
                    render={"priority": 8, "track": placement_track},
                )
            )

        return (promoter_features, promoter_effects, promoter_labels)

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

        features, fixed_entries = self._parse_annotations(row.get(ann_col), sequence=sequence, record_id=record_id)
        fixed_features, fixed_effects, fixed_labels = self._parse_promoter_entries_from_annotations(
            fixed_entries,
            sequence=sequence,
            record_id=record_id,
        )
        if fixed_features:
            features.extend(fixed_features)
        promoter_detail_col = self.columns.get("promoter_detail")
        if promoter_detail_col is None and "densegen__promoter_detail" in row:
            promoter_detail_col = "densegen__promoter_detail"
        if fixed_features and promoter_detail_col is not None and row.get(str(promoter_detail_col)) is not None:
            raise SchemaError(
                "DenseGen row provided fixed elements in annotations and promoter_detail. "
                "Provide a single source of fixed-element placements."
            )
        promoter_features, promoter_effects, promoter_labels = self._parse_promoter_detail(
            row.get(str(promoter_detail_col)) if promoter_detail_col is not None else None,
            sequence=sequence,
            record_id=record_id,
        )
        if promoter_features:
            features.extend(promoter_features)

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
                    tf_label = tag[3:]
                    tf_label = _title_case_first(tf_label)
                    tag_labels[tag] = tf_label
        tag_labels.update(promoter_labels)
        tag_labels.update(fixed_labels)

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
            effects=tuple([*promoter_effects, *fixed_effects]),
            display=Display(overlay_text=overlay_text, tag_labels=tag_labels),
            meta={"row_index": row_index, "adapter": "densegen_tfbs"},
        )
        return record.validate()
