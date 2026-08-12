"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/usr/genbank_annotations_v1.py

Adapter for USR GenBank annotation overlays projected into Record v1 features.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...core import ContractError, Effect, Record, SchemaError, SkipRecord, Span
from ...core.record import Display, Feature

_POLICY_DEFAULTS: dict[str, object] = {
    "on_invalid_row": "error",
    "require_non_empty": True,
    "min_per_record": 1,
    "include_untyped_features": True,
    "overlay_text_template": None,
}

_TRAILING_REGULATOR_SIGN_RE = re.compile(r"(?<=[A-Za-z0-9])[+-]$")
_PROMOTER_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]*p(?:\d+)?$", re.IGNORECASE)


def _to_list(value: Any, *, ctx: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise SchemaError(f"{ctx} is a string but not valid JSON") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise SchemaError(f"{ctx} must be a list")
    return value


def _to_mapping(value: Any, *, ctx: str) -> Mapping[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} entries must be dicts")
    return value


def _clean_regulator_label(label: str) -> str:
    raw = str(label).strip()
    if raw == "":
        return "unknown"
    return _TRAILING_REGULATOR_SIGN_RE.sub("", raw).strip() or raw


def _tf_display_label(label: str) -> str:
    cleaned = _clean_regulator_label(label)
    if cleaned.lower() == "background":
        return "Neutral sites"
    return f"{cleaned} sites"


def _looks_like_tfbs_label(label: str) -> bool:
    return "tfbs" in str(label).strip().lower()


def _tf_regulator_from_label(label: str) -> str:
    cleaned = str(label).strip()
    cleaned = re.sub(r"\bpred(?:icted)?\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btfbs\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbinding\s+sites?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsites?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = _TRAILING_REGULATOR_SIGN_RE.sub("", cleaned.strip())
    cleaned = cleaned.strip(" .:_-/")
    return cleaned or _clean_regulator_label(label)


def _is_source_fragment_label(label: str, *, start: int, end: int, sequence_length: int) -> bool:
    normalized = str(label).strip().lower()
    if "upstream" not in normalized:
        return False
    if int(start) != 0:
        return False
    return int(end) >= max(1, int(sequence_length) - 1)


def _looks_like_promoter_region_label(label: str, *, start: int, end: int, sequence_length: int) -> bool:
    normalized = str(label).strip().lower()
    if not normalized:
        return False
    if "promoter" in normalized:
        return True
    if normalized.startswith("pred."):
        candidate = normalized.removeprefix("pred.").strip()
        return bool(_PROMOTER_LABEL_RE.match(candidate))
    span_len = max(0, int(end) - int(start))
    if span_len > min(140, max(1, int(sequence_length))):
        return False
    if any(token in normalized for token in (" ", "(", ")", "tfbs", "operator")):
        return False
    return bool(_PROMOTER_LABEL_RE.match(normalized))


def _looks_like_operator_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return "operator" in normalized or bool(re.match(r"^[a-z]{2,}o\d+[a-z]*$", normalized))


def _annotation_semantic(
    *,
    label: str,
    feature_type: str,
    start: int,
    end: int,
    sequence_length: int,
) -> tuple[str, str]:
    if _is_source_fragment_label(label, start=start, end=end, sequence_length=sequence_length):
        return "source_fragment", label or "Source fragment"
    if str(feature_type).strip().lower() == "promoter":
        return "promoter_region", label or "Promoter region"
    if _looks_like_operator_label(label):
        return "operator_site", label or "Operator site"
    if _looks_like_promoter_region_label(label, start=start, end=end, sequence_length=sequence_length):
        return "promoter_region", label or "Promoter region"
    return "misc_annotation", label or "Additional annotation"


def _strand(value: object) -> str:
    if value in {-1, "-1", "rev", "reverse", "reverse_complement"}:
        return "rev"
    return "fwd"


def _render_template(template: str, values: Mapping[str, object], *, ctx: str) -> str:
    normalized = {str(key): "" if value is None else str(value) for key, value in values.items()}
    try:
        return template.format_map(normalized).strip()
    except KeyError as exc:
        missing = str(exc.args[0])
        raise SchemaError(f"{ctx} references missing field: {missing}") from exc
    except (IndexError, ValueError) as exc:
        raise SchemaError(f"{ctx} is not a valid format template") from exc


def _feature_sequence(sequence: str, span: Span) -> str:
    segment = sequence[span.start : span.end]
    if span.strand == "rev":
        from ...core.record import revcomp

        return revcomp(segment).upper()
    return segment.upper()


def _feature_id(record_id: str, raw: Mapping[str, Any], idx: int) -> str:
    raw_id = str(raw.get("feature_id") or "").strip()
    if raw_id:
        return f"{record_id}:genbank:{raw_id}"
    return f"{record_id}:genbank:feature_{idx}"


@dataclass(frozen=True)
class UsrGenbankAnnotationsV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def __post_init__(self) -> None:
        merged = dict(_POLICY_DEFAULTS)
        merged.update(dict(self.policies or {}))
        object.__setattr__(self, "policies", merged)

    def _invalid(self, message: str) -> None:
        if str(self.policies["on_invalid_row"]).lower() == "skip":
            raise SkipRecord(message)
        raise SchemaError(message)

    def _parse_annotations(self, row: Mapping[str, Any]) -> list[Any]:
        annotations_col = str(self.columns.get("annotations"))
        annotations = row.get(annotations_col)
        if annotations is None:
            self._invalid(f"usr_genbank_annotations_v1 missing GenBank annotations column '{annotations_col}'")
        items = _to_list(annotations, ctx=annotations_col)
        if bool(self.policies["require_non_empty"]) and not items:
            self._invalid(f"usr_genbank_annotations_v1 missing GenBank annotations in '{annotations_col}'")
        return items

    def _annotation_span(self, item: Mapping[str, Any], *, idx: int, sequence_length: int) -> tuple[int, int, Span]:
        start_raw = item.get("start_0")
        end_raw = item.get("end_0")
        if start_raw is None or end_raw is None:
            self._invalid(f"GenBank annotation {idx} is missing start_0/end_0")
        if isinstance(start_raw, bool) or isinstance(end_raw, bool):
            self._invalid(f"GenBank annotation {idx} start_0/end_0 must be integers")
        try:
            start = int(start_raw)
            end = int(end_raw)
        except Exception as exc:
            raise SchemaError(f"GenBank annotation {idx} start_0/end_0 must be integers") from exc
        if start < 0:
            self._invalid(f"GenBank annotation {idx} start_0 must be >= 0")
        if end <= start:
            self._invalid(f"GenBank annotation {idx} end_0 must be > start_0")
        if end > sequence_length:
            self._invalid(f"GenBank annotation {idx} span [{start}, {end}) exceeds sequence length {sequence_length}")
        return start, end, Span(start=start, end=end, strand=_strand(item.get("strand")))

    def _feature_from_annotation(
        self,
        item: Mapping[str, Any],
        *,
        idx: int,
        record_id: str,
        sequence: str,
    ) -> Feature | None:
        start, end, span = self._annotation_span(item, idx=idx, sequence_length=len(sequence))
        label = str(item.get("label") or "").strip()
        role_hint = str(item.get("role_hint") or "").strip()
        feature_type = str(item.get("feature_type") or "").strip()
        confidence = str(item.get("confidence") or "").strip()

        attrs: dict[str, object] = {
            "source": "usr_genbank",
            "feature_id": str(item.get("feature_id") or ""),
            "feature_type": feature_type,
            "role_hint": role_hint,
            "confidence": confidence,
        }
        if item.get("location_raw") is not None:
            attrs["location_raw"] = str(item.get("location_raw"))

        feature_id = _feature_id(record_id, item, idx)
        priority = 9
        if role_hint == "sigma70_minus35" or label == "-35":
            attrs.update(
                {
                    "name": "sigma70_core",
                    "component": "upstream",
                    "display_label": "-35 site",
                    "source": "usr_genbank_promoter",
                }
            )
            return Feature(
                id=feature_id,
                kind="kmer",
                span=span,
                label=_feature_sequence(sequence, span),
                tags=("promoter:sigma70_core:upstream",),
                attrs=attrs,
                render={"priority": priority},
            )
        if role_hint == "sigma70_minus10" or label == "-10":
            attrs.update(
                {
                    "name": "sigma70_core",
                    "component": "downstream",
                    "display_label": "-10 site",
                    "source": "usr_genbank_promoter",
                }
            )
            return Feature(
                id=feature_id,
                kind="kmer",
                span=span,
                label=_feature_sequence(sequence, span),
                tags=("promoter:sigma70_core:downstream",),
                attrs=attrs,
                render={"priority": priority},
            )
        if role_hint == "TFBS" or _looks_like_tfbs_label(label):
            regulator = _tf_regulator_from_label(label)
            attrs.update({"tf": regulator, "display_label": _tf_display_label(regulator)})
            return Feature(
                id=feature_id,
                kind="kmer",
                span=span,
                label=_feature_sequence(sequence, span),
                tags=(f"tf:{regulator}",),
                attrs=attrs,
                render={"priority": 10},
            )

        if not bool(self.policies["include_untyped_features"]):
            return None
        semantic, display_label = _annotation_semantic(
            label=label,
            feature_type=feature_type,
            start=start,
            end=end,
            sequence_length=len(sequence),
        )
        if semantic == "source_fragment":
            return None
        return Feature(
            id=feature_id,
            kind="interval_annotation",
            span=Span(start=start, end=end, strand="fwd"),
            label=display_label or feature_type,
            tags=(f"genbank:{semantic}",),
            attrs={
                "lane": "annotation",
                "shape": "rounded_rect",
                "semantic": semantic,
                "source": "usr_genbank",
                "style_token": f"genbank:{semantic}",
            },
            render={"priority": 3},
        )

    def _promoter_pair_effect(self, features: list[Feature]) -> Effect | None:
        upstream = next((f for f in features if "promoter:sigma70_core:upstream" in f.tags), None)
        downstream = next((f for f in features if "promoter:sigma70_core:downstream" in f.tags), None)
        if upstream is None or downstream is None or upstream.id is None or downstream.id is None:
            return None
        spacer_bp = downstream.span.start - upstream.span.end
        if spacer_bp < 0:
            return None
        return Effect(
            kind="span_link",
            target={"from_feature_id": upstream.id, "to_feature_id": downstream.id},
            params={"label": f"{spacer_bp} bp", "lane": "top"},
            render={"priority": 8, "track": 0},
        )

    def _display(self, row: Mapping[str, Any], *, record_id: str) -> Display:
        overlay_text = None
        overlay_text_col = self.columns.get("overlay_text")
        if overlay_text_col is not None:
            raw = row.get(str(overlay_text_col))
            if raw is not None and str(raw).strip():
                overlay_text = str(raw).strip()
        template = self.policies.get("overlay_text_template")
        if template is not None:
            values = dict(row)
            values["id"] = record_id
            values["record_id"] = record_id
            values["overlay_text"] = overlay_text or ""
            overlay_text = (
                _render_template(
                    str(template),
                    values,
                    ctx="usr_genbank_annotations_v1 policies.overlay_text_template",
                )
                or None
            )

        video_subtitle = None
        video_subtitle_col = self.columns.get("video_subtitle")
        if video_subtitle_col is not None:
            raw = row.get(str(video_subtitle_col))
            if raw is not None and str(raw).strip():
                video_subtitle = str(raw).strip()

        return Display(
            overlay_text=overlay_text,
            video_subtitle=video_subtitle,
            tag_labels={
                "promoter:sigma70_core:upstream": "-35 site",
                "promoter:sigma70_core:downstream": "-10 site",
                "genbank:promoter_region": "Promoter region",
                "genbank:operator_site": "Operator site",
                "genbank:misc_annotation": "Additional annotation",
            },
        )

    def apply(self, row: dict, *, row_index: int) -> Record:
        sequence_col = str(self.columns.get("sequence"))
        sequence_raw = row.get(sequence_col)
        if sequence_raw is None or str(sequence_raw).strip() == "":
            self._invalid(f"usr_genbank_annotations_v1 row missing sequence column '{sequence_col}'")
        sequence = str(sequence_raw).upper()

        id_col = self.columns.get("id")
        if id_col is None:
            record_id = f"row_{row_index}"
        else:
            raw_id = row.get(str(id_col))
            if raw_id is None or str(raw_id).strip() == "":
                self._invalid(f"usr_genbank_annotations_v1 row missing id column '{id_col}'")
            record_id = str(raw_id)

        features: list[Feature] = []
        for idx, raw_item in enumerate(self._parse_annotations(row)):
            item = _to_mapping(raw_item, ctx=f"{self.columns.get('annotations')}[{idx}]")
            feature = self._feature_from_annotation(item, idx=idx, record_id=record_id, sequence=sequence)
            if feature is not None:
                features.append(feature)

        min_required = int(self.policies["min_per_record"])
        if len(features) < min_required:
            self._invalid(f"USR GenBank row produced {len(features)} features < min_per_record={min_required}")

        display = self._display(row, record_id=record_id)
        tag_labels = dict(display.tag_labels)
        for feature in features:
            for tag in feature.tags:
                if tag.startswith("tf:") and tag not in tag_labels:
                    tag_labels[tag] = _tf_display_label(tag[3:])
        display = Display(
            overlay_text=display.overlay_text,
            video_subtitle=display.video_subtitle,
            tag_labels=tag_labels,
        )

        effects = [effect for effect in [self._promoter_pair_effect(features)] if effect is not None]
        meta = {
            "row_index": row_index,
            "adapter": "usr_genbank_annotations_v1",
        }
        source_file_col = self.columns.get("source_file")
        if source_file_col is not None and row.get(str(source_file_col)) is not None:
            meta["source_file"] = str(row.get(str(source_file_col)))
            meta["source_basename"] = Path(str(row.get(str(source_file_col)))).name
        product_kind_col = self.columns.get("product_kind")
        if product_kind_col is not None and row.get(str(product_kind_col)) is not None:
            meta["product_kind"] = str(row.get(str(product_kind_col)))

        record = Record(
            id=record_id,
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=display,
            meta=meta,
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
