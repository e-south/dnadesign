"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/generic_features.py

Generic adapter for pre-normalized feature/effect/display columns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..core import ContractError, Record, SchemaError, Span, reject_unknown_keys
from ..core.record import Display, Effect, Feature


def _to_mapping(value: Any, *, ctx: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise SchemaError(f"{ctx} is a string but not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} must be a mapping/dict")
    return value


def _to_list(value: Any, *, ctx: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise SchemaError(f"{ctx} is a string but not valid JSON") from exc
    if not isinstance(value, list):
        raise SchemaError(f"{ctx} must be a list")
    return value


@dataclass(frozen=True)
class GenericFeaturesAdapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def _parse_feature(self, raw: Any, idx: int) -> Feature:
        feature = _to_mapping(raw, ctx=f"features[{idx}]")
        reject_unknown_keys(feature, {"id", "kind", "span", "label", "tags", "attrs", "render"}, f"features[{idx}]")

        kind = str(feature.get("kind", "")).strip()
        if kind == "":
            raise SchemaError(f"features[{idx}].kind is required")

        span_raw = _to_mapping(feature.get("span"), ctx=f"features[{idx}].span")
        reject_unknown_keys(span_raw, {"start", "end", "strand"}, f"features[{idx}].span")
        if "start" not in span_raw or "end" not in span_raw:
            raise SchemaError(f"features[{idx}].span requires start and end")
        span = Span(start=int(span_raw["start"]), end=int(span_raw["end"]), strand=span_raw.get("strand"))

        tags_raw = feature.get("tags", [])
        if not isinstance(tags_raw, (list, tuple)):
            raise SchemaError(f"features[{idx}].tags must be a list")
        tags = tuple(str(tag) for tag in tags_raw)

        attrs = feature.get("attrs", {})
        if attrs is None:
            attrs = {}
        if not isinstance(attrs, Mapping):
            raise SchemaError(f"features[{idx}].attrs must be a mapping")

        render = feature.get("render", {})
        if render is None:
            render = {}
        if not isinstance(render, Mapping):
            raise SchemaError(f"features[{idx}].render must be a mapping")

        feature_id = feature.get("id")
        if feature_id is not None:
            feature_id = str(feature_id)

        label = feature.get("label")
        if label is not None:
            label = str(label)

        return Feature(
            id=feature_id,
            kind=kind,
            span=span,
            label=label,
            tags=tags,
            attrs=dict(attrs),
            render=dict(render),
        )

    def _parse_effect(self, raw: Any, idx: int) -> Effect:
        effect = _to_mapping(raw, ctx=f"effects[{idx}]")
        reject_unknown_keys(effect, {"kind", "target", "params", "render"}, f"effects[{idx}]")

        kind = str(effect.get("kind", "")).strip()
        if kind == "":
            raise SchemaError(f"effects[{idx}].kind is required")

        target = _to_mapping(effect.get("target"), ctx=f"effects[{idx}].target")
        params = effect.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise SchemaError(f"effects[{idx}].params must be a mapping")

        render = effect.get("render", {})
        if render is None:
            render = {}
        if not isinstance(render, Mapping):
            raise SchemaError(f"effects[{idx}].render must be a mapping")

        return Effect(kind=kind, target=dict(target), params=dict(params), render=dict(render))

    def _parse_display(self, raw: Any) -> Display:
        if raw is None:
            return Display()
        display = _to_mapping(raw, ctx="display")
        reject_unknown_keys(display, {"overlay_text", "tag_labels"}, "display")

        overlay = display.get("overlay_text")
        if overlay is not None:
            overlay = str(overlay)

        tag_labels_raw = display.get("tag_labels", {})
        if tag_labels_raw is None:
            tag_labels_raw = {}
        if not isinstance(tag_labels_raw, Mapping):
            raise SchemaError("display.tag_labels must be a mapping")
        tag_labels = {str(k): str(v) for k, v in tag_labels_raw.items()}

        return Display(overlay_text=overlay, tag_labels=tag_labels)

    def apply(self, row: dict, *, row_index: int) -> Record:
        seq_col = str(self.columns.get("sequence"))
        features_col = str(self.columns.get("features"))
        effects_col = self.columns.get("effects")
        display_col = self.columns.get("display")
        id_col = self.columns.get("id")

        sequence_raw = row.get(seq_col)
        if sequence_raw is None or str(sequence_raw).strip() == "":
            raise SchemaError(f"generic_features row missing sequence column '{seq_col}'")
        sequence = str(sequence_raw)

        record_id: str
        if id_col is None:
            record_id = f"row_{row_index}"
        else:
            rid_raw = row.get(str(id_col))
            if rid_raw is None or str(rid_raw).strip() == "":
                raise SchemaError(f"generic_features row missing id column '{id_col}'")
            record_id = str(rid_raw)

        features_raw = _to_list(row.get(features_col), ctx=f"{features_col}")
        features = tuple(self._parse_feature(item, i) for i, item in enumerate(features_raw))

        effects: tuple[Effect, ...]
        if effects_col is None:
            effects = ()
        else:
            raw_effects = row.get(str(effects_col))
            if raw_effects is None:
                effects = ()
            else:
                effects_list = _to_list(raw_effects, ctx=str(effects_col))
                effects = tuple(self._parse_effect(item, i) for i, item in enumerate(effects_list))

        if display_col is None:
            display = Display()
        else:
            display = self._parse_display(row.get(str(display_col)))

        record = Record(
            id=record_id,
            alphabet=self.alphabet,
            sequence=sequence,
            features=features,
            effects=effects,
            display=display,
            meta={"row_index": row_index, "adapter": "generic_features"},
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
