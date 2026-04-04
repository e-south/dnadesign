"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/registry.py

Feature and effect contract registries for Record pre-render validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .contracts import ensure, reject_unknown_keys
from .errors import ContractError
from .record import Effect, Feature, Record


class FeatureKindContract(Protocol):
    kind: str

    def validate_feature(self, feature: Feature, record: Record) -> None: ...


class EffectKindContract(Protocol):
    kind: str

    def validate_effect(self, effect: Effect, record: Record) -> None: ...


@dataclass(frozen=True)
class _KmerFeatureContract:
    kind: str = "kmer"

    def validate_feature(self, feature: Feature, record: Record) -> None:
        ensure(feature.label is not None and feature.label != "", "kmer feature requires label", ContractError)
        ensure(feature.span.strand in {"fwd", "rev"}, "kmer feature requires strand=fwd|rev", ContractError)
        ensure(
            len(feature.label or "") == feature.span.length(),
            "kmer label length must equal span length",
            ContractError,
        )
        expected = record.segment_for(feature.span).upper()
        got = str(feature.label).upper()
        ensure(
            expected == got,
            (
                "kmer label must match sequence segment at span "
                f"[{feature.span.start}, {feature.span.end}) on {feature.span.strand}; "
                f"expected '{expected}', got '{got}'"
            ),
            ContractError,
        )


@dataclass(frozen=True)
class _RegulatorWindowFeatureContract:
    kind: str = "regulator_window"

    def validate_feature(self, feature: Feature, record: Record) -> None:
        ensure(
            feature.label is not None and feature.label != "",
            "regulator_window feature requires label",
            ContractError,
        )
        ensure(
            feature.span.strand in {"fwd", "rev"},
            "regulator_window feature requires strand=fwd|rev",
            ContractError,
        )
        ensure(
            len(feature.label or "") == feature.span.length(),
            "regulator_window label length must equal span length",
            ContractError,
        )


@dataclass(frozen=True)
class _IntervalAnnotationFeatureContract:
    kind: str = "interval_annotation"

    def validate_feature(self, feature: Feature, record: Record) -> None:
        reject_unknown_keys(
            feature.attrs,
            {"lane", "shape", "semantic", "intent", "style_token"},
            "interval_annotation.attrs",
        )
        lane = str(feature.attrs.get("lane", "")).strip().lower()
        ensure(lane != "", "interval_annotation.attrs.lane is required", ContractError)
        shape = str(feature.attrs.get("shape", "")).strip().lower()
        ensure(
            shape in {"rounded_rect", "band", "underline"},
            "interval_annotation.attrs.shape is invalid",
            ContractError,
        )
        semantic = str(feature.attrs.get("semantic", "")).strip()
        ensure(semantic != "", "interval_annotation.attrs.semantic is required", ContractError)


@dataclass(frozen=True)
class _SpanLinkEffectContract:
    kind: str = "span_link"

    def validate_effect(self, effect: Effect, record: Record) -> None:
        target = effect.target
        reject_unknown_keys(
            target,
            {"from_feature_id", "to_feature_id", "from_span", "to_span"},
            "span_link.target",
        )
        has_feature_refs = ("from_feature_id" in target) and ("to_feature_id" in target)
        has_spans = ("from_span" in target) and ("to_span" in target)
        ensure(
            has_feature_refs ^ has_spans,
            "span_link target must include either feature ids or spans, but not both",
            ContractError,
        )

        if has_feature_refs:
            ids = {f.id for f in record.features if f.id is not None}
            ensure(
                target["from_feature_id"] in ids,
                "span_link.from_feature_id not found in record features",
                ContractError,
            )
            ensure(
                target["to_feature_id"] in ids, "span_link.to_feature_id not found in record features", ContractError
            )

        if has_spans:
            for key in ("from_span", "to_span"):
                span_obj = target[key]
                ensure(isinstance(span_obj, Mapping), f"span_link {key} must be a mapping", ContractError)
                reject_unknown_keys(span_obj, {"start", "end", "strand"}, f"span_link.target.{key}")
                for req in ("start", "end"):
                    ensure(req in span_obj, f"span_link {key} missing '{req}'", ContractError)

        params = effect.params
        reject_unknown_keys(params, {"label", "inner_margin_bp", "lane"}, "span_link.params")


@dataclass(frozen=True)
class _MotifLogoEffectContract:
    kind: str = "motif_logo"

    def validate_effect(self, effect: Effect, record: Record) -> None:
        target = effect.target
        reject_unknown_keys(target, {"feature_id"}, "motif_logo.target")
        ensure("feature_id" in target, "motif_logo target must include feature_id", ContractError)
        feature_id = target["feature_id"]
        feat = next((f for f in record.features if f.id == feature_id), None)
        ensure(feat is not None, f"motif_logo target feature '{feature_id}' not found", ContractError)
        ensure(
            feat.kind in {"kmer", "regulator_window"},
            "motif_logo target feature must be kind='kmer' or 'regulator_window'",
            ContractError,
        )
        reject_unknown_keys(
            effect.params,
            {"matrix", "motif_ref", "render_span", "observed_sequence_5to3"},
            "motif_logo.params",
        )
        matrix = effect.params.get("matrix")
        motif_ref = effect.params.get("motif_ref")
        render_span = effect.params.get("render_span")
        expected_length = feat.span.length()
        if render_span is not None:
            ensure(isinstance(render_span, Mapping), "motif_logo params.render_span must be a mapping", ContractError)
            reject_unknown_keys(render_span, {"start", "end"}, "motif_logo.params.render_span")
            ensure(
                "start" in render_span and "end" in render_span,
                "motif_logo params.render_span must include start and end",
                ContractError,
            )
            try:
                render_start = int(render_span["start"])
                render_end = int(render_span["end"])
            except Exception as exc:
                raise ContractError("motif_logo params.render_span start/end must be integers") from exc
            ensure(
                0 <= render_start < render_end <= len(record.sequence),
                "motif_logo params.render_span must fit within the record sequence",
                ContractError,
            )
            expected_length = render_end - render_start
        ensure(
            matrix is not None or motif_ref is not None,
            "motif_logo params must include matrix and/or motif_ref",
            ContractError,
        )
        if matrix is not None:
            ensure(
                isinstance(matrix, list) and len(matrix) > 0,
                "motif_logo params.matrix must be a non-empty list",
                ContractError,
            )
            ensure(
                len(matrix) == expected_length,
                "motif_logo matrix length must match target render span length",
                ContractError,
            )
            for row in matrix:
                ensure(
                    isinstance(row, (list, tuple)) and len(row) >= 4,
                    "motif_logo matrix rows must contain at least 4 values [A,C,G,T]",
                    ContractError,
                )
        if motif_ref is not None:
            ensure(isinstance(motif_ref, Mapping), "motif_logo params.motif_ref must be a mapping", ContractError)
            reject_unknown_keys(motif_ref, {"source", "motif_id"}, "motif_logo.params.motif_ref")
            ensure(
                str(motif_ref.get("source", "")).strip() != "",
                "motif_logo params.motif_ref.source is required",
                ContractError,
            )
            ensure(
                str(motif_ref.get("motif_id", "")).strip() != "",
                "motif_logo params.motif_ref.motif_id is required",
                ContractError,
            )
        observed_sequence = effect.params.get("observed_sequence_5to3")
        if observed_sequence is not None:
            ensure(
                isinstance(observed_sequence, str) and observed_sequence != "",
                "motif_logo params.observed_sequence_5to3 must be a non-empty string",
                ContractError,
            )
            ensure(
                len(observed_sequence) == expected_length,
                "motif_logo params.observed_sequence_5to3 length must match target render span length",
                ContractError,
            )


@dataclass(frozen=True)
class _BoundaryMarkerEffectContract:
    kind: str = "boundary_marker"

    def validate_effect(self, effect: Effect, record: Record) -> None:
        target = effect.target
        reject_unknown_keys(target, {"boundary", "lane"}, "boundary_marker.target")
        ensure("boundary" in target, "boundary_marker target must include boundary", ContractError)
        ensure("lane" in target, "boundary_marker target must include lane", ContractError)
        ensure(
            isinstance(target["boundary"], int),
            "boundary_marker.target.boundary must be int",
            ContractError,
        )
        ensure(
            int(target["boundary"]) >= 0,
            "boundary_marker.target.boundary must be >= 0",
            ContractError,
        )
        ensure(
            int(target["boundary"]) <= len(record.sequence),
            "boundary_marker.target.boundary must be within sequence boundaries",
            ContractError,
        )
        ensure(
            str(target["lane"]).lower() in {"primary", "complement"},
            "boundary_marker.target.lane is invalid",
            ContractError,
        )
        reject_unknown_keys(effect.params, {"label", "semantic", "intent"}, "boundary_marker.params")


@dataclass(frozen=True)
class _PairMapEffectContract:
    kind: str = "pair_map"

    def validate_effect(self, effect: Effect, record: Record) -> None:
        target = effect.target
        reject_unknown_keys(target, {"pairs"}, "pair_map.target")
        ensure("pairs" in target, "pair_map target must include pairs", ContractError)
        pairs = target["pairs"]
        ensure(
            isinstance(pairs, list) and len(pairs) > 0,
            "pair_map.target.pairs must be a non-empty list",
            ContractError,
        )
        for index, pair in enumerate(pairs):
            ensure(isinstance(pair, Mapping), f"pair_map.target.pairs[{index}] must be a mapping", ContractError)
            reject_unknown_keys(pair, {"left_index", "right_index"}, f"pair_map.target.pairs[{index}]")
            ensure(
                "left_index" in pair and "right_index" in pair,
                f"pair_map.target.pairs[{index}] must include left_index and right_index",
                ContractError,
            )
        reject_unknown_keys(effect.params, {"semantic"}, "pair_map.params")


_FEATURE_CONTRACTS: dict[str, FeatureKindContract] = {}
_EFFECT_CONTRACTS: dict[str, EffectKindContract] = {}


def clear_feature_effect_contracts() -> None:
    _FEATURE_CONTRACTS.clear()
    _EFFECT_CONTRACTS.clear()


def register_feature_contract(contract: FeatureKindContract) -> None:
    _FEATURE_CONTRACTS[contract.kind] = contract


def register_effect_contract(contract: EffectKindContract) -> None:
    _EFFECT_CONTRACTS[contract.kind] = contract


def get_feature_contract(kind: str) -> FeatureKindContract:
    contract = _FEATURE_CONTRACTS.get(kind)
    if contract is None:
        raise ContractError(f"Unknown feature kind: {kind}")
    return contract


def get_effect_contract(kind: str) -> EffectKindContract:
    contract = _EFFECT_CONTRACTS.get(kind)
    if contract is None:
        raise ContractError(f"Unknown effect kind: {kind}")
    return contract


def validate_record_kinds(record: Record) -> None:
    for feature in record.features:
        get_feature_contract(feature.kind).validate_feature(feature, record)
    for effect in record.effects:
        get_effect_contract(effect.kind).validate_effect(effect, record)


def register_builtin_contracts() -> None:
    register_feature_contract(_KmerFeatureContract())
    register_feature_contract(_RegulatorWindowFeatureContract())
    register_feature_contract(_IntervalAnnotationFeatureContract())
    register_effect_contract(_SpanLinkEffectContract())
    register_effect_contract(_MotifLogoEffectContract())
    register_effect_contract(_BoundaryMarkerEffectContract())
    register_effect_contract(_PairMapEffectContract())
