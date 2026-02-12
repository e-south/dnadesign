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
        ensure(feat.kind == "kmer", "motif_logo target feature must be kind='kmer'", ContractError)
        reject_unknown_keys(effect.params, {"matrix"}, "motif_logo.params")
        matrix = effect.params.get("matrix")
        ensure(
            isinstance(matrix, list) and len(matrix) > 0,
            "motif_logo params.matrix must be a non-empty list",
            ContractError,
        )


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
    register_effect_contract(_SpanLinkEffectContract())
    register_effect_contract(_MotifLogoEffectContract())
