"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/record.py

Record v1 dataclasses and strict validation for features, effects, and display metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .contracts import ensure
from .errors import AlphabetError, ContractError
from .types import Alphabet, Span

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _render_mapping(obj: Mapping[str, object] | None) -> Mapping[str, object]:
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise ContractError("render must be a mapping/dict")
    return obj


def _attrs_mapping(obj: Mapping[str, object] | None) -> Mapping[str, object]:
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise ContractError("attrs must be a mapping/dict")
    return obj


def _validate_render_keys(render: Mapping[str, object], *, ctx: str) -> None:
    allowed = {"track", "priority", "lane"}
    unknown = sorted(set(render.keys()) - allowed)
    if unknown:
        raise ContractError(f"Unknown render keys in {ctx}: {unknown}")
    if "track" in render:
        if not isinstance(render["track"], int):
            raise ContractError(f"{ctx}.track must be int")
    if "priority" in render:
        if not isinstance(render["priority"], int):
            raise ContractError(f"{ctx}.priority must be int")


@dataclass(frozen=True)
class Feature:
    id: str | None
    kind: str
    span: Span
    label: str | None
    tags: tuple[str, ...] = field(default_factory=tuple)
    attrs: Mapping[str, object] = field(default_factory=dict)
    render: Mapping[str, object] = field(default_factory=dict)

    def validate(self, *, seq_len: int, alphabet: Alphabet, ctx: str) -> "Feature":
        ensure(isinstance(self.kind, str) and self.kind.strip() != "", f"{ctx}.kind must be a non-empty string")
        self.span.validate_within(seq_len, alphabet)
        ensure(all(isinstance(tag, str) and tag.strip() != "" for tag in self.tags), f"{ctx}.tags must be strings")
        _validate_render_keys(_render_mapping(self.render), ctx=f"{ctx}.render")
        _attrs_mapping(self.attrs)
        return self


@dataclass(frozen=True)
class Effect:
    kind: str
    target: Mapping[str, object]
    params: Mapping[str, object] = field(default_factory=dict)
    render: Mapping[str, object] = field(default_factory=dict)

    def validate(self, *, ctx: str) -> "Effect":
        ensure(isinstance(self.kind, str) and self.kind.strip() != "", f"{ctx}.kind must be a non-empty string")
        ensure(isinstance(self.target, Mapping), f"{ctx}.target must be a mapping/dict")
        ensure(isinstance(self.params, Mapping), f"{ctx}.params must be a mapping/dict")
        _validate_render_keys(_render_mapping(self.render), ctx=f"{ctx}.render")
        return self


@dataclass(frozen=True)
class Display:
    overlay_text: str | None = None
    tag_labels: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> "Display":
        ensure(isinstance(self.tag_labels, Mapping), "display.tag_labels must be a mapping/dict")
        for k, v in self.tag_labels.items():
            ensure(isinstance(k, str) and k.strip() != "", "display.tag_labels keys must be non-empty strings")
            ensure(isinstance(v, str) and v.strip() != "", "display.tag_labels values must be non-empty strings")
        if self.overlay_text is not None:
            ensure(isinstance(self.overlay_text, str), "display.overlay_text must be a string or null")
        return self


@dataclass(frozen=True)
class Record:
    id: str
    alphabet: Alphabet
    sequence: str
    features: tuple[Feature, ...] = field(default_factory=tuple)
    effects: tuple[Effect, ...] = field(default_factory=tuple)
    display: Display = field(default_factory=Display)
    meta: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> "Record":
        ensure(isinstance(self.id, str) and self.id.strip() != "", "record.id must be a non-empty string")
        ensure(self.alphabet in {"DNA", "RNA", "PROTEIN"}, f"Unsupported alphabet: {self.alphabet}", AlphabetError)
        ensure(isinstance(self.sequence, str) and self.sequence != "", "record.sequence must be a non-empty string")
        self._validate_sequence_alphabet()

        seq_len = len(self.sequence)
        for i, f in enumerate(self.features):
            f.validate(seq_len=seq_len, alphabet=self.alphabet, ctx=f"record.features[{i}]")
        for i, e in enumerate(self.effects):
            e.validate(ctx=f"record.effects[{i}]")

        self.display.validate()
        ensure(isinstance(self.meta, Mapping), "record.meta must be a mapping/dict")
        return self

    def _validate_sequence_alphabet(self) -> None:
        if self.alphabet == "DNA":
            allowed = set("ACGTNacgtn")
        elif self.alphabet == "RNA":
            allowed = set("ACGUNacgun")
        else:
            allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*")
        bad = sorted({ch for ch in self.sequence if ch not in allowed})
        if bad:
            raise AlphabetError(f"Sequence contains invalid characters for {self.alphabet}: {bad}")

    def segment_for(self, span: Span) -> str:
        seg = self.sequence[span.start : span.end]
        if span.strand == "rev":
            return revcomp(seg)
        return seg
