"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/record.py

Record v1 dataclasses and strict validation for features, effects, and display metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
class TrajectoryInset:
    x: tuple[float, ...]
    y: tuple[float, ...]
    point_index: int
    corner: str = "top_right"
    label: str | None = None

    def validate(self) -> "TrajectoryInset":
        ensure(len(self.x) >= 2, "display.trajectory_inset.x must contain at least 2 points")
        ensure(len(self.y) == len(self.x), "display.trajectory_inset.y must match display.trajectory_inset.x length")
        ensure(
            isinstance(self.point_index, int) and 0 <= self.point_index < len(self.x),
            "display.trajectory_inset.point_index is out of bounds",
        )
        ensure(
            str(self.corner).strip().lower() in {"top_left", "top_right", "bottom_left", "bottom_right"},
            "display.trajectory_inset.corner must be top_left|top_right|bottom_left|bottom_right",
        )
        if self.label is not None:
            ensure(
                isinstance(self.label, str) and self.label.strip() != "",
                "display.trajectory_inset.label must be a non-empty string when set",
            )
        for idx, value in enumerate(self.x):
            ensure(
                isinstance(value, (int, float)) and math.isfinite(float(value)),
                f"display.trajectory_inset.x[{idx}] must be finite numeric",
            )
        for idx, value in enumerate(self.y):
            ensure(
                isinstance(value, (int, float)) and math.isfinite(float(value)),
                f"display.trajectory_inset.y[{idx}] must be finite numeric",
            )
        return self

    @staticmethod
    def from_mapping(raw: Mapping[str, object]) -> "TrajectoryInset":
        x_raw = raw.get("x")
        y_raw = raw.get("y")
        point_index_raw = raw.get("point_index")
        ensure(
            isinstance(x_raw, Sequence) and not isinstance(x_raw, (str, bytes)),
            "display.trajectory_inset.x must be a list",
        )
        ensure(
            isinstance(y_raw, Sequence) and not isinstance(y_raw, (str, bytes)),
            "display.trajectory_inset.y must be a list",
        )
        ensure(point_index_raw is not None, "display.trajectory_inset.point_index is required")
        corner = str(raw.get("corner", "top_right"))
        label_raw = raw.get("label")
        label = None if label_raw is None else str(label_raw)
        return TrajectoryInset(
            x=tuple(float(v) for v in x_raw),
            y=tuple(float(v) for v in y_raw),
            point_index=int(point_index_raw),
            corner=corner,
            label=label,
        ).validate()


@dataclass(frozen=True)
class Display:
    overlay_text: str | None = None
    tag_labels: Mapping[str, str] = field(default_factory=dict)
    trajectory_inset: TrajectoryInset | None = None

    def validate(self) -> "Display":
        ensure(isinstance(self.tag_labels, Mapping), "display.tag_labels must be a mapping/dict")
        for k, v in self.tag_labels.items():
            ensure(isinstance(k, str) and k.strip() != "", "display.tag_labels keys must be non-empty strings")
            ensure(isinstance(v, str) and v.strip() != "", "display.tag_labels values must be non-empty strings")
        if self.overlay_text is not None:
            ensure(isinstance(self.overlay_text, str), "display.overlay_text must be a string or null")
        if self.trajectory_inset is not None:
            ensure(
                isinstance(self.trajectory_inset, TrajectoryInset),
                "display.trajectory_inset must be a trajectory inset object or null",
            )
            self.trajectory_inset.validate()
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
