"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/model.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, Mapping, Sequence

from .contracts import AlphabetError, BoundsError, ensure

Strand = Literal["fwd", "rev"]

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _comp(seq: str) -> str:
    return seq.translate(_DNA_COMP)


def _revcomp(seq: str) -> str:
    return _comp(seq)[::-1]


@dataclass(frozen=True)
class Annotation:
    start: int
    length: int
    strand: Strand
    label: str
    tag: str
    payload: Mapping[str, object] | None = None

    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class Guide:
    kind: str
    # x positions are in base indices; y positioning is derived by renderer
    start: int
    end: int
    label: str | None = None
    payload: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SeqRecord:
    id: str
    alphabet: str
    sequence: str
    annotations: Sequence[Annotation] = field(default_factory=tuple)
    guides: Sequence[Guide] = field(default_factory=tuple)

    def validate(self) -> "SeqRecord":
        seq = self.sequence
        ensure(
            self.alphabet in {"DNA", "RNA", "PROTEIN"},
            f"Unsupported alphabet: {self.alphabet}",
            AlphabetError,
        )
        if self.alphabet == "DNA":
            # allow ACGTN only, case-insensitive
            allowed = set("ACGTNacgtn")
            bad = set(ch for ch in seq if ch not in allowed)
            ensure(not bad, f"Non-DNA characters present: {sorted(bad)}", AlphabetError)
        for a in self.annotations:
            ensure(0 <= a.start < len(seq), f"Annotation {a} out of bounds", BoundsError)
            ensure(
                a.end() <= len(seq),
                f"Annotation {a} exceeds sequence length",
                BoundsError,
            )
            ensure(a.strand in {"fwd", "rev"}, f"Invalid strand: {a.strand}", BoundsError)

            # Fail-fast: the letters must match the correct strand at this offset.
            seg = seq[a.start : a.end()]
            expected = seg.upper() if a.strand == "fwd" else _revcomp(seg).upper()
            got = a.label.upper()
            ensure(
                expected == got,
                (
                    f"Annotation letters mismatch at {a.start}:{a.end()} on {a.strand} "
                    f"(record '{self.id}'): expected '{expected}', got '{got}'"
                ),
                BoundsError,
            )
        return self

    def with_extra(self, *, annotations: Iterable[Annotation] = (), guides: Iterable[Guide] = ()) -> "SeqRecord":
        return replace(
            self,
            annotations=tuple(list(self.annotations) + list(annotations)),
            guides=tuple(list(self.guides) + list(guides)),
        ).validate()
