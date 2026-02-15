"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/types.py

Core sequence span and alphabet type contracts used by records and renderers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import ensure
from .errors import AlphabetError, BoundsError

Alphabet = Literal["DNA", "RNA", "PROTEIN"]
Strand = Literal["fwd", "rev"]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    strand: Strand | None = None

    def length(self) -> int:
        return self.end - self.start

    def validate_within(self, seq_len: int, alphabet: Alphabet) -> "Span":
        ensure(isinstance(self.start, int), "Span.start must be int", BoundsError)
        ensure(isinstance(self.end, int), "Span.end must be int", BoundsError)
        ensure(self.start >= 0, "Span.start must be >= 0", BoundsError)
        ensure(self.end > self.start, "Span.end must be > Span.start", BoundsError)
        ensure(self.end <= seq_len, f"Span [{self.start}, {self.end}) exceeds sequence length {seq_len}", BoundsError)
        if self.strand is not None:
            ensure(self.strand in {"fwd", "rev"}, f"Unsupported strand: {self.strand}", BoundsError)
            if alphabet != "DNA":
                ensure(
                    self.strand != "rev",
                    f"Reverse strand is only supported for DNA records (alphabet={alphabet})",
                    AlphabetError,
                )
        return self
