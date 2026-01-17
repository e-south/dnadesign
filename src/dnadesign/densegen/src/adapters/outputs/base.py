"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/base.py

Base output sink interfaces for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from .record import OutputRecord

DEFAULT_NAMESPACE = "densegen"


@dataclass(frozen=True)
class AlignmentDigest:
    id_count: int
    xor_hash: str


class SinkBase:
    def add(self, record: OutputRecord) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def existing_ids(self) -> set[str]:  # pragma: no cover - default
        return set()

    def alignment_digest(self) -> AlignmentDigest | None:  # pragma: no cover - default
        return None


class USRSink(SinkBase):
    def __init__(self, writer):
        self.writer = writer

    def add(self, record: OutputRecord) -> bool:
        return bool(self.writer.add(record))

    def flush(self) -> None:
        self.writer.flush()

    def existing_ids(self) -> set[str]:
        return set(self.writer.existing_ids())

    def alignment_digest(self) -> AlignmentDigest | None:
        digest = getattr(self.writer, "alignment_digest", None)
        if digest is None:
            return None
        return digest()
