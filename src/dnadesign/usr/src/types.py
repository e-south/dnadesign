"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/types.py

Public data structures for USR API responses.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Fingerprint:
    rows: int
    cols: int
    size_bytes: int
    sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        if data.get("sha256") is None:
            data.pop("sha256", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Fingerprint":
        return cls(
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            size_bytes=int(data["size_bytes"]),
            sha256=data.get("sha256"),
        )


@dataclass(frozen=True)
class OverlayInfo:
    namespace: str
    key: Optional[str]
    created_at: Optional[str]
    path: str
    columns: List[str]
    fingerprint: Fingerprint

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint.to_dict()
        return data


@dataclass(frozen=True)
class Manifest:
    name: str
    path: str
    metadata: Dict[str, str]
    fingerprint: Fingerprint
    overlays: List[OverlayInfo]
    snapshots: List[str]
    events_count: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint.to_dict(),
            "overlays": [o.to_dict() for o in self.overlays],
            "snapshots": list(self.snapshots),
            "events": {"count": self.events_count},
        }


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    path: str
    rows: int
    columns: List[str]
    namespaces: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "rows": self.rows,
            "columns": list(self.columns),
            "namespaces": list(self.namespaces),
        }


@dataclass(frozen=True)
class AddSequencesResult:
    added: int
    skipped: int
    ids: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {"added": self.added, "skipped": self.skipped, "ids": list(self.ids)}
