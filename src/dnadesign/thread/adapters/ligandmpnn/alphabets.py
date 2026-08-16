"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/alphabets.py

Deterministic residue-alphabet sidecars for LigandMPNN.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnRequest


@dataclass(frozen=True)
class LigandMpnnResidueAlphabetSidecar:
    """Digest-bound receipt for one materialized omission JSON sidecar."""

    request_id: str
    path: Path
    sha256: str
    residue_count: int
    materialized_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.sha256.startswith("sha256:") or len(self.sha256) != 71:
            raise ValueError("sha256 must be a prefixed SHA256 digest")
        if not isinstance(self.path, Path) or self.path.suffix.lower() != ".json":
            raise ValueError("path must be a Path ending in .json")
        if self.materialized_path is not None and (
            not isinstance(self.materialized_path, Path) or self.materialized_path.suffix.lower() != ".json"
        ):
            raise ValueError("materialized_path must be a Path ending in .json")
        if isinstance(self.residue_count, bool) or self.residue_count <= 0:
            raise ValueError("residue_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.residue_alphabet_sidecar",
            "schema_version": 1,
            "request_id": self.request_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "residue_count": self.residue_count,
        }

    def validate_for(self, request: LigandMpnnRequest) -> None:
        """Fail if this receipt is not the exact sidecar for ``request``."""

        self._validate_request_binding(request)
        materialized_path = self.materialized_path or self.path
        if not materialized_path.is_file() or _digest(materialized_path.read_bytes()) != self.sha256:
            raise ValueError("residue alphabet sidecar file SHA256 does not match receipt")

    def validate_execution_file(self, request: LigandMpnnRequest) -> None:
        """Validate the final command-bound file after staging is promoted."""

        self._validate_request_binding(request)
        if not self.path.is_file() or _digest(self.path.read_bytes()) != self.sha256:
            raise ValueError("execution residue alphabet sidecar SHA256 does not match receipt")

    def _validate_request_binding(self, request: LigandMpnnRequest) -> None:
        if self.request_id != request.request_id:
            raise ValueError("residue alphabet sidecar request_id does not match request")
        expected = _canonical_bytes(request)
        if self.residue_count != len(request.residue_alphabets):
            raise ValueError("residue alphabet sidecar residue_count does not match request")
        if self.sha256 != _digest(expected):
            raise ValueError("residue alphabet sidecar digest does not match request")


def materialize_residue_alphabet_sidecar(
    request: LigandMpnnRequest,
    path: Path,
    *,
    write_path: Path | None = None,
) -> LigandMpnnResidueAlphabetSidecar:
    """Materialize omission JSON while binding the eventual execution path."""

    if not request.residue_alphabets:
        raise ValueError("request has no residue alphabets to materialize")
    if not isinstance(path, Path) or path.suffix.lower() != ".json":
        raise ValueError("path must be a Path ending in .json")
    target = write_path or path
    if not isinstance(target, Path) or target.suffix.lower() != ".json":
        raise ValueError("write_path must be a Path ending in .json")
    content = _canonical_bytes(request)
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite different residue alphabet sidecar: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return LigandMpnnResidueAlphabetSidecar(
        request_id=request.request_id,
        path=path,
        sha256=_digest(content),
        residue_count=len(request.residue_alphabets),
        materialized_path=target,
    )


def _canonical_bytes(request: LigandMpnnRequest) -> bytes:
    payload = {
        alphabet.residue.upstream_id: alphabet.omitted_amino_acids
        for alphabet in sorted(request.residue_alphabets, key=lambda item: item.residue.upstream_id)
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()
